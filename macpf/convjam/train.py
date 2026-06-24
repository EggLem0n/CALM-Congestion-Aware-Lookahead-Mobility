"""Train the congestion-prediction model (ConvLSTM).

Input  : per-episode shards from `macpf.generate_heatmap` (x: (T,5,H,W), y: (T,1,H,W)).
Output : a trained model checkpoint under models/.

The model is an encoder-forecaster ConvLSTM (see `macpf.convjam.convlstm`) trained to
predict the *future* congestion cost map from an observed window of frames. Data is
sliced into forecasting windows by `macpf.convjam.dataset.CongestionWindowDataset`.
"""
from __future__ import annotations

import random
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from macpf.config import DATA_DIR, FIGURES_DIR, MODELS_DIR
from macpf.features import iter_episode_files

app = typer.Typer()


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning(
            "CUDA를 사용할 수 없어 CPU로 학습합니다. GPU로 돌리려면 CUDA 빌드 torch가 필요합니다: "
            "pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logger.error(
            "CUDA를 사용할 수 없습니다 (torch.cuda.is_available()=False). 설치된 torch가 CPU 전용 "
            "빌드일 가능성이 큽니다. CUDA 빌드로 재설치하세요:\n"
            "  pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            "(CPU로 강제하려면 --device cpu)"
        )
        raise typer.Exit(code=1)
    return device


def _make_grad_scaler(torch, enabled: bool):
    """GradScaler across torch versions (torch.amp is preferred, torch.cuda.amp legacy)."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _epoch_metrics(sum_sq_res: float, count: int, sum_y: float, sum_y2: float) -> tuple[float, float]:
    """Collapse an epoch's streaming sums into (mse, accuracy).

    Accuracy is the coefficient of determination R² = 1 - SS_res/SS_tot: 1.0 is a
    perfect fit, 0.0 is "no better than predicting the mean", and it can go negative
    early in training. R² is used (rather than a thresholded pixel match) because the
    congestion label is sparse / mostly zero, where an all-zero prediction would score
    a deceptively high pixel accuracy but a (correct) R² near 0. Swap THIS function to
    redefine "accuracy" (e.g. PSNR = -10*log10(mse), or a |pred-target|<tol pixel ratio).
    """
    n = max(1, count)
    mse = sum_sq_res / n
    ss_tot = sum_y2 - (sum_y * sum_y) / n  # = Σ(y - ȳ)²
    r2 = 1.0 - sum_sq_res / ss_tot if ss_tot > 1e-12 else 0.0
    return mse, r2


def _setup_curves(enabled: bool, acc_label: str = "R²"):
    """Create the live 1×2 figure: [ loss (MSE) | accuracy ], each with a train and a
    test line. Returns a state dict, or None when disabled."""
    if not enabled:
        return None
    import matplotlib
    import matplotlib.pyplot as plt

    plt.ion()
    # Only a handful of backends are truly non-interactive (QtAgg/TkAgg contain "agg").
    _non_interactive = {"agg", "pdf", "ps", "svg", "cairo", "template", "pgf"}
    live = matplotlib.get_backend().lower() not in _non_interactive
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    lines = {}
    (lines["train_mse"],) = ax_loss.plot([], [], "-o", ms=3, color="#1f77b4", label="train")
    (lines["val_mse"],) = ax_loss.plot([], [], "-o", ms=3, color="#d62728", label="test")
    ax_loss.set_title("Loss (MSE)")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("MSE")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="upper right")
    (lines["train_acc"],) = ax_acc.plot([], [], "-o", ms=3, color="#1f77b4", label="train")
    (lines["val_acc"],) = ax_acc.plot([], [], "-o", ms=3, color="#d62728", label="test")
    ax_acc.set_title(f"Accuracy ({acc_label})")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel(acc_label)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # leave room for the suptitle
    path = FIGURES_DIR / "train_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not live:
        logger.warning(f"Non-interactive matplotlib backend; curve frames -> {path}")
    return dict(plt=plt, fig=fig, ax_loss=ax_loss, ax_acc=ax_acc, lines=lines,
                acc_label=acc_label, live=live, path=path)


def _update_curves(state, history) -> None:
    """Redraw both curves from the running history after each epoch."""
    if not state:
        return
    xs = history["epoch"]
    state["lines"]["train_mse"].set_data(xs, history["train_mse"])
    state["lines"]["train_acc"].set_data(xs, history["train_acc"])
    has_val = bool(xs) and history["val_mse"][-1] is not None
    if has_val:
        state["lines"]["val_mse"].set_data(xs, history["val_mse"])
        state["lines"]["val_acc"].set_data(xs, history["val_acc"])
    for ax in (state["ax_loss"], state["ax_acc"]):
        ax.relim()
        ax.autoscale_view()
    # at-a-glance train-vs-test comparison in the figure title
    txt = f"epoch {xs[-1]}   MSE: train {history['train_mse'][-1]:.4f}"
    if has_val:
        txt += f" / test {history['val_mse'][-1]:.4f}"
    txt += f"      {state['acc_label']}: train {history['train_acc'][-1]:.3f}"
    if has_val:
        txt += f" / test {history['val_acc'][-1]:.3f}"
    state["fig"].suptitle(txt, fontsize=10)
    if state["live"]:
        state["fig"].canvas.draw_idle()
        state["plt"].pause(0.001)
    else:
        state["fig"].savefig(state["path"], dpi=120)


def _save_curves(state, history) -> None:
    """Final high-res save of the curve figure + a CSV of the per-epoch metrics."""
    if not state:
        return
    state["fig"].savefig(state["path"], dpi=130)
    logger.info(f"Saved training curves -> {state['path']}")
    import csv

    csv_path = state["path"].with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "test_mse", "train_acc", "test_acc"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i], history["train_mse"][i], history["val_mse"][i],
                history["train_acc"][i], history["val_acc"][i],
            ])
    logger.info(f"Saved metric history -> {csv_path}")


@app.command()
def main(
    dataset_dir: Path = DATA_DIR / "heatmap_dataset",
    model_path: Path = MODELS_DIR / "congestion_convlstm.pt",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    t_in: int = 60,
    t_out: int = 10,
    stride: int = 8,
    hidden_dims: str = "64,64",
    kernel_size: int = 3,
    val_frac: float = 0.2,
    cache_size: int = 16,
    num_workers: int = 0,
    max_train_windows: int = 0,
    max_val_windows: int = 0,
    tf32: bool = True,
    device: str = "cuda",
    amp: bool = True,
    preview: bool = typer.Option(False, help="Live predicted-vs-ground-truth window during training."),
    preview_every: int = typer.Option(50, help="Refresh the preview every N training steps."),
    all_runs: bool = typer.Option(
        False,
        "--all-runs",
        help="When dataset_dir contains timestamped run directories, train on all of them instead of only the latest.",
    ),
    curves: bool = typer.Option(
        True, "--curves/--no-curves",
        help="Live train/test loss(MSE) & accuracy(R²) curves in a 1x2 plot; "
             "saves figures/train_curves.png (+ .csv).",
    ),
    label_norm: float = 0.0,
    seed: int = 0,
):
    """Train a ConvLSTM that forecasts the future congestion heatmap."""
    import torch
    from torch.utils.data import DataLoader, Subset

    from macpf.convjam.convlstm import CongestionConvLSTM
    from macpf.convjam.dataset import CongestionWindowDataset, compute_label_norm

    run_dirs = (
        [dataset_dir]
        if any(dataset_dir.glob("episode_*.npz"))
        else sorted(p for p in dataset_dir.glob("*") if p.is_dir())
    )
    if not run_dirs:
        logger.error(
            f"No dataset found under {dataset_dir}. Generate one with `python -m macpf.generate_heatmap`."
        )
        raise typer.Exit(code=1)

    selected_run_dirs = run_dirs if all_runs else [run_dirs[-1]]
    episodes = []
    for run_dir in selected_run_dirs:
        episodes.extend(iter_episode_files(run_dir))
    if not episodes:
        logger.error(f"No episode_*.npz shards in {dataset_dir}.")
        raise typer.Exit(code=1)
    logger.info(f"Found {len(episodes)} episode(s) from {len(selected_run_dirs)} run dir(s)")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(device)
    use_amp = bool(amp) and device == "cuda"
    logger.info(f"Training on device={device} (amp={use_amp})")
    if device == "cuda":
        import torch

        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cudnn.benchmark = True
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA speed flags: amp={use_amp}, tf32={bool(tf32)}, cudnn.benchmark=True")

    # Split by episode (not by window) to keep train/val independent.
    rng = random.Random(seed)
    shuffled = episodes[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_frac) if len(shuffled) > 1 else 0
    val_eps = shuffled[:n_val]
    train_eps = shuffled[n_val:]
    logger.info(f"Episodes: {len(train_eps)} train / {len(val_eps)} val")

    if label_norm <= 0:
        label_norm = compute_label_norm(train_eps)
        logger.info(f"Computed label_norm={label_norm:.1f} (override with --label-norm)")

    hidden = [int(s) for s in hidden_dims.split(",") if s.strip()]

    train_ds = CongestionWindowDataset(train_eps, t_in, t_out, stride, label_norm, cache_size)
    if max_train_windows and len(train_ds) > max_train_windows:
        rng_subset = random.Random(seed + 101)
        idx = sorted(rng_subset.sample(range(len(train_ds)), int(max_train_windows)))
        train_ds = Subset(train_ds, idx)
        logger.info(f"Using max_train_windows={len(train_ds)}")

    loader_kwargs = {
        "num_workers": max(0, int(num_workers)),
        "pin_memory": (device == "cuda"),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = None
    if val_eps:
        val_ds = CongestionWindowDataset(val_eps, t_in, t_out, stride, label_norm, cache_size)
        if max_val_windows and len(val_ds) > max_val_windows:
            rng_subset = random.Random(seed + 202)
            idx = sorted(rng_subset.sample(range(len(val_ds)), int(max_val_windows)))
            val_ds = Subset(val_ds, idx)
            logger.info(f"Using max_val_windows={len(val_ds)}")
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        logger.info(f"Windows: {len(train_ds)} train / {len(val_ds)} val")
    else:
        logger.info(f"Windows: {len(train_ds)} train / 0 val")

    model = CongestionConvLSTM(
        in_channels=5,
        hidden_dims=hidden,
        kernel_size=kernel_size,
        t_in=t_in,
        t_out=t_out,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    scaler = _make_grad_scaler(torch, use_amp)

    # --- optional live preview: predicted vs ground truth, refreshed while training ---
    preview_state = None
    if preview:
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np

        plt.ion()
        # Interactive backends (QtAgg/TkAgg/...) all contain "agg", so a substring
        # check misfires. Only a handful of backends are truly non-interactive.
        _non_interactive = {"agg", "pdf", "ps", "svg", "cairo", "template", "pgf"}
        live = matplotlib.get_backend().lower() not in _non_interactive
        sample_ds = val_ds if val_eps else train_ds
        enc_p, dec_p, tgt_p = sample_ds[0]
        enc_p = enc_p.unsqueeze(0).to(device)
        dec_p = dec_p.unsqueeze(0).to(device)
        gt_disp = tgt_p[-1, 0].numpy() * label_norm  # the t_out-th (10s-ahead) frame
        vmax = float(gt_disp.max()) or 1.0
        fig_p, axes_p = plt.subplots(1, 2, figsize=(12.4, 5.2))
        im_pred = axes_p[0].imshow(np.zeros_like(gt_disp), cmap="inferno", vmin=0, vmax=vmax)
        axes_p[0].set_title("predicted (NN)")
        axes_p[1].imshow(gt_disp, cmap="inferno", vmin=0, vmax=vmax)
        axes_p[1].set_title("ground truth")
        for ax in axes_p:
            ax.set_xticks([])
            ax.set_yticks([])
        fig_p.colorbar(im_pred, ax=axes_p, fraction=0.046, pad=0.04)
        sup_p = fig_p.suptitle("warming up...")
        preview_path = FIGURES_DIR / "train_preview.png"
        if not live:
            logger.warning(f"Non-interactive matplotlib backend; preview frames -> {preview_path}")
        preview_state = dict(plt=plt, fig=fig_p, im=im_pred, sup=sup_p,
                             enc=enc_p, dec=dec_p, live=live, path=preview_path)

    def draw_preview(epoch: int, step: int, loss: float) -> None:
        if not preview_state:
            return
        was_training = model.training
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", enabled=use_amp):
            out = model(preview_state["enc"], preview_state["dec"])
        model.train(was_training)
        preview_state["im"].set_data(out[0, -1, 0].float().cpu().numpy() * label_norm)
        preview_state["sup"].set_text(
            f"epoch {epoch} · step {step} · loss {loss:.4f} · predicted {t_out}s ahead"
        )
        if preview_state["live"]:
            preview_state["fig"].canvas.draw_idle()
            preview_state["plt"].pause(0.001)
        else:
            preview_state["fig"].savefig(preview_state["path"], dpi=120)

    def run_epoch(loader, train: bool, epoch: int) -> tuple[float, float]:
        model.train(train)
        total, n_batches = 0.0, 0
        # Streaming sums for the exact epoch MSE + R² ("accuracy"), kept on-device so
        # we sync (.item()) once per epoch instead of once per batch.
        sum_sq_res = torch.zeros((), device=device)
        sum_y = torch.zeros((), device=device)
        sum_y2 = torch.zeros((), device=device)
        count = 0
        map_hw = None
        phase = "train" if train else " val "
        pbar = tqdm(
            loader,
            desc=f"epoch {epoch:>3}/{epochs} [{phase}]",
            leave=False,
            dynamic_ncols=True,
        )
        with torch.set_grad_enabled(train):
            for enc_in, dec_in, target in pbar:
                enc_in = enc_in.to(device, non_blocking=True)
                dec_in = dec_in.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                map_hw = tuple(enc_in.shape[-2:])
                if train:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=use_amp):
                    pred = model(enc_in, dec_in)
                    loss = loss_fn(pred, target)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                with torch.no_grad():
                    p = pred.detach().float()
                    t = target.detach().float()
                    sum_sq_res += ((p - t) ** 2).sum()
                    sum_y += t.sum()
                    sum_y2 += (t * t).sum()
                    count += t.numel()
                total += float(loss.item())
                n_batches += 1
                pbar.set_postfix(loss=f"{total / n_batches:.5f}")
                if train:
                    if preview_state and run_epoch.gstep % preview_every == 0:
                        draw_preview(epoch, run_epoch.gstep, float(loss.item()))
                    elif preview_state and preview_state["live"]:
                        # Pump the GUI event loop every step so the window stays
                        # responsive (not "Not Responding") between full redraws.
                        preview_state["fig"].canvas.flush_events()
                    run_epoch.gstep += 1
        pbar.close()
        run_epoch.last_map_hw = map_hw
        mse, acc = _epoch_metrics(
            float(sum_sq_res.item()), count, float(sum_y.item()), float(sum_y2.item())
        )
        return mse, acc

    run_epoch.last_map_hw = None
    run_epoch.gstep = 0
    if preview_state:
        draw_preview(0, 0, 0.0)  # populate the window with the initial (untrained) prediction

    curves_state = _setup_curves(curves, acc_label="R²")
    history = {"epoch": [], "train_mse": [], "val_mse": [], "train_acc": [], "val_acc": []}

    best_metric = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, True, epoch)
        if val_loader:
            val_loss, val_acc = run_epoch(val_loader, False, epoch)
        else:
            val_loss, val_acc = None, None

        if val_loss is not None:
            logger.info(
                f"epoch {epoch:>3}/{epochs}  "
                f"train_mse={train_loss:.5f} test_mse={val_loss:.5f}  "
                f"train_R2={train_acc:.4f} test_R2={val_acc:.4f}"
            )
        else:
            logger.info(
                f"epoch {epoch:>3}/{epochs}  train_mse={train_loss:.5f}  train_R2={train_acc:.4f}"
            )

        history["epoch"].append(epoch)
        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        _update_curves(curves_state, history)

        metric = val_loss if val_loss is not None else train_loss
        if metric < best_metric:
            best_metric = metric
            map_hw = run_epoch.last_map_hw or (None, None)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_cfg": {
                        "in_channels": 5,
                        "hidden_dims": hidden,
                        "kernel_size": kernel_size,
                        "t_in": t_in,
                        "t_out": t_out,
                    },
                    "label_norm": label_norm,
                    "map_shape_hw": [int(map_hw[0]), int(map_hw[1])] if map_hw[0] else None,
                },
                model_path,
            )
            logger.info(f"  saved best checkpoint -> {model_path} (metric={best_metric:.5f})")

    logger.success(f"Training complete. Best metric={best_metric:.5f}. Checkpoint: {model_path}")

    _save_curves(curves_state, history)

    if preview_state and preview_state["live"]:
        draw_preview(epochs, run_epoch.gstep, best_metric)
    any_live = (preview_state and preview_state["live"]) or (curves_state and curves_state["live"])
    if any_live:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()  # keep the final curve / preview window(s) open


if __name__ == "__main__":
    app()
