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
    device: str = "cuda",
    amp: bool = True,
    preview: bool = typer.Option(False, help="Live predicted-vs-ground-truth window during training."),
    preview_every: int = typer.Option(50, help="Refresh the preview every N training steps."),
    label_norm: float = 0.0,
    seed: int = 0,
):
    """Train a ConvLSTM that forecasts the future congestion heatmap."""
    import torch
    from torch.utils.data import DataLoader

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

    run_dir = run_dirs[-1]
    episodes = list(iter_episode_files(run_dir))
    if not episodes:
        logger.error(f"No episode_*.npz shards in {run_dir}.")
        raise typer.Exit(code=1)
    logger.info(f"Found {len(episodes)} episode(s) in {run_dir}")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(device)
    use_amp = bool(amp) and device == "cuda"
    logger.info(f"Training on device={device} (amp={use_amp})")
    if device == "cuda":
        import torch

        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

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
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )
    val_loader = None
    if val_eps:
        val_ds = CongestionWindowDataset(val_eps, t_in, t_out, stride, label_norm, cache_size)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device == "cuda"),
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

    def run_epoch(loader, train: bool, epoch: int) -> float:
        model.train(train)
        total, n_batches = 0.0, 0
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
        return total / max(1, n_batches)

    run_epoch.last_map_hw = None
    run_epoch.gstep = 0
    if preview_state:
        draw_preview(0, 0, 0.0)  # populate the window with the initial (untrained) prediction
    best_metric = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, True, epoch)
        val_loss = run_epoch(val_loader, False, epoch) if val_loader else None
        if val_loss is not None:
            logger.info(f"epoch {epoch:>3}/{epochs}  train={train_loss:.5f}  val={val_loss:.5f}")
        else:
            logger.info(f"epoch {epoch:>3}/{epochs}  train={train_loss:.5f}")

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

    if preview_state and preview_state["live"]:
        draw_preview(epochs, run_epoch.gstep, best_metric)
        preview_state["plt"].ioff()
        preview_state["plt"].show()  # keep the final window open


if __name__ == "__main__":
    app()
