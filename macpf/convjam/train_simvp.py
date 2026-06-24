"""Train the OpenSTL-style SimVP congestion-prediction model."""
from __future__ import annotations

import random
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from macpf.config import DATA_DIR, FIGURES_DIR, MODELS_DIR
from macpf.features import iter_episode_files
from macpf.convjam.train import (
    _epoch_metrics,
    _make_grad_scaler,
    _resolve_device,
    _save_curves,
    _setup_curves,
    _update_curves,
)

app = typer.Typer()


@app.command()
def main(
    dataset_dir: Path = DATA_DIR / "heatmap_dataset_pibt",
    model_path: Path = MODELS_DIR / "congestion_simvp_pibt.pt",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    t_in: int = 60,
    t_out: int = 10,
    stride: int = 8,
    variant: str = typer.Option("openstl", help="Model variant: openstl or compact."),
    hidden_dim: int = typer.Option(48, help="Compact variant hidden channels."),
    translator_layers: int = typer.Option(4, help="Compact variant translator depth."),
    kernel_size: int = typer.Option(3, help="Compact variant conv kernel."),
    hid_s: int = typer.Option(16, help="OpenSTL spatial hidden channels."),
    hid_t: int = typer.Option(256, help="OpenSTL temporal/meta hidden channels."),
    n_s: int = typer.Option(4, help="OpenSTL spatial encoder/decoder depth."),
    n_t: int = typer.Option(4, help="OpenSTL temporal translator depth."),
    model_type: str = typer.Option("gsta", help="OpenSTL mid-block type; lightweight port supports gsta."),
    mlp_ratio: float = typer.Option(4.0, help="OpenSTL gSTA MLP expansion ratio."),
    drop: float = typer.Option(0.0, help="OpenSTL dropout."),
    drop_path: float = typer.Option(0.1, help="OpenSTL stochastic depth."),
    spatio_kernel_enc: int = typer.Option(3, help="OpenSTL encoder spatial kernel."),
    spatio_kernel_dec: int = typer.Option(3, help="OpenSTL decoder spatial kernel."),
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
        True,
        "--curves/--no-curves",
        help="Save train/test MSE and R² curves.",
    ),
    label_norm: float = 0.0,
    seed: int = 0,
):
    """Train SimVP to forecast future congestion heatmaps from 5-channel map sequences."""
    import torch
    from torch.utils.data import DataLoader, Subset

    from macpf.convjam.dataset import CongestionWindowDataset, compute_label_norm
    from macpf.convjam.simvp import CongestionSimVP, OpenSTLCongestionSimVP

    run_dirs = (
        [dataset_dir]
        if any(dataset_dir.glob("episode_*.npz"))
        else sorted(p for p in dataset_dir.glob("*") if p.is_dir())
    )
    if not run_dirs:
        logger.error(f"No dataset found under {dataset_dir}.")
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
    logger.info(f"Training SimVP variant={variant} on device={device} (amp={use_amp})")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cudnn.benchmark = True
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA speed flags: amp={use_amp}, tf32={bool(tf32)}, cudnn.benchmark=True")

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

    variant = str(variant).lower()
    if variant == "openstl":
        model = OpenSTLCongestionSimVP(
            in_channels=5,
            out_channels=1,
            hid_s=hid_s,
            hid_t=hid_t,
            n_s=n_s,
            n_t=n_t,
            model_type=model_type,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            spatio_kernel_enc=spatio_kernel_enc,
            spatio_kernel_dec=spatio_kernel_dec,
            t_in=t_in,
            t_out=t_out,
        ).to(device)
    elif variant == "compact":
        model = CongestionSimVP(
            in_channels=5,
            hidden_dim=hidden_dim,
            translator_layers=translator_layers,
            kernel_size=kernel_size,
            t_in=t_in,
            t_out=t_out,
        ).to(device)
    else:
        logger.error(f"Unknown SimVP variant: {variant}. Use openstl or compact.")
        raise typer.Exit(code=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    scaler = _make_grad_scaler(torch, use_amp)

    preview_state = None
    if preview:
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np

        plt.ion()
        _non_interactive = {"agg", "pdf", "ps", "svg", "cairo", "template", "pgf"}
        live = matplotlib.get_backend().lower() not in _non_interactive
        enc_p, dec_p, tgt_p = train_ds[0]
        enc_p = enc_p.unsqueeze(0).to(device)
        dec_p = dec_p.unsqueeze(0).to(device)
        gt_disp = tgt_p[-1, 0].numpy() * label_norm
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
        preview_path = FIGURES_DIR / "train_simvp_preview.png"
        if not live:
            logger.warning(f"Non-interactive matplotlib backend; preview frames -> {preview_path}")
        preview_state = dict(plt=plt, fig=fig_p, im=im_pred, sup=sup_p, enc=enc_p, dec=dec_p, live=live, path=preview_path)

    def draw_preview(epoch: int, step: int, loss: float) -> None:
        if not preview_state:
            return
        was_training = model.training
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", enabled=use_amp):
            out = model(preview_state["enc"], preview_state["dec"])
        model.train(was_training)
        preview_state["im"].set_data(out[0, -1, 0].float().cpu().numpy() * label_norm)
        preview_state["sup"].set_text(f"epoch {epoch} - step {step} - loss {loss:.4f} - predicted {t_out}s ahead")
        if preview_state["live"]:
            preview_state["fig"].canvas.draw_idle()
            preview_state["plt"].pause(0.001)
        else:
            preview_state["fig"].savefig(preview_state["path"], dpi=120)

    def run_epoch(loader, train: bool, epoch: int) -> tuple[float, float]:
        model.train(train)
        sum_sq_res = torch.zeros((), device=device)
        sum_y = torch.zeros((), device=device)
        sum_y2 = torch.zeros((), device=device)
        count = 0
        phase = "train" if train else " val "
        pbar = tqdm(loader, desc=f"epoch {epoch:>3}/{epochs} [{phase}]", leave=False, dynamic_ncols=True)
        with torch.set_grad_enabled(train):
            for enc_in, dec_in, target in pbar:
                enc_in = enc_in.to(device, non_blocking=True)
                dec_in = dec_in.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
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
                pbar.set_postfix(loss=f"{float(loss.item()):.5f}")
                if train:
                    if preview_state and run_epoch.gstep % max(1, int(preview_every)) == 0:
                        draw_preview(epoch, run_epoch.gstep, float(loss.item()))
                    elif preview_state and preview_state["live"]:
                        preview_state["fig"].canvas.flush_events()
                    run_epoch.gstep += 1
        pbar.close()
        return _epoch_metrics(float(sum_sq_res.item()), count, float(sum_y.item()), float(sum_y2.item()))

    run_epoch.gstep = 0
    if preview_state:
        draw_preview(0, 0, 0.0)

    curves_state = _setup_curves(curves, acc_label="R²")
    history = {"epoch": [], "train_mse": [], "val_mse": [], "train_acc": [], "val_acc": []}
    best_metric = float("inf")

    for epoch in range(1, epochs + 1):
        train_mse, train_r2 = run_epoch(train_loader, True, epoch)
        if val_loader:
            val_mse, val_r2 = run_epoch(val_loader, False, epoch)
        else:
            val_mse, val_r2 = None, None

        if val_mse is not None:
            logger.info(
                f"epoch {epoch:>3}/{epochs}  train_mse={train_mse:.5f} "
                f"test_mse={val_mse:.5f}  train_R2={train_r2:.4f} test_R2={val_r2:.4f}"
            )
        else:
            logger.info(f"epoch {epoch:>3}/{epochs}  train_mse={train_mse:.5f} train_R2={train_r2:.4f}")

        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["train_acc"].append(train_r2)
        history["val_acc"].append(val_r2)
        _update_curves(curves_state, history)

        metric = val_mse if val_mse is not None else train_mse
        if metric < best_metric:
            best_metric = metric
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_type": "simvp_openstl" if variant == "openstl" else "simvp",
                    "model_cfg": (
                        {
                            "in_channels": 5,
                            "out_channels": 1,
                            "hid_s": int(hid_s),
                            "hid_t": int(hid_t),
                            "n_s": int(n_s),
                            "n_t": int(n_t),
                            "model_type": str(model_type),
                            "mlp_ratio": float(mlp_ratio),
                            "drop": float(drop),
                            "drop_path": float(drop_path),
                            "spatio_kernel_enc": int(spatio_kernel_enc),
                            "spatio_kernel_dec": int(spatio_kernel_dec),
                            "t_in": int(t_in),
                            "t_out": int(t_out),
                        }
                        if variant == "openstl"
                        else {
                            "in_channels": 5,
                            "hidden_dim": int(hidden_dim),
                            "translator_layers": int(translator_layers),
                            "kernel_size": int(kernel_size),
                            "t_in": int(t_in),
                            "t_out": int(t_out),
                        }
                    ),
                    "label_norm": float(label_norm),
                },
                model_path,
            )
            logger.info(f"  saved best {variant} SimVP checkpoint -> {model_path} (metric={best_metric:.5f})")

    logger.success(f"SimVP training complete. Best metric={best_metric:.5f}. Checkpoint: {model_path}")
    _save_curves(curves_state, history)

    if preview_state and preview_state["live"]:
        draw_preview(epochs, run_epoch.gstep, best_metric)
    any_live = (preview_state and preview_state["live"]) or (curves_state and curves_state["live"])
    if any_live:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    app()
