"""Real-time visualization of predicted vs ground-truth congestion.

Plays an episode back like a simulator: the AMRs move (dots) while we show, at each
time t, the model's forecast for t (made `lead` seconds earlier from a `t_in`-second
observation window) next to the ground-truth congestion and their absolute error.

    python -m macpf.convjam.visualize                       # live window
    python -m macpf.convjam.visualize --save out.gif        # write a GIF/MP4

Without a trained checkpoint it still animates ground truth + robots so you can
preview the scenario.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
import numpy as np
import typer

from macpf.config import DATA_DIR, MODELS_DIR
from macpf.features import iter_episode_files, load_episode

app = typer.Typer()


def _resolve_episode(input_path: Path) -> Path:
    if input_path.is_file() and input_path.suffix == ".npz":
        return input_path
    run_dirs = (
        [input_path]
        if any(input_path.glob("episode_*.npz"))
        else sorted(p for p in input_path.glob("*") if p.is_dir())
    )
    for run_dir in reversed(run_dirs):
        episodes = list(iter_episode_files(run_dir))
        if episodes:
            return episodes[0]
    raise FileNotFoundError(f"No episode_*.npz scenario found under {input_path}.")


def _load_agent_positions(path: Path) -> Optional[np.ndarray]:
    with np.load(path) as data:
        if "agent_positions" in data:
            return np.asarray(data["agent_positions"])  # (T, num_agents, 2) as (x, y)
    return None


def _predict_for_frames(
    model,
    x: np.ndarray,
    frames: List[int],
    t_in: int,
    t_out: int,
    lead: int,
    label_norm: float,
    device: str,
    batch: int,
) -> Dict[int, np.ndarray]:
    """For each frame f, forecast congestion at f from a window ending `lead` steps
    earlier (encoder = [f-lead-t_in+1 .. f-lead], decoder = next t_out, occ zeroed)."""
    import torch

    T = x.shape[0]
    enc_list, dec_list, owners = [], [], []
    for f in frames:
        c = f - lead  # last observed step
        s = c - t_in + 1
        if s < 0 or c + 1 + t_out > T:
            continue
        dec = x[c + 1 : c + 1 + t_out].copy()
        dec[:, 0] = 0.0  # future occupancy is unknown
        enc_list.append(x[s : c + 1])
        dec_list.append(dec)
        owners.append(f)

    preds: Dict[int, np.ndarray] = {}
    use_amp = device == "cuda"
    for i in range(0, len(enc_list), batch):
        enc_b = torch.from_numpy(np.stack(enc_list[i : i + batch])).to(device)
        dec_b = torch.from_numpy(np.stack(dec_list[i : i + batch])).to(device)
        with torch.no_grad(), torch.autocast("cuda", enabled=use_amp):
            out = model(enc_b, dec_b)  # (b, t_out, 1, H, W)
        out = out[:, lead - 1, 0].float().cpu().numpy() * label_norm  # (b, H, W)
        out = np.clip(out, 0.0, None)
        for j, f in enumerate(owners[i : i + batch]):
            preds[f] = out[j]
    return preds


@app.command()
def main(
    model_path: Path = MODELS_DIR / "congestion_convlstm.pt",
    input_path: Path = DATA_DIR / "heatmap_dataset",
    save: Path = typer.Option(None, help="Write GIF/MP4 instead of a live window."),
    lead: int = typer.Option(0, help="Forecast lead in seconds (0 = model t_out)."),
    fps: int = 15,
    frame_step: int = typer.Option(2, help="Show every Nth simulation second."),
    max_frames: int = typer.Option(600, help="Cap animated frames (0 = no cap)."),
    batch: int = 32,
    device: str = "cuda",
):
    """Animate predicted vs ground-truth congestion with the AMRs moving."""
    import matplotlib

    if save is not None:
        matplotlib.use("Agg")  # headless render to file
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    episode = _resolve_episode(input_path)
    logger.info(f"Scenario: {episode}")
    x, y = load_episode(episode)  # (T,5,H,W), (T,1,H,W)
    gt = y[:, 0]  # (T, H, W)
    T, H, W = gt.shape
    obstacle = x[0, 4] > 0  # static obstacle map
    agents = _load_agent_positions(episode)  # (T, N, 2) or None

    # Load the model if a checkpoint exists; otherwise ground-truth-only preview.
    model = None
    t_in = t_out = 0
    if model_path.exists():
        import torch

        from macpf.convjam.convlstm import CongestionConvLSTM

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(model_path, map_location="cpu")
        cfg = ckpt["model_cfg"]
        label_norm = float(ckpt.get("label_norm", 1.0)) or 1.0
        t_in, t_out = int(cfg["t_in"]), int(cfg["t_out"])
        model = CongestionConvLSTM(**cfg).to(device).eval()
        model.load_state_dict(ckpt["state_dict"])
        logger.info(f"Model loaded (t_in={t_in}, t_out={t_out}, label_norm={label_norm:.1f}) on {device}")
    else:
        logger.warning(f"No checkpoint at {model_path}: showing ground truth + robots only.")

    lead = (lead or t_out) if model is not None else 0

    # Frames to animate. Start late enough that a forecast exists for the first frame.
    start = (t_in + lead) if model is not None else 0
    frames = list(range(start, T, max(1, frame_step)))
    if max_frames and len(frames) > max_frames:
        frames = frames[:max_frames]
    if not frames:
        logger.error("No frames to animate (window longer than the episode?).")
        raise typer.Exit(code=1)

    preds: Dict[int, np.ndarray] = {}
    if model is not None:
        logger.info(f"Forecasting {len(frames)} frames (lead={lead}s)...")
        preds = _predict_for_frames(model, x, frames, t_in, t_out, lead, label_norm, device, batch)

    # Shared color scale (robust to a few extreme cells).
    vmax = float(np.percentile(gt[frames], 99.5)) or 1.0

    # 2 columns x 1 row: predicted (left) | ground truth (right).
    panels = []
    if model is not None:
        panels.append(("predicted (NN)", "pred"))
    panels.append(("ground truth", "gt"))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.2))
    if len(panels) == 1:
        axes = [axes]

    images, scatters = {}, {}
    f0 = frames[0]
    for ax, (title, key) in zip(axes, panels):
        if key == "pred":
            data = preds.get(f0, np.zeros((H, W), np.float32))
        elif key == "gt":
            data = gt[f0]
        else:
            data = np.abs(preds.get(f0, np.zeros((H, W), np.float32)) - gt[f0])
        im = ax.imshow(data, cmap="inferno", vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~obstacle, obstacle), cmap="gray", alpha=0.35, vmin=0, vmax=1)
        sc = ax.scatter([], [], s=14, c="cyan", edgecolors="black", linewidths=0.4, zorder=3)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        images[key] = im
        scatters[key] = sc

    lead_txt = f"  |  predicted {lead}s ahead" if model is not None else ""
    suptitle = fig.suptitle("")

    def update(f: int):
        for key in images:
            if key == "pred":
                images[key].set_data(preds.get(f, np.zeros((H, W), np.float32)))
            elif key == "gt":
                images[key].set_data(gt[f])
            else:
                images[key].set_data(np.abs(preds.get(f, np.zeros((H, W), np.float32)) - gt[f]))
        if agents is not None:
            pts = agents[f]  # (N, 2) as (x, y)
            for sc in scatters.values():
                sc.set_offsets(pts)
        suptitle.set_text(f"t = {f}s / {T - 1}s{lead_txt}")
        return list(images.values()) + list(scatters.values()) + [suptitle]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / max(1, fps), blit=False)
    fig.tight_layout()

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        if save.suffix.lower() == ".mp4":
            from matplotlib.animation import FFMpegWriter

            anim.save(str(save), writer=FFMpegWriter(fps=fps))
        else:
            from matplotlib.animation import PillowWriter

            anim.save(str(save), writer=PillowWriter(fps=fps))
        logger.success(f"Saved animation ({len(frames)} frames) -> {save}")
    else:
        plt.show()


if __name__ == "__main__":
    app()
