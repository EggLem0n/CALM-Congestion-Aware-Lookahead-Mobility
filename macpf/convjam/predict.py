"""Run the trained congestion model to produce an AI congestion-cost field.

Output : a (future_steps, H, W) float32 array saved as .npy, which the classical
planner consumes via `load_ai_congestion_cost` when `use_ai_congestion_cost:
true` and `ai_cost_path` points at this file (see configs/default.yaml).

The model (`macpf.convjam.convlstm.CongestionConvLSTM`) forecasts future congestion
from an observed window: the first `t_in` frames of the scenario are the observation,
and the next `t_out` frames supply the known exogenous channels (occupancy zeroed).
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import typer

from macpf.config import DATA_DIR, MODELS_DIR
from macpf.features import iter_episode_files, load_episode

app = typer.Typer()


def _resolve_episode(input_path: Path) -> Path:
    """Pick a scenario shard: a direct .npz, or the first episode of the latest run."""
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


def _pad_to(frames: np.ndarray, length: int, mode: str) -> np.ndarray:
    """Pad a (n, C, H, W) stack to `length` along axis 0 by repeating an edge frame."""
    n = frames.shape[0]
    if n >= length:
        return frames[:length] if mode == "tail" else frames[-length:]
    edge = frames[-1:] if mode == "tail" else frames[:1]
    pad = np.repeat(edge, length - n, axis=0)
    return np.concatenate([frames, pad], axis=0) if mode == "tail" else np.concatenate([pad, frames], axis=0)


@app.command()
def main(
    model_path: Path = MODELS_DIR / "congestion_convlstm.pt",
    input_path: Path = DATA_DIR / "heatmap_dataset",
    congestion_cost_path: Path = MODELS_DIR / "congestion_cost.npy",
    horizon: int = 0,
    device: str = "auto",
):
    """Predict a congestion-cost field for the planner to use as a soft cost."""
    import torch

    from macpf.convjam.convlstm import CongestionConvLSTM

    if not model_path.exists():
        logger.error(
            f"No trained model at {model_path}. Train one with `python -m macpf.convjam.train`."
        )
        raise typer.Exit(code=1)

    congestion_cost_path.parent.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_path, map_location="cpu")
    cfg = checkpoint["model_cfg"]
    label_norm = float(checkpoint.get("label_norm", 1.0)) or 1.0
    t_in, t_out = int(cfg["t_in"]), int(cfg["t_out"])

    model = CongestionConvLSTM(**cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    episode = _resolve_episode(input_path)
    logger.info(f"Scenario: {episode}")
    x, _ = load_episode(episode)  # (T, 5, H, W) float32

    # Observation window = first t_in frames; decoder exogenous = the following t_out
    # frames with occupancy zeroed (pad by repeating an edge frame if the shard is short).
    enc_in = _pad_to(x[:t_in], t_in, mode="lead")
    dec_in = _pad_to(x[t_in : t_in + t_out], t_out, mode="tail").copy()
    dec_in[:, 0] = 0.0  # occupancy is unknown in the future

    enc_t = torch.from_numpy(enc_in[None]).to(device)
    dec_t = torch.from_numpy(dec_in[None]).to(device)
    with torch.no_grad():
        pred = model(enc_t, dec_t)  # (1, t_out, 1, H, W)

    pred = pred.squeeze(0).squeeze(1).cpu().numpy() * label_norm  # (t_out, H, W)
    pred = np.clip(pred, 0.0, None).astype(np.float32)

    # Optionally stretch/trim to a requested planner horizon (planner also clamps,
    # but pre-padding lets one file cover the whole plan).
    if horizon and horizon != pred.shape[0]:
        if horizon > pred.shape[0]:
            tail = np.repeat(pred[-1:], horizon - pred.shape[0], axis=0)
            pred = np.concatenate([pred, tail], axis=0)
        else:
            pred = pred[:horizon]

    np.save(congestion_cost_path, pred)
    logger.success(
        f"Saved congestion cost {pred.shape} (future_steps, H, W) -> {congestion_cost_path}. "
        "Enable it via use_ai_congestion_cost / ai_cost_path in configs/default.yaml."
    )


if __name__ == "__main__":
    app()
