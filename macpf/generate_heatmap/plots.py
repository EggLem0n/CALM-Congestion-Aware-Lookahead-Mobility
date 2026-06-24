"""Human-facing plots for the congestion-prediction side of the project.

Path/animation plots for the classical planner live in
`macpf.classical_mapf.viz`. This module renders congestion heatmaps (labels or
model predictions) and, like every human-facing figure in this project, writes
them under reports/figures/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import typer

from macpf.config import DATA_DIR, FIGURES_DIR

app = typer.Typer()


def _squeeze_congestion(array: np.ndarray) -> np.ndarray:
    """Accept (T, H, W) or (T, 1, H, W) and return (T, H, W)."""
    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0, :, :]
    if array.ndim == 3:
        return array
    raise ValueError(f"Expected congestion array of shape (T,H,W) or (T,1,H,W), got {array.shape}")


def plot_congestion_frame(
    congestion: np.ndarray,
    frame: int,
    save_path: Path,
    obstacle_map: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> Path:
    """Render one congestion heatmap frame and save it under reports/figures/."""
    congestion = _squeeze_congestion(np.asarray(congestion))
    frame = int(np.clip(frame, 0, congestion.shape[0] - 1))
    heatmap = congestion[frame]

    fig, ax = plt.subplots(figsize=(12, 8))
    if obstacle_map is not None:
        ax.imshow(np.asarray(obstacle_map), cmap="gray_r", origin="upper", alpha=0.25)
    image = ax.imshow(heatmap, cmap="inferno", origin="upper", alpha=0.85)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Additive congestion")
    ax.set_title(title or f"Congestion heatmap (t={frame})")
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


@app.command()
def main(
    congestion_path: Path = typer.Argument(
        ...,
        help="A congestion_labels.npy (T,H,W) from a classical run, or an episode .npz (key 'y').",
    ),
    frame: int = typer.Option(-1, help="Frame index to plot. -1 = the peak-congestion frame."),
    output_path: Path = FIGURES_DIR / "congestion" / "congestion_frame.png",
):
    """Plot a single congestion heatmap frame to reports/figures/."""
    if congestion_path.suffix == ".npz":
        with np.load(congestion_path) as data:
            congestion = data["y"]
    else:
        congestion = np.load(congestion_path)
    congestion = _squeeze_congestion(np.asarray(congestion, dtype=np.float32))

    if frame < 0:
        frame = int(congestion.reshape(congestion.shape[0], -1).sum(axis=1).argmax())

    obstacle_path = DATA_DIR / "maps" / "obstacle_map_v3_oneway.npy"
    obstacle_map = np.load(obstacle_path) if obstacle_path.exists() else None

    saved = plot_congestion_frame(congestion, frame, output_path, obstacle_map=obstacle_map)
    logger.success(f"Saved congestion heatmap (t={frame}) to {saved}")


if __name__ == "__main__":
    app()
