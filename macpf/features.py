"""Feature loading for the congestion-prediction model.

The raw dataset shards are produced by `macpf.generate_heatmap` (the classical MAPF
simulator with additive congestion labels). Each episode is an .npz with:

    x : (T, 5, H, W) uint8   input channels
        [amr_occupancy, current_pickup_targets, current_delivery_targets,
         initial_start_positions, obstacles]
    y : (T, 1, H, W) float32 additive congestion heatmap label

This module turns those stored shards into model-ready float32 tensors. The
channels are stored as uint8 to keep files ~4x smaller; the only "feature"
step today is that cast plus optional label normalization, kept here so the
training loader (`macpf.convjam.train`) stays thin.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

from loguru import logger
import numpy as np
import typer

from macpf.config import DATA_DIR

app = typer.Typer()

INPUT_CHANNELS = [
    "amr_occupancy",
    "current_pickup_targets",
    "current_delivery_targets",
    "initial_start_positions",
    "obstacles",
]


def load_episode(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one episode shard and return (x, y) as float32 arrays.

    x: (T, 5, H, W) float32, y: (T, 1, H, W) float32.
    """
    with np.load(npz_path) as data:
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
    return x, y


def iter_episode_files(dataset_dir: Path) -> Iterator[Path]:
    """Yield episode_*.npz paths in a dataset run directory, in order."""
    yield from sorted(dataset_dir.glob("episode_*.npz"))


@app.command()
def main(
    dataset_dir: Path = DATA_DIR / "heatmap_dataset",
    normalize: bool = typer.Option(
        False, help="Report the dataset label max so labels can be scaled to [0, 1]."
    ),
):
    """Summarize a generated dataset run and sanity-check the feature tensors."""
    run_dirs = (
        [dataset_dir]
        if any(dataset_dir.glob("episode_*.npz"))
        else sorted(p for p in dataset_dir.glob("*") if p.is_dir())
    )
    if not run_dirs:
        logger.error(
            f"No episode shards found under {dataset_dir}. "
            "Run `python -m macpf.generate_heatmap` first."
        )
        raise typer.Exit(code=1)

    run_dir = run_dirs[-1]
    episodes = list(iter_episode_files(run_dir))
    logger.info(f"Inspecting {len(episodes)} episode(s) in {run_dir}")
    label_max = 0.0
    for episode_path in episodes:
        x, y = load_episode(episode_path)
        label_max = max(label_max, float(y.max()))
        logger.info(
            f"  {episode_path.name}: x{tuple(x.shape)} y{tuple(y.shape)} "
            f"label_max={float(y.max()):.1f}"
        )
    if normalize:
        logger.info(
            f"Dataset label max = {label_max:.1f} "
            "(divide y by this to normalize labels to [0, 1])."
        )
    logger.success("Feature inspection complete.")


if __name__ == "__main__":
    app()
