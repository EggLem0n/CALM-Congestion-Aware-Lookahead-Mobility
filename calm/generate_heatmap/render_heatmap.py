"""Render one MP4 per heatmap-dataset episode: congestion heatmap + AMR positions.

Reads the per-episode shards written by ``calm.generate_heatmap`` (``episode_*.npz``)
plus the dataset's ``obstacle_map.npy`` and animates, for each timestep:

  * the additive congestion label ``y`` (T, 1, H, W) as a "hot" heatmap, and
  * the AMR positions ``agent_positions`` (T, N, 2) as an overlay scatter,

over the static factory obstacle background. Output goes to ``<dataset>/videos/``.

Rendering is post-hoc (no re-simulation) and parallel across episodes. The heatmap
is normalized per episode (label magnitude scales with AMR count), so brightness is
comparable within a video but not across videos -- pass --vmax for a fixed scale.

Usage (from repo root, in the macpf env):
    python -m calm.generate_heatmap.render_heatmap --dataset data/heatmap_dataset/260623_0907
    python -m calm.generate_heatmap.render_heatmap --dataset <dir> --episodes 0 209 --fps 60
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: no display, safe under multiprocessing

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

# Point matplotlib's FFMpegWriter at the ffmpeg binary bundled with imageio-ffmpeg
# (no system ffmpeg needed).
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def episode_files(dataset_dir: Path, only: Optional[Sequence[int]]) -> List[Path]:
    files = sorted(dataset_dir.glob("episode_*.npz"))
    if only is None:
        return files
    wanted = set(int(i) for i in only)
    return [f for f in files if int(f.stem.split("_")[1]) in wanted]


def _episode_id(npz_path: Path) -> int:
    return int(npz_path.stem.split("_")[1])


def render_congestion_video(
    congestion: np.ndarray,
    positions: np.ndarray,
    obstacle: np.ndarray,
    out_path,
    *,
    fps: int = 30,
    stride: int = 1,
    dpi: int = 100,
    vmax: Optional[float] = None,
    vmax_pct: float = 99.0,
    codec: str = "libx264",
    label: str = "",
    title_fmt: Optional[str] = None,
) -> float:
    """Animate a (T,H,W) congestion field ("hot") + (T,N,2) AMR scatter over an
    (H,W) obstacle background to ``out_path``; returns the vmax used.

    ``congestion`` may be (T,1,H,W) or (T,H,W). With ``vmax=None`` the colour scale
    is the ``vmax_pct`` percentile of positive congestion (so a single hot cell can't
    wash out the map); pass a shared ``vmax`` to make two clips colour-comparable.
    ``label`` is drawn (static) top-left; ``title_fmt`` may use {t}/{T}/{agents}."""
    cong = np.asarray(congestion)
    if cong.ndim == 4:                  # (T,1,H,W) -> (T,H,W)
        cong = cong[:, 0]
    T, H, W = cong.shape
    positions = np.asarray(positions)
    obstacle = np.asarray(obstacle).astype(np.float32)

    if vmax is None:
        pos_vals = cong[cong > 0]
        vmax = float(np.percentile(pos_vals, vmax_pct)) if pos_vals.size else 1.0
    vmax = max(float(vmax), 1.0)

    fig, ax = plt.subplots(figsize=(W / 10.0, H / 10.0))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # origin='upper' to match array row order
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    # Static obstacle background (walls drawn faint gray under the heatmap).
    ax.imshow(obstacle, cmap="binary", vmin=0.0, vmax=1.0, alpha=0.30,
              interpolation="nearest", zorder=0)

    heat = ax.imshow(np.ma.masked_less_equal(cong[0], 0.0), cmap="hot",
                     vmin=0.0, vmax=vmax, interpolation="nearest", zorder=1)
    scat = ax.scatter(positions[0, :, 0], positions[0, :, 1],
                      s=6, c="cyan", edgecolors="black", linewidths=0.3, zorder=2)
    txt = None
    if label or title_fmt:
        txt = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
                      color="white", fontsize=8,
                      bbox=dict(facecolor="black", alpha=0.4, pad=2), zorder=3)

    n_agents = positions.shape[1]
    writer = FFMpegWriter(fps=fps, codec=codec, extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(out_path), dpi=dpi):
        for t in range(0, T, stride):
            heat.set_data(np.ma.masked_less_equal(cong[t], 0.0))
            scat.set_offsets(positions[t])
            if txt is not None:
                line = title_fmt.format(t=t, T=T - 1, agents=n_agents) if title_fmt else ""
                txt.set_text(f"{label}  {line}".strip())
            writer.grab_frame()
    plt.close(fig)
    return vmax


def render_one(
    npz_path_str: str,
    obstacle_path_str: str,
    out_dir_str: str,
    fps: int,
    stride: int,
    dpi: int,
    vmax_fixed: Optional[float],
    vmax_pct: float,
    overwrite: bool,
) -> str:
    """Render a single episode to <out_dir>/episode_XXXX.mp4. Returns a status string.
    Runs in a worker process: all args are picklable primitives."""
    npz_path = Path(npz_path_str)
    out_path = Path(out_dir_str) / f"{npz_path.stem}.mp4"
    if out_path.exists() and not overwrite:
        return f"skip {out_path.name} (exists)"

    data = np.load(npz_path)
    y = data["y"]                       # (T, 1, H, W) float32 congestion
    positions = data["agent_positions"] # (T, N, 2) int16 -> [col, row]
    obstacle = np.load(obstacle_path_str).astype(np.float32)  # (H, W)
    T = y.shape[0]

    vmax = render_congestion_video(
        y, positions, obstacle, out_path,
        fps=fps, stride=stride, dpi=dpi, vmax=vmax_fixed, vmax_pct=vmax_pct,
        title_fmt=f"{npz_path.stem}  t={{t}}/{{T}}  agents={{agents}}")
    data.close()
    return f"ok   {out_path.name}  (vmax={vmax:.0f}, frames={len(range(0, T, stride))})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=Path,
                   help="Dataset directory containing episode_*.npz and obstacle_map.npy.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir for mp4s. Default: <dataset>/videos.")
    p.add_argument("--episodes", type=int, nargs="+", default=None,
                   help="Only render these episode ids (default: all).")
    p.add_argument("--num_of_process", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="Parallel render processes (one episode per task).")
    p.add_argument("--fps", type=int, default=30, help="Output frames per second.")
    p.add_argument("--stride", type=int, default=1,
                   help="Render every Nth timestep (1 = all 1801 frames).")
    p.add_argument("--dpi", type=int, default=100, help="Render DPI (resolution).")
    p.add_argument("--vmax", type=float, default=None,
                   help="Fixed heatmap max for ALL videos (default: per-episode percentile).")
    p.add_argument("--vmax-percentile", type=float, default=99.0,
                   help="Percentile of positive congestion used as per-episode vmax.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-render even if the mp4 already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    obstacle_path = dataset_dir / "obstacle_map.npy"
    if not obstacle_path.exists():
        raise SystemExit(f"obstacle_map.npy not found in {dataset_dir}")

    files = episode_files(dataset_dir, args.episodes)
    if not files:
        raise SystemExit(f"No matching episode_*.npz in {dataset_dir}")

    out_dir = (args.out or (dataset_dir / "videos")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.num_of_process))
    print(f"Rendering {len(files)} episode video(s) -> {out_dir}  "
          f"({workers} processes, fps={args.fps}, stride={args.stride}, dpi={args.dpi})",
          flush=True)

    job_args = [
        (str(f), str(obstacle_path), str(out_dir), args.fps, args.stride, args.dpi,
         args.vmax, args.vmax_percentile, args.overwrite)
        for f in files
    ]

    started = time.perf_counter()
    done = 0
    total = len(job_args)
    if workers == 1:
        for ja in job_args:
            print(render_one(*ja), flush=True)
            done += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(render_one, *ja) for ja in job_args]
            for fut in as_completed(futures):
                done += 1
                elapsed = time.perf_counter() - started
                eta = (elapsed / done) * (total - done) / 60.0
                print(f"[{done}/{total} | {elapsed/60.0:.1f} min | eta {eta:.1f} min] "
                      f"{fut.result()}", flush=True)

    print(f"Done. {total} video(s) in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
