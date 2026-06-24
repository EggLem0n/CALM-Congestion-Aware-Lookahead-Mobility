"""Make an MP4 from a finished run's saved trajectory -- no re-planning.

A completed run already saved its full kinodynamic trajectory (agent_states.npy) and
grid paths, so the MP4 can be re-encoded at any playback speed cheaply, without re-running
the (slow) simulation. Speed is set by --subframes against the fixed 30 fps:

    subframes == 30  -> real-time (1 sim-second = 1 video-second)
    subframes == 3   -> 10x speed
    subframes == 8   -> ~3.75x speed (legacy default)

    python -m macpf.online_mapf.mp4maker data/online_runs/<ts>
    python -m macpf.online_mapf.mp4maker data/online_runs/<ts> --subframes 3 --out reports/figures/<...>

Importing the macpf.online_mapf package forces the headless Agg matplotlib backend, so
this renders fine on a server / inside a subprocess.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from macpf.classical_mapf import animate_paths
from macpf.classical_mapf import factory_map_generator as fmg
from macpf.classical_mapf.utils import load_config

PROJ_ROOT = Path(__file__).resolve().parents[2]


def make_mp4(run_dir: Path, out_dir: Path, subframes: int) -> Path:
    """Re-encode <run_dir>'s animation into out_dir at the given subframes; returns the mp4 path."""
    env = fmg.build_factory_map()
    agent_states = np.load(run_dir / "agent_states.npy")
    T = int(agent_states.shape[0])
    pj = json.loads((run_dir / "classical_paths.json").read_text(encoding="utf-8"))
    n = len(pj)
    paths = [[tuple(p) for p in pj[str(i)]] for i in range(n)]
    metrics = json.loads((run_dir / "classical_metrics.json").read_text(encoding="utf-8"))
    final_goals = metrics.get("final_goals")
    goals = [tuple(g) for g in final_goals] if final_goals else [paths[i][-1] for i in range(n)]
    starts = [paths[i][0] for i in range(n)]
    # Rebuild task_summary so the route lines + pickup/delivery target markers render
    # (same visuals as classical_mapf); without it the animation shows only static goals.
    task_assignments = metrics.get("task_assignments")
    task_summary = {"task_assignments": task_assignments} if task_assignments else None

    config = load_config().replace(animation_subframes=subframes, save_animation=True, max_time=T - 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    animate_paths(env, paths, starts, goals, out_dir, config,
                  agent_states=agent_states, task_summary=task_summary)
    return out_dir / "classical_mapf_animation.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a run's MP4 from its saved trajectory.")
    parser.add_argument("run_dir", help="Data run dir (needs agent_states.npy + classical_paths.json).")
    parser.add_argument("--out", default=None,
                        help="Output figures dir (default: reports/figures/mp4maker/<run>).")
    parser.add_argument("--subframes", type=int, default=30,
                        help="Frames per sim-step at 30 fps. 30=real-time, 3=10x, 8=~3.75x.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else (PROJ_ROOT / "reports" / "figures" / "mp4maker" / run_dir.name)
    mp4 = make_mp4(run_dir, out_dir, args.subframes)
    print(f"rendered -> {mp4.resolve()}")


if __name__ == "__main__":
    main()
