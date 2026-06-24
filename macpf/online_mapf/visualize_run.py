"""Render a saved MAPF run without re-running planning.

Usage:
    python -m macpf.online_mapf.visualize_run data/online_runs/260618_0506
    python -m macpf.online_mapf.visualize_run latest --subframes 3

The input run directory must contain:
    agent_states.npy or agent_positions.npy
    classical_paths.json
    classical_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from macpf.classical_mapf import animate_paths
from macpf.classical_mapf import factory_map_generator as fmg
from macpf.classical_mapf.utils import load_config

PROJ_ROOT = Path(__file__).resolve().parents[2]


def latest_run_dir(base: Path) -> Path:
    if not base.exists():
        raise FileNotFoundError(f"Run folder base does not exist: {base}")
    candidates = [
        p for p in base.iterdir()
        if p.is_dir()
        and ((p / "agent_states.npy").exists() or (p / "agent_positions.npy").exists())
        and (p / "classical_paths.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No visualizable runs under: {base}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_run_dir(value: str) -> Path:
    if value.lower() == "latest":
        return latest_run_dir(PROJ_ROOT / "data" / "online_runs")
    run_dir = Path(value)
    if not run_dir.is_absolute():
        run_dir = PROJ_ROOT / run_dir
    return run_dir


def load_paths(path_file: Path) -> List[List[Tuple[int, int]]]:
    raw = json.loads(path_file.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        keys = sorted(raw.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
        return [[tuple(map(int, point[:2])) for point in raw[key]] for key in keys]
    if isinstance(raw, list):
        return [[tuple(map(int, point[:2])) for point in agent_path] for agent_path in raw]
    raise ValueError(f"Unsupported path JSON format: {path_file}")


def build_task_summary(metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    task_assignments = metrics.get("task_assignments")
    if isinstance(task_assignments, list):
        return {"task_assignments": task_assignments}
    return None


def render_saved_run(
    run_dir: Path,
    out_dir: Path,
    subframes: int,
    max_steps: Optional[int],
    show_planned_routes: Optional[bool],
) -> Path:
    run_dir = run_dir.resolve()
    required = ["classical_paths.json", "classical_metrics.json"]
    missing = [name for name in required if not (run_dir / name).exists()]
    has_agent_states = (run_dir / "agent_states.npy").exists()
    has_agent_positions = (run_dir / "agent_positions.npy").exists()
    if not has_agent_states and not has_agent_positions:
        missing.append("agent_states.npy or agent_positions.npy")
    if missing:
        raise FileNotFoundError(f"{run_dir} is missing required files: {', '.join(missing)}")

    env = fmg.build_factory_map()
    if has_agent_states:
        agent_states = np.load(run_dir / "agent_states.npy")
    else:
        agent_positions = np.load(run_dir / "agent_positions.npy")
        # animate_paths can render plain paths directly. Keep agent_states=None
        # so reference-style PIBT point-agent runs are shown without fake
        # kinodynamic smoothing.
        agent_states = None
    if max_steps is not None:
        if agent_states is not None:
            keep = max(1, min(int(max_steps), int(agent_states.shape[0] - 1)))
            agent_states = agent_states[: keep + 1]
        else:
            keep = max(1, min(int(max_steps), int(agent_positions.shape[0] - 1)))

    paths = load_paths(run_dir / "classical_paths.json")
    if max_steps is not None:
        paths = [path[: max_steps + 1] if len(path) > max_steps + 1 else path for path in paths]

    metrics = json.loads((run_dir / "classical_metrics.json").read_text(encoding="utf-8"))
    final_goals = metrics.get("final_goals")
    goals: Sequence[Tuple[int, int]]
    if final_goals:
        goals = [tuple(map(int, goal[:2])) for goal in final_goals]
    else:
        goals = [path[-1] for path in paths]
    starts = [path[0] for path in paths]
    task_summary = build_task_summary(metrics)

    config = load_config().replace(
        animation_subframes=max(1, int(subframes)),
        save_animation=True,
        max_time=int((agent_states.shape[0] if agent_states is not None else agent_positions.shape[0]) - 1),
    )
    if show_planned_routes is not None:
        config = config.replace(show_planned_routes=bool(show_planned_routes))

    out_dir.mkdir(parents=True, exist_ok=True)
    animate_paths(
        env,
        paths,
        starts,
        goals,
        out_dir,
        config,
        agent_states=agent_states,
        task_summary=task_summary,
    )

    mp4 = out_dir / "classical_mapf_animation.mp4"
    gif = out_dir / "classical_mapf_animation.gif"
    return mp4 if mp4.exists() else gif


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize an already-saved online/classical MAPF run.")
    parser.add_argument(
        "run_dir",
        help="Run directory, e.g. data/online_runs/260618_0506, or 'latest' for latest online run.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: reports/figures/visualize_run/<run_name>.",
    )
    parser.add_argument(
        "--subframes",
        type=int,
        default=3,
        help="Video frames per simulation step at 30 fps. 3=10x speed, 30=real-time.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Render only the first N simulation steps for a quick preview.",
    )
    route_group = parser.add_mutually_exclusive_group()
    route_group.add_argument("--show-routes", action="store_true", help="Show faint planned route underlays.")
    route_group.add_argument("--hide-routes", action="store_true", help="Hide faint planned route underlays.")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    out_dir = Path(args.out) if args.out else PROJ_ROOT / "reports" / "figures" / "visualize_run" / run_dir.name
    show_routes: Optional[bool] = None
    if args.show_routes:
        show_routes = True
    elif args.hide_routes:
        show_routes = False

    rendered = render_saved_run(
        run_dir=run_dir,
        out_dir=out_dir,
        subframes=args.subframes,
        max_steps=args.max_steps,
        show_planned_routes=show_routes,
    )
    print(f"visualized run: {run_dir}")
    print(f"rendered video: {rendered.resolve()}")


if __name__ == "__main__":
    main()
