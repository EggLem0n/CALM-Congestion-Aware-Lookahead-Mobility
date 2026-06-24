"""Single-run classical MAPF simulation entry point.

Run as a module:  python -m macpf.classical_mapf.classical_mapf

The engine lives in sibling submodules of the `macpf.classical_mapf` package:
- utils    : config loader + shared types (Coord/PathType/AStarNode) + grid helpers
- solver   : reservation tables, A*, and prioritized planners
- motion   : kinodynamic motion model and safety controller
- metrics  : occupancy/congestion labels and run metrics
- viz      : path plots and GIF animation

The public API is re-exported by macpf/classical_mapf/__init__.py, so callers
just `from macpf import classical_mapf as mapf` and call mapf.<function>.
"""
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import *
from .solver import *
from .motion import *
from .metrics import *
from .viz import animate_paths, plot_map_background, visualize_paths

# This file lives at macpf/macpf/classical_mapf/classical_mapf.py, so the
# project root (which holds data/, reports/, configs/) is three levels up.
PROJ_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJ_ROOT / "reports" / "figures"


def load_factory_environment(config: Optional[MAPFConfig] = None) -> Dict[str, Any]:
    config = config or load_config()
    try:
        map_module = importlib.import_module(config.map_module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import the map module '{config.map_module_name}'. "
            "Install the macpf package (pip install -e .) so it is importable."
        ) from exc

    env = map_module.build_factory_map()
    required = {
        "factory_map",
        "walkable_map",
        "obstacle_map",
        "pickup_points",
        "delivery_points",
        "charging_points",
        "start_candidates",
        "labels",
    }
    missing = sorted(required.difference(env))
    if missing:
        raise KeyError(f"Factory map is missing required keys: {missing}")
    return env

def save_results(
    paths: Sequence[PathType],
    metrics: Dict[str, Any],
    agent_positions: np.ndarray,
    agent_states: Optional[np.ndarray],
    occupancy_sequence: np.ndarray,
    congestion_labels: np.ndarray,
    env: Dict[str, Any],
    config: MAPFConfig,
    run_id: str,
) -> Path:
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        # config.output_dir is relative to the project root (macpf/).
        output_dir = (PROJ_ROOT / output_dir).resolve()
    output_dir = output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable_paths = {str(i): [list(pos) for pos in path] for i, path in enumerate(paths)}
    (output_dir / "classical_paths.json").write_text(
        json.dumps(serializable_paths, indent=2), encoding="utf-8"
    )
    (output_dir / "classical_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.save(output_dir / "agent_positions.npy", agent_positions)
    if agent_states is not None:
        np.save(output_dir / "agent_states.npy", agent_states)
    np.save(output_dir / "occupancy_sequence.npy", occupancy_sequence)
    np.save(output_dir / "congestion_labels.npy", congestion_labels)
    np.save(
        output_dir / "map_info.npy",
        {
            "factory_map": np.asarray(env["factory_map"]),
            "walkable_map": np.asarray(env["walkable_map"]),
            "obstacle_map": np.asarray(env["obstacle_map"]),
            "pickup_points": normalize_points(env.get("pickup_points")),
            "pickup_point_groups": {
                name: [list(point) for point in points]
                for name, points in normalize_point_groups(env.get("pickup_point_groups")).items()
            },
            "delivery_points": normalize_points(env.get("delivery_points")),
            "delivery_point_groups": {
                name: [list(point) for point in points]
                for name, points in normalize_point_groups(env.get("delivery_point_groups")).items()
            },
            "charging_points": normalize_points(env.get("charging_points")),
            "start_candidates": normalize_points(env.get("start_candidates")),
            "labels": env.get("labels"),
            "colors": env.get("colors"),
            "config": config.as_dict(),
        },
        allow_pickle=True,
    )
    return output_dir

def resolve_config_path(raw_path: str) -> Path:
    """Resolve a config path relative to the project root (macpf/)."""
    path = Path(raw_path)
    if not path.is_absolute():
        path = (PROJ_ROOT / path).resolve()
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one classical MAPF simulation.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML config file. Relative paths are resolved from the project root.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config.seed.")
    parser.add_argument("--max-time", type=int, default=None, help="Override config.max_time.")
    parser.add_argument("--num-agents", type=int, default=None, help="Override config.num_agents.")
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="Sweep mode: write the metrics JSON here and skip the full data save + figures.",
    )
    parser.add_argument("--run-id", default=None, help="Run dir / figures name (default: timestamp).")
    parser.add_argument("--save-animation", action="store_true",
                        help="Force the animation on (overrides config.save_animation).")
    parser.add_argument("--subframes", type=int, default=None,
                        help="Override animation_subframes (3=10x speed, 30=real-time at 30 fps).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"Loaded config: {config_path}")
    else:
        config = load_config()
        print(f"Config file not found ({config_path}); using configs/default.yaml.")
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.max_time is not None:
        overrides["max_time"] = args.max_time
    if args.num_agents is not None:
        overrides["num_agents"] = args.num_agents
    if args.save_animation:
        overrides["save_animation"] = True
    if args.subframes is not None:
        overrides["animation_subframes"] = args.subframes
    if overrides:
        config = config.replace(**overrides)
    env = load_factory_environment(config)
    walkable_map = np.asarray(env["walkable_map"]).astype(bool)
    H, W = walkable_map.shape[:2]
    starts, goals = select_start_goal_pairs(env, walkable_map, config)
    congestion_cost = load_ai_congestion_cost(
        config.ai_cost_path if config.use_ai_congestion_cost else None,
        config.max_time + 1,
        H,
        W,
    )

    task_summary = None
    solver_summary: Dict[str, Any] = {
        "mapf_solver": "repeated_task_prioritized_with_dynamic_priority"
        if config.repeated_task_mode
        else "prioritized"
    }
    if config.repeated_task_mode:
        pickup_points = [
            p for p in normalize_points(env.get("pickup_points"))
            if is_walkable(*p, walkable_map)
        ]
        delivery_points = [
            p for p in normalize_points(env.get("delivery_points"))
            if is_walkable(*p, walkable_map)
        ]
        if not pickup_points:
            raise ValueError("Repeated task mode needs at least one walkable pickup point.")
        if not delivery_points:
            raise ValueError("Repeated task mode needs at least one walkable delivery point.")
        pickup_point_groups = normalize_point_groups(env.get("pickup_point_groups"))
        delivery_point_groups = normalize_point_groups(env.get("delivery_point_groups"))
        paths, task_summary = prioritized_planning_repeated_tasks(
            starts,
            pickup_points,
            delivery_points,
            walkable_map,
            congestion_cost,
            config,
            pickup_point_groups=pickup_point_groups,
            delivery_point_groups=delivery_point_groups,
        )
        goals = [tuple(p) for p in task_summary["final_goals"]]
        max_t = config.max_time
    else:
        paths = prioritized_planning(starts, goals, walkable_map, congestion_cost, config)
        solver_summary = {"mapf_solver": "prioritized"}
        max_t = max((len(path) - 1 for path in paths), default=0)

    repair_summary = None
    if config.hard_clearance_cells > 0:
        paths, repair_summary = repair_paths_with_clearance(paths, walkable_map, config)
        max_t = config.max_time

    agent_states = None
    safety_summary = None
    if config.use_kinodynamic_motion:
        agent_states = grid_paths_to_kinodynamic_states(paths, walkable_map, config)
        agent_states, safety_summary = apply_proximity_safety_controller(agent_states, config)
        agent_positions = states_to_agent_positions(agent_states)
        max_t = config.max_time
    else:
        agent_positions = paths_to_agent_positions(paths, max_t)

    metrics = compute_metrics(paths, starts, goals, task_summary=task_summary, max_t=max_t)
    metrics.update(solver_summary)
    if repair_summary:
        metrics.update(repair_summary)
    metrics["motion_model"] = (
        "kinodynamic_unicycle_with_speed_accel_decel_yaw_omega_turn_radius"
        if config.use_kinodynamic_motion
        else "grid_discrete"
    )
    metrics["soft_proximity_cost_radius_cells"] = int(config.soft_proximity_cost_radius_cells)
    metrics["soft_proximity_cost_weight"] = float(config.soft_proximity_cost_weight)
    metrics["use_local_amr_avoidance"] = bool(config.use_local_amr_avoidance)
    metrics["local_avoidance_radius_cells"] = float(config.local_avoidance_radius_cells)
    metrics["local_avoidance_strength"] = float(config.local_avoidance_strength)
    metrics["local_avoidance_max_offset_cells"] = float(config.local_avoidance_max_offset_cells)
    metrics["local_path_update_hz"] = float(config.local_path_update_hz)
    metrics["use_priority_yielding"] = bool(config.use_priority_yielding)
    metrics["yield_decision_hold_frames"] = int(config.yield_decision_hold_frames)
    metrics["use_local_detour_when_blocked"] = bool(config.use_local_detour_when_blocked)
    metrics["local_detour_max_step_cells"] = float(config.local_detour_max_step_cells)
    metrics["visual_safe_gap_cells"] = float(config.visual_safe_gap_cells)
    if agent_states is not None:
        metrics["max_observed_speed_mps"] = float(np.max(agent_states[:, :, 3]))
        metrics["min_nonzero_observed_speed_mps"] = float(
            np.min(agent_states[:, :, 3][agent_states[:, :, 3] > 0])
        )
        metrics["mean_observed_speed_mps"] = float(np.mean(agent_states[:, :, 3]))
        metrics["std_observed_speed_mps"] = float(np.std(agent_states[:, :, 3]))
        metrics["max_observed_abs_accel_mps2"] = float(np.max(np.abs(agent_states[:, :, 5])))
        metrics["max_observed_abs_omega_radps"] = float(np.max(np.abs(agent_states[:, :, 4])))
        metrics["vehicle_size_cells"] = float(config.vehicle_size_cells)
    if safety_summary:
        metrics.update(safety_summary)
    metrics["collision_count"] = int(compute_collision_count(agent_positions))
    metrics.update(compute_start_clearance_stats(starts, config))
    metrics.update(compute_clearance_stats(agent_positions, config.hard_clearance_cells))
    if agent_states is not None:
        metrics.update(compute_interpolated_clearance_stats(agent_states[:, :, :2], config))
    else:
        metrics.update(compute_interpolated_clearance_stats(agent_positions.astype(np.float32), config))

    occupancy_sequence = build_occupancy_sequence(agent_positions, H, W)
    congestion_labels = build_additive_congestion_label_sequence(
        agent_positions,
        H,
        W,
        center_value=config.congestion_center_value,
        step_value=config.congestion_step_value,
    )
    # Congestion KPIs (the project's headline objective): lower is better. Same keys
    # the online planner emits, so classical vs online runs compare directly. Total
    # congestion is ~constant (each robot's kernel sums to a fixed value), so we
    # measure *clustering*: peak overlap + cell-timesteps where >=2 robots stack.
    metrics["congestion_peak"] = float(congestion_labels.max())
    metrics["congestion_overlap_cell_count"] = int(
        (congestion_labels > float(config.congestion_center_value)).sum()
    )

    if args.metrics_out:
        # Sweep mode: emit just the metrics, skip the full data save + figures.
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"metrics -> {out.resolve()}")
        return

    run_id = args.run_id or datetime.now().strftime("%y%m%d_%H%M")
    output_dir = save_results(
        paths,
        metrics,
        agent_positions,
        agent_states,
        occupancy_sequence,
        congestion_labels,
        env,
        config,
        run_id,
    )
    # Human-facing plots go under reports/figures/, kept separate from the
    # data artifacts (json/npy) saved above under data/.
    figures_dir = FIGURES_DIR / "classical_runs" / run_id
    figures_dir.mkdir(parents=True, exist_ok=True)
    visualize_paths(env, paths, starts, goals, figures_dir)
    if config.save_animation:
        animate_paths(
            env,
            paths,
            starts,
            goals,
            figures_dir,
            config,
            agent_states=agent_states,
            task_summary=task_summary,
        )

    print(f"Saved Classical MAPF data to:    {output_dir.resolve()}")
    print(f"Saved Classical MAPF figures to: {figures_dir.resolve()}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
