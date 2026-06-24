from __future__ import annotations

import argparse
import json
import os
import signal
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from macpf import classical_mapf as mapf
from macpf.classical_mapf import factory_map_generator as factory_map_module
from macpf.generate_heatmap.generate import (
    build_start_marker_sequence,
    build_task_marker_sequences,
    save_episode,
)
from macpf.online_mapf.observe import FrameBuilder, build_dec, build_enc
from macpf.online_mapf.predictor import CongestionPredictor
from macpf.pibt.engine import PIBTEngine
from macpf.pibt.runner import DistanceCache, TaskManager, build_agent_ai_cost_weights


Coord = Tuple[int, int]
PROJ_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ConvLSTM congestion dataset shards from PIBT baseline runs."
    )
    parser.add_argument(
        "--output-dir",
        default="data/heatmap_dataset_pibt",
        help="Dataset output directory. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML config file. Relative paths are resolved from the project root.",
    )
    parser.add_argument("--episodes", type=int, default=167, help="Number of simulation episodes.")
    parser.add_argument(
        "--seconds",
        type=int,
        default=900,
        help="Episode duration in seconds at 1 Hz.",
    )
    parser.add_argument("--num-agents", type=int, default=30, help="AMR count per episode.")
    parser.add_argument("--base-seed", type=int, default=10, help="First random seed.")
    parser.add_argument("--mode", choices=["baseline", "ai"], default="baseline")
    parser.add_argument("--model", default="models/congestion_convlstm.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ai-cost-weight", type=float, default=None)
    parser.add_argument("--ai-cost-threshold", type=float, default=0.0)
    parser.add_argument("--ai-cost-cap", type=float, default=None)
    parser.add_argument("--ai-cost-mode", choices=["additive", "tiebreak"], default="tiebreak")
    parser.add_argument("--skip-ai-fraction", type=float, default=None)
    parser.add_argument("--center-value", type=float, default=None)
    parser.add_argument("--step-value", type=float, default=None)
    parser.add_argument(
        "--num_of_process",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Parallel episode processes. Use 1 to disable multiprocessing.",
    )
    parser.add_argument("--compress", action="store_true", help="Use np.savez_compressed.")
    parser.add_argument(
        "--kinodynamic",
        action="store_true",
        help="Also save agent_states.npy-style speed states in each npz. Training ignores this extra key.",
    )
    parser.add_argument(
        "--continuous-safe-gap",
        type=float,
        default=None,
        help="PIBT continuous segment safe gap in cells. Defaults to config.continuous_safe_gap_cells.",
    )
    return parser.parse_args()


def _resolve_config_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (PROJ_ROOT / path).resolve()
    return path


def _filter_groups(groups: Dict[str, List[Coord]], walkable_map: np.ndarray) -> Dict[str, List[Coord]]:
    return mapf.filter_point_groups_by_walkability(groups, walkable_map)


def generate_episode(
    env: Dict[str, Any],
    episode_id: int,
    args: argparse.Namespace,
    show_progress: bool = True,
) -> Dict[str, Any]:
    overrides = {
        "num_agents": int(args.num_agents),
        "seed": int(args.base_seed) + int(episode_id),
        "max_time": int(args.seconds),
        "congestion_center_value": float(args.center_value),
        "congestion_step_value": float(args.step_value),
        "save_animation": False,
        "use_kinodynamic_motion": False,
        "repeated_task_mode": True,
    }
    if args.ai_cost_weight is not None:
        overrides["ai_cost_weight"] = float(args.ai_cost_weight)
    if args.skip_ai_fraction is not None:
        overrides["online_congestion_skip_fraction"] = float(args.skip_ai_fraction)
    config = mapf.load_config(args.config_resolved).replace(**overrides)

    walkable_map = np.asarray(env["walkable_map"]).astype(bool)
    obstacle_map = np.asarray(env["obstacle_map"]).astype(bool)
    h, w = walkable_map.shape[:2]

    starts, _ = mapf.select_start_goal_pairs(env, walkable_map, config)
    pickup_groups = _filter_groups(
        mapf.normalize_point_groups(env.get("pickup_point_groups"))
        or {"pickup": mapf.normalize_points(env.get("pickup_points"))},
        walkable_map,
    )
    delivery_groups = _filter_groups(
        mapf.normalize_point_groups(env.get("delivery_point_groups"))
        or {"delivery": mapf.normalize_points(env.get("delivery_points"))},
        walkable_map,
    )

    task_manager = TaskManager(
        pickup_groups,
        delivery_groups,
        len(starts),
        int(config.seed) + 3000,
    )
    distance_cache = DistanceCache(walkable_map)
    ai_weights = build_agent_ai_cost_weights(config, len(starts))
    current_cost = np.zeros((1, h, w), dtype=np.float32)
    predictor: Optional[CongestionPredictor] = None
    if args.mode == "ai":
        predictor = CongestionPredictor(_resolve_config_path(args.model), device=args.device)

    def distance_fn(agent_id: int, cell: Coord, goal: Coord) -> float:
        del agent_id
        return distance_cache.distance(cell, goal)

    def normalized_ai_cost(cell: Coord) -> float:
        if args.mode != "ai" or current_cost.size == 0:
            return 0.0
        x, y = cell
        cost_index = min(1, current_cost.shape[0] - 1)
        raw_cost = float(current_cost[cost_index, y, x])
        raw_cost = max(0.0, raw_cost - max(0.0, float(args.ai_cost_threshold)))
        if args.ai_cost_cap is not None:
            raw_cost = min(raw_cost, max(0.0, float(args.ai_cost_cap)))
        return raw_cost

    def candidate_cost_fn(agent_id: int, cell: Coord, goal: Coord, timestep: int) -> float:
        del goal, timestep
        if args.mode != "ai":
            return 0.0
        return float(ai_weights[agent_id]) * normalized_ai_cost(cell)

    engine = PIBTEngine(
        walkable_map,
        len(starts),
        seed=int(config.seed) + 4100,
        distance_fn=distance_fn,
        candidate_cost_fn=candidate_cost_fn if args.mode == "ai" else None,
        candidate_cost_mode=str(args.ai_cost_mode) if args.mode == "ai" else "additive",
        continuous_safe_gap_cells=float(args.continuous_safe_gap),
    )

    if show_progress:
        print(
            f"[episode {episode_id}] PIBT {args.mode}: {config.num_agents} AMRs, "
            f"{config.max_time}s, seed={config.seed}",
            flush=True,
        )

    positions: List[Coord] = [tuple(start) for start in starts]
    paths: List[List[Coord]] = [[tuple(start)] for start in starts]
    frame_builder = FrameBuilder(starts, obstacle_map)
    if predictor is not None:
        initial_frame = frame_builder.frame(np.asarray(starts, dtype=np.int32), task_manager.goals_typed())
        history = deque([initial_frame], maxlen=max(1, int(predictor.t_in)))
    else:
        history = deque(maxlen=1)
    pibt_stats = {
        "pibt_inherited_count": 0,
        "pibt_backtrack_count": 0,
        "pibt_forced_wait_count": 0,
        "pibt_candidate_reject_vertex": 0,
        "pibt_candidate_reject_swap": 0,
        "pibt_candidate_reject_continuous": 0,
    }

    for t in range(int(config.max_time)):
        if predictor is not None and t % max(1, int(config.online_replan_every)) == 0:
            enc = build_enc(list(history), int(predictor.t_in))
            dec = build_dec(history[-1], int(predictor.t_out))
            pred = predictor.predict(enc, dec)
            pred = pred / max(1e-6, float(config.congestion_center_value))
            current_cost = np.concatenate([pred[:1], pred], axis=0)

        result = engine.step(
            positions,
            task_manager.goals,
            assigned=task_manager.assigned_flags(),
            assigned_priority_bonus=1_000_000.0,
            timestep=t,
        )
        pibt_stats["pibt_inherited_count"] += int(result.inherited_count)
        pibt_stats["pibt_backtrack_count"] += int(result.backtrack_count)
        pibt_stats["pibt_forced_wait_count"] += int(result.forced_wait_count)
        pibt_stats["pibt_candidate_reject_vertex"] += int(result.candidate_reject_vertex)
        pibt_stats["pibt_candidate_reject_swap"] += int(result.candidate_reject_swap)
        pibt_stats["pibt_candidate_reject_continuous"] += int(result.candidate_reject_continuous)

        positions = result.next_positions
        for agent_id, pos in enumerate(positions):
            paths[agent_id].append(tuple(pos))
        task_manager.update_arrivals(positions, t + 1)
        if predictor is not None:
            history.append(
                frame_builder.frame(
                    np.asarray(positions, dtype=np.int32),
                    task_manager.goals_typed(),
                )
            )

    task_summary = task_manager.summary(positions, int(config.max_time))
    if args.kinodynamic:
        agent_states = mapf.grid_paths_to_kinodynamic_states(paths, walkable_map, config)
        agent_positions = mapf.states_to_agent_positions(agent_states)
    else:
        agent_states = None
        agent_positions = mapf.paths_to_agent_positions(paths, int(config.max_time))

    occupancy = mapf.build_occupancy_sequence(agent_positions, h, w)
    congestion = mapf.build_additive_congestion_label_sequence(
        agent_positions,
        h,
        w,
        center_value=config.congestion_center_value,
        step_value=config.congestion_step_value,
    )
    pickup_targets, delivery_targets = build_task_marker_sequences(
        task_summary.get("task_assignments", []),
        int(config.max_time),
        h,
        w,
    )
    start_markers = build_start_marker_sequence(starts, int(config.max_time), h, w)
    obstacle = np.repeat(np.asarray(obstacle_map, dtype=np.uint8)[None, :, :], int(config.max_time) + 1, axis=0)

    x = np.stack(
        [occupancy, pickup_targets, delivery_targets, start_markers, obstacle],
        axis=1,
    ).astype(np.uint8)
    y = congestion[:, None, :, :].astype(np.float32)

    collision_count = mapf.compute_collision_count(agent_positions)
    meta = {
        "episode_id": int(episode_id),
        "seed": int(config.seed),
        "frames": int(config.max_time + 1),
        "num_agents": int(config.num_agents),
        "simulator": f"pibt_{args.mode}",
        "pibt_reference_point_agent_mode": True,
        "pibt_kinodynamic_output": bool(args.kinodynamic),
        "pibt_continuous_safe_gap_cells_used": float(args.continuous_safe_gap),
        "pibt_ai_cost_enabled": bool(args.mode == "ai"),
        "pibt_ai_cost_mode": str(args.ai_cost_mode) if args.mode == "ai" else "off",
        "pibt_skip_ai_fraction": float(config.online_congestion_skip_fraction),
        "pibt_ai_weight_mean": float(ai_weights.mean()) if len(ai_weights) else 0.0,
        "ai_model": str(args.model) if args.mode == "ai" else None,
        "collision_count": int(collision_count),
        "total_completed_deliveries": int(task_summary.get("total_completed_deliveries", 0)),
        "total_completed_targets": int(task_summary.get("total_completed_targets", 0)),
        **pibt_stats,
    }
    episode: Dict[str, Any] = {
        "x": x,
        "y": y,
        "agent_positions": agent_positions.astype(np.float32 if args.kinodynamic else np.int16),
        "starts": np.asarray(starts, dtype=np.int16),
        "completed_deliveries": np.asarray(task_summary.get("completed_deliveries", []), dtype=np.int16),
        "meta": meta,
        "config": config.as_dict(),
    }
    if agent_states is not None:
        episode["agent_states"] = agent_states.astype(np.float32)
    return episode


def _init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_episode_job(
    env: Dict[str, Any],
    episode_id: int,
    args: argparse.Namespace,
    output_dir_str: str,
    show_progress: bool,
) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    episode = generate_episode(env, episode_id, args, show_progress=show_progress)
    episode_file = Path(output_dir_str) / f"episode_{episode_id:04d}.npz"
    arrays = {
        "x": episode["x"],
        "y": episode["y"],
        "agent_positions": episode["agent_positions"],
        "starts": episode["starts"],
        "completed_deliveries": episode["completed_deliveries"],
    }
    if "agent_states" in episode:
        arrays["agent_states"] = episode["agent_states"]
    save_episode(episode_file, bool(args.compress), **arrays)
    return episode_id, {"file": episode_file.name, **episode["meta"]}, episode["config"]


def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(args.config)
    if config_path.exists():
        base_config = mapf.load_config(str(config_path))
        args.config_resolved = str(config_path)
        print(f"Loaded config: {config_path}", flush=True)
    else:
        base_config = mapf.load_config()
        args.config_resolved = None
        print(f"Config file not found ({config_path}); using default config.", flush=True)

    if args.center_value is None:
        args.center_value = float(base_config.congestion_center_value)
    if args.step_value is None:
        args.step_value = float(base_config.congestion_step_value)
    if args.continuous_safe_gap is None:
        args.continuous_safe_gap = float(base_config.continuous_safe_gap_cells)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJ_ROOT / output_dir).resolve()
    output_dir = output_dir / datetime.now().strftime("%y%m%d_%H%M")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = factory_map_module.build_factory_map()
    h, w = np.asarray(env["walkable_map"]).shape[:2]
    metadata: Dict[str, Any] = {
        "format": "npz-per-episode",
        "simulator": f"pibt_{args.mode}",
        "frequency_hz": 1,
        "frames_per_episode": int(args.seconds) + 1,
        "episodes": int(args.episodes),
        "target_total_frames": int(args.episodes) * (int(args.seconds) + 1),
        "num_agents": int(args.num_agents),
        "map_shape_hw": [int(h), int(w)],
        "input_channels": [
            "amr_occupancy",
            "current_pickup_targets",
            "current_delivery_targets",
            "initial_start_positions",
            "obstacles",
        ],
        "label_channels": ["additive_current_congestion_heatmap"],
        "storage": {
            "x_dtype": "uint8",
            "y_dtype": "float32",
            "note": "ConvLSTM training can read this with macpf.convjam.train unchanged.",
        },
        "label_definition": {
            "center_value": float(args.center_value),
            "step_value": float(args.step_value),
            "distance_metric": "manhattan",
            "aggregation": "sum over all AMRs without clipping or frame normalization",
        },
        "pibt_settings": {
            "reference_point_agent_mode": True,
            "continuous_safe_gap_cells": float(args.continuous_safe_gap),
            "kinodynamic_output": bool(args.kinodynamic),
            "ai_cost_enabled": bool(args.mode == "ai"),
            "ai_model": str(args.model) if args.mode == "ai" else None,
            "ai_cost_mode": str(args.ai_cost_mode) if args.mode == "ai" else "off",
            "skip_ai_fraction": float(args.skip_ai_fraction) if args.skip_ai_fraction is not None else float(base_config.online_congestion_skip_fraction),
        },
        "episode_files": [],
    }

    workers = max(1, int(args.num_of_process))
    episode_metas: Dict[int, Dict[str, Any]] = {}
    config_template: Dict[str, Any] = {}

    print(
        f"Generating PIBT dataset -> {output_dir} "
        f"({args.episodes} episodes, {args.seconds + 1} frames/episode, {workers} process(es))",
        flush=True,
    )
    try:
        if workers == 1:
            for episode_id in range(int(args.episodes)):
                episode_id, meta, config_template = run_episode_job(
                    env, episode_id, args, str(output_dir), True
                )
                episode_metas[episode_id] = meta
                print(
                    f"[done {episode_id + 1}/{args.episodes}] "
                    f"seed={meta['seed']} deliveries={meta['total_completed_deliveries']} "
                    f"collisions={meta['collision_count']}",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
                futures = {
                    pool.submit(run_episode_job, env, episode_id, args, str(output_dir), False): episode_id
                    for episode_id in range(int(args.episodes))
                }
                done = 0
                for future in as_completed(futures):
                    episode_id, meta, config_template = future.result()
                    episode_metas[episode_id] = meta
                    done += 1
                    print(
                        f"[done {done}/{args.episodes}] episode={episode_id} "
                        f"seed={meta['seed']} deliveries={meta['total_completed_deliveries']} "
                        f"collisions={meta['collision_count']}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("Interrupted. Already written episode files are left in place.", flush=True)
        raise

    total_frames = 0
    for episode_id in sorted(episode_metas):
        meta = episode_metas[episode_id]
        metadata["episode_files"].append(meta)
        total_frames += int(meta["frames"])

    metadata["actual_total_frames"] = int(total_frames)
    metadata["config_template"] = config_template
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "config_template.json").write_text(
        json.dumps(config_template, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved PIBT heatmap dataset: {output_dir}", flush=True)
    print(f"Total frames: {total_frames}", flush=True)


if __name__ == "__main__":
    main()
