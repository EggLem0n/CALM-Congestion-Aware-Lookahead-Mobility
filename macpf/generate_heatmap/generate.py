from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from multiprocessing import Value
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from macpf.classical_mapf import factory_map_generator as factory_map_module
from macpf import classical_mapf as mapf


Coord = Tuple[int, int]

# This file lives at macpf/macpf/generate_heatmap/generate.py, so the project root
# (which holds data/ and configs/) is three levels up. Computed locally rather than
# imported from macpf.config to avoid re-running config side effects in every worker
# process spawned below.
PROJ_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MAPF/AMR congestion heatmap dataset shards for ConvLSTM training. "
                    "Each round sweeps the AMR count from --min-agents to --max-agents (one episode "
                    "per count); the spawn layout is randomized per episode by the seed.",
    )
    parser.add_argument(
        "--num_of_process",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Parallel episode processes. 1 disables multiprocessing. "
             "Default is half the logical processors to keep CPU temperature in check.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=10,
        help="First random seed; episode i uses base_seed + i. Fixes the spawn layout per episode "
             "for reproducibility.",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=None,
        help="Episode duration in seconds at 1 Hz. Default: max_time from configs/default.yaml.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="How many times to sweep the AMR-count range. Total episodes = rounds * "
             "(max_agents - min_agents + 1). Each round reuses every count once with fresh seeds.",
    )
    parser.add_argument(
        "--min-agents",
        type=int,
        default=20,
        help="Lower bound of the per-episode AMR-count sweep (inclusive).",
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=50,
        help="Upper bound of the per-episode AMR-count sweep (inclusive).",
    )
    parser.add_argument(
        "--distributed-start-frac",
        type=float,
        default=0.0,
        help="Fraction of episodes (0..1) that spawn agents spread across the whole map "
             "(distributed_starts=true) instead of the staging-aisle default. 0 keeps every "
             "episode at staging. Chosen per episode, deterministic from the seed.",
    )
    parser.add_argument(
        "--center-value",
        type=float,
        default=None,
        help="Congestion value at each AMR's own cell. Default: from configs/default.yaml.",
    )
    parser.add_argument(
        "--step-value",
        type=float,
        default=None,
        help="Decrease per Manhattan cell, spreading until 0. Default: from configs/default.yaml.",
    )
    args = parser.parse_args()
    if args.min_agents < 1 or args.min_agents > args.max_agents:
        parser.error("Require 1 <= --min-agents <= --max-agents.")
    if args.rounds < 1:
        parser.error("--rounds must be >= 1.")
    if not (0.0 <= args.distributed_start_frac <= 1.0):
        parser.error("--distributed-start-frac must be in [0, 1].")
    return args


def build_task_marker_sequences(
    task_assignments: Sequence[Sequence[Dict[str, Any]]],
    max_time: int,
    h: int,
    w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mark each task's target cell as active over its [start_t, end_t] window.

    Each AMR's assignments are sequential (pickup -> delivery -> ...) and non-overlapping
    in time, so writing each assignment's window directly is equivalent to scanning every
    timestep for the active assignment, but O(assignments) instead of O(T * agents).
    """
    pickup_targets = np.zeros((max_time + 1, h, w), dtype=np.uint8)
    delivery_targets = np.zeros((max_time + 1, h, w), dtype=np.uint8)

    for assignments in task_assignments:
        for assignment in assignments:
            target = assignment.get("target")
            if not isinstance(target, list) or len(target) < 2:
                continue
            x, y = int(target[0]), int(target[1])
            if not (0 <= x < w and 0 <= y < h):
                continue
            t0 = max(0, int(assignment.get("start_t", 0)))
            t1 = min(max_time, int(assignment.get("end_t", -1)))
            if t1 < t0:
                continue
            markers = pickup_targets if assignment.get("action") == "pickup" else delivery_targets
            markers[t0 : t1 + 1, y, x] = 1

    return pickup_targets, delivery_targets


def build_start_marker_map(starts: Sequence[Coord], h: int, w: int) -> np.ndarray:
    """A single (H, W) frame marking initial start cells; broadcast over time by the caller."""
    start_map = np.zeros((h, w), dtype=np.uint8)
    for x, y in starts:
        if 0 <= x < w and 0 <= y < h:
            start_map[y, x] = 1
    return start_map


def save_episode(output_path: Path, **arrays: np.ndarray) -> None:
    """Write one episode shard as an uncompressed .npz (fastest to write and to mmap-load)."""
    np.savez(output_path, **arrays)


def generate_episode(
    env: Dict[str, Any],
    episode_id: int,
    args: argparse.Namespace,
    show_progress: bool = True,
) -> Dict[str, Any]:
    base_config = mapf.load_config()
    ep_seed = args.base_seed + episode_id

    # AMR count sweeps deterministically across [min, max]: episode i in a round gets
    # min + (i % sweep_size), so one round visits every count exactly once. The spawn
    # layout is what the seed randomizes (via select_start_goal_pairs below).
    sweep_size = args.max_agents - args.min_agents + 1
    ep_num_agents = args.min_agents + (episode_id % sweep_size)
    # Optional: spread some episodes across the whole map. A dedicated RNG stream (distinct
    # seed sequence from the start/goal sampler) keeps the toggle independent of spawn cells.
    if args.distributed_start_frac > 0.0:
        ep_distributed = bool(np.random.default_rng([ep_seed, 0xC0FFEE]).random() < args.distributed_start_frac)
    else:
        ep_distributed = False

    config = base_config.replace(
        num_agents=ep_num_agents,
        distributed_starts=ep_distributed,
        seed=ep_seed,
        max_time=args.seconds,
        congestion_center_value=args.center_value,
        congestion_step_value=args.step_value,
        save_animation=False,
        use_kinodynamic_motion=False,
        repeated_task_mode=True,
        show_planning_progress=show_progress,
    )
    walkable_map = np.asarray(env["walkable_map"]).astype(bool)
    h, w = walkable_map.shape[:2]
    starts, _ = mapf.select_start_goal_pairs(env, walkable_map, config)
    congestion_cost = np.zeros(
        (config.max_time + config.continuous_task_lookahead + 1, h, w),
        dtype=np.float32,
    )

    pickup_points = [
        point for point in mapf.normalize_points(env.get("pickup_points"))
        if mapf.is_walkable(*point, walkable_map)
    ]
    delivery_points = [
        point for point in mapf.normalize_points(env.get("delivery_points"))
        if mapf.is_walkable(*point, walkable_map)
    ]

    if show_progress:
        print(
            f"[episode {episode_id}] planning {config.num_agents} AMR paths "
            f"over a {config.max_time}s horizon (seed={config.seed})",
            flush=True,
        )
    paths, task_summary = mapf.prioritized_planning_repeated_tasks(
        starts,
        pickup_points,
        delivery_points,
        walkable_map,
        congestion_cost,
        config,
        pickup_point_groups=mapf.normalize_point_groups(env.get("pickup_point_groups")),
        delivery_point_groups=mapf.normalize_point_groups(env.get("delivery_point_groups")),
    )

    if show_progress:
        print(f"[episode {episode_id}] building input tensors and congestion labels", flush=True)
    agent_positions = mapf.paths_to_agent_positions(paths, config.max_time)
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
        config.max_time,
        h,
        w,
    )
    # Start positions and obstacles are time-invariant: broadcast a single (H, W) frame
    # to (T, H, W) as a view (no per-frame copy) -- np.stack materializes the copy once.
    frames = config.max_time + 1
    start_markers = np.broadcast_to(build_start_marker_map(starts, h, w), (frames, h, w))
    obstacle = np.broadcast_to(np.asarray(env["obstacle_map"], dtype=np.uint8), (frames, h, w))

    # Shape: (T, C, H, W). Inputs are 1 Hz frame sequences for ConvLSTM.
    # Stored as uint8 (all channels are 0/1 flags or small counts) to keep
    # files 4x smaller; convert to float32 in the training data loader.
    x = np.stack(
        [occupancy, pickup_targets, delivery_targets, start_markers, obstacle],
        axis=1,
    ).astype(np.uint8)
    y = congestion[:, None, :, :]

    collision_count = mapf.compute_collision_count(agent_positions)
    return {
        "x": x,
        "y": y,
        "agent_positions": agent_positions.astype(np.int16),
        "starts": np.asarray(starts, dtype=np.int16),
        "completed_deliveries": np.asarray(task_summary.get("completed_deliveries", []), dtype=np.int16),
        "meta": {
            "episode_id": episode_id,
            "seed": config.seed,
            "frames": int(config.max_time + 1),
            "num_agents": int(config.num_agents),
            "distributed_starts": bool(config.distributed_starts),
            "collision_count": int(collision_count),
            "total_completed_deliveries": int(task_summary.get("total_completed_deliveries", 0)),
            "priority_boost_event_count": int(task_summary.get("priority_boost_event_count", 0)),
            "repeated_planning_failed_count": int(task_summary.get("repeated_planning_failed_count", 0)),
            "safe_extension_strict_steps": int(task_summary.get("safe_extension_strict_steps", 0)),
            "safe_extension_relaxed_steps": int(task_summary.get("safe_extension_relaxed_steps", 0)),
            "safe_extension_failed_steps": int(task_summary.get("safe_extension_failed_steps", 0)),
        },
        "config": config.as_dict(),
    }


_progress_counter = None


def _init_progress_hook(counter) -> None:
    """Install the per-agent progress hook. Runs once in each worker process."""
    global _progress_counter
    _progress_counter = counter
    mapf.set_planning_progress_hook(_on_agent_planned)


def _init_worker(counter) -> None:
    """Worker process setup: progress hook + ignore Ctrl+C.

    Ctrl+C is handled by the main process, which terminates workers explicitly.
    Without this, each worker raises its own KeyboardInterrupt and the pool
    keeps starting the remaining episodes.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _init_progress_hook(counter)


def _on_agent_planned() -> None:
    if _progress_counter is not None:
        with _progress_counter.get_lock():
            _progress_counter.value += 1


class ProgressBox:
    """In-place box display: redraws itself with ANSI cursor moves."""

    INNER_WIDTH = 70
    BAR_WIDTH = 56

    def __init__(self, counter, total_agents: int, total_episodes: int, workers: int):
        self.counter = counter
        self.total_agents = max(1, total_agents)
        self.total_episodes = total_episodes
        self.workers = workers
        self.started = time.perf_counter()
        self.episodes_done = 0
        self.last_event = "(waiting for the first episode to finish)"
        self._lock = threading.RLock()
        self._lines_drawn = 0
        os.system("")  # enable ANSI escape sequences on Windows consoles

    def episode_finished(self, message: str) -> None:
        with self._lock:
            self.episodes_done += 1
            self.last_event = message
            self.draw()

    def _body_lines(self) -> List[str]:
        agents_done = min(int(self.counter.value), self.total_agents)
        frac = agents_done / self.total_agents
        filled = int(self.BAR_WIDTH * frac)
        elapsed = time.perf_counter() - self.started
        eta_text = (
            f"{((elapsed / frac) - elapsed) / 60.0:5.1f} min" if frac > 0 else "  --  "
        )
        return [
            f"MAPF congestion heatmap dataset  ({self.workers} processes)",
            f"[{'#' * filled}{'-' * (self.BAR_WIDTH - filled)}] {frac * 100.0:5.1f}%",
            f"agents planned : {agents_done}/{self.total_agents}",
            f"episodes done  : {self.episodes_done}/{self.total_episodes}",
            f"elapsed {elapsed / 60.0:5.1f} min | eta {eta_text}",
            f"last : {self.last_event}",
        ]

    def draw(self) -> None:
        with self._lock:
            width = self.INNER_WIDTH
            rows = ["+" + "-" * (width + 2) + "+"]
            rows += ["| " + text[:width].ljust(width) + " |" for text in self._body_lines()]
            rows.append("+" + "-" * (width + 2) + "+")

            out = ""
            if self._lines_drawn:
                out += f"\x1b[{self._lines_drawn}A"  # cursor up to redraw in place
            out += "".join("\x1b[2K" + row + "\n" for row in rows)
            print(out, end="", flush=True)
            self._lines_drawn = len(rows)


def _refresh_box_loop(box: "ProgressBox", stop_event: threading.Event) -> None:
    while not stop_event.wait(0.5):
        box.draw()


def run_episode_job(
    env: Dict[str, Any],
    episode_id: int,
    args: argparse.Namespace,
    output_dir_str: str,
    show_progress: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate one episode and save it. Runs in a worker process when --workers > 1.

    The npz is written here so only the small meta dict crosses process boundaries.
    """
    episode = generate_episode(env, episode_id, args, show_progress=show_progress)
    episode_file = Path(output_dir_str) / f"episode_{episode_id:04d}.npz"
    if show_progress:
        print(f"[episode {episode_id}] saving {episode_file.name}", flush=True)
    save_episode(
        episode_file,
        x=episode["x"],
        y=episode["y"],
        agent_positions=episode["agent_positions"],
        starts=episode["starts"],
        completed_deliveries=episode["completed_deliveries"],
    )
    meta = {"file": episode_file.name, **episode["meta"]}
    return meta, episode["config"]


def main() -> None:
    args = parse_args()

    base_config = mapf.load_config()  # configs/default.yaml
    # CLI options take precedence over the config file.
    if args.seconds is None:
        args.seconds = base_config.max_time
    if args.center_value is None:
        args.center_value = base_config.congestion_center_value
    if args.step_value is None:
        args.step_value = base_config.congestion_step_value

    sweep_size = args.max_agents - args.min_agents + 1
    args.episodes = args.rounds * sweep_size

    output_dir = (
        PROJ_ROOT / "data" / "heatmap_dataset" / datetime.now().strftime("%y%m%d_%H%M")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Generating {args.episodes} episodes "
        f"({args.rounds} round(s) x {sweep_size} AMR counts {args.min_agents}-{args.max_agents}), "
        f"{args.seconds}s each -> {output_dir}",
        flush=True,
    )

    env = factory_map_module.build_factory_map()
    h, w = np.asarray(env["walkable_map"]).shape[:2]
    metadata: Dict[str, Any] = {
        "format": "npz-per-episode",
        "frequency_hz": 1,
        "frames_per_episode": args.seconds + 1,
        "episodes": args.episodes,
        "rounds": args.rounds,
        "target_total_frames": args.episodes * (args.seconds + 1),
        "num_agents_sweep": [int(args.min_agents), int(args.max_agents)],
        "distributed_start_frac": float(args.distributed_start_frac),
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
            "note": "Convert x to float32 in the training data loader.",
        },
        "label_definition": {
            "center_value": args.center_value,
            "step_value": args.step_value,
            "distance_metric": "manhattan",
            "per_cell_value": f"max(0, {args.center_value:g} - {args.step_value:g} * manhattan_distance)",
            "aggregation": "sum over all AMRs without clipping or frame normalization",
        },
        "episode_files": [],
    }

    total_frames = 0
    workers = max(1, int(args.num_of_process))
    episode_metas: Dict[int, Dict[str, Any]] = {}
    config_template: Dict[str, Any] = {}

    counter = Value("i", 0)
    # Exact agent total: each round sweeps every count in [min, max] once.
    total_agents = args.rounds * sum(range(args.min_agents, args.max_agents + 1))
    box = ProgressBox(counter, total_agents, args.episodes, workers)
    stop_refresh = threading.Event()
    refresher = threading.Thread(target=_refresh_box_loop, args=(box, stop_refresh), daemon=True)
    box.draw()
    refresher.start()

    def episode_summary(episode_id: int, meta: Dict[str, Any]) -> str:
        return (
            f"episode {episode_id} (seed={meta['seed']}, agents={meta['num_agents']}, "
            f"deliveries={meta['total_completed_deliveries']}, "
            f"collisions={meta['collision_count']})"
        )

    try:
        if workers == 1:
            _init_progress_hook(counter)
            for episode_id in range(args.episodes):
                meta, config_template = run_episode_job(
                    env, episode_id, args, str(output_dir), show_progress=False
                )
                episode_metas[episode_id] = meta
                box.episode_finished(episode_summary(episode_id, meta))
        else:
            pool = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(counter,),
            )
            try:
                futures = {
                    pool.submit(
                        run_episode_job, env, episode_id, args, str(output_dir), False
                    ): episode_id
                    for episode_id in range(args.episodes)
                }
                for future in as_completed(futures):
                    episode_id = futures[future]
                    meta, config_template = future.result()
                    episode_metas[episode_id] = meta
                    box.episode_finished(episode_summary(episode_id, meta))
                pool.shutdown()
            except KeyboardInterrupt:
                # Kill workers immediately instead of letting the pool keep
                # running the remaining episodes.
                for process in list(getattr(pool, "_processes", {}).values()):
                    process.terminate()
                pool.shutdown(wait=False, cancel_futures=True)
                raise
    except KeyboardInterrupt:
        stop_refresh.set()
        print("\nInterrupted by Ctrl+C - all worker processes terminated.", flush=True)
        print(f"Partial episode files remain in: {output_dir}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        # Bypass normal interpreter shutdown. ProcessPoolExecutor registers an
        # atexit hook (_python_exit) that joins its manager thread and worker
        # queues; after we have force-terminated the workers those queues are
        # broken, so that join can hang forever -- which is exactly what made
        # Ctrl+C feel dead. os._exit() exits immediately without running it.
        os._exit(130)
    finally:
        stop_refresh.set()
        refresher.join(timeout=2.0)
    box.draw()

    for episode_id in range(args.episodes):
        meta = episode_metas[episode_id]
        metadata["episode_files"].append(meta)
        total_frames += int(meta["frames"])

    metadata["actual_total_frames"] = total_frames
    metadata["config_template"] = config_template
    metadata["map_info"] = {
        "pickup_point_groups": {
            name: [list(point) for point in points]
            for name, points in mapf.normalize_point_groups(env.get("pickup_point_groups")).items()
        },
        "delivery_point_groups": {
            name: [list(point) for point in points]
            for name, points in mapf.normalize_point_groups(env.get("delivery_point_groups")).items()
        },
        "labels": env.get("labels"),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    np.save(output_dir / "factory_map.npy", np.asarray(env["factory_map"]))
    np.save(output_dir / "walkable_map.npy", np.asarray(env["walkable_map"]))
    np.save(output_dir / "obstacle_map.npy", np.asarray(env["obstacle_map"]))

    print(f"Saved dataset to {output_dir.resolve()}")
    print(f"Total frames: {total_frames}")


if __name__ == "__main__":
    main()
