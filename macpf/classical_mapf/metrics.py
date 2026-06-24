"""Occupancy sequences, congestion labels, and run metrics."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .utils import *


def paths_to_agent_positions(paths: Sequence[PathType], max_t: int) -> np.ndarray:
    positions = np.zeros((max_t + 1, len(paths), 2), dtype=np.int32)
    for t in range(max_t + 1):
        for agent_id, path in enumerate(paths):
            positions[t, agent_id] = path_position_at(path, t)
    return positions


def _cell_counts(agent_positions: np.ndarray, H: int, W: int, dtype) -> np.ndarray:
    """(T, H, W) count of AMRs per cell per frame, scattered in one vectorized pass."""
    positions = np.asarray(agent_positions)
    T = positions.shape[0]
    counts = np.zeros((T, H, W), dtype=dtype)
    xs = positions[..., 0].astype(np.intp)
    ys = positions[..., 1].astype(np.intp)
    in_bounds = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    t_idx = np.broadcast_to(np.arange(T)[:, None], xs.shape)
    np.add.at(counts, (t_idx[in_bounds], ys[in_bounds], xs[in_bounds]), 1)
    return counts


def build_occupancy_sequence(agent_positions: np.ndarray, H: int, W: int) -> np.ndarray:
    return _cell_counts(agent_positions, H, W, np.uint8)


def build_additive_congestion_label_sequence(
    agent_positions: np.ndarray,
    H: int,
    W: int,
    center_value: float = 100.0,
    step_value: float = 25.0,
) -> np.ndarray:
    """
    Build full-grid additive congestion heatmaps from agent positions.

    Each AMR contributes max(0, center_value - step_value * manhattan_distance)
    to every cell, spreading until the value naturally reaches 0, and
    contributions from all AMRs are simply summed.
    No clipping or per-frame normalization. Output shape: (T, H, W)

    Summing every AMR's Manhattan "tent" equals convolving the per-cell AMR-count map
    with that tent kernel, so we scatter counts once and accumulate one shifted
    slice-add per kernel offset -- O(kernel) vectorized adds instead of a 4-deep loop.
    """
    if step_value <= 0:
        raise ValueError("step_value must be > 0 so each AMR's contribution reaches 0.")
    radius = max(0, math.ceil(center_value / step_value) - 1)
    counts = _cell_counts(agent_positions, H, W, np.float32)
    labels = np.zeros_like(counts)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            value = center_value - step_value * (abs(dx) + abs(dy))
            if value <= 0:
                continue
            # Add value * counts, with the source shifted by (dy, dx) and clipped to the grid.
            ys, ye = max(0, dy), min(H, H + dy)
            xs, xe = max(0, dx), min(W, W + dx)
            labels[:, ys:ye, xs:xe] += value * counts[:, ys - dy : ye - dy, xs - dx : xe - dx]
    return labels


def compute_collision_count(agent_positions: np.ndarray) -> int:
    collisions = 0
    for positions_at_t in agent_positions:
        seen: set[Coord] = set()
        for x, y in positions_at_t:
            coord = (int(x), int(y))
            if coord in seen:
                collisions += 1
            seen.add(coord)
    for t in range(1, agent_positions.shape[0]):
        prev_positions = [tuple(map(int, p)) for p in agent_positions[t - 1]]
        curr_positions = [tuple(map(int, p)) for p in agent_positions[t]]
        for i in range(len(curr_positions)):
            for j in range(i + 1, len(curr_positions)):
                if prev_positions[i] == curr_positions[j] and prev_positions[j] == curr_positions[i]:
                    collisions += 1
    return collisions


def compute_clearance_stats(agent_positions: np.ndarray, clearance_cells: int) -> Dict[str, Any]:
    violations = 0
    min_manhattan: Optional[int] = None
    for positions_at_t in agent_positions:
        coords = [tuple(map(int, p)) for p in positions_at_t]
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                distance = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                min_manhattan = distance if min_manhattan is None else min(min_manhattan, distance)
                if distance <= clearance_cells:
                    violations += 1
    return {
        "hard_clearance_cells": int(clearance_cells),
        "hard_clearance_violation_count": int(violations),
        "min_inter_amr_manhattan_distance_cells": int(min_manhattan) if min_manhattan is not None else None,
    }


def compute_interpolated_clearance_stats(
    positions: np.ndarray,
    config: MAPFConfig,
) -> Dict[str, Any]:
    if positions.shape[0] <= 1 or positions.shape[1] <= 1:
        return {
            "continuous_safe_gap_cells": float(config.continuous_safe_gap_cells),
            "interpolated_min_distance_cells": None,
            "interpolated_safe_gap_violation_count": 0,
        }

    coords = np.asarray(positions[:, :, :2], dtype=np.float32)
    subframes = max(1, int(config.animation_subframes))
    safe_gap = float(config.continuous_safe_gap_cells)
    min_distance = float("inf")
    violation_count = 0

    for t in range(coords.shape[0] - 1):
        for sub in range(subframes + 1):
            alpha = sub / subframes
            frame_positions = coords[t] * (1.0 - alpha) + coords[t + 1] * alpha
            for i in range(frame_positions.shape[0]):
                deltas = frame_positions[i + 1 :] - frame_positions[i]
                if deltas.size == 0:
                    continue
                distances = np.sqrt(np.sum(deltas * deltas, axis=1))
                frame_min = float(np.min(distances))
                min_distance = min(min_distance, frame_min)
                violation_count += int(np.sum(distances < safe_gap))

    return {
        "continuous_safe_gap_cells": float(safe_gap),
        "interpolated_min_distance_cells": float(min_distance) if math.isfinite(min_distance) else None,
        "interpolated_safe_gap_violation_count": int(violation_count),
    }


def compute_start_clearance_stats(starts: Sequence[Coord], config: MAPFConfig) -> Dict[str, Any]:
    min_start_distance: Optional[int] = None
    violations = 0
    for i in range(len(starts)):
        for j in range(i + 1, len(starts)):
            distance = manhattan_distance(starts[i], starts[j])
            min_start_distance = distance if min_start_distance is None else min(min_start_distance, distance)
            if distance <= config.initial_start_clearance_cells:
                violations += 1
    return {
        "initial_start_clearance_cells": int(config.initial_start_clearance_cells),
        "min_start_manhattan_distance_cells": int(min_start_distance) if min_start_distance is not None else None,
        "start_clearance_violation_count": int(violations),
    }


def compute_metrics(
    paths: Sequence[PathType],
    starts: Sequence[Coord],
    goals: Sequence[Coord],
    task_summary: Optional[Dict[str, Any]] = None,
    max_t: Optional[int] = None,
) -> Dict[str, Any]:
    makespan = max_t if max_t is not None else max((len(path) - 1 for path in paths), default=0)
    agent_positions = paths_to_agent_positions(paths, makespan)
    path_lengths = [len(path) - 1 for path in paths]
    waiting_times = [
        sum(1 for t in range(1, len(path)) if path[t] == path[t - 1])
        for path in paths
    ]
    successes = [bool(path and path[-1] == goal) for path, goal in zip(paths, goals)]
    metrics = {
        "num_agents": len(paths),
        "path_lengths": path_lengths,
        "total_path_length": int(sum(path_lengths)),
        "makespan": int(makespan),
        "waiting_times": waiting_times,
        "total_waiting_time": int(sum(waiting_times)),
        "collision_count": int(compute_collision_count(agent_positions)),
        "delivery_success_count": int(sum(successes)),
        "delivery_success_rate": float(sum(successes) / len(successes)) if successes else 0.0,
        "starts": [list(p) for p in starts],
        "goals": [list(p) for p in goals],
    }
    if task_summary:
        metrics.update(task_summary)
        completed_deliveries = task_summary.get("completed_deliveries", [])
        metrics["delivery_success_count"] = int(sum(1 for count in completed_deliveries if count > 0))
        metrics["delivery_success_rate"] = (
            float(metrics["delivery_success_count"] / len(completed_deliveries))
            if completed_deliveries
            else 0.0
        )
        metrics["throughput_deliveries"] = int(task_summary.get("total_completed_deliveries", 0))
        metrics["average_deliveries_per_agent"] = (
            float(metrics["throughput_deliveries"] / len(completed_deliveries))
            if completed_deliveries
            else 0.0
        )
    return metrics
