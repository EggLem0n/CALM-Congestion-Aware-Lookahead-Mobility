"""Occupancy sequences, additive congestion labels, and collision counting."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .grid import path_position_at
from .types import PathType


def paths_to_agent_positions(paths: Sequence[PathType], max_t: int) -> np.ndarray:
    """(T+1, N, 2) array of every agent's (x, y) at each timestep (paths clamp at end)."""
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
    """Full-grid additive congestion heatmaps from agent positions, shape (T, H, W).

    Each AMR contributes max(0, center_value - step_value * manhattan_distance) to
    every cell, spreading until it reaches 0; contributions from all AMRs are summed
    (no clipping, no per-frame normalization).

    Summing every AMR's Manhattan "tent" equals convolving the per-cell AMR-count
    map with that tent kernel, so we scatter counts once and accumulate one shifted
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
            ys, ye = max(0, dy), min(H, H + dy)
            xs, xe = max(0, dx), min(W, W + dx)
            labels[:, ys:ye, xs:xe] += value * counts[:, ys - dy : ye - dy, xs - dx : xe - dx]
    return labels


def compute_collision_count(agent_positions: np.ndarray) -> int:
    """Vertex collisions (two AMRs in one cell) + edge/swap collisions across frames.

    The swap check uses a per-frame reverse-move set (O(T*N)) instead of the old
    O(T*N^2) pairwise scan -- the quadratic version cost ~19 s for one 750-agent /
    1800-step episode, which dominated long high-agent runs.
    """
    positions = np.asarray(agent_positions)
    collisions = 0
    for positions_at_t in positions:
        seen: set = set()
        for x, y in positions_at_t:
            coord = (int(x), int(y))
            if coord in seen:
                collisions += 1
            seen.add(coord)
    for t in range(1, positions.shape[0]):
        prev = positions[t - 1]
        curr = positions[t]
        moves: set = set()
        for k in range(positions.shape[1]):
            px, py, cx, cy = int(prev[k, 0]), int(prev[k, 1]), int(curr[k, 0]), int(curr[k, 1])
            if (px, py) == (cx, cy):
                continue
            if (cx, cy, px, py) in moves:   # reverse move already seen this frame -> a swap
                collisions += 1
            moves.add((px, py, cx, cy))
    return collisions
