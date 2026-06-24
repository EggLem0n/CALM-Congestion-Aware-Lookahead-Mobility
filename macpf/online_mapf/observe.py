"""Build the ConvLSTM's 5-channel input frames from the live world state.

Channel order matches the training data (`macpf.features.INPUT_CHANNELS`):
    [amr_occupancy, current_pickup_targets, current_delivery_targets,
     initial_start_positions, obstacles]

The model was trained on 1 Hz *grid* occupancy (the dataset generator runs with
`use_kinodynamic_motion=False`), so the live occupancy channel is built from grid
cells too -- keeping inference on-distribution.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from macpf.classical_mapf import build_occupancy_sequence

OCCUPANCY_CHANNEL = 0


class FrameBuilder:
    """Builds 5-channel frames; holds the constant start/obstacle channels."""

    def __init__(self, starts: List[Tuple[int, int]], obstacle_map: np.ndarray):
        self.H, self.W = obstacle_map.shape[:2]
        self.obstacle = np.asarray(obstacle_map, dtype=np.float32)
        start_map = np.zeros((self.H, self.W), dtype=np.float32)
        for x, y in starts:
            if 0 <= x < self.W and 0 <= y < self.H:
                start_map[y, x] = 1.0
        self.start_map = start_map

    def frame(self, positions: np.ndarray, goals_typed: List[tuple]) -> np.ndarray:
        """Assemble one (5, H, W) float32 frame from the current state.

        positions   : (N, 2) current grid cells.
        goals_typed : [(goal_cell, is_pickup)] for agents with an active goal.
        """
        occupancy = build_occupancy_sequence(positions[None], self.H, self.W)[0].astype(np.float32)
        pickup = np.zeros((self.H, self.W), dtype=np.float32)
        delivery = np.zeros((self.H, self.W), dtype=np.float32)
        for (gx, gy), is_pickup in goals_typed:
            if 0 <= gx < self.W and 0 <= gy < self.H:
                (pickup if is_pickup else delivery)[gy, gx] = 1.0
        return np.stack([occupancy, pickup, delivery, self.start_map, self.obstacle], axis=0)


def build_enc(history: List[np.ndarray], t_in: int) -> np.ndarray:
    """(t_in, 5, H, W) encoder input from the rolling history, lead-padded if short.

    Lead-padding (repeat the earliest frame) matches `macpf.convjam.predict`, which
    pads a short observation window the same way.
    """
    frames = list(history)
    if not frames:
        raise ValueError("History is empty; push at least one frame before encoding.")
    if len(frames) < t_in:
        pad = [frames[0]] * (t_in - len(frames))
        frames = pad + frames
    else:
        frames = frames[-t_in:]
    return np.stack(frames, axis=0).astype(np.float32)


def build_dec(last_frame: np.ndarray, t_out: int) -> np.ndarray:
    """(t_out, 5, H, W) decoder input: known exogenous future, occupancy zeroed.

    Future occupancy is unknown, so it is zeroed (as in training/inference). The
    exogenous channels (targets/start/obstacle) are held at their current values
    across the forecast window.
    """
    dec = np.repeat(last_frame[None], t_out, axis=0).astype(np.float32)
    dec[:, OCCUPANCY_CHANNEL] = 0.0
    return dec
