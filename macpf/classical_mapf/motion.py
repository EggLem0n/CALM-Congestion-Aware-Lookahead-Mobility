"""Kinodynamic motion model and proximity safety controller."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .utils import *


def compute_contextual_speed(
    path: PathType,
    t: int,
    walkable_map: np.ndarray,
    config: MAPFConfig,
    agent_speed_factor: float,
    prev_yaw: float,
) -> Tuple[float, float, List[str]]:
    curr = path_position_at(path, t)
    prev = path_position_at(path, max(0, t - 1))
    nxt = path_position_at(path, min(config.max_time, t + 1))
    moved = t > 0 and curr != prev
    moving_next = nxt != curr

    yaw = movement_yaw(curr, nxt, movement_yaw(prev, curr, prev_yaw))
    if not moved:
        return yaw, 0.0, ["wait/yield"]

    base_speed = config.max_speed_mps * config.straight_speed_factor * agent_speed_factor
    speed = min(base_speed, (manhattan_distance(prev, curr) * config.cell_size_m) / max(config.dt_s, 1e-6))
    tags: List[str] = ["straight"]

    prev_yaw_for_turn = movement_yaw(prev, curr, prev_yaw)
    next_yaw = movement_yaw(curr, nxt, prev_yaw_for_turn)
    turn_angle = abs(normalize_angle(next_yaw - prev_yaw_for_turn))
    if moving_next and turn_angle > math.radians(20):
        speed *= config.turn_speed_factor
        tags.append("turn")

    if walkable_degree(curr, walkable_map) >= 3 or (moving_next and walkable_degree(nxt, walkable_map) >= 3):
        speed *= config.intersection_speed_factor
        tags.append("intersection")

    stop_distance = distance_to_next_stop_or_end(path, t, config.max_time)
    if 0 < stop_distance <= 3:
        factor = config.approach_speed_factor if stop_distance > 1 else config.docking_speed_factor
        speed *= factor
        tags.append("approach/docking")

    had_wait_before = t > 1 and path_position_at(path, t - 1) == path_position_at(path, t - 2)
    waits_next = nxt == curr
    if had_wait_before or waits_next:
        speed *= config.stop_go_speed_factor
        tags.append("stop-go")

    return yaw, clamp(speed, 0.0, config.max_speed_mps), tags


def grid_paths_to_kinodynamic_states(
    paths: Sequence[PathType],
    walkable_map: np.ndarray,
    config: MAPFConfig,
) -> np.ndarray:
    """
    Convert grid waypoint paths into AMR motion states with kinematic constraints.

    State layout:
        [x_cell, y_cell, yaw_rad, speed_mps, angular_speed_radps, accel_mps2]
    """
    states = np.zeros((config.max_time + 1, len(paths), 6), dtype=np.float32)
    rng = np.random.default_rng(config.seed + 2026)
    agent_speed_factors = rng.uniform(
        1.0 - config.agent_speed_variation,
        1.0 + config.agent_speed_variation,
        size=len(paths),
    )
    for agent_id, path in enumerate(paths):
        if not path:
            continue
        prev_yaw = 0.0
        prev_v = 0.0
        agent_speed_factor = float(agent_speed_factors[agent_id])
        for t in range(config.max_time + 1):
            curr = path_position_at(path, t)
            yaw, v, _ = compute_contextual_speed(
                path,
                t,
                walkable_map,
                config,
                agent_speed_factor,
                prev_yaw,
            )
            omega = clamp(
                normalize_angle(yaw - prev_yaw) / max(config.dt_s, 1e-6),
                -config.max_angular_speed_radps,
                config.max_angular_speed_radps,
            )
            if v > 0.05 and abs(omega) > 1e-6 and config.min_turn_radius_m > 0:
                max_v_for_turn = abs(omega) * config.min_turn_radius_m
                v = min(v, max_v_for_turn)
            raw_accel = (v - prev_v) / max(config.dt_s, 1e-6)
            accel = clamp(raw_accel, -config.max_decel_mps2, config.max_accel_mps2)

            states[t, agent_id] = [float(curr[0]), float(curr[1]), yaw, v, omega, accel]
            prev_yaw = yaw
            prev_v = v

    return states


def states_to_agent_positions(agent_states: np.ndarray) -> np.ndarray:
    positions = np.rint(agent_states[:, :, :2]).astype(np.int32)
    return positions


def apply_proximity_safety_controller(
    agent_states: np.ndarray,
    config: MAPFConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply a safety-radius speed controller.

    The grid reservation layer prevents cell collision. This controller models
    local AMR behavior: as another AMR enters the soft safety radius, speed is
    reduced; inside the emergency radius, the AMR stops.
    """
    controlled = agent_states.copy()
    T, N, _ = controlled.shape
    nearest_distances = np.full((T, N), np.inf, dtype=np.float32)
    slowdown_events = 0
    emergency_stop_events = 0
    car_following_events = 0
    ignored_rear_soft_events = 0

    for t in range(T):
        positions = controlled[t, :, :2]
        for i in range(N):
            if N <= 1:
                continue
            deltas = positions - positions[i]
            distances = np.sqrt(np.sum(deltas * deltas, axis=1))
            distances[i] = np.inf
            nearest = float(np.min(distances))
            nearest_distances[t, i] = nearest

            if nearest <= config.emergency_stop_radius_cells:
                speed_factor = 0.0
                emergency_stop_events += 1
            else:
                yaw = float(controlled[t, i, 2])
                heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
                soft_risk_distance = None
                rear_soft_count = 0
                for j in range(N):
                    if i == j:
                        continue
                    distance = float(distances[j])
                    if distance > config.safety_radius_cells:
                        continue
                    longitudinal = float(np.dot(deltas[j], heading))
                    if longitudinal < -0.10:
                        rear_soft_count += 1
                        continue
                    soft_risk_distance = (
                        distance
                        if soft_risk_distance is None
                        else min(soft_risk_distance, distance)
                    )

                if soft_risk_distance is None:
                    speed_factor = 1.0
                    ignored_rear_soft_events += rear_soft_count
                else:
                    span = max(config.safety_radius_cells - config.emergency_stop_radius_cells, 1e-6)
                    ratio = (soft_risk_distance - config.emergency_stop_radius_cells) / span
                    speed_factor = clamp(ratio, config.safety_min_speed_factor, 1.0)
                    slowdown_events += 1

            controlled[t, i, 3] *= speed_factor

        # Car-following behavior: a stopped or slow leader in the same aisle
        # creates a deceleration wave behind it.
        previous_speeds = controlled[max(0, t - 1), :, 3] if t > 0 else controlled[t, :, 3]
        for i in range(N):
            yaw = float(controlled[t, i, 2])
            heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
            lateral_axis = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
            leader_limit = None

            for j in range(N):
                if i == j:
                    continue
                delta = positions[j] - positions[i]
                longitudinal = float(np.dot(delta, heading))
                lateral = abs(float(np.dot(delta, lateral_axis)))
                if longitudinal <= 0.0 or longitudinal > config.car_following_radius_cells:
                    continue
                if lateral > config.car_following_lateral_cells:
                    continue

                gap_ratio = clamp(
                    (longitudinal - config.emergency_stop_radius_cells)
                    / max(config.car_following_radius_cells - config.emergency_stop_radius_cells, 1e-6),
                    0.0,
                    1.0,
                )
                leader_speed = float(previous_speeds[j])
                candidate_limit = leader_speed + gap_ratio * 0.35 * config.max_speed_mps
                leader_limit = candidate_limit if leader_limit is None else min(leader_limit, candidate_limit)

            if leader_limit is not None and controlled[t, i, 3] > leader_limit:
                controlled[t, i, 3] = max(0.0, leader_limit)
                car_following_events += 1

    for i in range(N):
        prev_v = 0.0
        for t in range(T):
            v = float(controlled[t, i, 3])
            controlled[t, i, 5] = clamp(
                (v - prev_v) / max(config.dt_s, 1e-6),
                -config.max_decel_mps2,
                config.max_accel_mps2,
            )
            prev_v = v

    finite_distances = nearest_distances[np.isfinite(nearest_distances)]
    summary = {
        "safety_radius_cells": float(config.safety_radius_cells),
        "emergency_stop_radius_cells": float(config.emergency_stop_radius_cells),
        "proximity_slowdown_events": int(slowdown_events),
        "proximity_emergency_stop_events": int(emergency_stop_events),
        "car_following_slowdown_events": int(car_following_events),
        "ignored_rear_soft_slowdown_candidates": int(ignored_rear_soft_events),
        "min_inter_amr_distance_cells": float(np.min(finite_distances)) if finite_distances.size else None,
        "mean_nearest_amr_distance_cells": float(np.mean(finite_distances)) if finite_distances.size else None,
    }
    return controlled, summary
