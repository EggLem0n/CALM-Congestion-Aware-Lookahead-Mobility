"""PIBT-based MAPD runner with optional ConvLSTM congestion cost.

Baseline and AI modes share the same PIBT engine. The only algorithmic
difference is whether the candidate ordering includes predicted congestion.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
except Exception:
    # Baseline mode can still run without torch; AI mode will raise a clearer
    # error when CongestionPredictor is constructed.
    pass

import numpy as np

from macpf.classical_mapf import (
    apply_proximity_safety_controller,
    build_additive_congestion_label_sequence,
    build_occupancy_sequence,
    compute_clearance_stats,
    compute_collision_count,
    compute_interpolated_clearance_stats,
    compute_metrics,
    compute_start_clearance_stats,
    filter_point_groups_by_walkability,
    grid_paths_to_kinodynamic_states,
    manhattan_distance,
    normalize_point_groups,
    normalize_points,
    paths_to_agent_positions,
    select_start_goal_pairs,
    states_to_agent_positions,
    visualize_paths,
)
from macpf.classical_mapf import animate_paths
from macpf.classical_mapf.classical_mapf import (
    FIGURES_DIR,
    load_factory_environment,
    resolve_config_path,
    save_results,
)
from macpf.classical_mapf.utils import load_config
from macpf.classical_mapf.utils.grid import Coord
from macpf.online_mapf.observe import FrameBuilder, build_dec, build_enc
from macpf.online_mapf.predictor import CongestionPredictor

from throughput_tuning.engine_tuned import PIBTEngine


def flatten_point_groups(groups: Dict[str, List[Coord]]) -> List[Coord]:
    points: List[Coord] = []
    for group_points in groups.values():
        for point in group_points:
            if point not in points:
                points.append(point)
    return points


def build_agent_ai_cost_weights(config, num_agents: int) -> np.ndarray:
    base_weight = float(config.ai_cost_weight)
    lo = float(config.online_agent_ai_weight_min_multiplier)
    hi = float(config.online_agent_ai_weight_max_multiplier)
    skip_fraction = min(1.0, max(0.0, float(config.online_congestion_skip_fraction)))
    if num_agents <= 0:
        return np.zeros(0, dtype=np.float32)
    if hi < lo:
        lo, hi = hi, lo
    multipliers = np.linspace(lo, hi, num_agents, dtype=np.float32)
    if bool(config.online_shuffle_agent_ai_weights):
        rng = np.random.default_rng(int(config.seed) + int(config.online_agent_ai_weight_seed_offset))
        rng.shuffle(multipliers)
    if skip_fraction > 0.0:
        skip_count = int(round(float(num_agents) * skip_fraction))
        if skip_count > 0:
            multipliers[:skip_count] = 0.0
    return (base_weight * multipliers).astype(np.float32)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_wait_diagnostics(paths: Sequence[Sequence[Coord]], max_time: int) -> Dict[str, Any]:
    """Return cumulative and consecutive wait diagnostics from saved grid paths."""
    per_agent_total: List[int] = []
    per_agent_max_streak: List[int] = []
    per_agent_streak_count_ge_5: List[int] = []
    moved_cells = 0

    for path in paths:
        total_wait = 0
        current_streak = 0
        max_streak = 0
        streak_count_ge_5 = 0
        in_ge_5_streak = False
        for t in range(1, min(len(path), int(max_time) + 1)):
            if tuple(path[t]) == tuple(path[t - 1]):
                total_wait += 1
                current_streak += 1
                max_streak = max(max_streak, current_streak)
                if current_streak >= 5 and not in_ge_5_streak:
                    streak_count_ge_5 += 1
                    in_ge_5_streak = True
            else:
                moved_cells += 1
                current_streak = 0
                in_ge_5_streak = False
        per_agent_total.append(total_wait)
        per_agent_max_streak.append(max_streak)
        per_agent_streak_count_ge_5.append(streak_count_ge_5)

    total_wait = int(sum(per_agent_total))
    total_agent_time = max(1, int(max_time) * max(1, len(paths)))
    return {
        "waiting_times": [int(v) for v in per_agent_total],
        "total_waiting_time": total_wait,
        "waiting_ratio": float(total_wait / total_agent_time),
        "actual_moved_cells": int(moved_cells),
        "deliveries_per_moved_cell": None,
        "max_consecutive_wait_per_agent": int(max(per_agent_max_streak) if per_agent_max_streak else 0),
        "mean_max_consecutive_wait_per_agent": float(np.mean(per_agent_max_streak)) if per_agent_max_streak else 0.0,
        "agents_consecutive_wait_ge_5": int(sum(1 for v in per_agent_max_streak if v >= 5)),
        "agents_consecutive_wait_ge_10": int(sum(1 for v in per_agent_max_streak if v >= 10)),
        "agents_consecutive_wait_ge_20": int(sum(1 for v in per_agent_max_streak if v >= 20)),
        "wait_streak_events_ge_5": int(sum(per_agent_streak_count_ge_5)),
    }


def compute_zone_occupancy_metrics(
    agent_positions: np.ndarray,
    factory_map: np.ndarray,
    labels: Optional[Dict[int, str]],
) -> Dict[str, Any]:
    """Compute simple smart-factory zone occupancy metrics from saved positions."""
    if agent_positions.size == 0:
        return {}
    T, N = agent_positions.shape[:2]
    total_agent_time = max(1, int(T) * int(N))
    h, w = factory_map.shape[:2]
    zone_counts: Counter[str] = Counter()
    aisle_counts = np.zeros((h, w), dtype=np.int32)
    label_map = labels or {}

    for t in range(T):
        for agent_id in range(N):
            x, y = agent_positions[t, agent_id, :2]
            x = int(x)
            y = int(y)
            if not (0 <= x < w and 0 <= y < h):
                continue
            zone_value = int(factory_map[y, x])
            zone_name = str(label_map.get(zone_value, f"zone_{zone_value}"))
            zone_counts[zone_name] += 1
            if zone_value == 0:
                aisle_counts[y, x] += 1

    zone_rates = {
        name: float(count / total_agent_time)
        for name, count in sorted(zone_counts.items())
    }
    positive_aisle = aisle_counts[aisle_counts > 0]
    return {
        "zone_occupancy_counts": {name: int(count) for name, count in sorted(zone_counts.items())},
        "zone_occupancy_rates": zone_rates,
        "road_occupied_cell_count": int(np.count_nonzero(aisle_counts)),
        "road_peak_cell_occupancy": int(positive_aisle.max()) if positive_aisle.size else 0,
        "road_mean_used_cell_occupancy": float(positive_aisle.mean()) if positive_aisle.size else 0.0,
    }


def bfs_distance_map(goal: Coord, walkable_map: np.ndarray) -> np.ndarray:
    h, w = walkable_map.shape[:2]
    dist = np.full((h, w), 1_000_000, dtype=np.int32)
    gx, gy = goal
    if not (0 <= gx < w and 0 <= gy < h and bool(walkable_map[gy, gx])):
        return dist
    q: deque[Coord] = deque([goal])
    dist[gy, gx] = 0
    while q:
        x, y = q.popleft()
        nd = int(dist[y, x]) + 1
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and bool(walkable_map[ny, nx]) and nd < dist[ny, nx]:
                dist[ny, nx] = nd
                q.append((nx, ny))
    return dist


class DistanceCache:
    def __init__(self, walkable_map: np.ndarray):
        self.walkable_map = walkable_map
        self._cache: Dict[Coord, np.ndarray] = {}

    def map_for(self, goal: Coord) -> np.ndarray:
        goal = tuple(goal)
        if goal not in self._cache:
            self._cache[goal] = bfs_distance_map(goal, self.walkable_map)
        return self._cache[goal]

    def distance(self, cell: Coord, goal: Coord) -> float:
        x, y = cell
        dist_map = self.map_for(goal)
        value = int(dist_map[y, x])
        if value >= 1_000_000:
            return float(manhattan_distance(cell, goal) + 1_000_000)
        return float(value)


class TaskManager:
    """Simple lifelong pickup -> delivery -> pickup task manager."""

    def __init__(
        self,
        pickup_groups: Dict[str, List[Coord]],
        delivery_groups: Dict[str, List[Coord]],
        num_agents: int,
        seed: int,
    ) -> None:
        self.pickup_groups = pickup_groups
        self.delivery_groups = delivery_groups
        self.pickups = flatten_point_groups(pickup_groups)
        self.deliveries = flatten_point_groups(delivery_groups)
        if not self.pickups:
            raise ValueError("PIBT runner needs at least one walkable pickup point.")
        if not self.deliveries:
            raise ValueError("PIBT runner needs at least one walkable delivery point.")
        self.seed = int(seed)
        self.agent_rngs = [
            np.random.default_rng(self.seed + 7919 * (agent_id + 1))
            for agent_id in range(num_agents)
        ]
        self.carrying = [False for _ in range(num_agents)]
        initial = [
            self._sample_from_groups(self.pickup_groups, agent_id)
            for agent_id in range(num_agents)
        ]
        self.goals: List[Coord] = [point for _, point in initial]
        self.goal_is_pickup = [True for _ in range(num_agents)]
        self.current_zone = [zone for zone, _ in initial]
        self.completed_targets = [0 for _ in range(num_agents)]
        self.completed_deliveries = [0 for _ in range(num_agents)]
        self.task_histories: List[List[Dict[str, Any]]] = [[] for _ in range(num_agents)]
        self.task_assignments: List[List[Dict[str, Any]]] = [
            [self._assignment(agent_id, 0)] for agent_id in range(num_agents)
        ]

    def _sample_from_groups(self, groups: Dict[str, List[Coord]], agent_id: int) -> Tuple[str, Coord]:
        all_points = [(name, point) for name, points in groups.items() for point in points]
        idx = int(self.agent_rngs[agent_id].integers(0, len(all_points)))
        return all_points[idx]

    def _sample_pickup(self, agent_id: int) -> Tuple[str, Coord]:
        return self._sample_from_groups(self.pickup_groups, agent_id)

    def _sample_delivery(self, agent_id: int) -> Tuple[str, Coord]:
        return self._sample_from_groups(self.delivery_groups, agent_id)

    def _assignment(self, agent_id: int, start_t: int) -> Dict[str, Any]:
        return {
            "action": "pickup" if self.goal_is_pickup[agent_id] else "delivery",
            "zone": self.current_zone[agent_id],
            "target": [int(self.goals[agent_id][0]), int(self.goals[agent_id][1])],
            "start_t": int(start_t),
            "end_t": None,
            "completed": False,
        }

    def goals_typed(self) -> List[Tuple[Coord, bool]]:
        return [(self.goals[i], self.goal_is_pickup[i]) for i in range(len(self.goals))]

    def assigned_flags(self) -> List[bool]:
        # Carrying/load-assigned agents are prioritized over free pickup-seeking agents.
        return [not is_pickup for is_pickup in self.goal_is_pickup]

    def update_arrivals(self, positions: Sequence[Coord], t: int) -> None:
        for agent_id, pos in enumerate(positions):
            if tuple(pos) != self.goals[agent_id]:
                continue
            active = self.task_assignments[agent_id][-1]
            active["end_t"] = int(t)
            active["completed"] = True
            self.completed_targets[agent_id] += 1
            if self.goal_is_pickup[agent_id]:
                self.carrying[agent_id] = True
                self.task_histories[agent_id].append(
                    {
                        "action": "pickup",
                        "zone": self.current_zone[agent_id],
                        "position": list(pos),
                        "arrival_t": int(t),
                        "carrying_load_after_action": True,
                    }
                )
                zone, goal = self._sample_delivery(agent_id)
                self.goals[agent_id] = goal
                self.current_zone[agent_id] = zone
                self.goal_is_pickup[agent_id] = False
            else:
                self.carrying[agent_id] = False
                self.completed_deliveries[agent_id] += 1
                self.task_histories[agent_id].append(
                    {
                        "action": "delivery",
                        "zone": self.current_zone[agent_id],
                        "position": list(pos),
                        "arrival_t": int(t),
                        "carrying_load_after_action": False,
                    }
                )
                zone, goal = self._sample_pickup(agent_id)
                self.goals[agent_id] = goal
                self.current_zone[agent_id] = zone
                self.goal_is_pickup[agent_id] = True
            self.task_assignments[agent_id].append(self._assignment(agent_id, t))

    def summary(self, positions: Sequence[Coord], max_time: int) -> Dict[str, Any]:
        assignments: List[List[Dict[str, Any]]] = []
        for agent_assignments in self.task_assignments:
            copied = []
            for item in agent_assignments:
                copied_item = dict(item)
                if copied_item["end_t"] is None:
                    copied_item["end_t"] = int(max_time)
                copied.append(copied_item)
            assignments.append(copied)
        in_progress = []
        for agent_id, pos in enumerate(positions):
            in_progress.append(
                {
                    "action": "pickup" if self.goal_is_pickup[agent_id] else "delivery",
                    "zone": self.current_zone[agent_id],
                    "target": list(self.goals[agent_id]),
                    "current_position_at_observation_end": list(pos),
                    "carrying_load": bool(self.carrying[agent_id]),
                }
            )
        return {
            "completed_targets": [int(v) for v in self.completed_targets],
            "completed_deliveries": [int(v) for v in self.completed_deliveries],
            "total_completed_targets": int(sum(self.completed_targets)),
            "total_completed_deliveries": int(sum(self.completed_deliveries)),
            "delivery_success_count": int(sum(self.completed_deliveries)),
            "delivery_success_rate": float(sum(self.completed_deliveries) / max(1, len(self.completed_deliveries))),
            "throughput_deliveries": int(sum(self.completed_deliveries)),
            "average_deliveries_per_agent": float(sum(self.completed_deliveries) / max(1, len(self.completed_deliveries))),
            "final_goals": [list(goal) for goal in self.goals],
            "task_histories": self.task_histories,
            "task_assignments": assignments,
            "in_progress_targets_at_observation_end": in_progress,
            "pickup_zone_groups": sorted(self.pickup_groups.keys()),
            "delivery_zone_groups": sorted(self.delivery_groups.keys()),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PIBT MAPD baseline or ConvLSTM-congestion-aware mode.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["baseline", "ai"], default="baseline")
    parser.add_argument("--model", default="models/congestion_convlstm.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-time", type=int, default=None)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--save-animation", action="store_true")
    parser.add_argument("--subframes", type=int, default=None)
    parser.add_argument("--ai-cost-weight", type=float, default=None)
    parser.add_argument("--ai-cost-threshold", type=float, default=0.0)
    parser.add_argument("--ai-cost-cap", type=float, default=None)
    parser.add_argument(
        "--ai-cost-mode",
        choices=["additive", "tiebreak"],
        default="additive",
        help="How AI congestion cost affects PIBT candidate ordering. "
        "additive can choose farther cells; tiebreak only ranks equally short candidates.",
    )
    parser.add_argument(
        "--ai-priority-weight",
        type=float,
        default=0.0,
        help="AI-only priority bonus for AMRs currently inside predicted congestion. "
        "Targets remain random; this only changes which AMR moves first under PIBT.",
    )
    parser.add_argument("--pickup-ai-multiplier", type=float, default=1.0)
    parser.add_argument("--delivery-ai-multiplier", type=float, default=1.0)
    parser.add_argument("--skip-ai-fraction", type=float, default=None)
    parser.add_argument(
        "--throughput-profile",
        choices=["off", "balanced", "aggressive"],
        default="balanced",
        help="AI-only throughput controller. Baseline ignores this option.",
    )
    parser.add_argument("--goal-near-radius", type=float, default=7.0)
    parser.add_argument("--goal-far-radius", type=float, default=22.0)
    parser.add_argument("--near-congestion-multiplier", type=float, default=0.10)
    parser.add_argument("--goal-progress-bonus", type=float, default=0.75)
    parser.add_argument("--target-entry-bonus", type=float, default=3.0)
    parser.add_argument("--wait-penalty", type=float, default=0.65)
    parser.add_argument("--completion-priority-weight", type=float, default=18.0)
    parser.add_argument("--delivery-priority-multiplier", type=float, default=2.0)
    parser.add_argument("--stalled-wait-threshold", type=int, default=5)
    parser.add_argument("--stalled-ai-cost-multiplier", type=float, default=0.20)
    parser.add_argument("--stalled-priority-weight", type=float, default=8.0)
    parser.add_argument(
        "--common-wait-priority-weight",
        type=float,
        default=1.0,
        help="Fair PIBT aging bonus applied to both baseline and AI after consecutive waits.",
    )
    parser.add_argument(
        "--amr-safety",
        action="store_true",
        help="Enable AMR-sized continuous gap checks and kinodynamic safety post-processing. "
        "By default, PIBT runs like the reference point-agent grid algorithm.",
    )
    parser.add_argument(
        "--continuous-safe-gap",
        type=float,
        default=None,
        help="Override PIBT continuous segment safe gap. Default is 0 for reference PIBT, "
        "or config.continuous_safe_gap_cells when --amr-safety is used.",
    )
    parser.add_argument(
        "--kinodynamic",
        action="store_true",
        help="Render/save kinodynamic AMR speed states without proximity safety slowdown. "
        "This keeps reference-style PIBT planning but shows turn/intersection/docking speed changes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = load_config(str(config_path)) if config_path.exists() else load_config()
    overrides = {"output_dir": "data/pibt_runs"}
    if args.max_time is not None:
        overrides["max_time"] = int(args.max_time)
    if args.num_agents is not None:
        overrides["num_agents"] = int(args.num_agents)
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.save_animation:
        overrides["save_animation"] = True
    if args.subframes is not None:
        overrides["animation_subframes"] = int(args.subframes)
    if args.ai_cost_weight is not None:
        overrides["ai_cost_weight"] = float(args.ai_cost_weight)
    if args.skip_ai_fraction is not None:
        overrides["online_congestion_skip_fraction"] = float(args.skip_ai_fraction)
    config = config.replace(**overrides)

    env = load_factory_environment(config)
    walkable_map = np.asarray(env["walkable_map"]).astype(bool)
    obstacle_map = np.asarray(env["obstacle_map"]).astype(bool)
    h, w = walkable_map.shape[:2]

    starts, _ = select_start_goal_pairs(env, walkable_map, config)
    pickup_groups = filter_point_groups_by_walkability(
        normalize_point_groups(env.get("pickup_point_groups")) or {"pickup": normalize_points(env.get("pickup_points"))},
        walkable_map,
    )
    delivery_groups = filter_point_groups_by_walkability(
        normalize_point_groups(env.get("delivery_point_groups")) or {"delivery": normalize_points(env.get("delivery_points"))},
        walkable_map,
    )

    distance_cache = DistanceCache(walkable_map)
    ai_weights = build_agent_ai_cost_weights(config, len(starts))
    current_cost = np.zeros((1, h, w), dtype=np.float32)
    consecutive_waits = [0 for _ in range(len(starts))]
    task_manager = TaskManager(
        pickup_groups,
        delivery_groups,
        len(starts),
        int(config.seed) + 3000,
    )

    predictor: Optional[CongestionPredictor] = None
    if args.mode == "ai":
        predictor = CongestionPredictor(resolve_config_path(args.model), device=args.device)
        print(f"Loaded ConvLSTM model for PIBT-AI: {args.model} on {predictor.device}")

    def distance_fn(agent_id: int, cell: Coord, goal: Coord) -> float:
        del agent_id
        return distance_cache.distance(cell, goal)

    def goal_distance(agent_id: int, cell: Coord, goal: Coord) -> float:
        return distance_cache.distance(cell, goal)

    def profile_scale() -> float:
        if args.throughput_profile == "off":
            return 0.0
        if args.throughput_profile == "aggressive":
            return 1.35
        return 1.0

    def congestion_gate(agent_id: int, goal: Coord) -> float:
        """Suppress congestion avoidance near a target so completion can happen."""
        if args.mode != "ai" or args.throughput_profile == "off":
            return 1.0
        dist_now = goal_distance(agent_id, positions[agent_id], goal)
        near = max(0.0, float(args.goal_near_radius))
        far = max(near + 1e-6, float(args.goal_far_radius))
        ramp = clamp01((dist_now - near) / (far - near))
        gate = float(args.near_congestion_multiplier) + (1.0 - float(args.near_congestion_multiplier)) * ramp
        if consecutive_waits[agent_id] >= int(args.stalled_wait_threshold):
            gate *= max(0.0, float(args.stalled_ai_cost_multiplier))
        return float(gate)

    def normalized_ai_cost(cell: Coord) -> float:
        if current_cost.size == 0:
            return 0.0
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
        del timestep
        if args.mode != "ai":
            return 0.0
        raw_cost = normalized_ai_cost(cell)
        task_multiplier = (
            float(args.pickup_ai_multiplier)
            if task_manager.goal_is_pickup[agent_id]
            else float(args.delivery_ai_multiplier)
        )
        dist_now = goal_distance(agent_id, positions[agent_id], goal)
        dist_next = goal_distance(agent_id, cell, goal)
        progress = max(0.0, dist_now - dist_next)
        scale = profile_scale()
        congestion_term = (
            float(ai_weights[agent_id])
            * task_multiplier
            * raw_cost
            * congestion_gate(agent_id, goal)
        )
        if scale <= 0.0:
            return congestion_term

        near_weight = 1.0 - clamp01((dist_now - float(args.goal_near_radius)) / max(float(args.goal_far_radius), 1e-6))
        delivery_multiplier = float(args.delivery_priority_multiplier) if not task_manager.goal_is_pickup[agent_id] else 1.0
        completion_bonus = scale * near_weight * delivery_multiplier * (
            float(args.goal_progress_bonus) * progress
            + (float(args.target_entry_bonus) if tuple(cell) == tuple(goal) else 0.0)
        )
        wait_penalty = float(args.wait_penalty) if tuple(cell) == tuple(positions[agent_id]) else 0.0
        return float(congestion_term + wait_penalty - completion_bonus)

    def ai_priority_bias() -> List[float]:
        biases: List[float] = []
        for agent_id, pos in enumerate(positions):
            common_wait_bias = 0.0
            if consecutive_waits[agent_id] >= int(args.stalled_wait_threshold):
                common_wait_bias = float(args.common_wait_priority_weight) * (
                    consecutive_waits[agent_id] - int(args.stalled_wait_threshold) + 1
                )
            if args.mode != "ai":
                biases.append(common_wait_bias)
                continue

            ai_bias = 0.0
            if float(args.ai_priority_weight) > 0.0:
                ai_bias += float(args.ai_priority_weight) * float(ai_weights[agent_id]) * normalized_ai_cost(pos)
            if args.throughput_profile != "off":
                goal = task_manager.goals[agent_id]
                dist_now = goal_distance(agent_id, pos, goal)
                near_weight = 1.0 - clamp01(dist_now / max(float(args.goal_near_radius), 1e-6))
                delivery_multiplier = float(args.delivery_priority_multiplier) if not task_manager.goal_is_pickup[agent_id] else 1.0
                ai_bias += profile_scale() * float(args.completion_priority_weight) * delivery_multiplier * near_weight
                if consecutive_waits[agent_id] >= int(args.stalled_wait_threshold):
                    ai_bias += float(args.stalled_priority_weight) * (
                        consecutive_waits[agent_id] - int(args.stalled_wait_threshold) + 1
                    )
            biases.append(float(common_wait_bias + ai_bias))
        return biases

    if args.continuous_safe_gap is not None:
        continuous_safe_gap_cells = float(args.continuous_safe_gap)
    elif args.amr_safety:
        continuous_safe_gap_cells = float(config.continuous_safe_gap_cells)
    else:
        continuous_safe_gap_cells = 0.0

    use_kinodynamic_output = bool(args.kinodynamic or args.amr_safety)
    use_proximity_safety_controller = bool(args.amr_safety)

    engine = PIBTEngine(
        walkable_map,
        len(starts),
        seed=int(config.seed) + 4100,
        distance_fn=distance_fn,
        candidate_cost_fn=candidate_cost_fn,
        candidate_cost_mode=args.ai_cost_mode if args.mode == "ai" else "additive",
        continuous_safe_gap_cells=continuous_safe_gap_cells,
    )

    frame_builder = FrameBuilder(starts, obstacle_map)
    initial_frame = frame_builder.frame(np.asarray(starts, dtype=np.int32), task_manager.goals_typed())
    t_in = predictor.t_in if predictor is not None else 1
    t_out = predictor.t_out if predictor is not None else 1
    history = deque([initial_frame], maxlen=max(1, t_in))
    positions: List[Coord] = [tuple(start) for start in starts]
    paths: List[List[Coord]] = [[tuple(start)] for start in starts]
    stats = Counter()
    progress_every = max(1, int(config.max_time) // 20)

    print(
        f"Running PIBT-{args.mode}: {len(starts)} AMRs, {int(config.max_time)} steps, "
        f"AI cost={'on' if args.mode == 'ai' else 'off'}, "
        f"point-agent={'no' if args.amr_safety else 'yes'}, "
        f"continuous_gap={continuous_safe_gap_cells:.3f}"
    )
    for t in range(int(config.max_time)):
        if predictor is not None and t % max(1, int(config.online_replan_every)) == 0:
            enc = build_enc(list(history), t_in)
            dec = build_dec(history[-1], t_out)
            pred = predictor.predict(enc, dec)
            pred = pred / max(1e-6, float(config.congestion_center_value))
            current_cost = np.concatenate([pred[:1], pred], axis=0)

        result = engine.step(
            positions,
            task_manager.goals,
            assigned=task_manager.assigned_flags(),
            assigned_priority_bonus=1_000_000.0,
            priority_bias=ai_priority_bias(),
            timestep=t,
        )
        stats["pibt_inherited_count"] += result.inherited_count
        stats["pibt_backtrack_count"] += result.backtrack_count
        stats["pibt_forced_wait_count"] += result.forced_wait_count
        stats["pibt_candidate_reject_vertex"] += result.candidate_reject_vertex
        stats["pibt_candidate_reject_swap"] += result.candidate_reject_swap
        stats["pibt_candidate_reject_continuous"] += result.candidate_reject_continuous

        previous_positions = list(positions)
        positions = result.next_positions
        for agent_id, pos in enumerate(positions):
            if tuple(pos) == tuple(previous_positions[agent_id]):
                consecutive_waits[agent_id] += 1
            else:
                consecutive_waits[agent_id] = 0
        for agent_id, pos in enumerate(positions):
            paths[agent_id].append(tuple(pos))
        task_manager.update_arrivals(positions, t + 1)
        history.append(
            frame_builder.frame(
                np.asarray(positions, dtype=np.int32),
                task_manager.goals_typed(),
            )
        )
        if (t + 1) % progress_every == 0 or t + 1 == int(config.max_time):
            print(
                f"  step {t + 1}/{int(config.max_time)} | "
                f"deliveries={sum(task_manager.completed_deliveries)} | "
                f"waits={sum(1 for p in paths if len(p) > 1 and p[-1] == p[-2])}",
                flush=True,
            )

    task_summary = task_manager.summary(positions, int(config.max_time))
    final_goals = [tuple(goal) for goal in task_summary["final_goals"]]
    if use_kinodynamic_output:
        agent_states = grid_paths_to_kinodynamic_states(paths, walkable_map, config)
        if use_proximity_safety_controller:
            agent_states, safety_summary = apply_proximity_safety_controller(agent_states, config)
        else:
            safety_summary = None
        agent_positions = states_to_agent_positions(agent_states)
    else:
        agent_states = None
        safety_summary = None
        agent_positions = paths_to_agent_positions(paths, int(config.max_time))

    occupancy_sequence = build_occupancy_sequence(agent_positions, h, w)
    congestion_labels = build_additive_congestion_label_sequence(
        agent_positions,
        h,
        w,
        center_value=config.congestion_center_value,
        step_value=config.congestion_step_value,
    )
    metrics = compute_metrics(paths, starts, final_goals, task_summary=task_summary, max_t=int(config.max_time))
    metrics.update(task_summary)
    wait_diagnostics = compute_wait_diagnostics(paths, int(config.max_time))
    wait_diagnostics["deliveries_per_moved_cell"] = float(
        task_summary["total_completed_deliveries"] / max(1, int(wait_diagnostics["actual_moved_cells"]))
    )
    metrics.update(wait_diagnostics)
    metrics.update(
        compute_zone_occupancy_metrics(
            agent_positions,
            np.asarray(env["factory_map"]),
            env.get("labels"),
        )
    )
    metrics["mapf_solver"] = "pibt_convlstm" if args.mode == "ai" else "pibt_baseline"
    metrics["pibt_mode"] = args.mode
    metrics["pibt_shared_engine"] = True
    metrics["pibt_ai_cost_enabled"] = args.mode == "ai"
    metrics["pibt_reference_point_agent_mode"] = not bool(args.amr_safety)
    metrics["pibt_continuous_safe_gap_cells_used"] = float(continuous_safe_gap_cells)
    metrics["pibt_kinodynamic_output"] = bool(use_kinodynamic_output)
    metrics["pibt_proximity_safety_controller_enabled"] = bool(use_proximity_safety_controller)
    metrics["pibt_ai_weight_min"] = float(ai_weights.min()) if len(ai_weights) else 0.0
    metrics["pibt_ai_weight_max"] = float(ai_weights.max()) if len(ai_weights) else 0.0
    metrics["pibt_ai_weight_mean"] = float(ai_weights.mean()) if len(ai_weights) else 0.0
    metrics["pibt_ai_weights"] = [float(v) for v in ai_weights]
    metrics["pibt_ai_cost_threshold"] = float(args.ai_cost_threshold)
    metrics["pibt_ai_cost_cap"] = None if args.ai_cost_cap is None else float(args.ai_cost_cap)
    metrics["pibt_ai_cost_mode"] = args.ai_cost_mode if args.mode == "ai" else "off"
    metrics["pibt_ai_priority_weight"] = float(args.ai_priority_weight) if args.mode == "ai" else 0.0
    metrics["pibt_pickup_ai_multiplier"] = float(args.pickup_ai_multiplier)
    metrics["pibt_delivery_ai_multiplier"] = float(args.delivery_ai_multiplier)
    metrics["pibt_skip_ai_fraction"] = float(config.online_congestion_skip_fraction)
    metrics["pibt_throughput_profile"] = args.throughput_profile if args.mode == "ai" else "baseline"
    metrics["pibt_goal_near_radius"] = float(args.goal_near_radius)
    metrics["pibt_goal_far_radius"] = float(args.goal_far_radius)
    metrics["pibt_near_congestion_multiplier"] = float(args.near_congestion_multiplier)
    metrics["pibt_goal_progress_bonus"] = float(args.goal_progress_bonus)
    metrics["pibt_target_entry_bonus"] = float(args.target_entry_bonus)
    metrics["pibt_wait_penalty"] = float(args.wait_penalty)
    metrics["pibt_completion_priority_weight"] = float(args.completion_priority_weight)
    metrics["pibt_delivery_priority_multiplier"] = float(args.delivery_priority_multiplier)
    metrics["pibt_stalled_wait_threshold"] = int(args.stalled_wait_threshold)
    metrics["pibt_stalled_ai_cost_multiplier"] = float(args.stalled_ai_cost_multiplier)
    metrics["pibt_stalled_priority_weight"] = float(args.stalled_priority_weight)
    metrics["pibt_common_wait_priority_weight"] = float(args.common_wait_priority_weight)
    for key, value in stats.items():
        metrics[key] = int(value)
    if safety_summary:
        metrics.update(safety_summary)
    metrics["collision_count"] = int(compute_collision_count(agent_positions))
    metrics.update(compute_start_clearance_stats(starts, config))
    metrics.update(compute_clearance_stats(agent_positions, config.hard_clearance_cells))
    interp_positions = agent_states[:, :, :2] if agent_states is not None else agent_positions.astype(np.float32)
    metrics.update(compute_interpolated_clearance_stats(interp_positions, config))
    metrics["continuous_safe_gap_cells"] = float(continuous_safe_gap_cells)
    if continuous_safe_gap_cells <= 0.0:
        metrics["interpolated_safe_gap_violation_count"] = 0
    metrics["congestion_peak"] = float(congestion_labels.max())
    metrics["congestion_overlap_cell_count"] = int(
        (congestion_labels > float(config.congestion_center_value)).sum()
    )

    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"metrics -> {out.resolve()}")
        return

    run_id = args.run_id or f"{datetime.now().strftime('%y%m%d_%H%M')}_{args.mode}"
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
    print(f"Saved PIBT {args.mode} data to: {output_dir.resolve()}")
    print(json.dumps(metrics, indent=2))

    if args.no_figures:
        print("Figures skipped (--no-figures).")
        return

    figures_dir = FIGURES_DIR / "pibt_runs" / run_id
    figures_dir.mkdir(parents=True, exist_ok=True)
    visualize_paths(env, paths, starts, final_goals, figures_dir)
    if bool(config.save_animation):
        animate_paths(
            env,
            paths,
            starts,
            final_goals,
            figures_dir,
            config,
            agent_states=agent_states,
            task_summary=task_summary,
        )
    print(f"Saved PIBT figures to: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
