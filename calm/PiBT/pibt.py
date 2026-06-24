"""Lifelong PIBT solver (Priority Inheritance with Backtracking).

Replaces the prioritized-planning solver. Standard PIBT (Okumura et al.): every
timestep, all agents move one cell simultaneously; each picks the neighbour closest
to its goal (by an obstacle-aware BFS distance field), higher-priority agents push
lower-priority ones out of the way recursively, and head-on swaps are forbidden.
The result is always vertex/edge collision-free, and it scales near-linearly with
agent count where prioritized planning blew up under congestion.

"Lifelong": when an agent reaches its target it is immediately assigned the next
one, alternating pickup -> delivery, so the episode is a continuous task stream.

``plan_pibt_repeated_tasks`` mirrors the old planner's signature and return shape
(paths + a summary whose ``task_assignments`` / ``completed_deliveries`` feed the
heatmap generator), so the generator needs almost no change.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import MAPFConfig
from .distance import DistanceFieldCache
from .grid import filter_point_groups_by_walkability, walkable_neighbors
from .metrics import build_additive_congestion_label_sequence
from .types import Coord, PathType

__all__ = ["plan_pibt_repeated_tasks", "set_planning_progress_hook"]

# Optional hook the dataset generator installs to advance its progress bar
# (called once per agent at the end of an episode, matching the old per-agent cadence).
PLANNING_PROGRESS_HOOK: Optional[Callable[[], None]] = None


def set_planning_progress_hook(hook: Optional[Callable[[], None]]) -> None:
    global PLANNING_PROGRESS_HOOK
    PLANNING_PROGRESS_HOOK = hook


def plan_pibt_repeated_tasks(
    starts: Sequence[Coord],
    pickup_points: Sequence[Coord],
    delivery_points: Sequence[Coord],
    walkable_map: np.ndarray,
    config: MAPFConfig,
    pickup_point_groups: Optional[Dict[str, List[Coord]]] = None,
    delivery_point_groups: Optional[Dict[str, List[Coord]]] = None,
    congestion_predictor: Any = None,
    congestion_weight: float = 0.0,
    predict_every: int = 10,
) -> Tuple[List[PathType], Dict[str, Any]]:
    """Lifelong PIBT. With ``congestion_predictor`` set and ``congestion_weight > 0``
    it becomes congestion-aware: every step the live congestion frame (same additive
    tent as the dataset label) is pushed into a 10-frame buffer; once full, the
    predictor forecasts the next 10 frames (re-run every ``predict_every`` steps,
    MPC-style) and each agent's move cost gains ``congestion_weight * predicted /
    center_value`` -- a soft penalty in distance-cell units that steers agents off
    soon-to-be-congested cells. The predictor is duck-typed (``.predict((10,1,H,W))
    -> (10,1,H,W)`` raw congestion) so this numpy-only module never imports torch.
    With no predictor / weight 0 the behaviour is byte-for-byte the original PIBT,
    which is exactly the A/B baseline."""
    rng = np.random.default_rng(config.seed + 1000)
    walkable = np.asarray(walkable_map).astype(bool)
    H, W = walkable.shape[:2]
    T = int(config.max_time)
    N = len(starts)
    cache = DistanceFieldCache(walkable)

    # --- congestion-aware (MPC) state ---
    use_congestion = congestion_predictor is not None and congestion_weight > 0.0
    center_v = float(config.congestion_center_value)
    step_v = float(config.congestion_step_value)
    predict_every = max(1, min(10, int(predict_every)))
    cong_buffer: deque = deque(maxlen=10)        # last 10 live congestion frames (raw)
    pred_future: Optional[np.ndarray] = None     # (10, H, W) raw predicted congestion
    pred_base_t = -1                             # time index of the latest frame fed to the model
    prediction_count = 0

    pickup_groups = filter_point_groups_by_walkability(
        pickup_point_groups or {"pickup": list(pickup_points)}, walkable
    )
    delivery_groups = filter_point_groups_by_walkability(
        delivery_point_groups or {"delivery": list(delivery_points)}, walkable
    )

    def choose_target(cur: Coord, is_pickup: bool) -> Optional[Tuple[str, Coord]]:
        """Pick a random reachable point from the relevant group set (!= cur)."""
        group_dict = pickup_groups if is_pickup else delivery_groups
        names = list(group_dict.keys())
        if not names:
            return None
        for _ in range(20):
            name = names[int(rng.integers(0, len(names)))]
            pts = group_dict[name]
            g = tuple(pts[int(rng.integers(0, len(pts)))])
            if g != cur and np.isfinite(cache.field(g)[cur[1], cur[0]]):
                return name, g
        for name in names:  # exhaustive fallback
            for g in group_dict[name]:
                g = tuple(g)
                if g != cur and np.isfinite(cache.field(g)[cur[1], cur[0]]):
                    return name, g
        return None

    # --- per-agent state ---
    pos: List[Coord] = [tuple(s) for s in starts]
    goal: List[Coord] = list(pos)
    current_is_pickup: List[bool] = [True] * N
    open_assignment: List[Optional[Dict[str, Any]]] = [None] * N
    assignments: List[List[Dict[str, Any]]] = [[] for _ in range(N)]
    completed_targets = [0] * N
    completed_deliveries = [0] * N
    since_goal = [0] * N                       # timesteps since last target reached (priority)
    tiebreak = [float(rng.random()) for _ in range(N)]  # fixed per-agent priority tiebreak
    paths: List[PathType] = [[tuple(s)] for s in starts]
    failed_count = 0

    def assign(i: int, is_pickup: bool, start_t: int) -> None:
        nonlocal failed_count
        chosen = choose_target(pos[i], is_pickup)
        if chosen is None:
            failed_count += 1
            goal[i] = pos[i]
            open_assignment[i] = None
            return
        zone, g = chosen
        goal[i] = g
        open_assignment[i] = {
            "action": "pickup" if is_pickup else "delivery",
            "zone": zone,
            "target": [int(g[0]), int(g[1])],
            "start_t": int(start_t),
        }

    for i in range(N):  # first target (pickup) at t=0
        assign(i, True, 0)

    # --- one collision-free PIBT timestep ---
    def pibt_step(pen_field: Optional[np.ndarray]) -> List[Coord]:
        next_v: List[Optional[Coord]] = [None] * N
        occupied_next: Dict[Coord, int] = {}
        occupied_now: Dict[Coord, int] = {pos[i]: i for i in range(N)}

        def pibt(ai: int, forbidden: Optional[Coord]) -> bool:
            field = cache.field(goal[ai])
            cands = [pos[ai]] + walkable_neighbors(pos[ai], walkable)
            # Ranking by distance-to-goal (+ optional congestion penalty) only changes
            # PREFERENCE; collision-freedom comes from the inheritance/backtracking below
            # and is unaffected, so the penalty can never make a step illegal.
            if pen_field is None:
                cands.sort(key=lambda c: (float(field[c[1], c[0]]), rng.random()))
            else:
                cands.sort(key=lambda c: (float(field[c[1], c[0]])
                                          + congestion_weight * float(pen_field[c[1], c[0]]),
                                          rng.random()))
            for v in cands:
                if v in occupied_next:
                    continue
                if forbidden is not None and v == forbidden:
                    continue
                occupied_next[v] = ai
                next_v[ai] = v
                ak = occupied_now.get(v)
                if ak is not None and ak != ai and next_v[ak] is None:
                    if pibt(ak, pos[ai]):  # push ak out; it may not swap into our cell
                        return True
                    del occupied_next[v]   # ak stuck -> roll back and try next cell
                    next_v[ai] = None
                    continue
                return True
            next_v[ai] = None  # leave no trace; caller rolls back (top-level always finds a stay)
            return False

        order = sorted(range(N), key=lambda i: (since_goal[i] + tiebreak[i]), reverse=True)
        for i in order:
            if next_v[i] is None:
                pibt(i, None)
        return [v if v is not None else pos[i] for i, v in enumerate(next_v)]

    # --- lifelong loop ---
    for t in range(1, T + 1):
        pen_field: Optional[np.ndarray] = None
        if use_congestion:
            # live congestion from the current (pre-move) positions == congestion at
            # time t-1; identical additive tent to the dataset label, so the predictor
            # sees in-distribution frames.
            frame = build_additive_congestion_label_sequence(
                np.asarray(pos, dtype=np.int32)[None, :, :], H, W, center_v, step_v)[0]
            cong_buffer.append(frame)
            if len(cong_buffer) == 10:
                # pred_future[k] forecasts time (pred_base_t + 1 + k); we are choosing the
                # move that lands at time t, so we want offset = t - pred_base_t - 1.
                offset = t - pred_base_t - 1
                if pred_future is None or offset >= predict_every:
                    past = np.stack(cong_buffer)[:, None, :, :]    # (10, 1, H, W) raw
                    pred_future = np.asarray(congestion_predictor.predict(past))[:, 0]
                    pred_base_t = t - 1
                    prediction_count += 1
                    offset = 0
                # normalize by center_value so congestion_weight is in distance-cell units
                # (~"extra cells of detour per one-agent-equivalent of congestion").
                pen_field = pred_future[min(offset, 9)] / center_v
        nxt = pibt_step(pen_field)
        for i in range(N):
            pos[i] = nxt[i]
            paths[i].append(pos[i])
            assignment = open_assignment[i]
            if assignment is not None and pos[i] == goal[i]:
                assignment["end_t"] = int(t)
                assignment["completed"] = True
                assignments[i].append(assignment)
                completed_targets[i] += 1
                if assignment["action"] == "delivery":
                    completed_deliveries[i] += 1
                since_goal[i] = 0
                current_is_pickup[i] = not current_is_pickup[i]
                assign(i, current_is_pickup[i], t)
            else:
                since_goal[i] += 1

    for i in range(N):  # close still-open assignments at the horizon
        assignment = open_assignment[i]
        if assignment is not None:
            assignment["end_t"] = int(T)
            assignment["completed"] = False
            assignments[i].append(assignment)
        if PLANNING_PROGRESS_HOOK is not None:
            PLANNING_PROGRESS_HOOK()

    summary = {
        "task_mode": "pibt_lifelong_pickup_delivery",
        "observation_horizon_max_time": int(T),
        "pickup_zone_groups": sorted(pickup_groups),
        "delivery_zone_groups": sorted(delivery_groups),
        "completed_targets": completed_targets,
        "completed_deliveries": completed_deliveries,
        "total_completed_targets": int(sum(completed_targets)),
        "total_completed_deliveries": int(sum(completed_deliveries)),
        "task_assignments": assignments,
        "repeated_planning_failed_count": int(failed_count),
        "priority_boost_event_count": 0,
        "safe_extension_strict_steps": 0,
        "safe_extension_relaxed_steps": 0,
        "safe_extension_failed_steps": 0,
        "congestion_aware": bool(use_congestion),
        "congestion_weight": float(congestion_weight) if use_congestion else 0.0,
        "congestion_prediction_count": int(prediction_count),
    }
    return paths, summary
