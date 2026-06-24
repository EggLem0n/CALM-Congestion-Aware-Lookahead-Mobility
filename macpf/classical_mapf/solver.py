"""MAPF solvers: reservation tables, conflicts, A*, and prioritized planning."""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import replace
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import *


# Optional hook called once per planned agent during repeated-task planning.
# Used by generate_heatmap.py to drive a cross-process progress bar.
PLANNING_PROGRESS_HOOK: Optional[Callable[[], None]] = None


def set_planning_progress_hook(hook: Optional[Callable[[], None]]) -> None:
    """Install/clear the per-agent planning progress callback."""
    global PLANNING_PROGRESS_HOOK
    PLANNING_PROGRESS_HOOK = hook


def choose_reachable_group_target(
    current: Coord,
    point_groups: Dict[str, List[Coord]],
    rng: np.random.Generator,
    walkable_map: np.ndarray,
    reservation_table: Dict[int, set[Coord]],
    edge_reservation_table: Dict[int, set[Tuple[Coord, Coord]]],
    congestion_cost: np.ndarray,
    planning_config: MAPFConfig,
    current_t: int,
    soft_cost_table: Optional["SoftCostTable"] = None,
    edge_buckets: Optional["EdgeBuckets"] = None,
    heuristic_provider: Optional[Callable[[Coord], Optional[np.ndarray]]] = None,
) -> Tuple[Optional[str], Optional[Coord], Optional[PathType]]:
    candidates_with_groups = [
        (group_name, point)
        for group_name, points in point_groups.items()
        for point in points
        if point != current
    ]
    if not candidates_with_groups:
        return None, None, None

    for candidate_index in rng.permutation(len(candidates_with_groups)):
        group_name, candidate_goal = candidates_with_groups[int(candidate_index)]
        candidate_segment = astar_single_agent(
            current,
            candidate_goal,
            walkable_map,
            reservation_table,
            edge_reservation_table,
            congestion_cost,
            planning_config,
            start_time=current_t,
            soft_cost_table=soft_cost_table,
            edge_buckets=edge_buckets,
            heuristic=heuristic_provider(candidate_goal) if heuristic_provider is not None else None,
        )
        if candidate_segment is not None and len(candidate_segment) > 1:
            return group_name, candidate_goal, candidate_segment
    return None, None, None

def add_path_to_reservation_tables(
    path: PathType,
    reservation_table: Dict[int, set[Coord]],
    edge_reservation_table: Dict[int, set[Tuple[Coord, Coord]]],
    max_time: int,
) -> None:
    """Register one finalized path into existing reservation tables.

    Incremental alternative to rebuilding tables from every prior path each
    time an agent plans a new task segment.
    """
    if not path:
        return
    for t in range(max_time + 1):
        curr = path_position_at(path, t)
        reservation_table.setdefault(t, set()).add(curr)
        if t > 0:
            prev = path_position_at(path, t - 1)
            edge_reservation_table.setdefault(t, set()).add((prev, curr))


def build_reservation_table(
    paths: Sequence[PathType], max_time: int
) -> Tuple[Dict[int, set[Coord]], Dict[int, set[Tuple[Coord, Coord]]]]:
    reservation_table: Dict[int, set[Coord]] = {}
    edge_reservation_table: Dict[int, set[Tuple[Coord, Coord]]] = {}
    for path in paths:
        add_path_to_reservation_tables(path, reservation_table, edge_reservation_table, max_time)
    return reservation_table, edge_reservation_table


SoftCostTable = Dict[int, Dict[Coord, float]]
EdgeBuckets = Dict[int, Dict[Coord, List[Tuple[Coord, Coord]]]]

# Bucket edge endpoints on a 3x3 cell grid. The continuous-motion check only
# looks at edges within Chebyshev distance 2, so the 3x3 neighbouring buckets
# always cover every relevant edge.
_EDGE_BUCKET_SIZE = 3


def _edge_bucket(cell: Coord) -> Coord:
    return (cell[0] // _EDGE_BUCKET_SIZE, cell[1] // _EDGE_BUCKET_SIZE)


@lru_cache(maxsize=None)
def _soft_cost_kernel(radius: int, weight: float) -> Tuple[Tuple[int, int, float], ...]:
    """(dx, dy, added_cost) offsets of the diamond stamped around each path cell.

    The geometry depends only on (radius, weight), not on the path, so cache it
    once instead of recomputing the abs()/division for every cell at every
    timestep of every path.
    """
    kernel: List[Tuple[int, int, float]] = []
    for dx in range(-radius, radius + 1):
        span = radius - abs(dx)
        for dy in range(-span, span + 1):
            distance = abs(dx) + abs(dy)
            if distance == 0:
                continue
            kernel.append((dx, dy, weight * (radius - distance + 1) / radius))
    return tuple(kernel)


def add_path_to_soft_cost_table(
    path: PathType,
    soft_cost_table: SoftCostTable,
    max_time: int,
    config: MAPFConfig,
) -> None:
    """Precompute soft proximity costs around one finalized path.

    Replaces the per-expansion scan over every reserved cell in
    get_soft_proximity_cost with an O(1) lookup during A*.
    """
    radius = int(config.soft_proximity_cost_radius_cells)
    weight = float(config.soft_proximity_cost_weight)
    if not path or radius <= 0 or weight <= 0.0:
        return
    kernel = _soft_cost_kernel(radius, weight)
    for t in range(max_time + 1):
        x, y = path_position_at(path, t)
        costs_at_t = soft_cost_table.setdefault(t, {})
        for dx, dy, added in kernel:
            cell = (x + dx, y + dy)
            costs_at_t[cell] = costs_at_t.get(cell, 0.0) + added


def add_path_to_edge_buckets(path: PathType, edge_buckets: EdgeBuckets, max_time: int) -> None:
    """Index one finalized path's edges by coarse position for fast lookup."""
    if not path:
        return
    for t in range(1, max_time + 1):
        prev = path_position_at(path, t - 1)
        curr = path_position_at(path, t)
        edge_buckets.setdefault(t, {}).setdefault(_edge_bucket(prev), []).append((prev, curr))


def has_vertex_conflict(x: int, y: int, t: int, reservation_table: Dict[int, set[Coord]]) -> bool:
    return (x, y) in reservation_table.get(t, set())


def has_clearance_conflict(
    x: int,
    y: int,
    t: int,
    reservation_table: Dict[int, set[Coord]],
    clearance_cells: int,
) -> bool:
    if clearance_cells <= 0:
        return False
    for ox, oy in reservation_table.get(t, set()):
        if abs(x - ox) + abs(y - oy) <= clearance_cells:
            return True
    return False


def get_soft_proximity_cost(
    x: int,
    y: int,
    t: int,
    reservation_table: Dict[int, set[Coord]],
    config: MAPFConfig,
) -> float:
    radius = int(config.soft_proximity_cost_radius_cells)
    if radius <= 0 or config.soft_proximity_cost_weight <= 0.0:
        return 0.0
    cost = 0.0
    for ox, oy in reservation_table.get(t, set()):
        distance = abs(x - ox) + abs(y - oy)
        if distance == 0:
            continue
        if distance <= radius:
            cost += config.soft_proximity_cost_weight * (radius - distance + 1) / radius
    return cost


def has_edge_conflict(
    prev: Coord,
    curr: Coord,
    t: int,
    edge_reservation_table: Dict[int, set[Tuple[Coord, Coord]]],
) -> bool:
    return (curr, prev) in edge_reservation_table.get(t, set())


def sampled_segment_min_distance(
    a0: Coord,
    a1: Coord,
    b0: Coord,
    b1: Coord,
    samples: int = 8,
) -> float:
    del samples
    dx0 = float(a0[0] - b0[0])
    dy0 = float(a0[1] - b0[1])
    dvx = float((a1[0] - a0[0]) - (b1[0] - b0[0]))
    dvy = float((a1[1] - a0[1]) - (b1[1] - b0[1]))
    denom = dvx * dvx + dvy * dvy
    if denom <= 1e-9:
        alpha = 0.0
    else:
        alpha = clamp(-(dx0 * dvx + dy0 * dvy) / denom, 0.0, 1.0)
    cx = dx0 + dvx * alpha
    cy = dy0 + dvy * alpha
    return math.hypot(cx, cy)


def has_continuous_motion_conflict(
    prev: Coord,
    curr: Coord,
    t: int,
    edge_reservation_table: Dict[int, set[Tuple[Coord, Coord]]],
    config: MAPFConfig,
    safe_gap: Optional[float] = None,
) -> bool:
    if safe_gap is None:
        safe_gap = float(config.continuous_safe_gap_cells)
    if safe_gap <= 0.0:
        return False
    for other_prev, other_curr in edge_reservation_table.get(t, set()):
        if (
            abs(prev[0] - other_prev[0]) > 2
            or abs(prev[1] - other_prev[1]) > 2
            or abs(curr[0] - other_curr[0]) > 2
            or abs(curr[1] - other_curr[1]) > 2
        ):
            continue
        if curr == other_curr or (prev == other_curr and curr == other_prev):
            continue
        if sampled_segment_min_distance(prev, curr, other_prev, other_curr) < safe_gap:
            return True
    return False


def has_continuous_motion_conflict_indexed(
    prev: Coord,
    curr: Coord,
    t: int,
    edge_buckets: EdgeBuckets,
    config: MAPFConfig,
    safe_gap: Optional[float] = None,
) -> bool:
    """Same check as has_continuous_motion_conflict, but only visits edges
    whose start cell lies in the 3x3 buckets around prev."""
    if safe_gap is None:
        safe_gap = float(config.continuous_safe_gap_cells)
    if safe_gap <= 0.0:
        return False
    buckets_at_t = edge_buckets.get(t)
    if not buckets_at_t:
        return False
    px, py = prev
    cx, cy = curr
    bx = px // _EDGE_BUCKET_SIZE
    by = py // _EDGE_BUCKET_SIZE
    get_bucket = buckets_at_t.get
    for bxx in (bx - 1, bx, bx + 1):
        for byy in (by - 1, by, by + 1):
            edges = get_bucket((bxx, byy))
            if not edges:
                continue
            for other_prev, other_curr in edges:
                opx, opy = other_prev
                ocx, ocy = other_curr
                if abs(px - opx) > 2 or abs(py - opy) > 2 or abs(cx - ocx) > 2 or abs(cy - ocy) > 2:
                    continue
                if curr == other_curr or (prev == other_curr and curr == other_prev):
                    continue
                if sampled_segment_min_distance(prev, curr, other_prev, other_curr) < safe_gap:
                    return True
    return False


def gather_reserved_edges_near(
    prev: Coord,
    t: int,
    edge_buckets: EdgeBuckets,
) -> List[Tuple[Coord, Coord]]:
    """Reserved edges at time `t` whose start cell is within Chebyshev 2 of `prev`.

    Hoisted out of the per-neighbour continuous-motion check: when A* expands one
    node it tests all five neighbours against the *same* `prev`, so the 3x3 edge
    buckets only need scanning once per node instead of once per neighbour. The
    Chebyshev-2 prefilter on the start cell is the `prev` half of the original
    4-way abs() gate; the `curr` half stays in the per-neighbour test below, so the
    pair of stages accepts exactly the same edges as the inline gate did.
    """
    buckets_at_t = edge_buckets.get(t)
    if not buckets_at_t:
        return []
    px, py = prev
    bx = px // _EDGE_BUCKET_SIZE
    by = py // _EDGE_BUCKET_SIZE
    get_bucket = buckets_at_t.get
    nearby: List[Tuple[Coord, Coord]] = []
    for bxx in (bx - 1, bx, bx + 1):
        for byy in (by - 1, by, by + 1):
            edges = get_bucket((bxx, byy))
            if not edges:
                continue
            for other_prev, other_curr in edges:
                if abs(px - other_prev[0]) <= 2 and abs(py - other_prev[1]) <= 2:
                    nearby.append((other_prev, other_curr))
    return nearby


def continuous_motion_conflict_against_edges(
    prev: Coord,
    curr: Coord,
    nearby_edges: List[Tuple[Coord, Coord]],
    safe_gap: float,
) -> bool:
    """Per-neighbour half of the continuous-motion check against pre-gathered edges.

    `nearby_edges` is already filtered to edges near `prev` (see
    gather_reserved_edges_near); here we apply the `curr` proximity gate and the
    swap/shared-cell exemptions, then the sampled min-distance test. Equivalent to
    has_continuous_motion_conflict_indexed for a single (prev, curr) pair.
    """
    cx, cy = curr
    for other_prev, other_curr in nearby_edges:
        if abs(cx - other_curr[0]) > 2 or abs(cy - other_curr[1]) > 2:
            continue
        if curr == other_curr or (prev == other_curr and curr == other_prev):
            continue
        if sampled_segment_min_distance(prev, curr, other_prev, other_curr) < safe_gap:
            return True
    return False


def min_continuous_motion_distance_to_reserved_edges(
    prev: Coord,
    curr: Coord,
    t: int,
    edge_reservation_table: Dict[int, set[Tuple[Coord, Coord]]],
) -> float:
    min_distance = float("inf")
    for other_prev, other_curr in edge_reservation_table.get(t, set()):
        min_distance = min(
            min_distance,
            sampled_segment_min_distance(prev, curr, other_prev, other_curr),
        )
    return min_distance


def load_ai_congestion_cost(path: Optional[str], horizon: int, H: int, W: int) -> np.ndarray:
    """
    Placeholder for future AI-predicted congestion cost.
    Expected output shape:
        (horizon, H, W)
    If path is None, return zeros.
    """
    if path is None:
        return np.zeros((horizon, H, W), dtype=np.float32)

    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "congestion_cost" in loaded:
            data = loaded["congestion_cost"]
        else:
            first_key = loaded.files[0]
            data = loaded[first_key]
    else:
        data = loaded

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3 or data.shape[1:] != (H, W):
        raise ValueError(f"AI cost must have shape (future_steps, {H}, {W}), got {data.shape}")
    if data.shape[0] < horizon:
        pad = np.repeat(data[-1:, :, :], horizon - data.shape[0], axis=0)
        data = np.concatenate([data, pad], axis=0)
    return data[:horizon]


def get_congestion_cost(congestion_cost: np.ndarray, t: int, x: int, y: int) -> float:
    """
    Return predicted congestion cost at position (x, y) and time t.
    If t exceeds the prediction horizon, use the last available timestep.
    """
    if congestion_cost.size == 0:
        return 0.0
    cost_t = min(t, congestion_cost.shape[0] - 1)
    return float(congestion_cost[cost_t, y, x])


def compute_distance_field(walkable_map: np.ndarray, goal: Coord) -> np.ndarray:
    """True shortest grid-distance (4-connectivity) from every walkable cell to `goal`.

    A single BFS over the static map; unreachable cells stay +inf. Used as A*'s
    heuristic in place of Manhattan distance: on the factory map the long conveyor
    obstacles make Manhattan a weak lower bound, so A* fans out across huge areas
    before threading around them. This field is the exact obstacle-aware distance,
    so it stays admissible/consistent (reservations and waiting only add cost on
    top) while collapsing the explored frontier to near the real corridor.

    The field depends only on (static map, goal); callers cache it per goal and
    reuse it across every agent and every re-plan.
    """
    H, W = walkable_map.shape[:2]
    field = np.full((H, W), np.inf, dtype=np.float32)
    gx, gy = int(goal[0]), int(goal[1])
    if not (0 <= gx < W and 0 <= gy < H) or not walkable_map[gy, gx]:
        return field
    wmap = walkable_map
    field[gy, gx] = 0.0
    queue = deque([(gx, gy)])
    while queue:
        x, y = queue.popleft()
        nd = field[y, x] + 1.0
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < W and 0 <= ny < H and wmap[ny, nx] and field[ny, nx] == np.inf:
                field[ny, nx] = nd
                queue.append((nx, ny))
    return field


def astar_single_agent(
    start: Coord,
    goal: Coord,
    walkable_map: np.ndarray,
    reservation_table: Dict[int, set[Coord]],
    edge_reservation_table: Dict[int, set[Tuple[Coord, Coord]]],
    congestion_cost: np.ndarray,
    config: MAPFConfig,
    start_time: int = 0,
    soft_cost_table: Optional[SoftCostTable] = None,
    edge_buckets: Optional[EdgeBuckets] = None,
    heuristic: Optional[np.ndarray] = None,
) -> Optional[PathType]:
    if start == goal:
        return [start]
    if not is_walkable(*start, walkable_map) or not is_walkable(*goal, walkable_map):
        return None

    # Hoist hot config reads / grid data into locals. The inner loop below runs
    # millions of times per planning pass, so each saved Config.__getattr__ or
    # numpy scalar lookup matters. walkable_map is read as a nested Python list
    # so neighbour checks avoid numpy scalar-indexing overhead.
    max_time = int(config.max_time)
    hard_clearance = int(config.hard_clearance_cells)
    use_ai = bool(config.use_ai_congestion_cost)
    ai_weight = float(config.ai_cost_weight)
    safe_gap = float(config.continuous_safe_gap_cells)
    gx, gy = goal
    wmap = walkable_map.tolist()
    H = len(wmap)
    W = len(wmap[0]) if H else 0
    INF = float("inf")

    # Optional precomputed true-distance heuristic (compute_distance_field). When
    # absent we fall back to Manhattan -- the exact original behaviour. As a nested
    # list for the same reason as wmap: avoid numpy scalar-indexing in the hot loop.
    hmap = heuristic.tolist() if heuristic is not None else None
    if hmap is not None and hmap[start[1]][start[0]] == INF:
        return None  # goal unreachable from start on the static map

    open_heap: List[Tuple[float, float, int, int, int, int]] = []
    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    best_g: Dict[Tuple[int, int, int], float] = {(start[0], start[1], start_time): 0.0}
    tie_breaker = 0

    if hmap is not None:
        h0 = float(hmap[start[1]][start[0]])
    else:
        h0 = float(abs(start[0] - gx) + abs(start[1] - gy))
    heapq.heappush(open_heap, (h0, h0, start_time, tie_breaker, start[0], start[1]))

    while open_heap:
        _, _, t, _, x, y = heapq.heappop(open_heap)
        state = (x, y, t)
        g = best_g.get(state)
        if g is None:
            continue
        if x == gx and y == gy:
            return reconstruct_path(came_from, state)
        if t >= max_time:
            continue

        nt = t + 1
        res_t = reservation_table.get(nt)
        edge_t = edge_reservation_table.get(nt)
        soft_t = soft_cost_table.get(nt) if soft_cost_table is not None else None
        prev_cell = (x, y)
        # Gather the reserved edges near this node ONCE. The continuous-motion check
        # is identical-`prev` for all five neighbours, so scanning the 3x3 edge
        # buckets per neighbour re-did the same work 5x (it was ~40% of planning
        # time). edge_buckets is the path used by every real caller; the unindexed
        # fallback below keeps the original per-neighbour scan.
        nearby_edges: Optional[List[Tuple[Coord, Coord]]] = None
        if safe_gap > 0.0 and edge_buckets is not None:
            nearby_edges = gather_reserved_edges_near(prev_cell, nt, edge_buckets)
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x, y)):
            if not (0 <= nx < W and 0 <= ny < H and wmap[ny][nx]):
                continue
            cell = (nx, ny)
            # vertex conflict
            if res_t is not None and cell in res_t:
                continue
            # hard-clearance conflict
            if hard_clearance > 0 and res_t:
                blocked = False
                for ox, oy in res_t:
                    if abs(nx - ox) + abs(ny - oy) <= hard_clearance:
                        blocked = True
                        break
                if blocked:
                    continue
            # edge (head-on swap) conflict
            if edge_t is not None and (cell, prev_cell) in edge_t:
                continue
            # continuous-motion conflict
            if edge_buckets is not None:
                if nearby_edges and continuous_motion_conflict_against_edges(
                    prev_cell, cell, nearby_edges, safe_gap
                ):
                    continue
            elif has_continuous_motion_conflict(prev_cell, cell, nt, edge_reservation_table, config, safe_gap):
                continue

            proximity_cost = 0.0
            if soft_cost_table is not None:
                if soft_t:
                    proximity_cost = soft_t.get(cell, 0.0)
            else:
                proximity_cost = get_soft_proximity_cost(nx, ny, nt, reservation_table, config)
            new_g = g + 1.0 + proximity_cost
            next_state = (nx, ny, nt)
            if new_g >= best_g.get(next_state, INF):
                continue
            if hmap is not None:
                h = hmap[ny][nx]
                if h == INF:
                    continue  # cell cannot reach the goal -- never on a valid path
            else:
                h = float(abs(nx - gx) + abs(ny - gy))
            if use_ai:
                f = new_g + h + ai_weight * get_congestion_cost(congestion_cost, nt, nx, ny)
            else:
                f = new_g + h
            best_g[next_state] = new_g
            came_from[next_state] = state
            tie_breaker += 1
            heapq.heappush(open_heap, (f, h, nt, tie_breaker, nx, ny))

    return None


def reconstruct_path(
    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]],
    goal_state: Tuple[int, int, int],
) -> PathType:
    states = [goal_state]
    while states[-1] in came_from:
        states.append(came_from[states[-1]])
    states.reverse()
    return [(x, y) for x, y, _ in states]


def prioritized_planning(
    starts: Sequence[Coord],
    goals: Sequence[Coord],
    walkable_map: np.ndarray,
    congestion_cost: np.ndarray,
    config: MAPFConfig,
) -> List[PathType]:
    paths: List[PathType] = []
    reservation_table: Dict[int, set[Coord]] = {}
    edge_reservation_table: Dict[int, set[Tuple[Coord, Coord]]] = {}
    soft_cost_table: SoftCostTable = {}
    edge_buckets: EdgeBuckets = {}
    for agent_id, (start, goal) in enumerate(zip(starts, goals)):
        path = astar_single_agent(
            start,
            goal,
            walkable_map,
            reservation_table,
            edge_reservation_table,
            congestion_cost,
            config,
            soft_cost_table=soft_cost_table,
            edge_buckets=edge_buckets,
        )
        if path is None:
            if config.verbose_planning:
                print(f"[planning failed] agent={agent_id}, start={start}, goal={goal}")
            paths.append([start])
        else:
            paths.append(path)
        add_path_to_reservation_tables(
            paths[-1], reservation_table, edge_reservation_table, config.max_time
        )
        add_path_to_soft_cost_table(paths[-1], soft_cost_table, config.max_time, config)
        add_path_to_edge_buckets(paths[-1], edge_buckets, config.max_time)
    return paths


def extend_path_safely(
    current_path: PathType,
    reservation_table: Dict[int, set[Coord]],
    edge_reservation_table: Dict[int, set[Tuple[Coord, Coord]]],
    walkable_map: np.ndarray,
    config: MAPFConfig,
) -> Tuple[PathType, Dict[str, int]]:
    stats = {
        "strict_steps": 0,
        "relaxed_steps": 0,
        "failed_steps": 0,
    }
    while len(current_path) < config.max_time + 1:
        t = len(current_path) - 1
        x, y = current_path[-1]
        candidates = get_neighbors(x, y, t, walkable_map)
        base_safe_candidates = []
        strict_safe_candidates = []
        for nx, ny, nt in candidates:
            if has_vertex_conflict(nx, ny, nt, reservation_table):
                continue
            if has_clearance_conflict(nx, ny, nt, reservation_table, config.hard_clearance_cells):
                continue
            if has_edge_conflict((x, y), (nx, ny), nt, edge_reservation_table):
                continue
            base_safe_candidates.append((nx, ny))
            if not has_continuous_motion_conflict((x, y), (nx, ny), nt, edge_reservation_table, config):
                strict_safe_candidates.append((nx, ny))

        safe_candidates = strict_safe_candidates
        if safe_candidates:
            if (x, y) in safe_candidates:
                current_path.append((x, y))
            else:
                current_path.append(safe_candidates[0])
            stats["strict_steps"] += 1
        elif base_safe_candidates:
            relaxed_gap = max(0.0, float(config.extension_relaxed_safe_gap_cells))

            def candidate_score(candidate: Coord) -> Tuple[int, float, int]:
                continuous_distance = min_continuous_motion_distance_to_reserved_edges(
                    (x, y),
                    candidate,
                    t + 1,
                    edge_reservation_table,
                )
                if math.isinf(continuous_distance):
                    continuous_distance = 999.0
                relaxed_ok = 1 if continuous_distance >= relaxed_gap else 0
                return (relaxed_ok, continuous_distance, -manhattan_distance((x, y), candidate))

            chosen = max(base_safe_candidates, key=candidate_score)
            if config.verbose_planning:
                score = candidate_score(chosen)
                print(
                    f"[safe extension relaxed] t={t}, position={(x, y)}, "
                    f"chosen={chosen}, min_segment_distance={score[1]:.3f}"
                )
            current_path.append(chosen)
            stats["relaxed_steps"] += 1
        else:
            if config.verbose_planning:
                print(f"[safe extension failed] t={t}, position={(x, y)}")
            current_path.append((x, y))
            stats["failed_steps"] += 1
    return current_path, stats


def repair_paths_with_clearance(
    paths: Sequence[PathType],
    walkable_map: np.ndarray,
    config: MAPFConfig,
) -> Tuple[List[PathType], Dict[str, Any]]:
    repaired: List[PathType] = [[path_position_at(path, 0)] for path in paths]
    walkable_cells = [(int(x), int(y)) for y, x in np.argwhere(walkable_map)]
    local_repairs = 0
    global_repairs = 0
    unresolved_repairs = 0

    for t in range(1, config.max_time + 1):
        accepted: List[Coord] = []
        accepted_edges: set[Tuple[Coord, Coord]] = set()
        for agent_id, path in enumerate(paths):
            prev = repaired[agent_id][-1]
            desired = path_position_at(path, t)
            local_candidates = [desired, prev]
            local_candidates.extend((nx, ny) for nx, ny, _ in get_neighbors(prev[0], prev[1], t - 1, walkable_map))
            local_candidates = unique_preserving_order(local_candidates)

            def candidate_is_safe(candidate: Coord) -> bool:
                if not is_walkable(*candidate, walkable_map):
                    return False
                if any(abs(candidate[0] - other[0]) + abs(candidate[1] - other[1]) <= config.hard_clearance_cells for other in accepted):
                    return False
                if (candidate, prev) in accepted_edges:
                    return False
                return True

            safe_local = [c for c in local_candidates if candidate_is_safe(c)]
            if safe_local:
                chosen = min(safe_local, key=lambda c: (manhattan_distance(c, desired), manhattan_distance(c, prev)))
                if chosen != desired:
                    local_repairs += 1
            elif config.allow_clearance_teleport_repair:
                safe_global = [c for c in walkable_cells if candidate_is_safe(c)]
                if safe_global:
                    chosen = min(
                        safe_global,
                        key=lambda c: (
                            manhattan_distance(c, prev) + manhattan_distance(c, desired),
                            manhattan_distance(c, desired),
                        ),
                    )
                    global_repairs += 1
                else:
                    chosen = prev
                    local_repairs += 1
                    unresolved_repairs += 1
            else:
                chosen = prev
                local_repairs += 1
                unresolved_repairs += 1

            repaired[agent_id].append(chosen)
            accepted.append(chosen)
            accepted_edges.add((prev, chosen))

    return repaired, {
        "clearance_local_repair_count": int(local_repairs),
        "clearance_global_repair_count": int(global_repairs),
        "clearance_unresolved_repair_count": int(unresolved_repairs),
        "clearance_teleport_repair_enabled": bool(config.allow_clearance_teleport_repair),
    }


def compute_longest_stationary_runs(paths: Sequence[PathType], max_time: int) -> List[int]:
    longest_runs: List[int] = []
    for path in paths:
        longest = 0
        current_run = 0
        for t in range(1, max_time + 1):
            if path_position_at(path, t) == path_position_at(path, t - 1):
                current_run += 1
                longest = max(longest, current_run)
            else:
                current_run = 0
        longest_runs.append(int(longest))
    return longest_runs


def prioritized_planning_repeated_tasks(
    starts: Sequence[Coord],
    pickup_points: Sequence[Coord],
    delivery_points: Sequence[Coord],
    walkable_map: np.ndarray,
    congestion_cost: np.ndarray,
    config: MAPFConfig,
    pickup_point_groups: Optional[Dict[str, List[Coord]]] = None,
    delivery_point_groups: Optional[Dict[str, List[Coord]]] = None,
) -> Tuple[List[PathType], Dict[str, Any]]:
    rng = np.random.default_rng(config.seed + 1000)
    planning_config = config.replace(
        max_time=config.max_time + max(0, int(config.continuous_task_lookahead)),
    )
    pickup_groups = filter_point_groups_by_walkability(
        pickup_point_groups or {"pickup": list(pickup_points)},
        walkable_map,
    )
    delivery_groups = filter_point_groups_by_walkability(
        delivery_point_groups or {"delivery": list(delivery_points)},
        walkable_map,
    )
    def plan_once(priority_order: Sequence[int], iteration: int) -> Tuple[
        List[PathType],
        List[int],
        List[int],
        List[Coord],
        List[List[Dict[str, Any]]],
        List[List[Dict[str, Any]]],
        List[Optional[Dict[str, Any]]],
        Dict[str, int],
    ]:
        local_rng = np.random.default_rng(config.seed + 1000)
        paths_by_agent: List[PathType] = [[] for _ in starts]
        completed_targets_by_agent = [0 for _ in starts]
        completed_deliveries_by_agent = [0 for _ in starts]
        final_goals_by_agent: List[Coord] = [tuple(start) for start in starts]
        histories_by_agent: List[List[Dict[str, Any]]] = [[] for _ in starts]
        assignments_by_agent: List[List[Dict[str, Any]]] = [[] for _ in starts]
        in_progress_by_agent: List[Optional[Dict[str, Any]]] = [None for _ in starts]
        prior_paths: List[PathType] = []
        # Built once and grown incrementally as each agent's path is finalized.
        reservation_table: Dict[int, set[Coord]] = {}
        edge_reservation_table: Dict[int, set[Tuple[Coord, Coord]]] = {}
        soft_cost_table: SoftCostTable = {}
        edge_buckets: EdgeBuckets = {}
        extension_stats = {
            "safe_extension_strict_steps": 0,
            "safe_extension_relaxed_steps": 0,
            "safe_extension_failed_steps": 0,
            "repeated_planning_failed_count": 0,
        }

        for agent_id in priority_order:
            start = starts[agent_id]
            current_path: PathType = [start]
            current = start
            current_t = 0
            target_count = 0
            delivery_count = 0
            next_target_is_pickup = True
            carrying_load = False
            last_goal = start
            task_history: List[Dict[str, Any]] = []
            agent_assignments: List[Dict[str, Any]] = []
            in_progress_target: Optional[Dict[str, Any]] = None

            while current_t < config.max_time:
                target_start_t = current_t
                target_groups = pickup_groups if next_target_is_pickup else delivery_groups
                if not target_groups:
                    break
                target_group, goal, segment = choose_reachable_group_target(
                    current,
                    target_groups,
                    local_rng,
                    walkable_map,
                    reservation_table,
                    edge_reservation_table,
                    congestion_cost,
                    planning_config,
                    current_t,
                    soft_cost_table=soft_cost_table,
                    edge_buckets=edge_buckets,
                )

                if target_group is None or goal is None or segment is None or len(segment) <= 1:
                    extension_stats["repeated_planning_failed_count"] += 1
                    if config.verbose_planning:
                        print(
                            f"[repeated planning failed] agent={agent_id}, "
                            f"t={current_t}, start={current}, target_type="
                            f"{'pickup' if next_target_is_pickup else 'delivery'}"
                        )
                    current_path, stats = extend_path_safely(
                        current_path, reservation_table, edge_reservation_table, walkable_map, config
                    )
                    extension_stats["safe_extension_strict_steps"] += stats["strict_steps"]
                    extension_stats["safe_extension_relaxed_steps"] += stats["relaxed_steps"]
                    extension_stats["safe_extension_failed_steps"] += stats["failed_steps"]
                    break

                remaining = config.max_time - current_t
                observed_segment = segment[: remaining + 1]
                current_path.extend(observed_segment[1:])
                current_t = len(current_path) - 1
                current = current_path[-1]
                last_goal = goal
                action = "pickup" if next_target_is_pickup else "delivery"
                assignment = {
                    "action": action,
                    "zone": target_group,
                    "target": [int(goal[0]), int(goal[1])],
                    "start_t": int(target_start_t),
                    "end_t": int(current_t),
                    "completed": bool(current == goal),
                }
                agent_assignments.append(assignment)
                if current == goal:
                    target_count += 1
                    if not next_target_is_pickup:
                        delivery_count += 1
                    carrying_load = next_target_is_pickup
                    task_history.append(
                        {
                            "action": action,
                            "zone": target_group,
                            "position": [int(goal[0]), int(goal[1])],
                            "arrival_t": int(current_t),
                            "carrying_load_after_action": bool(carrying_load),
                        }
                    )
                    next_target_is_pickup = not next_target_is_pickup
                    in_progress_target = None
                else:
                    in_progress_target = {
                        "action": action,
                        "zone": target_group,
                        "target": [int(goal[0]), int(goal[1])],
                        "current_position_at_observation_end": [int(current[0]), int(current[1])],
                        "carrying_load": bool(carrying_load),
                    }
                    break

            if len(current_path) < config.max_time + 1:
                current_path, stats = extend_path_safely(
                    current_path, reservation_table, edge_reservation_table, walkable_map, config
                )
                extension_stats["safe_extension_strict_steps"] += stats["strict_steps"]
                extension_stats["safe_extension_relaxed_steps"] += stats["relaxed_steps"]
                extension_stats["safe_extension_failed_steps"] += stats["failed_steps"]
            current_path = current_path[: config.max_time + 1]
            paths_by_agent[agent_id] = current_path
            prior_paths.append(current_path)
            add_path_to_reservation_tables(
                current_path,
                reservation_table,
                edge_reservation_table,
                planning_config.max_time,
            )
            add_path_to_soft_cost_table(
                current_path, soft_cost_table, planning_config.max_time, config
            )
            add_path_to_edge_buckets(current_path, edge_buckets, planning_config.max_time)
            completed_targets_by_agent[agent_id] = target_count
            completed_deliveries_by_agent[agent_id] = delivery_count
            final_goals_by_agent[agent_id] = last_goal
            histories_by_agent[agent_id] = task_history
            assignments_by_agent[agent_id] = agent_assignments
            in_progress_by_agent[agent_id] = in_progress_target
            if PLANNING_PROGRESS_HOOK is not None:
                PLANNING_PROGRESS_HOOK()
            if config.show_planning_progress:
                if config.planning_progress_label:
                    # One tagged line per agent: safe when several processes
                    # share the same console.
                    print(
                        f"{config.planning_progress_label} planning iteration "
                        f"{iteration + 1}: agent {len(prior_paths)}/{len(starts)} done",
                        flush=True,
                    )
                else:
                    print(
                        f"\r  planning iteration {iteration + 1}: "
                        f"agent {len(prior_paths)}/{len(starts)} done",
                        end="",
                        flush=True,
                    )

        if config.show_planning_progress and not config.planning_progress_label:
            print(flush=True)
        return (
            paths_by_agent,
            completed_targets_by_agent,
            completed_deliveries_by_agent,
            final_goals_by_agent,
            histories_by_agent,
            assignments_by_agent,
            in_progress_by_agent,
            extension_stats,
        )

    priority_order = list(range(len(starts)))
    priority_boost_events: List[Dict[str, Any]] = []
    max_iterations = max(0, int(config.priority_boost_replan_iterations))
    paths: List[PathType] = []
    completed_targets: List[int] = []
    completed_deliveries: List[int] = []
    final_goals: List[Coord] = []
    task_histories: List[List[Dict[str, Any]]] = []
    task_assignments: List[List[Dict[str, Any]]] = []
    in_progress_targets: List[Optional[Dict[str, Any]]] = []
    extension_stats: Dict[str, int] = {}
    longest_stationary_runs: List[int] = []

    for iteration in range(max_iterations + 1):
        (
            paths,
            completed_targets,
            completed_deliveries,
            final_goals,
            task_histories,
            task_assignments,
            in_progress_targets,
            extension_stats,
        ) = plan_once(priority_order, iteration)
        longest_stationary_runs = compute_longest_stationary_runs(paths, config.max_time)
        if not longest_stationary_runs:
            break
        worst_agent = int(np.argmax(longest_stationary_runs))
        worst_wait = int(longest_stationary_runs[worst_agent])
        if worst_wait <= config.max_stationary_steps_before_replan:
            break
        if iteration >= max_iterations:
            break

        priority_order = [worst_agent] + [agent_id for agent_id in priority_order if agent_id != worst_agent]
        priority_boost_events.append(
            {
                "iteration": int(iteration + 1),
                "boosted_agent": worst_agent,
                "longest_stationary_run": worst_wait,
                "new_priority_order": priority_order[:],
            }
        )

    summary = {
        "task_mode": "continuous_random_pickup_delivery_observed_until_max_time",
        "target_sampling_mode": "point_level_uniform_random_reachable_first",
        "observation_horizon_max_time": int(config.max_time),
        "planning_horizon_max_time": int(planning_config.max_time),
        "max_stationary_steps_before_replan": int(config.max_stationary_steps_before_replan),
        "priority_boost_replan_iterations": int(config.priority_boost_replan_iterations),
        "priority_boost_event_count": int(len(priority_boost_events)),
        "priority_boost_events": priority_boost_events,
        **extension_stats,
        "final_priority_order": priority_order,
        "longest_stationary_runs": longest_stationary_runs,
        "pickup_zone_groups": sorted(pickup_groups),
        "delivery_zone_groups": sorted(delivery_groups),
        "completed_targets": completed_targets,
        "completed_deliveries": completed_deliveries,
        "total_completed_targets": int(sum(completed_targets)),
        "total_completed_deliveries": int(sum(completed_deliveries)),
        "final_goals": [list(p) for p in final_goals],
        "task_histories": task_histories,
        "task_assignments": task_assignments,
        "in_progress_targets_at_observation_end": in_progress_targets,
    }
    return paths, summary

def expand_start_pool(
    base_points: Sequence[Coord],
    walkable_map: np.ndarray,
    needed: int,
    clearance_cells: int = 0,
) -> List[Coord]:
    base_points = unique_preserving_order([p for p in base_points if is_walkable(*p, walkable_map)])
    spaced_base: List[Coord] = []
    for point in base_points:
        if is_clear_of_points(point, spaced_base, clearance_cells):
            spaced_base.append(point)
    if len(spaced_base) >= needed:
        return spaced_base

    walkable_cells = [
        (int(x), int(y))
        for y, x in np.argwhere(walkable_map)
    ]

    def distance_to_base(cell: Coord) -> Tuple[int, int, int]:
        if not spaced_base:
            return (0, cell[1], cell[0])
        nearest = min(manhattan_distance(cell, base) for base in spaced_base)
        return (nearest, cell[1], cell[0])

    expanded = spaced_base[:]
    for cell in sorted(walkable_cells, key=distance_to_base):
        if cell not in expanded and is_clear_of_points(cell, expanded, clearance_cells):
            expanded.append(cell)
        if len(expanded) >= needed:
            break
    return expanded


def select_distributed_starts(
    walkable_map: np.ndarray,
    num_agents: int,
    seed: int,
    min_clearance: int = 1,
) -> List[Coord]:
    """Pick `num_agents` start cells spread across the whole walkable map.

    Poisson-disk style: shuffle the walkable cells, greedily keep ones at least
    `spacing` apart, and relax `spacing` until all agents fit. Deterministic for a
    given seed. This is the alternative to the default behaviour where every agent
    spawns in the single staging aisle (start_candidates are nearly all on y=42) and
    has to fan out from one cluster.
    """
    cells = [(int(x), int(y)) for y, x in np.argwhere(walkable_map)]
    if len(cells) < num_agents:
        raise ValueError(f"Only {len(cells)} walkable cells for {num_agents} agents.")
    target = max(int(min_clearance) + 1, int((len(cells) / max(1, num_agents)) ** 0.5))
    for spacing in range(target, -1, -1):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(cells))
        chosen: List[Coord] = []
        for i in order:
            c = cells[int(i)]
            if all(abs(c[0] - o[0]) + abs(c[1] - o[1]) > spacing for o in chosen):
                chosen.append(c)
                if len(chosen) >= num_agents:
                    return chosen
    return cells[:num_agents]


def select_start_goal_pairs(env: Dict[str, Any], walkable_map: np.ndarray, config: MAPFConfig) -> Tuple[List[Coord], List[Coord]]:
    rng = np.random.default_rng(config.seed)
    goals_pool = [p for p in normalize_points(env.get("delivery_points")) if is_walkable(*p, walkable_map)]
    if not goals_pool:
        raise ValueError("No valid walkable goals found in delivery_points.")

    if bool(getattr(config, "distributed_starts", False)):
        # Spread agents across the whole map instead of the one staging aisle.
        starts = select_distributed_starts(
            walkable_map,
            config.num_agents,
            config.seed,
            config.initial_start_clearance_cells,
        )
    else:
        starts_pool = normalize_points(env.get("start_candidates")) or normalize_points(env.get("charging_points"))
        starts_pool = expand_start_pool(
            starts_pool,
            walkable_map,
            config.num_agents,
            clearance_cells=config.initial_start_clearance_cells,
        )
        if not starts_pool:
            raise ValueError("No valid walkable starts found in start_candidates or charging_points.")
        starts_idx = rng.choice(len(starts_pool), size=config.num_agents, replace=False)
        starts = [starts_pool[int(i)] for i in starts_idx]

    replace_goals = len(goals_pool) < config.num_agents
    goals_idx = rng.choice(len(goals_pool), size=config.num_agents, replace=replace_goals)
    goals = [goals_pool[int(i)] for i in goals_idx]
    for agent_id, (start, goal) in enumerate(zip(starts, goals)):
        if start == goal and len(goals_pool) > 1:
            alternatives = [p for p in goals_pool if p != start]
            goals[agent_id] = alternatives[int(rng.integers(0, len(alternatives)))]
    return starts, goals
