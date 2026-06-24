"""Start/goal selection for an episode (deterministic from config.seed)."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .config import MAPFConfig
from .grid import (
    is_clear_of_points,
    is_walkable,
    manhattan_distance,
    normalize_points,
    unique_preserving_order,
)
from .types import Coord


def expand_start_pool(
    base_points: Sequence[Coord],
    walkable_map: np.ndarray,
    needed: int,
    clearance_cells: int = 0,
) -> List[Coord]:
    """Grow a start pool to at least ``needed`` cells, spreading out from the base
    staging points to the nearest walkable cells while keeping a min clearance."""
    base_points = unique_preserving_order([p for p in base_points if is_walkable(*p, walkable_map)])
    spaced_base: List[Coord] = []
    for point in base_points:
        if is_clear_of_points(point, spaced_base, clearance_cells):
            spaced_base.append(point)
    if len(spaced_base) >= needed:
        return spaced_base

    walkable_cells = [(int(x), int(y)) for y, x in np.argwhere(walkable_map)]

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
    """Pick ``num_agents`` start cells spread across the whole walkable map.

    Poisson-disk style: shuffle walkable cells, greedily keep ones at least
    ``spacing`` apart, relax ``spacing`` until all agents fit. Deterministic for a
    given seed. The alternative to spawning every agent in the staging aisle.
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


def select_mixed_starts(
    walkable_map: np.ndarray,
    num_agents: int,
    fraction: float,
    seed: int,
    aisle_base_points: Sequence[Coord],
    clearance_cells: int = 0,
) -> List[Coord]:
    """Spawn ``fraction`` of the agents spread across the whole map and the rest in the
    charging/staging aisles. ``fraction=0`` -> all aisle, ``fraction=1`` -> all distributed.

    Per-agent mix (not an episode-level toggle): a single episode genuinely blends the two
    spawn styles, so the dispersion axis is continuous. Deterministic for a given seed.
    """
    fraction = min(1.0, max(0.0, float(fraction)))
    n_dist = int(round(fraction * num_agents))
    n_aisle = num_agents - n_dist
    rng = np.random.default_rng(seed)

    dist_cells = (
        select_distributed_starts(walkable_map, n_dist, seed, clearance_cells)
        if n_dist > 0 else []
    )
    used = set(dist_cells)

    aisle_cells: List[Coord] = []
    if n_aisle > 0:
        # Over-provision the aisle pool so n_aisle remain after dropping any that collide
        # with an already-chosen distributed cell.
        pool = expand_start_pool(
            aisle_base_points, walkable_map, n_aisle + len(used), clearance_cells=clearance_cells
        )
        pool = [c for c in pool if c not in used]
        if len(pool) > n_aisle:
            idx = rng.choice(len(pool), size=n_aisle, replace=False)
            aisle_cells = [pool[int(i)] for i in idx]
        else:
            aisle_cells = pool

    starts = list(dist_cells) + aisle_cells
    if len(starts) < num_agents:  # dedupe/clearance left us short: top up from any free cell
        used = set(starts)
        for y, x in np.argwhere(walkable_map):
            c = (int(x), int(y))
            if c not in used:
                starts.append(c)
                used.add(c)
                if len(starts) >= num_agents:
                    break
    return starts[:num_agents]


def select_start_goal_pairs(
    env: Dict[str, Any], walkable_map: np.ndarray, config: MAPFConfig
) -> Tuple[List[Coord], List[Coord]]:
    """Return (starts, goals) for ``config.num_agents`` agents.

    Goals here are just an initial delivery-point assignment; the lifelong PIBT
    solver reassigns pickup/delivery targets over the episode. Kept for interface
    parity with the heatmap generator.
    """
    rng = np.random.default_rng(config.seed)
    goals_pool = [p for p in normalize_points(env.get("delivery_points")) if is_walkable(*p, walkable_map)]
    if not goals_pool:
        raise ValueError("No valid walkable goals found in delivery_points.")

    # Dispersion fraction = share of agents spawned map-wide; rest spawn in the aisles.
    # Legacy distributed_starts=True is honoured as fraction=1.0.
    fraction = float(getattr(config, "distributed_fraction", 0.0))
    if bool(getattr(config, "distributed_starts", False)):
        fraction = 1.0
    aisle_base = normalize_points(env.get("start_candidates")) or normalize_points(env.get("charging_points"))
    if fraction < 1.0 and not aisle_base:
        raise ValueError("No valid walkable starts found in start_candidates or charging_points.")
    starts = select_mixed_starts(
        walkable_map, config.num_agents, fraction, config.seed, aisle_base,
        config.initial_start_clearance_cells,
    )

    replace_goals = len(goals_pool) < config.num_agents
    goals_idx = rng.choice(len(goals_pool), size=config.num_agents, replace=replace_goals)
    goals = [goals_pool[int(i)] for i in goals_idx]
    for agent_id, (start, goal) in enumerate(zip(starts, goals)):
        if start == goal and len(goals_pool) > 1:
            alternatives = [p for p in goals_pool if p != start]
            goals[agent_id] = alternatives[int(rng.integers(0, len(alternatives)))]
    return starts, goals
