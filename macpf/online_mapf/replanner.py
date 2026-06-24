"""Receding-horizon prioritized re-planning from the live state.

This is the closed-loop counterpart of `solver.prioritized_planning_repeated_tasks`.
The offline planner plans every agent's whole task chain once, starting at t=0 from
the fixed start cells. Here, every call re-plans from each agent's *current* cell
using the freshly predicted congestion, in a local timeframe where t=0 is "now".

Reuses the classical engine wholesale: `astar_single_agent`,
`choose_reachable_group_target`, the reservation/soft-cost/edge tables, and the
incremental table builders. Only the orchestration (start from current state,
persist the current goal across cycles) is new.

Target continuity: an agent keeps pursuing `agent.goal` until it arrives; only when
`goal is None` (initial, or just completed) does it roll a new reachable target.
This avoids the agent re-rolling a random target every second.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from macpf.classical_mapf import (
    Coord,
    MAPFConfig,
    PathType,
    add_path_to_edge_buckets,
    add_path_to_reservation_tables,
    add_path_to_soft_cost_table,
    astar_single_agent,
    choose_reachable_group_target,
    extend_path_safely,
)

from .world_state import World


def replan(
    world: World,
    congestion_cost: np.ndarray,
    pickup_groups: Dict[str, List[Coord]],
    delivery_groups: Dict[str, List[Coord]],
    walkable_map: np.ndarray,
    planning_config: MAPFConfig,
    rng: np.random.Generator,
    priority_order: Optional[List[int]] = None,
) -> List[PathType]:
    """Re-plan all agents from their current cells. Returns committed grid paths.

    Each returned path is in the local timeframe (index 0 == current cell). The
    runner executes index 1.. of each path before the next re-plan.
    """
    if priority_order is None:
        priority_order = list(range(world.num_agents))

    reservation_table: Dict[int, set] = {}
    edge_reservation_table: Dict[int, set] = {}
    soft_cost_table: Dict[int, Dict[Coord, float]] = {}
    edge_buckets: Dict = {}
    committed: List[Optional[PathType]] = [None] * world.num_agents
    horizon = int(planning_config.max_time)

    for agent_id in priority_order:
        agent = world.agents[agent_id]
        pos = agent.pos

        # Defensive: if we are already sitting on the active goal, close it out so
        # a fresh target is rolled below (normally handled at advance time).
        if agent.goal is not None and pos == agent.goal:
            agent.complete_goal(world.t)

        segment: Optional[PathType] = None
        if agent.goal is None:
            groups = pickup_groups if agent.next_target_is_pickup else delivery_groups
            zone, goal, segment = choose_reachable_group_target(
                pos,
                groups,
                rng,
                walkable_map,
                reservation_table,
                edge_reservation_table,
                congestion_cost,
                planning_config,
                0,
                soft_cost_table=soft_cost_table,
                edge_buckets=edge_buckets,
            )
            if goal is not None and segment is not None and len(segment) > 1:
                agent.begin_assignment(zone, goal, world.t)
            else:
                segment = None
        else:
            segment = astar_single_agent(
                pos,
                agent.goal,
                walkable_map,
                reservation_table,
                edge_reservation_table,
                congestion_cost,
                planning_config,
                start_time=0,
                soft_cost_table=soft_cost_table,
                edge_buckets=edge_buckets,
            )

        if not segment or len(segment) <= 1:
            # Blocked / no reachable target: hold position, but *safely* -- waiting on
            # the current cell is unsafe because higher-priority agents (planned first)
            # may pass through it, causing collisions. extend_path_safely picks moves
            # that avoid the reservation tables, exactly as the offline planner does.
            segment, _ = extend_path_safely(
                [pos], reservation_table, edge_reservation_table, walkable_map, planning_config
            )

        committed[agent_id] = segment
        # Register the full planned path so lower-priority agents plan around it,
        # exactly as the offline prioritized planner does.
        add_path_to_reservation_tables(segment, reservation_table, edge_reservation_table, horizon)
        add_path_to_soft_cost_table(segment, soft_cost_table, horizon, planning_config)
        add_path_to_edge_buckets(segment, edge_buckets, horizon)

    return committed  # type: ignore[return-value]
