"""Priority Inheritance with Backtracking (PIBT) one-step engine.

This module implements the shared, algorithm-only coordination layer. It does
not know about pickup/delivery tasks or ConvLSTM directly; callers provide the
current positions, current goals, and optionally a candidate cost function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from macpf.classical_mapf.utils.grid import Coord, get_neighbors, manhattan_distance
from macpf.classical_mapf.solver import sampled_segment_min_distance


CandidateCostFn = Callable[[int, Coord, Coord, int], float]
DistanceFn = Callable[[int, Coord, Coord], float]


@dataclass
class PIBTAgentState:
    """Persistent per-agent PIBT state.

    `priority` is the adaptive priority p_i in the paper. `epsilon` is a stable
    unique tie-breaker, so priorities remain deterministic and unique.
    """

    priority: float
    epsilon: float


@dataclass
class PIBTStepResult:
    """Result of one synchronized PIBT timestep."""

    next_positions: list[Coord]
    priorities: list[float]
    priority_order: list[int]
    valid: list[bool]
    inherited_count: int = 0
    backtrack_count: int = 0
    forced_wait_count: int = 0
    candidate_reject_vertex: int = 0
    candidate_reject_swap: int = 0
    candidate_reject_continuous: int = 0
    debug_events: list[dict] = field(default_factory=list)


class PIBTEngine:
    """Shared PIBT engine for one-step multi-agent coordination.

    The engine repeats exactly one PIBT decision step per call. A higher-level
    runner should call `step(...)`, apply the returned synchronized movement,
    update task goals, and call `step(...)` again.
    """

    def __init__(
        self,
        walkable_map: np.ndarray,
        num_agents: int,
        *,
        seed: int = 0,
        distance_fn: Optional[DistanceFn] = None,
        candidate_cost_fn: Optional[CandidateCostFn] = None,
        candidate_cost_mode: str = "additive",
        continuous_safe_gap_cells: float = 0.0,
        debug: bool = False,
    ) -> None:
        self.walkable_map = walkable_map
        self.num_agents = int(num_agents)
        self.distance_fn = distance_fn
        self.candidate_cost_fn = candidate_cost_fn
        if candidate_cost_mode not in {"additive", "tiebreak"}:
            raise ValueError("candidate_cost_mode must be 'additive' or 'tiebreak'")
        self.candidate_cost_mode = candidate_cost_mode
        self.continuous_safe_gap_cells = float(continuous_safe_gap_cells)
        self.debug = bool(debug)

        self.rng = np.random.default_rng(seed)
        epsilons = np.linspace(0.0, 1.0, self.num_agents, endpoint=False, dtype=np.float64)
        self.rng.shuffle(epsilons)
        self.agent_states = [
            PIBTAgentState(priority=float(eps), epsilon=float(eps))
            for eps in epsilons
        ]

    def reset_priorities(self) -> None:
        for state in self.agent_states:
            state.priority = state.epsilon

    def update_priorities(
        self,
        positions: Sequence[Coord],
        goals: Sequence[Coord],
        *,
        active: Optional[Sequence[bool]] = None,
        assigned: Optional[Sequence[bool]] = None,
        assigned_priority_bonus: float = 0.0,
        priority_bias: Optional[Sequence[float]] = None,
    ) -> list[float]:
        """Update adaptive PIBT priorities.

        Paper rule: if an agent is already at its goal, reset to epsilon;
        otherwise, increase priority by one. `assigned_priority_bonus` is an
        optional MAPD hook: task-carrying/assigned agents can be ranked above
        free agents while preserving the same PIBT mechanism.
        """
        priorities: list[float] = []
        for agent_id, state in enumerate(self.agent_states):
            is_active = True if active is None else bool(active[agent_id])
            if not is_active:
                state.priority = state.epsilon
                priorities.append(state.priority)
                continue

            pos = tuple(positions[agent_id])
            goal = tuple(goals[agent_id])
            if pos == goal:
                state.priority = state.epsilon
            else:
                state.priority += 1.0

            effective = state.priority
            if assigned is not None and bool(assigned[agent_id]):
                effective += float(assigned_priority_bonus)
            if priority_bias is not None:
                effective += float(priority_bias[agent_id])
            priorities.append(float(effective))
        return priorities

    def step(
        self,
        positions: Sequence[Coord],
        goals: Sequence[Coord],
        *,
        active: Optional[Sequence[bool]] = None,
        assigned: Optional[Sequence[bool]] = None,
        assigned_priority_bonus: float = 0.0,
        priority_bias: Optional[Sequence[float]] = None,
        timestep: int = 0,
    ) -> PIBTStepResult:
        """Plan one collision-free synchronized movement step.

        Prevents:
        - vertex conflicts: two agents reserving the same next cell
        - swap conflicts: two agents exchanging cells in one timestep

        Uses recursive priority inheritance and backtracking when a desired
        candidate cell is currently occupied by an undecided lower-priority agent.
        """
        if len(positions) != self.num_agents or len(goals) != self.num_agents:
            raise ValueError("positions/goals length must match num_agents")

        current = [tuple(p) for p in positions]
        target = [tuple(g) for g in goals]
        is_active = [True] * self.num_agents if active is None else [bool(v) for v in active]
        priorities = self.update_priorities(
            current,
            target,
            active=is_active,
            assigned=assigned,
            assigned_priority_bonus=assigned_priority_bonus,
            priority_bias=priority_bias,
        )
        priority_order = sorted(
            range(self.num_agents),
            key=lambda agent_id: (-priorities[agent_id], agent_id),
        )

        next_positions: list[Optional[Coord]] = [None for _ in range(self.num_agents)]
        requested: set[Coord] = set()
        occupied_now: dict[Coord, int] = {pos: agent_id for agent_id, pos in enumerate(current)}
        valid = [False for _ in range(self.num_agents)]
        stats = {
            "inherited_count": 0,
            "backtrack_count": 0,
            "forced_wait_count": 0,
            "candidate_reject_vertex": 0,
            "candidate_reject_swap": 0,
            "candidate_reject_continuous": 0,
        }
        debug_events: list[dict] = []

        def distance(agent_id: int, cell: Coord) -> float:
            if self.distance_fn is not None:
                return float(self.distance_fn(agent_id, cell, target[agent_id]))
            return float(manhattan_distance(cell, target[agent_id]))

        def candidate_score(agent_id: int, cell: Coord) -> tuple[float, int, float]:
            extra = 0.0
            if self.candidate_cost_fn is not None:
                extra = float(self.candidate_cost_fn(agent_id, cell, target[agent_id], timestep))
            occupied_tie = 1 if cell in occupied_now and occupied_now[cell] != agent_id else 0
            base_distance = distance(agent_id, cell)
            if self.candidate_cost_mode == "tiebreak":
                return (
                    base_distance,
                    occupied_tie,
                    extra,
                )
            return (
                base_distance + extra,
                occupied_tie,
                extra,
            )

        def candidates_for(agent_id: int) -> list[Coord]:
            x, y = current[agent_id]
            cells = [(nx, ny) for nx, ny, _ in get_neighbors(x, y, timestep, self.walkable_map)]
            cells = list(dict.fromkeys(cells))
            # Match the reference PIBT behavior: shuffle candidates first, then
            # stable-sort by distance/occupancy so exact ties stay randomized.
            self.rng.shuffle(cells)
            return sorted(cells, key=lambda cell: candidate_score(agent_id, cell))

        def would_swap_with_assigned(agent_id: int, candidate: Coord, parent_id: Optional[int]) -> bool:
            if parent_id is not None and current[parent_id] == candidate:
                return True
            for other_id, other_next in enumerate(next_positions):
                if other_id == agent_id or other_next is None:
                    continue
                if current[other_id] == candidate and other_next == current[agent_id]:
                    return True
            return False

        def would_close_pass_assigned(agent_id: int, candidate: Coord) -> bool:
            if self.continuous_safe_gap_cells <= 0.0:
                return False
            for other_id, other_next in enumerate(next_positions):
                if other_id == agent_id or other_next is None:
                    continue
                if candidate == other_next:
                    continue
                if current[agent_id] == other_next and candidate == current[other_id]:
                    continue
                if (
                    sampled_segment_min_distance(
                        current[agent_id],
                        candidate,
                        current[other_id],
                        other_next,
                    )
                    < self.continuous_safe_gap_cells
                ):
                    return True
            return False

        def restore(snapshot_next: list[Optional[Coord]], snapshot_requested: set[Coord]) -> None:
            next_positions[:] = snapshot_next
            requested.clear()
            requested.update(snapshot_requested)

        def assign(agent_id: int, parent_id: Optional[int]) -> bool:
            if next_positions[agent_id] is not None:
                return True
            if not is_active[agent_id]:
                next_positions[agent_id] = current[agent_id]
                valid[agent_id] = True
                return True

            for candidate in candidates_for(agent_id):
                if candidate in requested:
                    stats["candidate_reject_vertex"] += 1
                    continue
                if would_swap_with_assigned(agent_id, candidate, parent_id):
                    stats["candidate_reject_swap"] += 1
                    continue
                if would_close_pass_assigned(agent_id, candidate):
                    stats["candidate_reject_continuous"] += 1
                    continue

                snapshot_next = list(next_positions)
                snapshot_requested = set(requested)
                next_positions[agent_id] = candidate
                requested.add(candidate)

                blocker_id = occupied_now.get(candidate)
                if (
                    blocker_id is not None
                    and blocker_id != agent_id
                    and next_positions[blocker_id] is None
                    and is_active[blocker_id]
                ):
                    stats["inherited_count"] += 1
                    if self.debug:
                        debug_events.append(
                            {
                                "type": "inherit",
                                "from": agent_id,
                                "to": blocker_id,
                                "cell": list(candidate),
                                "timestep": int(timestep),
                            }
                        )
                    if not assign(blocker_id, agent_id):
                        stats["backtrack_count"] += 1
                        restore(snapshot_next, snapshot_requested)
                        continue

                valid[agent_id] = True
                return True

            # Invalid: no candidate worked. Stay only as a fallback so every
            # agent receives a next position; the caller can inspect valid=False.
            if current[agent_id] not in requested:
                next_positions[agent_id] = current[agent_id]
                requested.add(current[agent_id])
            else:
                next_positions[agent_id] = current[agent_id]
            stats["forced_wait_count"] += 1
            valid[agent_id] = False
            if self.debug:
                debug_events.append(
                    {
                        "type": "invalid_wait",
                        "agent": agent_id,
                        "cell": list(current[agent_id]),
                        "timestep": int(timestep),
                    }
                )
            return False

        for agent_id in priority_order:
            if next_positions[agent_id] is None:
                assign(agent_id, None)

        finalized = [
            tuple(pos) if pos is not None else current[agent_id]
            for agent_id, pos in enumerate(next_positions)
        ]
        return PIBTStepResult(
            next_positions=finalized,
            priorities=[self.agent_states[i].priority for i in range(self.num_agents)],
            priority_order=priority_order,
            valid=valid,
            inherited_count=int(stats["inherited_count"]),
            backtrack_count=int(stats["backtrack_count"]),
            forced_wait_count=int(stats["forced_wait_count"]),
            candidate_reject_vertex=int(stats["candidate_reject_vertex"]),
            candidate_reject_swap=int(stats["candidate_reject_swap"]),
            candidate_reject_continuous=int(stats["candidate_reject_continuous"]),
            debug_events=debug_events,
        )
