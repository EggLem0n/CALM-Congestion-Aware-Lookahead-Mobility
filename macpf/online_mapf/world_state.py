"""Live world state for the online closed-loop planner.

Tracks, per agent, the current grid cell plus the repeated-task bookkeeping the
classical planner keeps internally (which target it is heading to, whether it is
carrying a load, delivery counts) -- but as *persistent* state that survives
across 1 Hz re-plans, so an agent keeps pursuing the same target until it
actually arrives instead of re-rolling a random target every second.

Also holds the rolling observation buffer: the last `t_in` 5-channel frames the
ConvLSTM needs as its encoder input.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from macpf.classical_mapf import Coord


@dataclass
class AgentState:
    """One AMR's live state in the closed loop."""

    pos: Coord                                  # current grid cell
    goal: Optional[Coord] = None                # cell currently being pursued (None => pick a new target)
    zone: Optional[str] = None                  # pickup/delivery zone name of the current goal
    next_target_is_pickup: bool = True          # type of the *current* goal (flips on arrival)
    carrying_load: bool = False
    completed_targets: int = 0
    completed_deliveries: int = 0
    path: List[Coord] = field(default_factory=list)        # full recorded trajectory (grid)
    assignments: List[Dict[str, Any]] = field(default_factory=list)
    _open_assignment: Optional[Dict[str, Any]] = None      # in-flight assignment being timed

    def begin_assignment(self, zone: str, goal: Coord, t: int) -> None:
        self.goal = tuple(goal)
        self.zone = zone
        self._open_assignment = {
            "action": "pickup" if self.next_target_is_pickup else "delivery",
            "zone": zone,
            "target": [int(goal[0]), int(goal[1])],
            "start_t": int(t),
            "end_t": int(t),
            "completed": False,
        }

    def complete_goal(self, t: int) -> None:
        """Mark the current goal reached and set up the next pickup/delivery leg."""
        self.completed_targets += 1
        if not self.next_target_is_pickup:
            self.completed_deliveries += 1
        self.carrying_load = self.next_target_is_pickup
        if self._open_assignment is not None:
            self._open_assignment["end_t"] = int(t)
            self._open_assignment["completed"] = True
            self.assignments.append(self._open_assignment)
            self._open_assignment = None
        self.next_target_is_pickup = not self.next_target_is_pickup
        self.goal = None
        self.zone = None

    def all_assignments(self) -> List[Dict[str, Any]]:
        """Completed assignments plus the in-flight one, so the animation can draw
        the route line / target marker for the leg the agent is currently on."""
        if self._open_assignment is not None:
            return self.assignments + [self._open_assignment]
        return self.assignments


class World:
    """Container for all agent states + the model's observation history buffer."""

    def __init__(self, starts: List[Coord], t_in: int):
        self.agents: List[AgentState] = [
            AgentState(pos=tuple(s), path=[tuple(s)]) for s in starts
        ]
        self.t: int = 0
        self.history: Deque[np.ndarray] = deque(maxlen=max(1, t_in))

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def positions_array(self) -> np.ndarray:
        """(N, 2) int array of current cells, for occupancy/marker building."""
        return np.asarray([a.pos for a in self.agents], dtype=np.int32)

    def goals_typed(self) -> List[tuple]:
        """[(goal_cell, is_pickup)] for agents that currently have an active goal."""
        return [
            (a.goal, a.next_target_is_pickup)
            for a in self.agents
            if a.goal is not None
        ]

    def push_frame(self, frame: np.ndarray) -> None:
        self.history.append(frame)

    def assembled_paths(self) -> List[List[Coord]]:
        return [a.path for a in self.agents]
