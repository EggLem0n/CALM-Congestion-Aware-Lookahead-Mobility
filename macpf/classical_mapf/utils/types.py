"""Shared MAPF type aliases and the A* search node."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


Coord = Tuple[int, int]
PathType = List[Coord]


@dataclass(frozen=True)
class AStarNode:
    f: float
    g: float
    h: float
    t: int
    x: int
    y: int

    def heap_item(self, tie_breaker: int) -> Tuple[float, float, int, int, int, int]:
        return (self.f, self.h, self.t, tie_breaker, self.x, self.y)
