"""True (obstacle-aware) shortest-distance fields, used by PIBT to rank moves."""
from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np

from .types import Coord


def compute_distance_field(walkable_map: np.ndarray, goal: Coord) -> np.ndarray:
    """4-connected shortest grid-distance from every walkable cell to ``goal``.

    A single BFS over the static map; unreachable cells stay +inf. PIBT ranks an
    agent's candidate moves by this field, so on the factory map (long conveyor
    obstacles make Manhattan a weak guide) agents thread real corridors instead of
    walking into dead ends. The field depends only on (static map, goal), so it is
    cached per goal and reused across agents and timesteps.
    """
    H, W = walkable_map.shape[:2]
    field = np.full((H, W), np.inf, dtype=np.float32)
    gx, gy = int(goal[0]), int(goal[1])
    if not (0 <= gx < W and 0 <= gy < H) or not walkable_map[gy, gx]:
        return field
    field[gy, gx] = 0.0
    queue = deque([(gx, gy)])
    while queue:
        x, y = queue.popleft()
        nd = field[y, x] + 1.0
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < W and 0 <= ny < H and walkable_map[ny, nx] and field[ny, nx] == np.inf:
                field[ny, nx] = nd
                queue.append((nx, ny))
    return field


class DistanceFieldCache:
    """Lazily computes and caches one distance field per goal cell."""

    def __init__(self, walkable_map: np.ndarray):
        self._walkable = walkable_map
        self._cache: Dict[Coord, np.ndarray] = {}

    def field(self, goal: Coord) -> np.ndarray:
        key = (int(goal[0]), int(goal[1]))
        field = self._cache.get(key)
        if field is None:
            field = compute_distance_field(self._walkable, key)
            self._cache[key] = field
        return field
