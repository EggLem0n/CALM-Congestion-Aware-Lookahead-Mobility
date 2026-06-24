"""Grid helpers: walkability, neighbours, distances, and point normalisation."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .types import Coord, PathType

if TYPE_CHECKING:
    from .config import MAPFConfig


def normalize_points(points: Any) -> List[Coord]:
    if points is None:
        return []
    if isinstance(points, dict):
        iterable: Iterable[Any] = points.values()
    else:
        iterable = points

    coords: List[Coord] = []
    for item in iterable:
        if isinstance(item, dict):
            if "x" in item and "y" in item:
                coords.append((int(item["x"]), int(item["y"])))
            elif "coord" in item:
                x, y = item["coord"]
                coords.append((int(x), int(y)))
            elif "position" in item:
                x, y = item["position"]
                coords.append((int(x), int(y)))
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[:2]
            coords.append((int(x), int(y)))
    return coords


def normalize_point_groups(groups: Any) -> Dict[str, List[Coord]]:
    if not isinstance(groups, dict):
        return {}
    normalized: Dict[str, List[Coord]] = {}
    for name, points in groups.items():
        coords = normalize_points(points)
        if coords:
            normalized[str(name)] = coords
    return normalized


def filter_point_groups_by_walkability(
    groups: Dict[str, List[Coord]],
    walkable_map: np.ndarray,
) -> Dict[str, List[Coord]]:
    return {
        name: [point for point in points if is_walkable(*point, walkable_map)]
        for name, points in groups.items()
        if any(is_walkable(*point, walkable_map) for point in points)
    }

def is_walkable(x: int, y: int, walkable_map: np.ndarray) -> bool:
    h, w = walkable_map.shape[:2]
    return 0 <= x < w and 0 <= y < h and bool(walkable_map[y, x])


def get_neighbors(x: int, y: int, t: int, walkable_map: np.ndarray) -> List[Tuple[int, int, int]]:
    candidates = [
        (x + 1, y, t + 1),
        (x - 1, y, t + 1),
        (x, y + 1, t + 1),
        (x, y - 1, t + 1),
        (x, y, t + 1),
    ]
    return [(nx, ny, nt) for nx, ny, nt in candidates if is_walkable(nx, ny, walkable_map)]


def manhattan_distance(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def path_position_at(path: PathType, t: int) -> Coord:
    if not path:
        raise ValueError("Empty path cannot be queried.")
    return path[min(t, len(path) - 1)]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def nearest_cell(x: float, y: float) -> Coord:
    return (int(round(x)), int(round(y)))


def footprint_cells(x: float, y: float, config: "MAPFConfig") -> List[Coord]:
    # The requested vehicle size is one grid cell, so occupancy is the nearest cell.
    # This function keeps the footprint logic centralized for later larger vehicles.
    return [nearest_cell(x, y)]


def walkable_degree(cell: Coord, walkable_map: np.ndarray) -> int:
    x, y = cell
    return sum(
        1
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        if is_walkable(nx, ny, walkable_map)
    )


def movement_yaw(a: Coord, b: Coord, fallback: float) -> float:
    if a == b:
        return fallback
    return math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))


def distance_to_next_stop_or_end(path: PathType, t: int, max_time: int) -> int:
    current = path_position_at(path, t)
    distance = 0
    for future_t in range(t + 1, max_time + 1):
        nxt = path_position_at(path, future_t)
        if nxt == current:
            return distance
        distance += manhattan_distance(current, nxt)
        current = nxt
    return distance

def unique_preserving_order(points: Sequence[Coord]) -> List[Coord]:
    seen: set[Coord] = set()
    unique: List[Coord] = []
    for point in points:
        if point not in seen:
            seen.add(point)
            unique.append(point)
    return unique


def is_clear_of_points(point: Coord, points: Sequence[Coord], clearance_cells: int) -> bool:
    return all(abs(point[0] - other[0]) + abs(point[1] - other[1]) > clearance_cells for other in points)
