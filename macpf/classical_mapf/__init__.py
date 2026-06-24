"""Public API for the classical MAPF package.

The package used to import every solver, motion, metrics, and visualization
module at import time. That made lightweight imports fragile on Windows because
optional native dependencies were loaded even for commands that did not need
them. Public names are now resolved lazily when requested.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "Config": "macpf.classical_mapf.utils.config",
    "DEFAULT_CONFIG_PATH": "macpf.classical_mapf.utils.config",
    "MAPFConfig": "macpf.classical_mapf.utils.config",
    "load_config": "macpf.classical_mapf.utils.config",
    "AStarNode": "macpf.classical_mapf.utils.types",
    "Coord": "macpf.classical_mapf.utils.types",
    "PathType": "macpf.classical_mapf.utils.types",
    "clamp": "macpf.classical_mapf.utils.grid",
    "distance_to_next_stop_or_end": "macpf.classical_mapf.utils.grid",
    "filter_point_groups_by_walkability": "macpf.classical_mapf.utils.grid",
    "footprint_cells": "macpf.classical_mapf.utils.grid",
    "get_neighbors": "macpf.classical_mapf.utils.grid",
    "is_clear_of_points": "macpf.classical_mapf.utils.grid",
    "is_walkable": "macpf.classical_mapf.utils.grid",
    "manhattan_distance": "macpf.classical_mapf.utils.grid",
    "movement_yaw": "macpf.classical_mapf.utils.grid",
    "nearest_cell": "macpf.classical_mapf.utils.grid",
    "normalize_angle": "macpf.classical_mapf.utils.grid",
    "normalize_point_groups": "macpf.classical_mapf.utils.grid",
    "normalize_points": "macpf.classical_mapf.utils.grid",
    "path_position_at": "macpf.classical_mapf.utils.grid",
    "unique_preserving_order": "macpf.classical_mapf.utils.grid",
    "walkable_degree": "macpf.classical_mapf.utils.grid",
    "add_path_to_edge_buckets": "macpf.classical_mapf.solver",
    "add_path_to_reservation_tables": "macpf.classical_mapf.solver",
    "add_path_to_soft_cost_table": "macpf.classical_mapf.solver",
    "astar_single_agent": "macpf.classical_mapf.solver",
    "build_reservation_table": "macpf.classical_mapf.solver",
    "choose_reachable_group_target": "macpf.classical_mapf.solver",
    "compute_longest_stationary_runs": "macpf.classical_mapf.solver",
    "extend_path_safely": "macpf.classical_mapf.solver",
    "get_congestion_cost": "macpf.classical_mapf.solver",
    "get_soft_proximity_cost": "macpf.classical_mapf.solver",
    "has_clearance_conflict": "macpf.classical_mapf.solver",
    "has_continuous_motion_conflict": "macpf.classical_mapf.solver",
    "has_continuous_motion_conflict_indexed": "macpf.classical_mapf.solver",
    "has_edge_conflict": "macpf.classical_mapf.solver",
    "has_vertex_conflict": "macpf.classical_mapf.solver",
    "load_ai_congestion_cost": "macpf.classical_mapf.solver",
    "min_continuous_motion_distance_to_reserved_edges": "macpf.classical_mapf.solver",
    "prioritized_planning": "macpf.classical_mapf.solver",
    "prioritized_planning_repeated_tasks": "macpf.classical_mapf.solver",
    "repair_paths_with_clearance": "macpf.classical_mapf.solver",
    "sampled_segment_min_distance": "macpf.classical_mapf.solver",
    "select_distributed_starts": "macpf.classical_mapf.solver",
    "select_start_goal_pairs": "macpf.classical_mapf.solver",
    "set_planning_progress_hook": "macpf.classical_mapf.solver",
    "apply_proximity_safety_controller": "macpf.classical_mapf.motion",
    "compute_contextual_speed": "macpf.classical_mapf.motion",
    "grid_paths_to_kinodynamic_states": "macpf.classical_mapf.motion",
    "states_to_agent_positions": "macpf.classical_mapf.motion",
    "build_additive_congestion_label_sequence": "macpf.classical_mapf.metrics",
    "build_occupancy_sequence": "macpf.classical_mapf.metrics",
    "compute_clearance_stats": "macpf.classical_mapf.metrics",
    "compute_collision_count": "macpf.classical_mapf.metrics",
    "compute_interpolated_clearance_stats": "macpf.classical_mapf.metrics",
    "compute_metrics": "macpf.classical_mapf.metrics",
    "compute_start_clearance_stats": "macpf.classical_mapf.metrics",
    "paths_to_agent_positions": "macpf.classical_mapf.metrics",
    "animate_paths": "macpf.classical_mapf.viz",
    "plot_map_background": "macpf.classical_mapf.viz",
    "visualize_paths": "macpf.classical_mapf.viz",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
