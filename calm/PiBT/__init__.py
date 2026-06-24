"""calm.PiBT: PIBT MAPF engine.

`from calm import PiBT as mapf` then `mapf.load_config()`,
`mapf.plan_pibt_repeated_tasks(...)`, `mapf.build_additive_congestion_label_sequence(...)`,
`mapf.select_start_goal_pairs(...)`, etc.

`factory_map_generator` is not re-exported (import it directly); it is the only
member that may pull in matplotlib (lazily, for visualization).
"""
from .types import Coord, PathType
from .config import Config, MAPFConfig, load_config
from .grid import (
    filter_point_groups_by_walkability,
    is_walkable,
    manhattan_distance,
    normalize_point_groups,
    normalize_points,
    path_position_at,
    walkable_neighbors,
)
from .distance import DistanceFieldCache, compute_distance_field
from .scenario import (
    expand_start_pool,
    select_distributed_starts,
    select_start_goal_pairs,
)
from .metrics import (
    build_additive_congestion_label_sequence,
    build_occupancy_sequence,
    compute_collision_count,
    paths_to_agent_positions,
)
from .pibt import plan_pibt_repeated_tasks, set_planning_progress_hook
