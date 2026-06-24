"""macpf.classical_mapf: classical MAPF simulation engine.

This sub-package holds the conventional (non-learned) MAPF planner and AMR
simulator. It is intentionally kept separate from the congestion-prediction
pipeline (macpf.generate_heatmap / macpf.features / macpf.convjam), which
is the learned part of the project.

Importing this package re-exports the public API from focused submodules, so
`from macpf import classical_mapf as mapf` then `mapf.load_config()`,
`mapf.prioritized_planning_repeated_tasks(...)`,
`mapf.build_additive_congestion_label_sequence(...)`, etc.

Engine submodules (re-exported here):
- utils    : utils/{config,types,grid} sub-package (load_config, Coord/PathType/AStarNode, grid helpers)
- solver   : reservation tables, A*, and prioritized planners
- motion   : kinodynamic motion model and safety controller
- metrics  : occupancy/congestion labels and run metrics
- viz      : path plots and GIF animation

Other package members (not re-exported; import directly):
- factory_map_generator : builds the 50x80 factory environment the engine consumes
- classical_mapf        : single-run entry point -> python -m macpf.classical_mapf.classical_mapf

The congestion dataset generator that consumes this engine lives one level up at
macpf/generate_heatmap.py -> python -m macpf.generate_heatmap.
"""
from .utils import *
from .solver import *
from .motion import *
from .metrics import *
from .viz import animate_paths, plot_map_background, visualize_paths
