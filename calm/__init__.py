"""PIBT-MACPF: a PIBT-based MAPF engine + congestion heatmap dataset generator.

This is a fresh start from the prioritized-planning MACPF project: the solver is
replaced by lifelong PIBT (``calm.PiBT.pibt``), which scales near-linearly
with agent count where the old prioritized planner blew up under congestion.

Only the pieces needed to produce the congestion heatmap dataset are ported:
the factory map, config, grid/scenario helpers, distance fields, occupancy/
congestion metrics, and the heatmap generator. The congestion-prediction model
(training/online replanning) comes later.
"""
