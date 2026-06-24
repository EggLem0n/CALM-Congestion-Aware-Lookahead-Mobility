"""macpf.online_mapf: closed-loop (receding-horizon) MAPF with live congestion prediction.

Where `macpf.classical_mapf` plans every agent's full task chain once up front and
then plays the trajectory back, this package runs a closed loop:

    every 1 s (1 Hz):  observe current grid state
                       -> ConvLSTM forecasts the next `t_out` s of congestion
                       -> prioritized A* re-plans every agent from its current cell
                       -> commit, advance one grid step

The kinodynamic motion model + proximity safety controller (10 Hz feel via
animation sub-frames) are applied as the execution/visualization layer on top of
the resulting trajectory, exactly as `classical_mapf` does. The learned model is
queried *live* each second instead of being baked into a static cost file.

Run as a module:  python -m macpf.online_mapf --config configs/default.yaml

Submodules
----------
- predictor   : load the ConvLSTM checkpoint + run live inference.
- observe     : build the model's 5-channel input frames from the live state.
- world_state : per-agent live state + rolling observation-history buffer.
- replanner   : receding-horizon prioritized re-plan from the current state.
- runner      : the loop + CLI + result saving (reuses classical_mapf I/O).
"""
import os

# Match macpf.convjam: tolerate duplicate OpenMP runtimes before torch loads.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# This runner is a batch/file producer (it writes PNGs + an optional GIF), so it
# must not touch an interactive GUI backend -- doing so hard-crashes when launched
# non-interactively on Windows. Force the headless Agg backend before viz.py
# imports matplotlib.pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")
