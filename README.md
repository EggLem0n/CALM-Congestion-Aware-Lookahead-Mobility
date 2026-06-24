# MAPF with congestion prediction

MAPF with a learned congestion-prediction term added on top of the conventional planner, for more efficient path-generation in multi-agent systems.

The project has two halves:

- **Classical MAPF engine** (`macpf/classical_mapf/`) — the conventional planner + AMR
  simulator. Self-contained; not coupled to the learned pipeline.
- **Congestion-prediction pipeline** (`macpf/generate_heatmap/`, `features.py`, `convjam/`)
  — runs the engine to generate labeled data and learns a congestion heatmap that
  feeds back into planning as a soft cost.

## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
├── pyproject.toml     <- Project configuration and package metadata for macpf (ruff config too).
├── environment.yml    <- Conda environment specification.
│
├── configs            <- YAML simulation parameters (single source of truth).
│   ├── default.yaml          <- All classical-MAPF parameters.
│   └── heatmap_dataset.yaml  <- Overrides for dataset generation.
│
├── data
│   ├── maps                  <- Factory map arrays (factory/walkable/obstacle .npy).
│   ├── classical_runs        <- Per-run artifacts from a single classical MAPF run.
│   ├── heatmap_dataset       <- Generated congestion dataset shards (npz per episode).
│
├── models             <- Trained congestion models + predicted congestion-cost fields.
│
├── reports
│   └── figures               <- Human-facing plots (path plots, animation GIFs, heatmaps).
│
└── macpf              <- Source package.
    │
    ├── config.py             <- Paths (PROJ_ROOT, DATA_DIR, MODELS_DIR, FIGURES_DIR, ...).
    ├── generate_heatmap      <- Generate the congestion heatmap dataset (runs the engine).
    │   ├── generate.py           <- Dataset generation pipeline (episodes -> npz shards).
    │   ├── plots.py              <- Congestion heatmap figures -> reports/figures.
    │   └── __main__.py           <- CLI: `python -m macpf.generate_heatmap`.
    ├── features.py           <- Load episode shards into model-ready tensors.
    │
    ├── convjam
    │   ├── convlstm.py       <- ConvLSTM model definition (encoder-forecaster).
    │   ├── dataset.py        <- Windowed forecasting Dataset.
    │   ├── train.py          <- Train the ConvLSTM congestion predictor.
    │   ├── predict.py        <- Produce an AI congestion-cost field.
    │   └── visualize.py      <- Live predicted-vs-ground-truth animation.
    │
    └── classical_mapf        <- Classical MAPF engine (conventional planner + simulator).
        ├── classical_mapf.py     <- Single-run entry point.
        ├── factory_map_generator.py
        ├── solver.py             <- Reservation tables, A*, prioritized planners.
        ├── motion.py             <- Kinodynamic motion model + safety controller.
        ├── metrics.py            <- Occupancy/congestion labels + run metrics.
        ├── viz.py                <- Path plots + GIF animation.
        └── utils                 <- config loader, shared types, grid helpers.
```

## Setup

```bash
conda env create -f environment.yml
conda activate macpf
```

### Clone & set up on another machine

The dataset, trained checkpoints, and map arrays are **git-ignored** (kept out of the
repo to stay small), so a fresh clone must regenerate them — they are reproducible
from fixed seeds.

```bash
# 1) Clone + environment (installs deps + the CUDA cu128 PyTorch wheel)
git clone https://github.com/EggLem0n/MACPF.git
cd MACPF
conda env create -f environment.yml
conda activate macpf

# 2) Verify the GPU build of torch (needs an NVIDIA driver)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#   if it prints False (a CPU-only wheel got installed), reinstall the CUDA build:
#   pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3) Regenerate the map arrays and a dataset (not in the repo)
python -m macpf.classical_mapf.factory_map_generator        # -> data/maps/*.npy
python -m macpf.generate_heatmap --episodes 30 --seconds 300 --num_of_process 4
```

Then train / visualize / predict as below. To reuse an existing machine's data instead
of regenerating, copy `data/heatmap_dataset/`, `data/maps/`, and `models/*.pt` over
manually. Pull later updates with `git pull`.

### Updating an existing clone (don't re-clone)

If a machine was already cloned, just pull — but the changes must be pushed first.

```bash
# On the source machine: publish your changes
git add -A
git commit -m "..."
git push

# On the other machine (the existing clone):
cd MACPF
git pull
conda env update -f environment.yml      # if environment.yml changed (e.g. cu128 torch)
```

Renames/moves (e.g. `modeling/` -> `convjam/`) are applied automatically by `git pull`.
Your generated `data/` and `models/` are git-ignored, so pull never touches them.

If `git pull` is rejected because of local changes:

```bash
git stash && git pull && git stash pop      # keep local edits, or
git fetch origin && git reset --hard origin/main   # discard local edits, match remote
```

`data/` and `models/` survive a hard reset (they're ignored). After updating, re-verify
the GPU build: `python -c "import torch; print(torch.cuda.is_available())"`.

## Usage

```bash
# Generate the factory map preview / arrays
python -m macpf.classical_mapf.factory_map_generator

# Run a single classical MAPF simulation
#   data  -> data/classical_runs/<timestamp>/
#   plots -> reports/figures/classical_runs/<timestamp>/
python -m macpf.classical_mapf.classical_mapf --config configs/default.yaml

# Generate the congestion heatmap dataset (-> data/heatmap_dataset/<timestamp>/)
python -m macpf.generate_heatmap --episodes 100 --num_of_process 4

# Inspect the generated dataset tensors
python -m macpf.features

# Plot a congestion heatmap frame (-> reports/figures/congestion/)
python -m macpf.generate_heatmap.plots data/classical_runs/<timestamp>/congestion_labels.npy
```

## Congestion model (ConvLSTM)

The learned term is a **forecasting** ConvLSTM: it predicts the *future* congestion
cost map from the observed scenario state, which is exactly what the planner queries
(`solver.load_ai_congestion_cost` / `get_congestion_cost` index future timesteps).
Note that the same-step label `y[t]` is a deterministic Manhattan-kernel blur of the
occupancy channel, so a same-step mapping would be trivial — the model instead learns
the spatiotemporal evolution of congestion.

- **Architecture** — Encoder–Forecaster ConvLSTM (seq2seq). The encoder reads `T_in`
  observed 5-channel frames; the forecaster rolls out `T_out` future steps fed the
  *known exogenous* channels (obstacle / start / pickup-delivery markers, occupancy
  zeroed) and emits a non-negative congestion map per step.
  - `macpf/convjam/convlstm.py` — `ConvLSTMCell`, `CongestionConvLSTM`.
  - `macpf/convjam/dataset.py` — `CongestionWindowDataset` (windowed forecasting
    samples, label normalization, episode caching; reuses `features.load_episode`).
  - `macpf/convjam/train.py` — Adam + MSE + AMP loop, best-val checkpoint, optional
    `--preview` live window. `predict.py` / `visualize.py` — inference + animation.
- **I/O** — input `x:(T,5,H,W)`, label `y:(T,1,H,W)`; checkpoint stores `model_cfg` +
  `label_norm`; inference writes `(future_steps, H, W) float32` for the planner.

Full design notes: `../../docs/ConvLSTM_구현계획.md`.

```bash
# 0) Generate a dataset first (data/heatmap_dataset is empty by default)
python -m macpf.generate_heatmap --episodes 30 --seconds 300 --num_of_process 4

# 1) Train (defaults: 20 epochs, batch 8, CUDA). --preview shows a live
#    predicted-vs-ground-truth window while training. -> models/congestion_convlstm.pt
python -m macpf.convjam.train --preview

# 2) Replay predicted vs ground truth (after training), with the AMRs moving
python -m macpf.convjam.visualize

# 3) Produce a cost field for the planner -> models/congestion_cost.npy (future_steps, H, W)
python -m macpf.convjam.predict

# 4) Feed it back into planning: in configs/default.yaml set
#    use_ai_congestion_cost: true, ai_cost_path: <models/congestion_cost.npy>
python -m macpf.classical_mapf.classical_mapf --config configs/default.yaml
```

Requires a CUDA build of PyTorch (environment.yml installs the cu128 wheel). On an
8 GB GPU keep the batch small (default 8) — large batches OOM because the model
unrolls 70 timesteps over the full 50×80 map. The duplicate-OpenMP abort
("OMP: Error #15") is handled automatically in `macpf/convjam/__init__.py`.

## Online closed-loop planning (`macpf.online_mapf`)

`classical_mapf` plans every route once up front (with a *static* congestion file)
and replays it. `online_mapf` instead runs a **receding-horizon closed loop**:
every step (1 Hz) it observes the current grid state, runs the trained ConvLSTM
*live* to forecast the next `t_out` seconds of congestion, re-plans every agent
from its current cell with that fresh cost, then advances one cell. The kinodynamic
motion + proximity-safety controller are applied over the resulting trajectory just
as in `classical_mapf` (the 10 Hz "feel" comes from the animation sub-frames).

```bash
# Needs only a trained checkpoint (models/congestion_convlstm.pt). No
# congestion_cost.npy is read -- the model is queried live every step.
python -m macpf.online_mapf --config configs/default.yaml

# Handy overrides:
#   --max-time N --num-agents N --seed N --device cpu|cuda
#   --replan-every N   # steps between re-plans (default config.online_replan_every=1)
#   --no-animation     # skip the GIF
#   --no-figures       # skip ALL matplotlib output (headless / broken matplotlib)
```

Output: data → `data/online_runs/<timestamp>/`, figures →
`reports/figures/online_runs/<timestamp>/`.

The live prediction is rescaled to "robot-equivalents" (divided by
`congestion_center_value`) so it acts as a *soft* bias on A*; raw label peaks of a
few hundred would otherwise wall off crowded cells and stall the step-by-step
re-planning. Tune its strength with `ai_cost_weight` in the config.
