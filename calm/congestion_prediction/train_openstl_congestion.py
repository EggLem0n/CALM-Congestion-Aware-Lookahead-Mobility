# -*- coding: utf-8 -*-
"""
OpenSTL custom training script  --  follows examples/tutorial.ipynb step by step.

Task  : spatio-temporal prediction of the AMR factory CONGESTION heatmap.
        past `pre_seq_length` frames  ->  future `aft_seq_length` frames   (1 channel -> 1 channel)
Data  : C:\\Robot\\260614_0043   (100 episodes stored as per-episode .npz)
        each npz holds:
           x  (3601, 5, 50, 80) uint8     <- 5 input state channels (NOT used here)
           y  (3601, 1, 50, 80) float32   <- congestion heatmap      (THIS is what we predict)

Two (and only two) deviations from the tutorial, both forced by the data:
  1) We feed the single-channel congestion stream `y` as both the input video and the target
     video (past frames -> future frames).  Stock SimVP requires in_channels == out_channels
     (its aft>pre branch re-feeds predictions as inputs), so a 5ch->1ch model is impossible
     without editing the library.  Predicting future frames from past frames of the SAME
     1-channel video is exactly what the tutorial does, just with C=1 instead of C=3.
  2) N_S = 2 instead of the tutorial's 4.  SimVP down/up-samples by 2**(N_S/2); with N_S=4 the
     height 50 does not round-trip (50->25->13->26->52 != 50) and the encoder/decoder skip-add
     crashes.  N_S=2 round-trips exactly (50->25->50, 80->40->80).  N_S is an explicit tutorial
     hyperparameter, so tuning it stays within the tutorial.

Everything else (CustomDataset, dataloaders, custom_training_config / custom_model_config,
BaseExperiment, train()/test(), visualization) is taken straight from the tutorial.
"""

import os
# Windows/conda OpenMP shim: torch (libiomp) + numpy/opencv (MKL) may load two OpenMP
# runtimes in one process -> "OMP: Error #15". Allow it so training/eval can proceed.
# Must be set BEFORE importing numpy/torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import glob
import numpy as np

# ---------------------------------------------------------------------------
# make `import openstl` work when this file is run from C:\Robot
# (the tutorial assumes openstl is pip-installed; we just point at the local repo)
# ---------------------------------------------------------------------------
OPENSTL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OpenSTL")
if OPENSTL_ROOT not in sys.path:
    sys.path.insert(0, OPENSTL_ROOT)

import torch
from torch.utils.data import Dataset


# ===========================================================================
#  USER KNOBS
# ===========================================================================
# Moved into calm/congestion/: the dataset lives in the repo root's data/, two
# levels up from this folder, so reach up with "../..".
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "data", "heatmap_dataset", "260623_0907")

# ---- the tutorial's two key hyperparameters --------------------------------
pre_seq_length = 10     # number of given (past) frames
aft_seq_length = 10     # number of frames to predict (future)

# ---- how much data to load -------------------------------------------------
# Episodes are split by FILE (train/val/test never share frames).
# The split is STRATIFIED BY AMR ROBOT COUNT: each robot-count group (300, 350,
# ... 750 AMRs) is split on its own, so every count is represented in
# train / val / test in the same proportion (fair across robot counts).
# Each count has 33 episodes, so 19:7:7 divides them exactly (19 + 7 + 7 = 33).
# Uses all available episodes by default; lower MAX_EPISODES to train faster / use less RAM.
SPLIT_RATIO  = (19, 7, 7)    # train : val : test  (per 33-episode group -> 19/7/7)
MAX_EPISODES = None          # None = use all; else an even cap PER robot-count group
SPLIT_SEED   = 0             # shuffle seed inside each group (reproducible splits)

# stride of the sliding window along each 3601-frame episode timeline.
# == pre+aft  -> non-overlapping windows (keeps the dataset small).
window_stride = pre_seq_length + aft_seq_length

batch_size = 16
epoch      = 100        # tutorial: 3 is a toy value, ~100 is a good real starting point
lr         = 0.001

ex_name    = "custom_exp_congestion"

# Run testing only (skip training) on the best checkpoint already on disk.
# Set the env var TEST_ONLY=1 to salvage a run you stopped early:
#     (PowerShell)  $env:TEST_ONLY=1; python train_openstl_congestion.py
# It rebuilds the same data pipeline (so y_scale matches), loads best.ckpt, runs
# exp.test(), and writes saved/*.npy + the tutorial visualizations.
TEST_ONLY = os.environ.get("TEST_ONLY") == "1"


# ===========================================================================
# 1. Process your data
#    (the tutorial turns videos into a  B x T x C x H x W  array; here we turn
#     each episode's congestion stream `y` into the same B x T x C x H x W layout)
# ===========================================================================
def make_windows(y, pre_slen, aft_slen, stride):
    """y: (T_total, C, H, W) congestion stream of ONE episode.
    Returns data_x (n, pre_slen, C, H, W), data_y (n, aft_slen, C, H, W)."""
    total = pre_slen + aft_slen
    xs, ys = [], []
    for s in range(0, y.shape[0] - total + 1, stride):
        clip = y[s:s + total]            # (total, C, H, W)
        xs.append(clip[:pre_slen])       # past   -> input
        ys.append(clip[pre_slen:])       # future -> target
    if not xs:
        c, h, w = y.shape[1:]
        return (np.zeros((0, pre_slen, c, h, w), np.float32),
                np.zeros((0, aft_slen, c, h, w), np.float32))
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def build_split(files, pre_slen, aft_slen, stride):
    """Load a list of episode files and stack into X, Y  (B x T x C x H x W)."""
    X_list, Y_list = [], []
    for f in files:
        y = np.load(f)["y"].astype(np.float32)   # (3601, 1, 50, 80) congestion heatmap
        dx, dy = make_windows(y, pre_slen, aft_slen, stride)
        X_list.append(dx)
        Y_list.append(dy)
    return np.concatenate(X_list, 0), np.concatenate(Y_list, 0)


# ===========================================================================
# 1.5 Stratified split by AMR robot count
#     Each episode runs a fixed number of AMRs (300, 350, ... 750).  That count
#     equals the number of start positions stored in the npz, so we can read it
#     cheaply (the `starts` array is tiny) without loading the heatmap stream.
#     Splitting each count group on its own keeps train/val/test balanced across
#     robot counts -- otherwise a plain front/back file split leaves whole counts
#     out of val/test.
# ===========================================================================
def robot_count(f):
    """Number of AMRs in an episode = number of start positions in the npz."""
    with np.load(f) as z:
        return int(z["starts"].shape[0])


def stratified_split(files, ratio, max_per_group=None, seed=0):
    """Group `files` by AMR robot count, then split each group by `ratio`.
    Returns (train_files, val_files, test_files), each sorted, every robot
    count represented in all three in the same proportion."""
    import random
    groups = {}
    for f in files:
        groups.setdefault(robot_count(f), []).append(f)

    r_tr, r_va, r_te = ratio
    denom = r_tr + r_va + r_te
    rng = random.Random(seed)
    train, val, test = [], [], []
    summary = []
    for n_robots in sorted(groups):
        g = sorted(groups[n_robots])
        rng.shuffle(g)                       # de-bias within-group ordering
        if max_per_group is not None:
            g = g[:max_per_group]
        n = len(g)
        n_tr = n * r_tr // denom
        n_va = n * r_va // denom
        train += g[:n_tr]
        val   += g[n_tr:n_tr + n_va]
        test  += g[n_tr + n_va:]
        summary.append((n_robots, n_tr, n_va, n - n_tr - n_va))

    print("stratified split by AMR count (robots: train/val/test):")
    for n_robots, a, b, c in summary:
        print(f"   {n_robots:>4} AMRs -> {a:>3} / {b:>3} / {c:>3}")
    return sorted(train), sorted(val), sorted(test)


# ===========================================================================
# 2.1 Define the dataset   (verbatim from the tutorial)
# ===========================================================================
class CustomDataset(Dataset):
    def __init__(self, X, Y, normalize=False, data_name='custom'):
        super(CustomDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.mean = None
        self.std = None
        self.data_name = data_name

        if normalize:
            # get the mean/std values along the channel dimension
            mean = data.mean(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            std = data.std(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            data = (data - mean) / std
            self.mean = mean
            self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index]).float()
        labels = torch.tensor(self.Y[index]).float()
        return data, labels


def main():
    # ---- 1. build the dataset ---------------------------------------------
    files = sorted(glob.glob(os.path.join(DATA_DIR, "episode_*.npz")))
    assert files, f"no episode_*.npz found in {DATA_DIR}"

    # split episodes by the 6:2:2 ratio, STRATIFIED across AMR robot counts so
    # every count appears in train / val / test in the same proportion.
    train_files, val_files, test_files = stratified_split(
        files, SPLIT_RATIO, max_per_group=MAX_EPISODES, seed=SPLIT_SEED)
    print(f"episodes: {len(files)} total -> "
          f"train {len(train_files)} / val {len(val_files)} / test {len(test_files)}")

    print("building dataset ...")
    X_train, Y_train = build_split(train_files, pre_seq_length, aft_seq_length, window_stride)
    X_val,   Y_val   = build_split(val_files,   pre_seq_length, aft_seq_length, window_stride)
    X_test,  Y_test  = build_split(test_files,  pre_seq_length, aft_seq_length, window_stride)

    # The tutorial keeps frames in [0, 1] (it divides pixels by 255).
    # Congestion is unbounded, so we rescale by the training maximum into ~[0, 1].
    y_scale = float(max(X_train.max(), Y_train.max(), 1.0))
    for arr in (X_train, Y_train, X_val, Y_val, X_test, Y_test):
        arr /= y_scale

    # shape is B x T x C x H x W :  C = 1 (congestion), H = 50, W = 80
    print(f"  X_train {X_train.shape}  Y_train {Y_train.shape}   (y_scale = {y_scale:.1f})")
    print(f"  X_val   {X_val.shape}  Y_val   {Y_val.shape}")
    print(f"  X_test  {X_test.shape}  Y_test  {Y_test.shape}")

    # ---- 2.2 get the dataloaders  (verbatim from the tutorial) ------------
    train_set = CustomDataset(X=X_train, Y=Y_train)
    val_set   = CustomDataset(X=X_val,   Y=Y_val)
    test_set  = CustomDataset(X=X_test,  Y=Y_test)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader_val = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, pin_memory=True)

    # ===================================================================
    # 3.1 Define the custom configs   (tutorial)
    # ===================================================================
    custom_training_config = {
        'pre_seq_length': pre_seq_length,
        'aft_seq_length': aft_seq_length,
        'total_length': pre_seq_length + aft_seq_length,
        'batch_size': batch_size,
        'val_batch_size': batch_size,
        'epoch': epoch,
        'lr': lr,
        'metrics': ['mse', 'mae'],

        'ex_name': ex_name,
        'dataname': 'custom',
        'in_shape': [pre_seq_length, 1, 50, 80],   # T, C=1, H=50, W=80
    }

    custom_model_config = {
        # For MetaVP models, the most important hyperparameters are:
        # N_S, N_T, hid_S, hid_T, model_type
        'method': 'SimVP',
        'model_type': 'gSTA',
        'N_S': 2,        # 4 -> 2 so the 50x80 maps round-trip (50 is not divisible by 4)
        'N_T': 8,
        'hid_S': 64,
        'hid_T': 256,
    }

    # ===================================================================
    # 3.2 Setup the experiment   (tutorial)
    # ===================================================================
    from openstl.api import BaseExperiment
    from openstl.utils import create_parser, default_parser

    args = create_parser().parse_args([])
    config = args.__dict__

    # update the training config
    config.update(custom_training_config)
    # update the model config
    config.update(custom_model_config)
    # fulfill with default values
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]

    exp = BaseExperiment(
        args,
        dataloaders=(dataloader_train, dataloader_val, dataloader_test),
        strategy='auto')

    # ===================================================================
    # 3.3 Start training and evaluation   (tutorial)
    # ===================================================================
    if not TEST_ONLY:
        print('>' * 35 + ' training ' + '<' * 35)
        try:
            exp.train()
        except KeyboardInterrupt:
            # Ctrl+C: don't just die -- fall through to testing so saved/*.npy still
            # get written from the best checkpoint (behaves "as if not interrupted").
            print("\n[interrupted] training stopped early; continuing to test with "
                  "the best checkpoint so far.")

    print('>' * 35 + ' testing  ' + '<' * 35)
    # Evaluate the BEST (lowest val-loss) checkpoint, not the last in-memory weights
    # -- important when the model overfits late. exp.test() loads best.ckpt when
    # args.test is True; fall back to in-memory weights only if no checkpoint exists.
    best_ckpt = os.path.join("work_dirs", ex_name, "checkpoints", "best.ckpt")
    exp.args.test = os.path.exists(best_ckpt)
    exp.test()

    # ===================================================================
    # 4. Visualization   (tutorial; use_rgb=False because congestion is 1-channel)
    # ===================================================================
    try:
        from openstl.utils import show_video_line, show_video_gif_multiple

        saved = os.path.join("work_dirs", ex_name, "saved")
        inputs = np.load(os.path.join(saved, "inputs.npy"))
        preds  = np.load(os.path.join(saved, "preds.npy"))
        trues  = np.load(os.path.join(saved, "trues.npy"))

        ex = 0
        vmax = float(trues[ex].max())
        show_video_line(trues[ex], ncols=aft_seq_length, vmax=vmax, cbar=False,
                        out_path=os.path.join(saved, "true.png"), format='png', use_rgb=False)
        show_video_line(preds[ex], ncols=aft_seq_length, vmax=vmax, cbar=False,
                        out_path=os.path.join(saved, "pred.png"), format='png', use_rgb=False)
        show_video_gif_multiple(inputs[ex], trues[ex], preds[ex], use_rgb=False,
                                out_path=os.path.join(saved, "example.gif"))
        print("saved visualizations to", saved)
    except Exception as e:
        print("[viz skipped]", repr(e))


if __name__ == "__main__":
    main()
