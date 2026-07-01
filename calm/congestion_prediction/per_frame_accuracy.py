# -*- coding: utf-8 -*-
"""Per-horizon (per predicted frame) accuracy for the congestion predictor.

The model predicts 10 future frames (t+1 ... t+10) at once. Every other metric in
this repo (roll_episode, exp.test()'s ['mse','mae']) averages the error over the
WHOLE horizon, so you can't see how the prediction degrades from the near frame
(t+1, easy) to the far frame (t+10, hard). This script breaks the error down PER
FRAME and writes it as a CSV (+ an optional error-vs-horizon curve).

It reads the arrays exp.test() already saved -- no model, no dataset, no CUDA
needed -- so it runs instantly on whichever machine finished training:

    work_dirs/custom_exp_congestion/saved/preds.npy   (N, 10, 1, 50, 80)
    work_dirs/custom_exp_congestion/saved/trues.npy    "     "     "

Those arrays are in the model's NORMALIZED units (training divided congestion by
Y_SCALE into ~[0,1]). We report the metric in those normalized units AND, using
the same Y_SCALE constant as predict.py, converted back to raw congestion units
(raw_mse = norm_mse * Y_SCALE**2).

Run (OpenSTL conda env, on the machine that trained):
    conda activate OpenSTL
    cd calm/congestion_prediction
    python per_frame_accuracy.py                 # -> reports/.../per_frame_accuracy.csv + .png
    python per_frame_accuracy.py --saved <dir>   # point at a copied-over saved/ dir
    python per_frame_accuracy.py --no-plot       # CSV only (no matplotlib)
"""
from __future__ import annotations

import os
import csv
import argparse
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
EX_NAME = "custom_exp_congestion"
DEFAULT_SAVED = os.path.join(HERE, "work_dirs", EX_NAME, "saved")

# Must match predict.py's Y_SCALE (train split's windowed max). Only used to convert
# the normalized metric back into raw congestion units for the *_raw CSV columns.
Y_SCALE = 1100.0


def per_frame_metrics(preds: np.ndarray, trues: np.ndarray):
    """preds/trues: (N, T, C, H, W). Reduce over everything EXCEPT the frame axis.

    Returns a dict of length-T arrays: mse, mae, rmse (all in the arrays' own units).
    """
    assert preds.shape == trues.shape, (preds.shape, trues.shape)
    err = preds.astype(np.float64) - trues.astype(np.float64)
    axes = (0, 2, 3, 4)                       # keep axis 1 = frame / horizon
    mse = (err ** 2).mean(axis=axes)          # (T,)
    mae = np.abs(err).mean(axis=axes)         # (T,)
    return {"mse": mse, "mae": mae, "rmse": np.sqrt(mse)}


def write_csv(path: str, m: dict, y_scale: float) -> None:
    """One row per predicted frame + a final overall (horizon-averaged) row."""
    T = len(m["mse"])
    s2 = y_scale ** 2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["horizon", "mse_norm", "mae_norm", "rmse_norm",
                    "mse_raw", "mae_raw", "rmse_raw"])
        for t in range(T):
            w.writerow([f"t+{t + 1}",
                        f"{m['mse'][t]:.6e}", f"{m['mae'][t]:.6e}", f"{m['rmse'][t]:.6e}",
                        f"{m['mse'][t] * s2:.6f}", f"{m['mae'][t] * y_scale:.6f}",
                        f"{m['rmse'][t] * y_scale:.6f}"])
        # overall = mean over all frames (matches roll_episode / exp.test() reduction)
        mse_o, mae_o = float(m["mse"].mean()), float(m["mae"].mean())
        rmse_o = float(np.sqrt((m["rmse"] ** 2).mean()))
        w.writerow(["overall",
                    f"{mse_o:.6e}", f"{mae_o:.6e}", f"{rmse_o:.6e}",
                    f"{mse_o * s2:.6f}", f"{mae_o * y_scale:.6f}", f"{rmse_o * y_scale:.6f}"])


def plot_curve(path: str, m: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(m["mse"])
    xs = np.arange(1, T + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(xs, m["mse"], marker="o", color="tab:red", label="MSE (norm)")
    ax1.set_xlabel("prediction horizon  (t+k)")
    ax1.set_ylabel("MSE (normalized units)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_xticks(xs)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, m["mae"], marker="s", color="tab:blue", label="MAE (norm)")
    ax2.set_ylabel("MAE (normalized units)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_title("Per-frame prediction error vs horizon\n(higher k = further into the future)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saved", default=DEFAULT_SAVED,
                    help="dir holding preds.npy + trues.npy (default: this run's saved/)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the CSV/PNG (default: reports/.../<timestamp>/)")
    ap.add_argument("--y-scale", type=float, default=Y_SCALE,
                    help="normalized->raw scale; must match predict.py (default %(default)s)")
    ap.add_argument("--no-plot", action="store_true", help="write CSV only, skip the PNG")
    args = ap.parse_args()

    p_pred = os.path.join(args.saved, "preds.npy")
    p_true = os.path.join(args.saved, "trues.npy")
    if not (os.path.exists(p_pred) and os.path.exists(p_true)):
        raise SystemExit(
            f"[per_frame_accuracy] preds.npy / trues.npy not found in:\n    {args.saved}\n"
            "These are written by exp.test() (train_openstl_congestion.py). Run this on the\n"
            "machine that finished training, or point --saved at a copied-over saved/ dir.")

    preds = np.load(p_pred, mmap_mode="r")
    trues = np.load(p_true, mmap_mode="r")
    print(f"[per_frame_accuracy] preds {preds.shape}  trues {trues.shape}  "
          f"(N windows, T={preds.shape[1]} frames)")

    m = per_frame_metrics(np.asarray(preds), np.asarray(trues))

    out_dir = args.out_dir or os.path.join(
        REPO_ROOT, "reports", "congestion_prediction",
        datetime.now().strftime("%y%m%d_%H%M"))
    csv_path = os.path.join(out_dir, "per_frame_accuracy.csv")
    write_csv(csv_path, m, args.y_scale)

    # console table
    print("  horizon |   MSE(norm)   MAE(norm)  RMSE(norm) |    MSE(raw)    MAE(raw)  RMSE(raw)")
    print("  --------+-----------------------------------+-------------------------------------")
    s2 = args.y_scale ** 2
    for t in range(len(m["mse"])):
        print(f"   t+{t + 1:<2}   | {m['mse'][t]:.4e}  {m['mae'][t]:.4e}  {m['rmse'][t]:.4e} | "
              f"{m['mse'][t] * s2:11.2f}  {m['mae'][t] * args.y_scale:9.3f}  "
              f"{m['rmse'][t] * args.y_scale:9.3f}")
    print(f"[per_frame_accuracy] wrote {csv_path}")

    if not args.no_plot:
        png_path = os.path.join(out_dir, "per_frame_accuracy.png")
        plot_curve(png_path, m)
        print(f"[per_frame_accuracy] wrote {png_path}")


if __name__ == "__main__":
    main()
