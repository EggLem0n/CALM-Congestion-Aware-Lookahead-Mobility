# -*- coding: utf-8 -*-
"""
All-in-one visualization for an OpenSTL congestion run.  Reads what a run has
produced and writes every figure into reports/.  Safe to run any time: each
output is independent (loss curve needs only the log; the rest need the test
arrays saved by exp.test()).

Reads : work_dirs/<ex_name>/train_*.log              -> loss curve
        work_dirs/<ex_name>/saved/{preds,trues}.npy  -> GT-vs-pred image + videos
Writes: reports/loss_curve.png
        reports/compare_gt_vs_pred.png               (one window: GT over Pred)
        reports/videos/ep###_r<robots>.mp4           (ONE video PER test episode,
                                                       GT left vs Pred right)

The per-episode videos use OpenCV (fast for thousands of frames).  mp4 via the
'mp4v' codec (bundled with opencv-python -- no external ffmpeg needed here).

Run:
    conda activate OpenSTL
    python C:\\Robot\\congestion\\visualize.py
"""
import os
import re
import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- config (match train_openstl_congestion.py) ----------------------------
ex_name   = "custom_exp_congestion"
ex        = 0       # which window to draw for the single static compare image
VIDEO_FPS = 10      # playback speed of the per-episode videos
VIDEO_SCALE = 6     # upscale the 50x80 heatmaps by this factor for visibility

here      = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(here))   # CALM/ (two up from calm/<this folder>/)
log_dir = os.path.join(here, "work_dirs", ex_name)
saved   = os.path.join(log_dir, "saved")        # exp.test() arrays (read)
# Figures go under reports/ in a per-code, per-run subfolder so they're attributable
# to this congestion-prediction code and to a specific run (reports/ is shared).
reports = os.path.join(repo_root, "reports", "congestion_prediction",
                       datetime.now().strftime("%y%m%d_%H%M"))  # this run's figures
os.makedirs(reports, exist_ok=True)


# ===========================================================================
# 1. train / val loss curve   (needs only the per-epoch log -> works mid-run)
# ===========================================================================
def plot_loss():
    logs = sorted(glob.glob(os.path.join(log_dir, "train_*.log")))
    if not logs:
        print("[loss curve skipped] no train_*.log in", log_dir)
        return
    log_path = logs[-1]
    pat = re.compile(r"Epoch\s+(\d+):.*?Train Loss:\s+([\d.eE+-]+).*?Vali Loss:\s+([\d.eE+-]+)")
    epochs, train, val = [], [], []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train.append(float(m.group(2)))
                val.append(float(m.group(3)))
    if not epochs:
        print("[loss curve skipped] no epoch loss lines in", log_path)
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train, marker='o', ms=3, label="train loss")
    ax.plot(epochs, val,   marker='s', ms=3, label="val loss")
    best = min(range(len(val)), key=lambda i: val[i])
    ax.axvline(epochs[best], color='gray', ls='--', lw=1)
    ax.annotate(f"best val {val[best]:.4g}\n@ epoch {epochs[best]}",
                xy=(epochs[best], val[best]),
                xytext=(0.98, 0.95), textcoords='axes fraction',
                ha='right', va='top', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (MSE)")
    ax.set_title(f"Train / Val loss  ({os.path.basename(log_path)})")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(reports, "loss_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[loss curve] {len(epochs)} epochs from {os.path.basename(log_path)} -> {out}")


def _load_test_arrays():
    p_pred = os.path.join(saved, "preds.npy")
    p_true = os.path.join(saved, "trues.npy")
    if not (os.path.exists(p_pred) and os.path.exists(p_true)):
        return None, None
    return np.load(p_pred), np.load(p_true)


# ===========================================================================
# 2. single static image: GT (top) vs prediction (bottom) for one window
# ===========================================================================
def compare_image():
    preds, trues = _load_test_arrays()
    if preds is None:
        print("[compare image skipped] preds/trues.npy not found in", saved)
        return
    T    = trues[ex].shape[0]
    vmax = float(trues[ex].max())
    fig, axes = plt.subplots(2, T, figsize=(1.6 * T, 3.6))
    for t in range(T):
        axes[0, t].imshow(trues[ex][t, 0], vmin=0, vmax=vmax, cmap='jet')
        axes[1, t].imshow(preds[ex][t, 0], vmin=0, vmax=vmax, cmap='jet')
        axes[0, t].set_title(f"t+{t + 1}", fontsize=8)
        for r in (0, 1):
            axes[r, t].set_xticks([]); axes[r, t].set_yticks([])
    axes[0, 0].set_ylabel("GT",   fontsize=11)
    axes[1, 0].set_ylabel("Pred", fontsize=11)
    fig.suptitle("Congestion: ground truth (top) vs prediction (bottom)")
    fig.tight_layout()
    png = os.path.join(reports, "compare_gt_vs_pred.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print("[compare image]", png)


# ===========================================================================
# 3. ONE video per test episode: GT (left) vs prediction (right)
#    Test windows are stored episode-by-episode in order, every episode has the
#    same window count, so we chunk preds/trues back into per-episode clips.
# ===========================================================================
def _episode_robot_counts(n_windows):
    """Return (n_episodes, windows_per_ep, [robot_count per episode]).
    Re-derives the exact test split used to build the arrays so each clip can be
    labelled by its AMR count; falls back to plain indexing if that fails."""
    try:
        import train_openstl_congestion as T
        files = sorted(glob.glob(os.path.join(T.DATA_DIR, "episode_*.npz")))
        _, _, test_files = T.stratified_split(
            files, T.SPLIT_RATIO, max_per_group=T.MAX_EPISODES, seed=T.SPLIT_SEED)
        n_ep = len(test_files)
        if n_ep and n_windows % n_ep == 0:
            counts = [T.robot_count(f) for f in test_files]
            return n_ep, n_windows // n_ep, counts
    except Exception as e:
        print("  (robot-count labelling unavailable:", repr(e), "-> index only)")
    # fallback: assume every episode is 1801 frames -> 90 windows
    wpe = 90
    n_ep = max(1, n_windows // wpe)
    return n_ep, wpe, [None] * n_ep


def compare_videos_per_episode():
    import cv2
    preds, trues = _load_test_arrays()
    if preds is None:
        print("[episode videos skipped] preds/trues.npy not found in", saved)
        return

    N, T, _, H, W = preds.shape
    n_ep, wpe, counts = _episode_robot_counts(N)
    vids = os.path.join(reports, "videos")
    os.makedirs(vids, exist_ok=True)

    fw, fh = W * VIDEO_SCALE, H * VIDEO_SCALE
    gap, bar = 16, 34                      # column gap + top label bar height
    out_w, out_h = fw * 2 + gap, fh + bar
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    font = cv2.FONT_HERSHEY_SIMPLEX

    def colorize(frame, vmax):
        f8 = (np.clip(frame / vmax, 0, 1) * 255).astype(np.uint8)
        c = cv2.applyColorMap(f8, cv2.COLORMAP_JET)
        return cv2.resize(c, (fw, fh), interpolation=cv2.INTER_NEAREST)

    made = 0
    for e in range(n_ep):
        gt = trues[e * wpe:(e + 1) * wpe, :, 0].reshape(-1, H, W)   # (wpe*T, H, W)
        pr = preds[e * wpe:(e + 1) * wpe, :, 0].reshape(-1, H, W)
        vmax = float(max(gt.max(), pr.max(), 1e-6))
        tag = f"r{counts[e]}" if counts[e] is not None else "rNA"
        path = os.path.join(vids, f"ep{e:03d}_{tag}.mp4")
        vw = cv2.VideoWriter(path, fourcc, VIDEO_FPS, (out_w, out_h))
        for i in range(gt.shape[0]):
            canvas = np.zeros((out_h, out_w, 3), np.uint8)
            canvas[bar:bar + fh, 0:fw]                 = colorize(gt[i], vmax)
            canvas[bar:bar + fh, fw + gap:fw + gap + fw] = colorize(pr[i], vmax)
            cv2.putText(canvas, "Ground truth", (6, 23), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Prediction", (fw + gap + 6, 23), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            label = f"{counts[e]} AMRs  frame {i + 1}/{gt.shape[0]}" if counts[e] is not None \
                    else f"ep{e} frame {i + 1}/{gt.shape[0]}"
            cv2.putText(canvas, label, (out_w - 250, 23), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            vw.write(canvas)
        vw.release()
        made += 1
    print(f"[episode videos] {made} clips ({wpe * T} frames each) -> {vids}")


if __name__ == "__main__":
    plot_loss()
    compare_image()
    compare_videos_per_episode()
