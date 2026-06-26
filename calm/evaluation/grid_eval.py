# -*- coding: utf-8 -*-
"""3-D grid evaluation of congestion-aware PIBT + per-episode movement videos.

Skeleton mirrors ``calm.generate_heatmap.generate`` (the heatmap dataset
generator): the same (AMR count) x (dispersion fraction) grid of cells, one parallel
job per cell, a timestamped output dir with ``metadata.json``. The difference: there
are no "rounds" -- instead each cell is solved at every congestion weight lambda in
``--weights`` (default 0, 0.25, 0.5, 0.75, 1.0), so the result is a 3-D grid
(count x frac x lambda). lambda = 0 is plain "vanilla" PIBT; lambda > 0 turns on the
trained SimVP predictor.

The seed is fixed PER CELL (base_seed + cell index) and shared across all lambdas, so
within a cell the only thing that changes is the congestion penalty -- a controlled A/B.

Per (cell, lambda) metrics written to ``metrics.csv``:
  deliveries          throughput (higher better)
  energy              total Manhattan distance travelled by all AMRs (simple distance)
  energy_per_delivery distance per completed delivery (lower = more efficient)
  density_uniformity  entropy of time-averaged occupancy over walkable cells in [0,1]
                      (higher = AMRs more evenly spread)
  occ_cv              coeff. of variation of that occupancy (lower = more uniform)
  mean_robot_cong / p99_cong / peak_cong   ground-truth congestion (at robots / tail / peak)
  collisions          sanity (always 0)

Per cell it also writes ONE 2-panel MP4 PER lambda>0 -- vanilla (left) vs that lambda
(right) -- of the ACTUAL AMR movement (dots on the factory map), so weights
{0,0.25,0.5,0.75,1} give 4 videos per cell. Rendered with matplotlib FFMpegWriter
(MACPF's mp4 method; ffmpeg comes bundled via imageio-ffmpeg -- no system ffmpeg needed).

Run (OpenSTL conda env, from this folder) -- see commands at the bottom of the file.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
# Mute PyTorch Lightning's harmless pkg_resources deprecation chatter (set before the
# lazy predict/openstl import; also re-applied per worker in _init_worker).
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")
import sys
import csv
import json
import shutil
import signal
import time
import argparse
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import zip_longest
from multiprocessing import Array, Manager, Value
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PRED_DIR = REPO_ROOT / "calm" / "congestion_prediction"      # predict.py lives here (lazy import)
for p in (str(HERE), str(PRED_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from calm import PiBT as mapf                                    # noqa: E402
from calm.PiBT import factory_map_generator as fmg       # noqa: E402

# Lazily-built, per-process singleton (avoid importing torch / loading the 220MB model in
# workers that only run vanilla baselines, and avoid re-loading it per run).
_PREDICTOR = None


def get_predictor():
    """Per-process SimVP predictor singleton. Inference always runs on the GPU."""
    global _PREDICTOR
    if _PREDICTOR is None:
        from predict import CongestionPredictor      # local import: keeps torch out of vanilla paths
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU inference is required but CUDA is not available. Install a CUDA build of torch "
                "(e.g. pip install torch --index-url https://download.pytorch.org/whl/cu121).")
        _PREDICTOR = CongestionPredictor(device="cuda")
    return _PREDICTOR


# ---------------------------------------------------------------------------
# grid axes (same helpers as generate.py)
# ---------------------------------------------------------------------------
def agent_count_sweep(args):
    return list(range(args.min_agents, args.max_agents + 1, args.agent_step))


def frac_sweep(args):
    n = int(round((args.max_frac - args.min_frac) / args.frac_step)) + 1
    out = []
    for i in range(n):
        f = round(args.min_frac + i * args.frac_step, 6)
        if f <= args.max_frac + 1e-9:
            out.append(min(1.0, max(0.0, f)))
    return out


def grid_cells(args):
    return [(c, f) for c in agent_count_sweep(args) for f in frac_sweep(args)]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def evaluate(paths, summary, walkable, config, wall):
    H, W = walkable.shape[:2]
    ap = mapf.paths_to_agent_positions(paths, config.max_time)            # (T, N, 2)
    T = ap.shape[0]
    cong = mapf.build_additive_congestion_label_sequence(
        ap, H, W, config.congestion_center_value, config.congestion_step_value)

    tidx = np.arange(T)[:, None]
    robot_cong = cong[tidx, ap[..., 1], ap[..., 0]]                       # (T, N)
    energy = int(np.abs(np.diff(ap.astype(np.int32), axis=0)).sum())      # total cells moved
    deliveries = int(summary["total_completed_deliveries"])

    occ = mapf.build_occupancy_sequence(ap, H, W).astype(np.float64)      # (T, H, W)
    occ_mean = occ.mean(axis=0)[walkable]
    total = occ_mean.sum()
    if total > 0:
        p = occ_mean[occ_mean > 0] / total
        uniformity = float(-(p * np.log(p)).sum() / np.log(int(walkable.sum())))
        occ_cv = float(occ_mean.std() / (occ_mean.mean() + 1e-12))
    else:
        uniformity, occ_cv = 0.0, 0.0

    return {
        "deliveries": deliveries,
        "energy": energy,
        "energy_per_delivery": (energy / deliveries) if deliveries else float("nan"),
        "density_uniformity": uniformity,
        "occ_cv": occ_cv,
        "mean_robot_cong": float(robot_cong.mean()),
        "p99_cong": float(np.percentile(cong, 99.0)),
        "peak_cong": float(cong.max()),
        "collisions": int(mapf.compute_collision_count(ap)),
        "preds": int(summary["congestion_prediction_count"]),
        "wall_s": float(wall),
    }


def solve(weight, env, config, starts, predict_every, *,
          gamma=0.73, horizon=10, min_depth=2, depth_mode="peaked"):
    walkable = np.asarray(env["walkable_map"]).astype(bool)
    pickup = [p for p in mapf.normalize_points(env.get("pickup_points")) if mapf.is_walkable(*p, walkable)]
    delivery = [p for p in mapf.normalize_points(env.get("delivery_points")) if mapf.is_walkable(*p, walkable)]
    predictor = get_predictor() if weight > 0 else None
    t0 = time.perf_counter()
    paths, summary = mapf.plan_pibt_repeated_tasks(
        starts, pickup, delivery, walkable, config,
        pickup_point_groups=mapf.normalize_point_groups(env.get("pickup_point_groups")),
        delivery_point_groups=mapf.normalize_point_groups(env.get("delivery_point_groups")),
        congestion_predictor=predictor, congestion_weight=weight,
        congestion_gamma=gamma, congestion_horizon=horizon,
        congestion_min_depth=min_depth, congestion_depth_mode=depth_mode,
        predict_every=predict_every,
    )
    return paths, summary, evaluate(paths, summary, walkable, config, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# movement video: MACPF's animate_paths (vendored verbatim in calm.PiBT.viz).
# One MACPF panel per run; vanilla | congestion are hstacked with ffmpeg.
# ---------------------------------------------------------------------------
def _anim_config(base, args):
    """Config carrying MACPF's animation knobs + the adjustable robot/line sizes."""
    return base.replace(
        animation_subframes=args.anim_subframes,
        animation_interval_ms=35,
        local_path_update_hz=1.0,
        show_planned_routes=args.planned_routes,
        viz_robot_size=args.robot_size,
        viz_start_size=args.robot_size,
        viz_route_linewidth=args.route_linewidth,
        viz_planned_route_linewidth=args.planned_route_linewidth,
        viz_target_size=args.target_size,
        show_planning_progress=False,
        viz_video_dpi=args.video_dpi,
        viz_video_cq=args.video_cq,
    )


def _animate_scenario(env, starts, paths, summary, anim_cfg, out_mp4, tmp_dir):
    """Render ONE scenario with the vendored MACPF animate_paths -> out_mp4."""
    from calm.PiBT import viz
    tmp_dir.mkdir(parents=True, exist_ok=True)
    viz.animate_paths(env, paths, list(starts), list(starts), tmp_dir, anim_cfg, task_summary=summary)
    src = tmp_dir / "classical_mapf_animation.mp4"
    if not src.exists():                       # ffmpeg missing -> animate_paths fell back to gif
        src = tmp_dir / "classical_mapf_animation.gif"
        out_mp4 = out_mp4.with_suffix(".gif")
    if out_mp4.exists():
        out_mp4.unlink()
    shutil.move(str(src), str(out_mp4))
    return out_mp4


def _hstack(left_mp4, right_mp4, out_mp4, cq=15):
    """Combine two equal-size clips side by side (ffmpeg hstack). Re-encodes on the GPU
    (NVENC H.264) at high quality so this second pass barely adds generation loss; falls
    back to high-quality CPU x264 if NVENC is unavailable on this machine."""
    import subprocess
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    base = [ff, "-y", "-loglevel", "error", "-i", str(left_mp4), "-i", str(right_mp4),
            "-filter_complex", "[0:v][1:v]hstack=inputs=2"]
    nvenc = ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr",
             "-cq", str(cq), "-b:v", "0", "-pix_fmt", "yuv420p", str(out_mp4)]
    x264 = ["-c:v", "libx264", "-preset", "slow", "-crf", str(cq + 1),
            "-pix_fmt", "yuv420p", str(out_mp4)]
    try:
        subprocess.run(base + nvenc, check=True)
    except subprocess.CalledProcessError:
        subprocess.run(base + x264, check=True)


def _heatmap_pair(env, ap_base, ap_cong, cfg, out_mp4, tmp, args, labels):
    """Render the GROUND-TRUTH congestion fields of the vanilla and congestion-aware
    runs as 'hot' heatmaps and hstack them (vanilla | congestion) -> out_mp4 -- the same
    side-by-side layout as the movement video, but of the congestion the AMRs actually
    produced. A shared colour scale (percentile over BOTH fields) makes the two panels
    directly comparable, so you can see the applied run flatten the hotspots."""
    from calm.generate_heatmap.render_heatmap import render_congestion_video
    walk = np.asarray(env["walkable_map"]).astype(bool)
    H, W = walk.shape[:2]
    obstacle = np.asarray(env.get("obstacle_map", (~walk).astype(np.uint8)), dtype=np.float32)
    cv, sv = cfg.congestion_center_value, cfg.congestion_step_value
    cong_b = mapf.build_additive_congestion_label_sequence(ap_base, H, W, cv, sv)
    cong_c = mapf.build_additive_congestion_label_sequence(ap_cong, H, W, cv, sv)
    both = np.concatenate([cong_b[cong_b > 0].ravel(), cong_c[cong_c > 0].ravel()])
    vmax = float(np.percentile(both, args.heatmap_vmax_pct)) if both.size else 1.0
    fps = max(1, round(30 / max(1, args.anim_subframes)))    # match the movement clip's duration
    lb, lc = labels
    bmp4, cmp4 = tmp / "heat_vanilla.mp4", tmp / "heat_cong.mp4"
    render_congestion_video(cong_b, ap_base, obstacle, bmp4, fps=fps, dpi=args.video_dpi,
                            vmax=vmax, label=lb, title_fmt="t={t}/{T}")
    render_congestion_video(cong_c, ap_cong, obstacle, cmp4, fps=fps, dpi=args.video_dpi,
                            vmax=vmax, label=lc, title_fmt="t={t}/{T}")
    _hstack(bmp4, cmp4, out_mp4, args.video_cq)


# ---------------------------------------------------------------------------
# work unit = ONE planner run (a single grid point), so every worker stays busy:
# the total run count vastly outnumbers the workers. env / base config are
# per-process singletons (built once per worker, reused across that worker's runs).
# ---------------------------------------------------------------------------
_ENV = None
_BASE_CFG = None
_RUNNING = None        # pid -> label of the run this worker is doing now (for the live board)


def _job_label(job):
    """Short human label of a run: 'n300 f0.0  peaked md1 g0.61 λ0.5' / '... baseline'."""
    head = f"n{job['count']:>3} f{job['frac']:.1f}"
    if job["baseline"]:
        return f"{head}  baseline"
    return (f"{head}  {job['depth_mode']:>9} md{job['min_depth']} "
            f"g{job['gamma']:g} λ{job['weight']:g}")


def get_env():
    global _ENV
    if _ENV is None:
        _ENV = fmg.build_factory_map()
    return _ENV


def get_base_cfg():
    global _BASE_CFG
    if _BASE_CFG is None:
        _BASE_CFG = mapf.load_config()
    return _BASE_CFG


def _cell_cfg(count, frac, seed, args):
    """Deterministic per-cell config (same seed -> same starts & run, sharable across runs)."""
    return get_base_cfg().replace(
        num_agents=count, distributed_fraction=frac, seed=seed, max_time=args.seconds,
        congestion_center_value=args.center_value, congestion_step_value=args.step_value,
        show_planning_progress=False)


def run_job(job, args):
    """One planner run. Returns ONLY its CSV row (no path arrays cross the process boundary;
    videos re-run the chosen configs deterministically afterwards)."""
    if _RUNNING is not None:
        _RUNNING[os.getpid()] = _job_label(job)        # tell the board what this worker is on now
    try:
        env = get_env()
        walkable = np.asarray(env["walkable_map"]).astype(bool)
        cfg = _cell_cfg(job["count"], job["frac"], job["seed"], args)
        starts, _ = mapf.select_start_goal_pairs(env, walkable, cfg)
        weight = 0.0 if job["baseline"] else job["weight"]
        _, _, m = solve(weight, env, cfg, starts, args.predict_every,
                        gamma=job["gamma"], horizon=job["horizon"],
                        min_depth=job["min_depth"], depth_mode=job["depth_mode"])
        if job["baseline"]:
            row = {"episode": job["cell_idx"], "num_agents": job["count"], "frac": job["frac"],
                   "gamma": "", "horizon": job["horizon"], "min_depth": "", "depth_mode": "baseline",
                   "weight": 0.0, "seed": job["seed"], **m}
        else:
            row = {"episode": job["cell_idx"], "num_agents": job["count"], "frac": job["frac"],
                   "gamma": job["gamma"], "horizon": job["horizon"], "min_depth": job["min_depth"],
                   "depth_mode": job["depth_mode"], "weight": job["weight"], "seed": job["seed"], **m}
        return {"cell_idx": job["cell_idx"], "row": row}
    finally:
        if _RUNNING is not None:
            _RUNNING.pop(os.getpid(), None)


def video_job(vjob, args, out_dir_str):
    """Render one vanilla|congestion MP4 per gamma for a cell (primary config, best lambda).
    Re-runs the baseline + each chosen config deterministically, so no paths are shipped."""
    env = get_env()
    walkable = np.asarray(env["walkable_map"]).astype(bool)
    cfg = _cell_cfg(vjob["count"], vjob["frac"], vjob["seed"], args)
    starts, _ = mapf.select_start_goal_pairs(env, walkable, cfg)
    prim_mode, prim_md, horizon = vjob["prim_mode"], vjob["prim_md"], vjob["horizon"]
    best_by_gamma = vjob["best_by_gamma"]

    vdir = Path(out_dir_str) / "videos"
    tmp = vdir / f".tmp_ep{vjob['cell_idx']:03d}"
    anim_cfg = _anim_config(cfg, args)
    vs = args.video_seconds if 0 < args.video_seconds < args.seconds else (args.seconds + 1)
    clip = lambda paths: [p[:vs + 1] for p in paths]

    base_paths, base_summary, _ = solve(
        0.0, env, cfg, starts, args.predict_every,
        gamma=next(iter(best_by_gamma)), horizon=horizon, min_depth=prim_md, depth_mode=prim_mode)
    vanilla_mp4 = _animate_scenario(env, starts, clip(base_paths), base_summary,
                                    anim_cfg, tmp / "vanilla.mp4", tmp / "v")
    ap_base = mapf.paths_to_agent_positions(base_paths, vs) if args.heatmap else None
    names = []
    for g in sorted(best_by_gamma):
        w = best_by_gamma[g]
        cpaths, csumm, _ = solve(w, env, cfg, starts, args.predict_every,
                                 gamma=g, horizon=horizon, min_depth=prim_md, depth_mode=prim_mode)
        cong_mp4 = _animate_scenario(env, starts, clip(cpaths), csumm, anim_cfg,
                                     tmp / f"g{g:g}.mp4", tmp / f"c{g:g}")
        name = (f"ep{vjob['cell_idx']:03d}_n{vjob['count']}_f{vjob['frac']:.1f}_"
                f"{prim_mode}_md{prim_md}_g{g:g}_w{w:g}.mp4")
        _hstack(vanilla_mp4, cong_mp4, vdir / name, args.video_cq)
        names.append(name)
        if args.heatmap:
            ap_cong = mapf.paths_to_agent_positions(cpaths, vs)
            hname = name[:-4] + "_heatmap.mp4"
            _heatmap_pair(env, ap_base, ap_cong, cfg, vdir / hname, tmp, args,
                          labels=("vanilla λ0",
                                  f"{prim_mode} md{prim_md} g{g:g} λ{w:g}"))
            names.append(hname)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"cell_idx": vjob["cell_idx"], "videos": names}


def _init_worker(running=None):
    global _RUNNING
    _RUNNING = running
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # main process handles Ctrl+C
    warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")


def save_metrics_table_png(rows, weights, n_cells, out_path):
    """Render the by-lambda mean metrics as a table image (matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["lambda", "deliveries", "±sd", "energy", "energy/deliv", "uniformity",
            "occ_cv", "cong@robot", "p99 cong", "peak cong", "preds", "collisions"]
    body, best = [], {}
    # remember the best (max deliveries, max uniformity, min congestion) for highlighting
    for w in weights:
        sub = [r for r in rows if r["weight"] == w]
        if not sub:
            continue
        m = lambda k: float(np.mean([r[k] for r in sub]))
        dv = [r["deliveries"] for r in sub]
        sd = float(np.std(dv)) if len(dv) > 1 else 0.0
        body.append([f"{w:g}", f"{m('deliveries'):.1f}", f"{sd:.1f}", f"{m('energy'):.0f}",
                     f"{m('energy_per_delivery'):.1f}", f"{m('density_uniformity'):.3f}",
                     f"{m('occ_cv'):.2f}", f"{m('mean_robot_cong'):.1f}",
                     f"{m('p99_cong'):.0f}", f"{m('peak_cong'):.0f}", f"{m('preds'):.0f}",
                     f"{int(sum(r['collisions'] for r in sub))}"])

    fig, ax = plt.subplots(figsize=(1.35 * len(cols), 0.7 + 0.45 * (len(body) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=body, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    for j in range(len(cols)):                          # header styling
        c = tbl[0, j]; c.set_facecolor("#40466e"); c.set_text_props(color="white", weight="bold")
    for i in range(1, len(body) + 1):                   # zebra rows
        for j in range(len(cols)):
            tbl[i, j].set_facecolor("#f2f2f7" if i % 2 else "#ffffff")
    ax.set_title(f"Congestion-aware PIBT - metrics by lambda  (mean over {n_cells} cells)",
                 fontsize=12, pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_lambda_table(rows, weights, indent="", base_mean=None):
    """Print a per-lambda aggregate of `rows` (mean over whatever cells/configs they span).

    Columns mirror the CSV metrics, plus the across-row std of deliveries (±sd) and,
    when `base_mean` is given, the delivery gain over baseline (d-base)."""
    has_base = base_mean is not None
    cols = [("deliv", 7), ("±sd", 5)]
    if has_base:
        cols.append(("d-base", 7))
    cols += [("energy", 8), ("e/dlv", 7), ("unifrm", 6), ("occCV", 5),
             ("cong@r", 6), ("p99", 6), ("peak", 6), ("preds", 6), ("wall", 6), ("coll", 4)]
    hdr = f"{indent}{'lam':>5} | " + " ".join(f"{n:>{w}}" for n, w in cols)
    print(hdr)
    print(f"{indent}{'-' * (len(hdr) - len(indent))}")
    for lam in weights:
        sub = [r for r in rows if r["weight"] == lam]
        if not sub:
            continue
        dv = [r["deliveries"] for r in sub]
        mean = lambda k: float(np.mean([r[k] for r in sub]))
        sd = float(np.std(dv)) if len(dv) > 1 else 0.0
        vals = {
            "deliv":  f"{float(np.mean(dv)):>7.1f}",
            "±sd":    f"{sd:>5.1f}",
            "d-base": f"{(float(np.mean(dv)) - base_mean):>+7.1f}" if has_base else "",
            "energy": f"{mean('energy'):>8.0f}",
            "e/dlv":  f"{mean('energy_per_delivery'):>7.2f}",
            "unifrm": f"{mean('density_uniformity'):>6.3f}",
            "occCV":  f"{mean('occ_cv'):>5.2f}",
            "cong@r": f"{mean('mean_robot_cong'):>6.1f}",
            "p99":    f"{mean('p99_cong'):>6.0f}",
            "peak":   f"{mean('peak_cong'):>6.0f}",
            "preds":  f"{mean('preds'):>6.0f}",
            "wall":   f"{mean('wall_s'):>6.1f}",
            "coll":   f"{int(sum(r['collisions'] for r in sub)):>4}",
        }
        print(f"{indent}{lam:>5.2f} | " + " ".join(vals[n] for n, _ in cols))


class GridBoard:
    """In-place board of the whole (AMR count) x (dispersion frac) grid: every cell shows
    done / running / pending, redrawn via ANSI cursor moves so you can SEE what's running."""
    MARKS = {0: "·", 1: "▶", 2: "✓"}   # pending ·  running ▶  done ✓

    def __init__(self, counts, fracs, total_units, status, counter, running=None):
        self.counts, self.fracs = counts, fracs
        self.status, self.counter = status, counter
        self.running = running
        self.total_cells = len(counts) * len(fracs)
        self.total_units = max(1, total_units)
        self.started = time.perf_counter()
        self._lines = 0
        self._lock = threading.Lock()
        os.system("")                                # enable ANSI escapes on Windows consoles
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ✓/▶ never crash cp949
        except Exception:
            pass

    def _body(self):
        nf = len(self.fracs)
        lines = ["       " + "".join(f"{f:>5.1f}" for f in self.fracs)]
        for ci, c in enumerate(self.counts):
            row = f" n{c:<5}"
            for fi in range(nf):
                row += f"{self.MARKS.get(self.status[ci * nf + fi], '?'):^5}"
            lines.append(row)
        done = sum(1 for i in range(self.total_cells) if self.status[i] == 2)
        run = sum(1 for i in range(self.total_cells) if self.status[i] == 1)
        units = min(self.counter.value, self.total_units)
        elapsed = time.perf_counter() - self.started
        frac = units / self.total_units
        eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
        lines.append("")
        lines.append(f" cells {done}/{self.total_cells} done, {run} running  |  "
                     f"steps {units}/{self.total_units} ({frac * 100:4.0f}%)  |  "
                     f"elapsed {elapsed / 60:5.1f}m  eta {eta / 60:5.1f}m")
        try:
            active = sorted(self.running.values()) if self.running is not None else []
        except RuntimeError:                          # dict mutated mid-iteration; skip this frame
            active = []
        if active:
            lines.append(f" running now ({len(active)}):")
            lines.extend(f"   {lab}" for lab in active)
        return lines

    def draw(self):
        with self._lock:
            lines = self._body()
            out = f"\x1b[{self._lines}A" if self._lines else ""
            out += "".join("\x1b[2K" + ln + "\n" for ln in lines)
            extra = self._lines - len(lines)
            if extra > 0:                             # clear leftover lines from a taller frame
                out += "\x1b[2K\n" * extra + f"\x1b[{extra}A"
            sys.stdout.write(out)
            sys.stdout.flush()
            self._lines = len(lines)


# ---------------------------------------------------------------------------
# CLI (generate.py-style argument names)
# ---------------------------------------------------------------------------
def parse_args():
    base = mapf.load_config()
    ap = argparse.ArgumentParser(
        description="Grid A/B eval of congestion-aware PIBT over AMR count x dispersion frac x "
                    "congestion weight (lambda) x depth gamma (+ optional min_depth / depth_mode "
                    "ablations), with a per-(cell, gamma) vanilla-vs-best-lambda MP4.")
    ap.add_argument("--num_of_process", type=int, default=1,
                    help="Parallel cell jobs. NOTE: lambda>0 runs use the GPU; many processes "
                         "share one GPU (each loads the 220MB model). Raise only if VRAM allows.")
    ap.add_argument("--base-seed", type=int, default=42, help="cell i uses base_seed + i (shared across lambdas).")
    ap.add_argument("--seconds", type=int, default=900,
                    help="episode length in steps (metrics use the full length; dataset used 1800).")
    ap.add_argument("--video-seconds", type=int, default=900,
                    help="render only the first N steps in the MP4s (0 or >= --seconds = full). "
                         "Default 900 = full when --seconds is also 900.")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="congestion-weight (lambda) axis; 0 = vanilla PIBT (computed once per cell).")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.73],
                    help="depth-weight peak r axis (handoff usable band [0.607, 0.730]); "
                         "default [0.73] = single value (old behaviour). e.g. 0.61 0.66 0.70 0.73")
    ap.add_argument("--horizon", type=int, default=10, help="H: max descent depth read (<=10).")
    ap.add_argument("--min-depths", type=int, nargs="+", default=[2],
                    help="k_start ablation axis (handoff 7-1: '1 2' to compare include/exclude k=1).")
    ap.add_argument("--depth-modes", nargs="+", default=["peaked"], choices=["peaked", "frontload"],
                    help="depth-weight shape ablation axis (handoff 7-2: 'peaked frontload').")
    ap.add_argument("--min-agents", type=int, default=300)
    ap.add_argument("--max-agents", type=int, default=500)
    ap.add_argument("--agent-step", type=int, default=100)
    ap.add_argument("--min-frac", type=float, default=0.0)
    ap.add_argument("--max-frac", type=float, default=1.0)
    ap.add_argument("--frac-step", type=float, default=0.5)
    ap.add_argument("--center-value", type=float, default=base.congestion_center_value)
    ap.add_argument("--step-value", type=float, default=base.congestion_step_value)
    ap.add_argument("--predict-every", type=int, default=None,
                    help="MPC re-predict period; default = 11 - horizon (keeps full-depth lookahead, "
                         "handoff section 4). Raising it is cheaper but the penalty fades late in "
                         "each window.")
    ap.add_argument("--no-video", action="store_true", help="metrics only, skip MP4s")
    # --- MACPF animate_paths knobs (videos) ---
    ap.add_argument("--anim-subframes", type=int, default=1,
                    help="interpolated frames per sim-step at 30fps. 30=realtime/smooth but ~30x "
                         "slower to render; 1=cell-to-cell jumps, fastest. (MACPF default 30)")
    ap.add_argument("--robot-size", type=float, default=8.0,
                    help="robot marker area (matplotlib s=). MACPF default 95 is for tens of agents; "
                         "shrink for hundreds.")
    ap.add_argument("--route-linewidth", type=float, default=0.6,
                    help="dashed robot->current-target line width.")
    ap.add_argument("--planned-route-linewidth", type=float, default=0.3,
                    help="faint full planned-route underlay width (only if --planned-routes).")
    ap.add_argument("--target-size", type=float, default=40.0, help="pickup/delivery target marker area.")
    ap.add_argument("--video-dpi", type=int, default=200,
                    help="MP4 render resolution: the figure is 10x8 in, so dpi 200 -> 2000x1600 px "
                         "(the old default was effectively 100 = 1000x800). Higher = crisper dots but "
                         "slower to render and larger files.")
    ap.add_argument("--video-cq", type=int, default=15,
                    help="NVENC constant-quality level (0-51, lower = higher quality / larger file). "
                         "15 is near-visually-lossless for this flat dots-on-map content; push to ~10 for "
                         "max quality. The CPU x264 fallback uses crf = cq + 1. Encoding is on the GPU "
                         "(h264_nvenc); both the per-panel render and the side-by-side hstack use it.")
    ap.add_argument("--planned-routes", action="store_true",
                    help="also draw each robot's full planned route as a faint underlay "
                         "(off by default: 300-750 such polylines clutter the frame).")
    ap.add_argument("--heatmap", action="store_true",
                    help="also render a SEPARATE congestion-heatmap MP4 per gamma: the ground-truth "
                         "congestion fields of the vanilla vs congestion-aware runs side by side "
                         "(same vanilla|applied layout as the movement video, shared colour scale). "
                         "Reuses each run's paths -- no extra solves, just extra rendering.")
    ap.add_argument("--heatmap-vmax-pct", type=float, default=99.0,
                    help="percentile of positive congestion used as the shared heatmap colour-scale "
                         "max across BOTH panels (lower = brighter / more saturated).")
    ap.add_argument("--verbose", action="store_true",
                    help="scrolling per-cell/per-video log instead of the tqdm progress bar")
    args = ap.parse_args()
    if args.min_agents < 1 or args.min_agents > args.max_agents:
        ap.error("require 1 <= --min-agents <= --max-agents")
    if not (0.0 <= args.min_frac <= args.max_frac <= 1.0):
        ap.error("require 0.0 <= --min-frac <= --max-frac <= 1.0")
    return args


def main():
    args = parse_args()
    args.horizon = max(1, min(10, int(args.horizon)))
    if args.predict_every is None:
        args.predict_every = max(1, 11 - args.horizon)   # full-depth lookahead (handoff section 4)
    counts, fracs = agent_count_sweep(args), frac_sweep(args)
    grid = grid_cells(args)
    weights = sorted(args.weights)
    weights_pos = [w for w in weights if w > 0]
    gammas, min_depths, depth_modes = args.gammas, args.min_depths, args.depth_modes

    # group comparison runs under reports/CALM_comparison/<yymmdd_hhmm>/ (attributable to this code)
    out_dir = REPO_ROOT / "reports" / "CALM_comparison" / datetime.now().strftime("%y%m%d_%H%M")
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    env = fmg.build_factory_map()

    # per cell: 1 baseline (lambda=0) + (depth_mode x min_depth x gamma x lambda>0) congestion runs
    cong_per_cell = len(depth_modes) * len(min_depths) * len(gammas) * len(weights_pos)
    runs_per_cell = 1 + cong_per_cell
    n_vid = 0 if (args.no_video or not weights_pos) else len(grid) * len(gammas)
    print(f"grid: counts {counts} x fracs {fracs} | lambdas {weights} x gammas {gammas} "
          f"| min_depths {min_depths} x depth_modes {depth_modes}")
    print(f"  = {len(grid)} cells x {runs_per_cell} runs = {len(grid) * runs_per_cell} planner runs "
          f"| horizon {args.horizon} | predict_every {args.predict_every}"
          f"{'' if args.no_video else f'  (+{n_vid} MP4s: vanilla vs best-lambda per gamma)'}", flush=True)
    print(f"output -> {out_dir}\n", flush=True)

    all_rows, episode_videos = [], {}
    workers = max(1, int(args.num_of_process))
    t0 = time.perf_counter()

    # metrics.csv is written INCREMENTALLY (one cell's rows appended as it finishes), so a
    # long 550-run sweep that gets interrupted still keeps every completed cell's metrics
    # (and the per-cell MP4s are likewise already on disk).
    csv_path = out_dir / "metrics.csv"
    csv_fields = ["episode", "num_agents", "frac", "gamma", "horizon", "min_depth", "depth_mode",
                  "weight", "seed", "deliveries", "energy", "energy_per_delivery",
                  "density_uniformity", "occ_cv", "mean_robot_cong", "p99_cong", "peak_cong",
                  "collisions", "preds", "wall_s"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=csv_fields).writeheader()

    total = len(grid)

    # WORK UNIT = one planner run (not a whole cell): there are runs_per_cell runs per cell,
    # times every cell, which vastly outnumbers the workers, so no worker ever sits idle until
    # the very tail. Jobs are interleaved across cells (round-robin) so every cell has work in
    # the first wave. The board keeps its current form (one mark per (count, frac) cell); a
    # cell flips to done once all of its own runs complete.
    per_cell_jobs = []
    for ci, (count, frac) in enumerate(grid):
        seed = args.base_seed + ci               # per-cell seed, shared by that cell's runs (A/B)
        jl = [{"cell_idx": ci, "count": count, "frac": frac, "seed": seed, "baseline": True,
               "weight": 0.0, "gamma": gammas[0], "horizon": args.horizon,
               "min_depth": min_depths[0], "depth_mode": depth_modes[0]}]
        for mode in depth_modes:
            for md in min_depths:
                for g in gammas:
                    for w in weights_pos:
                        jl.append({"cell_idx": ci, "count": count, "frac": frac, "seed": seed,
                                   "baseline": False, "weight": w, "gamma": g,
                                   "horizon": args.horizon, "min_depth": md, "depth_mode": mode})
        per_cell_jobs.append(jl)
    jobs = [j for wave in zip_longest(*per_cell_jobs) for j in wave if j is not None]
    total_runs = len(jobs)

    # shared "currently running" map (pid -> label) so the board shows what every worker is on now.
    # Manager dict for the pool; a plain dict (exposed as the module global) for workers==1.
    if workers == 1:
        running = {}
        globals()["_RUNNING"] = running
        mgr = None
    else:
        mgr = Manager()
        running = mgr.dict()

    counter = Value("i", 0)
    status = Array("b", total)               # per cell; 0=pending 1=running 2=done
    for i in range(total):
        status[i] = 1                        # every cell is in flight from the start (no idle worker)

    # Live grid board (unchanged form): a ✓/▶/· per (count, frac) cell, redrawn ~2x/s by a ticker
    # thread. --verbose falls back to a scrolling per-cell line.
    board = None if args.verbose else GridBoard(counts, fracs, total_runs, status, counter, running)
    stop_tick = threading.Event()
    if board is not None:
        board.draw()

        def _tick():
            while not stop_tick.wait(0.5):
                board.draw()
        threading.Thread(target=_tick, daemon=True).start()

    cell_done = [0] * total
    rows_by_cell = {ci: [] for ci in range(total)}

    def absorb(res):
        ci, row = res["cell_idx"], res["row"]
        all_rows.append(row)
        rows_by_cell[ci].append(row)
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=csv_fields).writerow(row)
        cell_done[ci] += 1
        counter.value += 1
        if cell_done[ci] >= runs_per_cell:
            status[ci] = 2                   # all of this cell's runs done -> check mark
            if board is None:
                nd = sum(1 for d in cell_done if d >= runs_per_cell)
                el = time.perf_counter() - t0
                print(f"[{nd:>2}/{total}] cell {ci} ({row['num_agents']} AMRs frac {row['frac']:.1f}) "
                      f"done | {counter.value}/{total_runs} runs | elapsed {el / 60:.1f}m", flush=True)

    interrupted = False
    if workers == 1:
        try:
            for job in jobs:
                absorb(run_job(job, args))
        except KeyboardInterrupt:
            interrupted = True
            print("\n[interrupted] stopped; partial metrics.csv is kept.", flush=True)
    else:
        # Explicit pool (not `with`): on Ctrl+C, terminate workers immediately.
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(running,))
        try:
            futs = [pool.submit(run_job, job, args) for job in jobs]
            for fut in as_completed(futs):
                absorb(fut.result())
            pool.shutdown()
        except KeyboardInterrupt:
            interrupted = True
            print("\n[interrupted] terminating workers (partial results kept)...", flush=True)
            for proc in list(getattr(pool, "_processes", {}).values()):
                proc.terminate()
            pool.shutdown(wait=False, cancel_futures=True)

    stop_tick.set()
    if board is not None:
        board.draw()
        print()                                  # drop below the board for the summary
    cells_done = sum(1 for d in cell_done if d >= runs_per_cell)

    # ---- videos (separate phase): per (cell, gamma) best-lambda for the PRIMARY config.
    # Re-runs the chosen configs deterministically (same seed) instead of shipping path arrays;
    # rendered in parallel across cells.
    if not args.no_video and weights_pos and not interrupted:
        prim_mode, prim_md = depth_modes[0], min_depths[0]
        vjobs = []
        for ci, (count, frac) in enumerate(grid):
            best_by_gamma = {}
            for g in gammas:
                cand = [r for r in rows_by_cell[ci] if r["depth_mode"] == prim_mode
                        and r["min_depth"] == prim_md and r["gamma"] == g]
                if cand:
                    best_by_gamma[g] = max(cand, key=lambda r: r["deliveries"])["weight"]
            if best_by_gamma:
                vjobs.append({"cell_idx": ci, "count": count, "frac": frac,
                              "seed": args.base_seed + ci, "best_by_gamma": best_by_gamma,
                              "prim_mode": prim_mode, "prim_md": prim_md, "horizon": args.horizon})
        print(f"\nrendering videos for {len(vjobs)} cells (vanilla | best-lambda per gamma; "
              f"configs re-run deterministically)...", flush=True)
        vdone = [0]

        def absorb_v(res):
            episode_videos[res["cell_idx"]] = res["videos"]
            vdone[0] += 1
            print(f"  [{vdone[0]:>2}/{len(vjobs)}] cell {res['cell_idx']}: {len(res['videos'])} videos",
                  flush=True)
        try:
            if workers == 1:
                for vj in vjobs:
                    absorb_v(video_job(vj, args, str(out_dir)))
            else:
                vpool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker)
                vfuts = [vpool.submit(video_job, vj, args, str(out_dir)) for vj in vjobs]
                for fut in as_completed(vfuts):
                    absorb_v(fut.result())
                vpool.shutdown()
        except KeyboardInterrupt:
            print("\n[interrupted] video phase stopped (metrics already saved).", flush=True)
        except Exception as exc:  # noqa: BLE001  (videos are optional; never lose the metrics)
            print(f"[videos skipped] {exc!r}", flush=True)

    # ---- metadata.json ----
    (out_dir / "metadata.json").write_text(json.dumps({
        "tool": "grid_eval", "solver": "pibt_lifelong",
        "counts": counts, "fracs": fracs, "weights": weights,
        "gammas": gammas, "min_depths": min_depths, "depth_modes": depth_modes,
        "horizon": args.horizon, "predict_every": args.predict_every,
        "seconds": args.seconds, "base_seed": args.base_seed,
        "congestion_center_value": args.center_value, "congestion_step_value": args.step_value,
        "cells": total, "cells_completed": cells_done, "interrupted": interrupted,
        "runs": total_runs,
        "elapsed_min": (time.perf_counter() - t0) / 60.0,
        "episode_videos": episode_videos,
    }, indent=2), encoding="utf-8")

    # ---- detailed aggregates over completed cells ----
    cong_rows = [r for r in all_rows if r["depth_mode"] != "baseline"]
    base_rows = [r for r in all_rows if r["depth_mode"] == "baseline"]
    base_mean = float(np.mean([r["deliveries"] for r in base_rows])) if base_rows else None

    # (1) headline: mean over every completed cell, by congestion weight
    if all_rows:
        print(f"\n=== mean over {cells_done}/{total} completed cells, by congestion weight ===")
        _print_lambda_table(all_rows, weights, base_mean=base_mean)

    # (2) per (count, frac) cell: by congestion weight (mean over mode/md/gamma within the cell)
    if all_rows:
        print(f"\n=== per (count, frac) cell, by congestion weight ===")
        for ci in sorted(rows_by_cell):
            crows = rows_by_cell[ci]
            if not crows:
                continue
            count, frac = grid[ci]
            cbase = [r["deliveries"] for r in crows if r["depth_mode"] == "baseline"]
            cbm = float(np.mean(cbase)) if cbase else None
            tag = f"  [count={count}  frac={frac:g}]"
            if cbm is not None:
                tag += f"  baseline deliv={cbm:.1f}"
            print(f"\n{tag}")
            _print_lambda_table(crows, weights, indent="    ", base_mean=cbm)

    # (3) best lambda per (depth_mode, min_depth, gamma): compact winner summary
    if cong_rows:
        bm = base_mean if base_mean is not None else float("nan")
        print(f"\n=== best lambda per gamma (mean deliveries over cells; baseline {bm:.1f}) ===")
        hdr = f"{'mode':>9} {'md':>3} {'gamma':>6} | {'best lam':>8} {'deliv':>6} {'d-base':>7}"
        print(hdr); print("-" * len(hdr))
        for combo in sorted({(r["depth_mode"], r["min_depth"], r["gamma"]) for r in cong_rows}):
            mode, md, g = combo
            by_w = {}
            for r in cong_rows:
                if (r["depth_mode"], r["min_depth"], r["gamma"]) == combo:
                    by_w.setdefault(r["weight"], []).append(r["deliveries"])
            best_w = max(by_w, key=lambda w: float(np.mean(by_w[w])))
            best_d = float(np.mean(by_w[best_w]))
            print(f"{mode:>9} {md:>3} {g:>6g} | {best_w:>8g} {best_d:>6.1f} {best_d - bm:>+7.1f}")

    # (4) full lambda sweep per (depth_mode, min_depth, gamma): all lambdas, full metrics
    if cong_rows:
        print(f"\n=== full lambda sweep per (mode, md, gamma) (mean over cells) ===")
        for combo in sorted({(r["depth_mode"], r["min_depth"], r["gamma"]) for r in cong_rows}):
            mode, md, g = combo
            grp = [r for r in cong_rows
                   if (r["depth_mode"], r["min_depth"], r["gamma"]) == combo]
            print(f"\n  [mode={mode}  md={md}  gamma={g:g}]")
            _print_lambda_table(grp, weights_pos, indent="    ", base_mean=base_mean)

    # save the by-lambda table as an image too (out_dir; rides along when moved below)
    if all_rows:
        try:
            save_metrics_table_png(all_rows, weights, cells_done, out_dir / "metrics_table.png")
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics table png skipped] {exc!r}")

    print(f"\nmetrics -> {out_dir / 'metrics.csv'}")
    print(f"table   -> {out_dir / 'metrics_table.png'}")
    if not args.no_video:
        print(f"videos  -> {out_dir / 'videos'}")
    print(f"report  -> {out_dir}")
    print(f"{'INTERRUPTED — ' if interrupted else ''}elapsed "
          f"{(time.perf_counter() - t0) / 60.0:.1f} min "
          f"({cells_done}/{total} cells)")
    if interrupted:
        # hard-exit so a half-torn-down process pool can't hang the interpreter at shutdown
        sys.stdout.flush()
        os._exit(130)


if __name__ == "__main__":
    main()
