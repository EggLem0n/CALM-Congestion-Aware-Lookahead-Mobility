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
from multiprocessing import Array, Value
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for p in (str(HERE), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from calm import PiBT as mapf                                    # noqa: E402
from calm.PiBT import factory_map_generator as fmg       # noqa: E402

# Lazily-built, per-process singletons (avoid importing torch / loading the 220MB
# model in workers that only run vanilla, and avoid re-loading per episode).
_PREDICTOR = None
# Shared sub-step counter (solves + renders + hstacks) so the progress display advances
# WITHIN a cell, not just once per finished cell -- workers bump it across processes.
_PROGRESS = None
# Shared per-cell status array (0=pending, 1=running, 2=done), indexed by episode_id, so the
# main process can draw the whole grid with a ✓/▶/· per cell.
_STATUS = None


def _bump():
    if _PROGRESS is not None:
        with _PROGRESS.get_lock():
            _PROGRESS.value += 1


def _set_status(episode_id, value):
    if _STATUS is not None:
        _STATUS[episode_id] = value


def get_predictor(device):
    global _PREDICTOR
    if _PREDICTOR is None:
        from predict import CongestionPredictor      # local import: keeps torch out of --no-* paths
        _PREDICTOR = CongestionPredictor(device=device)
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


def solve(weight, env, config, starts, predict_every, device):
    walkable = np.asarray(env["walkable_map"]).astype(bool)
    pickup = [p for p in mapf.normalize_points(env.get("pickup_points")) if mapf.is_walkable(*p, walkable)]
    delivery = [p for p in mapf.normalize_points(env.get("delivery_points")) if mapf.is_walkable(*p, walkable)]
    predictor = get_predictor(device) if weight > 0 else None
    t0 = time.perf_counter()
    paths, summary = mapf.plan_pibt_repeated_tasks(
        starts, pickup, delivery, walkable, config,
        pickup_point_groups=mapf.normalize_point_groups(env.get("pickup_point_groups")),
        delivery_point_groups=mapf.normalize_point_groups(env.get("delivery_point_groups")),
        congestion_predictor=predictor, congestion_weight=weight, predict_every=predict_every,
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


def _hstack(left_mp4, right_mp4, out_mp4):
    """Combine two equal-size clips side by side (ffmpeg hstack)."""
    import subprocess
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ff, "-y", "-loglevel", "error", "-i", str(left_mp4), "-i", str(right_mp4),
         "-filter_complex", "[0:v][1:v]hstack=inputs=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)],
        check=True,
    )


# ---------------------------------------------------------------------------
# one episode = one (count, frac) cell, swept over all lambdas
# ---------------------------------------------------------------------------
def generate_episode(env, episode_id, args, out_dir_str):
    _set_status(episode_id, 1)                  # mark this cell running on the grid board
    cell = grid_cells(args)[episode_id]
    count, frac = cell
    seed = args.base_seed + episode_id          # per-cell seed, SHARED across lambdas (controlled A/B)
    base = mapf.load_config()
    walkable = np.asarray(env["walkable_map"]).astype(bool)

    # one fixed start layout per cell (same seed) -> shared by every lambda and the videos
    cell_cfg = base.replace(num_agents=count, distributed_fraction=frac, seed=seed,
                            max_time=args.seconds, congestion_center_value=args.center_value,
                            congestion_step_value=args.step_value, show_planning_progress=False)
    starts, _ = mapf.select_start_goal_pairs(env, walkable, cell_cfg)

    run_weights = sorted(set(args.weights) | ({0.0} if not args.no_video else set()))
    if args.verbose:
        print(f"  > ep{episode_id:03d} start  {count} AMRs frac {frac:.1f}: {len(run_weights)} runs",
              flush=True)

    rows = []
    solved = {}      # w -> (paths, summary)
    for w in run_weights:
        paths, summary, m = solve(w, env, cell_cfg, starts, args.predict_every, args.device)
        solved[w] = (paths, summary)
        _bump()                                   # one solve done
        if w in args.weights:
            rows.append({"episode": episode_id, "num_agents": count, "frac": frac,
                         "weight": w, "seed": seed, **m})

    # one 2-panel MP4 per lambda>0: vanilla (left) | that lambda (right), each panel rendered
    # by MACPF's animate_paths. Vanilla is rendered once per cell and reused for all lambdas.
    videos = []
    if not args.no_video and 0.0 in solved:
        vdir = Path(out_dir_str) / "videos"
        tmp = vdir / f".tmp_ep{episode_id:03d}"
        anim_cfg = _anim_config(base, args)
        # videos only show the first --video-seconds steps (full-length 1800-step videos would
        # take minutes each); metrics above already used the full episode.
        vs = args.video_seconds if 0 < args.video_seconds < args.seconds else (args.seconds + 1)
        clip = lambda paths: [p[:vs + 1] for p in paths]
        vpaths, vsumm = solved[0.0]
        vanilla_mp4 = _animate_scenario(env, starts, clip(vpaths), vsumm, anim_cfg, tmp / "vanilla.mp4", tmp / "v")
        _bump()                                   # vanilla panel rendered
        for w in sorted(solved):
            if w <= 0.0:
                continue
            if args.verbose:
                print(f"  > ep{episode_id:03d} render lambda={w:g}", flush=True)
            cpaths, csumm = solved[w]
            lam_mp4 = _animate_scenario(env, starts, clip(cpaths), csumm, anim_cfg, tmp / f"w{w:g}.mp4", tmp / f"c{w:g}")
            _bump()                               # lambda panel rendered
            name = f"ep{episode_id:03d}_n{count}_f{frac:.1f}_w{w:g}.mp4"
            _hstack(vanilla_mp4, lam_mp4, vdir / name)
            _bump()                               # 2-panel hstacked
            videos.append(name)
        shutil.rmtree(tmp, ignore_errors=True)
    _set_status(episode_id, 2)                  # mark this cell done on the grid board
    return {"episode": episode_id, "cell": cell, "rows": rows, "videos": videos}


def _init_worker(counter, status):
    global _PROGRESS, _STATUS
    _PROGRESS = counter
    _STATUS = status
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # main process handles Ctrl+C
    warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")


def save_metrics_table_png(rows, weights, n_cells, out_path):
    """Render the by-lambda mean metrics as a table image (matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["lambda", "deliveries", "energy", "energy/deliv", "uniformity",
            "occ_cv", "cong@robot", "p99 cong", "collisions"]
    body, best = [], {}
    # remember the best (max deliveries, max uniformity, min congestion) for highlighting
    for w in weights:
        sub = [r for r in rows if r["weight"] == w]
        if not sub:
            continue
        m = lambda k: float(np.mean([r[k] for r in sub]))
        body.append([f"{w:g}", f"{m('deliveries'):.1f}", f"{m('energy'):.0f}",
                     f"{m('energy_per_delivery'):.1f}", f"{m('density_uniformity'):.3f}",
                     f"{m('occ_cv'):.2f}", f"{m('mean_robot_cong'):.1f}",
                     f"{m('p99_cong'):.0f}", f"{int(sum(r['collisions'] for r in sub))}"])

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


class GridBoard:
    """In-place board of the whole (AMR count) x (dispersion frac) grid: every cell shows
    done / running / pending, redrawn via ANSI cursor moves so you can SEE what's running."""
    MARKS = {0: "·", 1: "▶", 2: "✓"}   # pending ·  running ▶  done ✓

    def __init__(self, counts, fracs, total_units, status, counter):
        self.counts, self.fracs = counts, fracs
        self.status, self.counter = status, counter
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
        return lines

    def draw(self):
        with self._lock:
            lines = self._body()
            out = f"\x1b[{self._lines}A" if self._lines else ""
            out += "".join("\x1b[2K" + ln + "\n" for ln in lines)
            sys.stdout.write(out)
            sys.stdout.flush()
            self._lines = len(lines)


# ---------------------------------------------------------------------------
# CLI (generate.py-style argument names)
# ---------------------------------------------------------------------------
def parse_args():
    base = mapf.load_config()
    ap = argparse.ArgumentParser(
        description="3-D grid (AMR count x dispersion frac x congestion weight) A/B eval "
                    "of congestion-aware PIBT, with a per-cell vanilla-vs-congestion MP4.")
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
                    help="congestion-weight axis (3rd grid dim); 0 = vanilla PIBT.")
    ap.add_argument("--min-agents", type=int, default=300)
    ap.add_argument("--max-agents", type=int, default=500)
    ap.add_argument("--agent-step", type=int, default=100)
    ap.add_argument("--min-frac", type=float, default=0.0)
    ap.add_argument("--max-frac", type=float, default=1.0)
    ap.add_argument("--frac-step", type=float, default=0.5)
    ap.add_argument("--center-value", type=float, default=base.congestion_center_value)
    ap.add_argument("--step-value", type=float, default=base.congestion_step_value)
    ap.add_argument("--predict-every", type=int, default=10)
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
    ap.add_argument("--planned-routes", action="store_true",
                    help="also draw each robot's full planned route as a faint underlay "
                         "(off by default: 300-750 such polylines clutter the frame).")
    ap.add_argument("--verbose", action="store_true",
                    help="scrolling per-cell/per-video log instead of the tqdm progress bar")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()
    if args.min_agents < 1 or args.min_agents > args.max_agents:
        ap.error("require 1 <= --min-agents <= --max-agents")
    if not (0.0 <= args.min_frac <= args.max_frac <= 1.0):
        ap.error("require 0.0 <= --min-frac <= --max-frac <= 1.0")
    return args


def main():
    args = parse_args()
    counts, fracs = agent_count_sweep(args), frac_sweep(args)
    grid = grid_cells(args)
    weights = sorted(args.weights)

    # group comparison runs under reports/CALM_comparison/<yymmdd_hhmm>/ (attributable to this code)
    out_dir = REPO_ROOT / "reports" / "CALM_comparison" / datetime.now().strftime("%y%m%d_%H%M")
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    env = fmg.build_factory_map()

    n_pos = len([w for w in weights if w > 0])
    n_vid = 0 if args.no_video else len(grid) * n_pos
    print(f"3-D grid: counts {counts} x fracs {fracs} x weights {weights}  "
          f"= {len(grid)} cells x {len(weights)} = {len(grid) * len(weights)} runs"
          f"{'' if args.no_video else f'  (+{n_vid} MP4s: vanilla vs each lambda>0)'}", flush=True)
    print(f"output -> {out_dir}\n", flush=True)

    all_rows, episode_videos = [], {}
    workers = max(1, int(args.num_of_process))
    t0 = time.perf_counter()

    # metrics.csv is written INCREMENTALLY (one cell's rows appended as it finishes), so a
    # long 550-run sweep that gets interrupted still keeps every completed cell's metrics
    # (and the per-cell MP4s are likewise already on disk).
    csv_path = out_dir / "metrics.csv"
    csv_fields = ["episode", "num_agents", "frac", "weight", "seed", "deliveries", "energy",
                  "energy_per_delivery", "density_uniformity", "occ_cv", "mean_robot_cong",
                  "p99_cong", "peak_cong", "collisions", "preds", "wall_s"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=csv_fields).writeheader()

    total = len(grid)

    # Progress = a shared sub-step counter (each solve / render / hstack) + a shared per-cell
    # status array (0 pending, 1 running, 2 done) that workers update, so the main process can
    # draw the WHOLE grid as a live board instead of a single opaque bar.
    run_weights_main = sorted(set(weights) | ({0.0} if not args.no_video else set()))
    n_pos = len([w for w in weights if w > 0])
    per_cell_units = len(run_weights_main) + (0 if args.no_video else (1 + n_pos) + n_pos)
    total_units = total * per_cell_units
    counter = Value("i", 0)
    status = Array("b", total)               # per episode_id; 0=pending 1=running 2=done
    global _PROGRESS, _STATUS
    _PROGRESS, _STATUS = counter, status     # used in-process when workers == 1

    # Live grid board (default): a ✓/▶/· per (count, frac) cell, redrawn ~2x/s by a ticker
    # thread. --verbose falls back to a scrolling per-cell line.
    board = None if args.verbose else GridBoard(counts, fracs, total_units, status, counter)
    stop_tick = threading.Event()
    if board is not None:
        board.draw()

        def _tick():
            while not stop_tick.wait(0.5):
                board.draw()
        threading.Thread(target=_tick, daemon=True).start()

    def absorb(res):
        all_rows.extend(res["rows"])
        episode_videos[res["episode"]] = res["videos"]
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=csv_fields).writerows(res["rows"])
        if board is not None:
            board.draw()
        else:
            c, f = res["cell"]
            done = len(episode_videos)
            elapsed = time.perf_counter() - t0
            eta = (elapsed / done * (total - done)) if done else 0.0
            vid = f" videos={len(res['videos'])}" if res["videos"] else ""
            print(f"[{done:>3}/{total} {100 * done / total:4.0f}%] ep{res['episode']:03d} "
                  f"{c} AMRs frac {f:.1f} done ({len(res['rows'])} lambdas){vid}"
                  f"  | elapsed {elapsed / 60:.1f}m  eta {eta / 60:.1f}m", flush=True)

    interrupted = False
    if workers == 1:
        try:
            for episode_id in range(total):
                absorb(generate_episode(env, episode_id, args, str(out_dir)))
        except KeyboardInterrupt:
            interrupted = True
            print("\n[interrupted] stopped; partial metrics.csv + finished videos are kept.", flush=True)
    else:
        # Explicit pool (not `with`): on Ctrl+C, terminate workers immediately instead of
        # blocking until in-flight cells finish (which is why Ctrl+C felt dead before).
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(counter, status))
        try:
            futs = {pool.submit(generate_episode, env, eid, args, str(out_dir)): eid
                    for eid in range(total)}
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

    # ---- metadata.json ----
    (out_dir / "metadata.json").write_text(json.dumps({
        "tool": "grid_eval", "solver": "pibt_lifelong",
        "counts": counts, "fracs": fracs, "weights": weights,
        "seconds": args.seconds, "base_seed": args.base_seed,
        "predict_every": args.predict_every,
        "congestion_center_value": args.center_value, "congestion_step_value": args.step_value,
        "cells": total, "cells_completed": len(episode_videos), "interrupted": interrupted,
        "runs": total * len(weights),
        "elapsed_min": (time.perf_counter() - t0) / 60.0,
        "episode_videos": episode_videos,
    }, indent=2), encoding="utf-8")

    # ---- aggregate by weight (mean over completed cells) ----
    if all_rows:
        print(f"\n=== mean over {len(episode_videos)}/{total} completed cells, by congestion weight ===")
        hdr = (f"{'lam':>5} | {'deliv':>6} {'energy':>8} {'e/deliv':>7} {'unifrm':>6} "
               f"{'occCV':>5} {'cong@r':>6} {'p99':>6} {'coll':>4}")
        print(hdr); print("-" * len(hdr))
        for w in weights:
            sub = [r for r in all_rows if r["weight"] == w]
            if not sub:
                continue
            mean = lambda k: float(np.mean([r[k] for r in sub]))
            print(f"{w:>5.2f} | {mean('deliveries'):>6.1f} {mean('energy'):>8.0f} "
                  f"{mean('energy_per_delivery'):>7.2f} {mean('density_uniformity'):>6.3f} "
                  f"{mean('occ_cv'):>5.2f} {mean('mean_robot_cong'):>6.1f} {mean('p99_cong'):>6.0f} "
                  f"{int(sum(r['collisions'] for r in sub)):>4}")

    # save the by-lambda table as an image too (out_dir; rides along when moved below)
    if all_rows:
        try:
            save_metrics_table_png(all_rows, weights, len(episode_videos), out_dir / "metrics_table.png")
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics table png skipped] {exc!r}")

    print(f"\nmetrics -> {out_dir / 'metrics.csv'}")
    print(f"table   -> {out_dir / 'metrics_table.png'}")
    if not args.no_video:
        print(f"videos  -> {out_dir / 'videos'}")
    print(f"report  -> {out_dir}")
    print(f"{'INTERRUPTED — ' if interrupted else ''}elapsed "
          f"{(time.perf_counter() - t0) / 60.0:.1f} min "
          f"({len(episode_videos)}/{total} cells)")
    if interrupted:
        # hard-exit so a half-torn-down process pool can't hang the interpreter at shutdown
        sys.stdout.flush()
        os._exit(130)


if __name__ == "__main__":
    main()
