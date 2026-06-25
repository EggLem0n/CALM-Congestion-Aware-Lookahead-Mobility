# -*- coding: utf-8 -*-
"""Gamma x lambda sweep for congestion-aware PIBT (handoff sections 5 & 7).

For each depth gamma ``r`` (which sets where the depth weight ``w_k`` peaks) we sweep the
congestion weight ``lambda`` and report throughput, so each ``r``'s best ``lambda`` can be
compared fairly -- the section-5 recipe: optimize ``lambda`` per ``r`` first, then compare
the optima (removes the per-``r`` penalty-scale difference). It also runs the section-7
ablations as extra axes: ``min_depth`` (1 vs 2, i.e. include/exclude the immediate cell)
and ``depth_mode`` (peaked ``k*r^(k-1)`` vs front-load ``r^(k-1)``).

The ``lambda = 0`` baseline is plain PIBT and is byte-for-byte identical regardless of
``r`` / ``min_depth`` / ``depth_mode``, so it is computed ONCE and reused as the comparison
point for every cell. ``predict_every`` defaults to ``11 - horizon`` (handoff section 4:
an old MPC forecast loses reach as the offset grows, so this keeps full-depth lookahead).

Outputs, under ``reports/gamma_sweep/<yymmdd_hhmm>/``:
  - a console table per (depth_mode, min_depth) block, with each gamma's best lambda,
  - ``metrics.csv`` (one row per run incl. the baseline) + ``metadata.json``,
  - unless ``--no-video``: one 2-panel ``vanilla | congestion`` MP4 per gamma (at that
    gamma's best lambda), rendered with the same MACPF animator grid_eval uses.

Same map / starts / task RNG across all runs (PIBT's own seed), so the only thing that
changes within the sweep is the congestion penalty. Congestion in the report is the
GROUND-TRUTH additive field recomputed from the resulting positions, not the forecast.

Run (OpenSTL conda env, from this folder):
    python gamma_sweep.py --no-video
    python gamma_sweep.py --gammas 0.61 0.66 0.70 0.73 --weights 0.25 0.5 1 2
    python gamma_sweep.py --min-depths 1 2 --depth-modes peaked frontload   # section-7 ablation
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import csv
import json
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, REPO_ROOT):                 # predict.py + grid_eval.py (here) + calm (repo root)
    if p not in sys.path:
        sys.path.insert(0, p)

from calm import PiBT as mapf                          # noqa: E402
from calm.PiBT import factory_map_generator as fmg     # noqa: E402
import grid_eval                                       # noqa: E402  (video helpers; torch-free import)
# CongestionPredictor is imported lazily in main() so --help / --no-video planning stays light.


def evaluate(paths, summary, walkable, config, wall):
    """Throughput + ground-truth congestion stats from the resulting paths.

    Vendored from ab_eval (kept verbatim) so this module's import never pulls in torch."""
    H, W = walkable.shape[:2]
    ap = mapf.paths_to_agent_positions(paths, config.max_time)            # (T, N, 2)
    cong = mapf.build_additive_congestion_label_sequence(
        ap, H, W, config.congestion_center_value, config.congestion_step_value)  # (T, H, W)
    T = ap.shape[0]
    tidx = np.arange(T)[:, None]
    agent_cong = cong[tidx, ap[..., 1], ap[..., 0]]          # (T, N) congestion at each robot
    return {
        "deliveries": int(summary["total_completed_deliveries"]),
        "mean_robot_cong": float(agent_cong.mean()),
        "p99_cong": float(np.percentile(cong, 99.0)),
        "peak_cong": float(cong.max()),
        "collisions": int(mapf.compute_collision_count(ap)),
        "preds": int(summary["congestion_prediction_count"]),
        "wall_s": float(wall),
    }


def run_one(env, config, walkable, starts, pickup, delivery, predictor, *,
            weight, gamma, horizon, min_depth, depth_mode, predict_every):
    """One planner run. Returns (metrics, paths, summary). predictor is only used when
    weight > 0 (weight 0 -> plain PIBT, the baseline)."""
    t0 = time.perf_counter()
    paths, summary = mapf.plan_pibt_repeated_tasks(
        starts, pickup, delivery, walkable, config,
        pickup_point_groups=mapf.normalize_point_groups(env.get("pickup_point_groups")),
        delivery_point_groups=mapf.normalize_point_groups(env.get("delivery_point_groups")),
        congestion_predictor=(predictor if weight > 0 else None),
        congestion_weight=weight,
        congestion_gamma=gamma,
        congestion_horizon=horizon,
        congestion_min_depth=min_depth,
        congestion_depth_mode=depth_mode,
        predict_every=predict_every,
    )
    return evaluate(paths, summary, walkable, config, time.perf_counter() - t0), paths, summary


CSV_FIELDS = ["gamma", "weight", "horizon", "min_depth", "depth_mode", "predict_every",
              "seed", "deliveries", "delta_vs_base", "mean_robot_cong", "p99_cong",
              "peak_cong", "collisions", "preds", "wall_s"]


def _row(gamma, weight, args, horizon, min_depth, depth_mode, predict_every, m, base_deliv):
    return {"gamma": gamma, "weight": weight, "horizon": horizon, "min_depth": min_depth,
            "depth_mode": depth_mode, "predict_every": predict_every, "seed": args.seed,
            "deliveries": m["deliveries"], "delta_vs_base": m["deliveries"] - base_deliv,
            "mean_robot_cong": round(m["mean_robot_cong"], 3), "p99_cong": round(m["p99_cong"], 3),
            "peak_cong": round(m["peak_cong"], 1), "collisions": m["collisions"],
            "preds": m["preds"], "wall_s": round(m["wall_s"], 2)}


def render_videos(env, starts, base_paths, base_summary, jobs, args, config, out_dir):
    """Render vanilla once, then hstack each (label, paths, summary) job beside it.

    Reuses grid_eval's MACPF animator helpers; matplotlib/ffmpeg are imported lazily there,
    so a --no-video sweep never touches them."""
    vdir = out_dir / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    tmp = vdir / ".tmp"
    anim_cfg = grid_eval._anim_config(config, args)
    # videos only show the first --video-seconds steps; metrics already used the full episode.
    vs = args.video_seconds if 0 < args.video_seconds < args.seconds else (args.seconds + 1)
    clip = lambda paths: [p[:vs + 1] for p in paths]
    print(f"\nrendering {len(jobs)} video(s) (vanilla | congestion)...")
    vanilla_mp4 = grid_eval._animate_scenario(
        env, starts, clip(base_paths), base_summary, anim_cfg, tmp / "vanilla.mp4", tmp / "v")
    for label, paths, summary in jobs:
        cong_mp4 = grid_eval._animate_scenario(
            env, starts, clip(paths), summary, anim_cfg, tmp / f"{label}.mp4", tmp / f"c_{label}")
        out_name = vdir / f"{label}.mp4"
        grid_eval._hstack(vanilla_mp4, cong_mp4, out_name)
        print(f"  video -> {out_name.name}")
    shutil.rmtree(tmp, ignore_errors=True)


def parse_args():
    base = mapf.load_config()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- scenario (shared by every run) ---
    ap.add_argument("--agents", type=int, default=300)
    ap.add_argument("--seconds", type=int, default=150, help="episode length (steps)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frac", type=float, default=0.0, help="distributed-spawn fraction (0..1)")
    # --- sweep axes ---
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.61, 0.66, 0.70, 0.73],
                    help="r values = depth-weight peak position (handoff usable band [0.607, 0.730])")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0],
                    help="lambda values swept per gamma; 0 is the always-computed baseline")
    ap.add_argument("--horizon", type=int, default=10, help="H: max descent depth read (<=10)")
    ap.add_argument("--min-depths", type=int, nargs="+", default=[2],
                    help="k_start ablation axis (handoff section 7-1: try '1 2')")
    ap.add_argument("--depth-modes", nargs="+", default=["peaked"],
                    choices=["peaked", "frontload"],
                    help="depth-weight ablation axis (handoff section 7-2)")
    ap.add_argument("--predict-every", type=int, default=None,
                    help="MPC re-predict period; default = 11 - horizon (full-depth lookahead)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    # --- output ---
    ap.add_argument("--no-video", action="store_true", help="metrics only, skip MP4s")
    ap.add_argument("--video-all", action="store_true",
                    help="render every (gamma, lambda>0) combo instead of just each gamma's best")
    ap.add_argument("--center-value", type=float, default=base.congestion_center_value)
    ap.add_argument("--step-value", type=float, default=base.congestion_step_value)
    # --- MACPF animate_paths knobs (names must match grid_eval._anim_config) ---
    ap.add_argument("--video-seconds", type=int, default=0,
                    help="render only first N steps (0 or >= --seconds = full)")
    ap.add_argument("--anim-subframes", type=int, default=1,
                    help="interpolated frames per sim-step (1 = fastest cell-to-cell jumps)")
    ap.add_argument("--robot-size", type=float, default=8.0, help="robot marker area")
    ap.add_argument("--route-linewidth", type=float, default=0.6)
    ap.add_argument("--planned-route-linewidth", type=float, default=0.3)
    ap.add_argument("--target-size", type=float, default=40.0)
    ap.add_argument("--planned-routes", action="store_true")
    args = ap.parse_args()
    if not (0.0 <= args.frac <= 1.0):
        ap.error("require 0.0 <= --frac <= 1.0")
    return args


def main():
    args = parse_args()
    horizon = max(1, min(10, int(args.horizon)))
    predict_every = args.predict_every if args.predict_every is not None else max(1, 11 - horizon)

    env = fmg.build_factory_map()
    base = mapf.load_config()
    config = base.replace(num_agents=args.agents, seed=args.seed, max_time=args.seconds,
                          distributed_fraction=args.frac,
                          congestion_center_value=args.center_value,
                          congestion_step_value=args.step_value,
                          show_planning_progress=False)
    walkable = np.asarray(env["walkable_map"]).astype(bool)
    starts, _ = mapf.select_start_goal_pairs(env, walkable, config)
    pickup = [p for p in mapf.normalize_points(env.get("pickup_points")) if mapf.is_walkable(*p, walkable)]
    delivery = [p for p in mapf.normalize_points(env.get("delivery_points")) if mapf.is_walkable(*p, walkable)]

    gammas = list(args.gammas)
    weights_pos = sorted({w for w in args.weights if w > 0})
    min_depths = list(args.min_depths)
    depth_modes = list(args.depth_modes)

    predictor = None
    if weights_pos:
        from predict import CongestionPredictor          # lazy: only load torch when needed
        predictor = CongestionPredictor(device=args.device)
        print(f"predictor: best.ckpt on {predictor.device} | y_scale={predictor.y_scale:g}")

    out_dir = Path(REPO_ROOT) / "reports" / "gamma_sweep" / datetime.now().strftime("%y%m%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cong = len(depth_modes) * len(min_depths) * len(gammas) * len(weights_pos)
    print(f"scenario: {args.agents} AMRs | {args.seconds} s | seed {args.seed} | frac {args.frac} "
          f"| horizon {horizon} | predict_every {predict_every}")
    print(f"axes: gammas {gammas} x weights {weights_pos}  | min_depths {min_depths} "
          f"| depth_modes {depth_modes}  = {n_cong} congestion runs + 1 baseline")
    print(f"output -> {out_dir}\n", flush=True)

    # --- baseline (lambda = 0), computed ONCE and reused for every cell ---
    base_m, base_paths, base_summary = run_one(
        env, config, walkable, starts, pickup, delivery, None,
        weight=0.0, gamma=gammas[0] if gammas else 0.73, horizon=horizon,
        min_depth=min_depths[0] if min_depths else 2,
        depth_mode=depth_modes[0] if depth_modes else "peaked", predict_every=predict_every)
    base_deliv = base_m["deliveries"]
    print(f"baseline (lambda=0): deliveries={base_deliv}  mean@robot={base_m['mean_robot_cong']:.1f}  "
          f"p99={base_m['p99_cong']:.1f}  peak={base_m['peak_cong']:.0f}\n", flush=True)

    rows = [_row("-", 0.0, args, horizon, "-", "-", predict_every, base_m, base_deliv)]
    best_per_gamma = {}     # (mode, md, gamma) -> (weight, metrics, paths, summary)
    video_runs = []         # only populated when --video-all

    for mode in depth_modes:
        for md in min_depths:
            print(f"=== depth_mode={mode}  min_depth={md} ===")
            hdr = (f"{'gamma':>5} {'lam':>5} | {'deliv':>6} {'d-base':>6} | "
                   f"{'mean@r':>7} {'p99':>7} {'preds':>5} {'wall_s':>6}")
            print(hdr); print("-" * len(hdr))
            for g in gammas:
                best = None
                for w in weights_pos:
                    m, paths, summary = run_one(
                        env, config, walkable, starts, pickup, delivery, predictor,
                        weight=w, gamma=g, horizon=horizon, min_depth=md, depth_mode=mode,
                        predict_every=predict_every)
                    rows.append(_row(g, w, args, horizon, md, mode, predict_every, m, base_deliv))
                    print(f"{g:>5.2f} {w:>5.2f} | {m['deliveries']:>6d} {m['deliveries']-base_deliv:>+6d} | "
                          f"{m['mean_robot_cong']:>7.1f} {m['p99_cong']:>7.1f} {m['preds']:>5d} "
                          f"{m['wall_s']:>6.2f}", flush=True)
                    label = f"{mode}_md{md}_g{g:.2f}_w{w:g}"
                    if args.video_all and not args.no_video:
                        video_runs.append((label, paths, summary))
                    if best is None or m["deliveries"] > best[1]["deliveries"]:
                        best = (w, m, paths, summary)
                if best is not None:
                    best_per_gamma[(mode, md, g)] = best
                    print(f"   -> best lambda @ gamma={g:.2f}: {best[0]:g}  "
                          f"(deliveries {best[1]['deliveries']}, {best[1]['deliveries']-base_deliv:+d} vs base)")
            print(flush=True)

    # --- write metrics.csv + metadata.json ---
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        wr.writeheader(); wr.writerows(rows)
    (out_dir / "metadata.json").write_text(json.dumps({
        "tool": "gamma_sweep", "solver": "pibt_lifelong",
        "agents": args.agents, "seconds": args.seconds, "seed": args.seed, "frac": args.frac,
        "gammas": gammas, "weights": weights_pos, "min_depths": min_depths,
        "depth_modes": depth_modes, "horizon": horizon, "predict_every": predict_every,
        "congestion_center_value": args.center_value, "congestion_step_value": args.step_value,
        "baseline_deliveries": base_deliv,
    }, indent=2), encoding="utf-8")

    # --- best-lambda-per-gamma summary (section-5 fair comparison) ---
    if best_per_gamma:
        print("=== best lambda per gamma (vs baseline) ===")
        hdr = f"{'mode':>9} {'md':>3} {'gamma':>5} | {'best lam':>8} {'deliv':>6} {'d-base':>6}"
        print(hdr); print("-" * len(hdr))
        for (mode, md, g), (w, m, _p, _s) in sorted(best_per_gamma.items()):
            print(f"{mode:>9} {md:>3} {g:>5.2f} | {w:>8g} {m['deliveries']:>6d} "
                  f"{m['deliveries']-base_deliv:>+6d}")
        gw, gm = max(((k, v) for k, v in best_per_gamma.items()), key=lambda kv: kv[1][1]["deliveries"])
        print(f"\nbest overall: depth_mode={gw[0]} min_depth={gw[1]} gamma={gw[2]:.2f} "
              f"lambda={gm[0]:g} -> deliveries {gm[1]['deliveries']} ({gm[1]['deliveries']-base_deliv:+d} vs base)")

    print(f"\nmetrics -> {out_dir / 'metrics.csv'}")

    # --- videos: each gamma's best lambda (default), or every combo (--video-all) ---
    if not args.no_video and (best_per_gamma or video_runs):
        if args.video_all:
            jobs = video_runs
        else:
            jobs = [(f"{mode}_md{md}_g{g:.2f}_w{w:g}", paths, summary)
                    for (mode, md, g), (w, _m, paths, summary) in sorted(best_per_gamma.items())]
        try:
            render_videos(env, starts, base_paths, base_summary, jobs, args, config, out_dir)
            print(f"videos  -> {out_dir / 'videos'}")
        except Exception as exc:  # noqa: BLE001  (video is optional; never fail the metrics run)
            print(f"[videos skipped] {exc!r}")

    print(f"report  -> {out_dir}")


if __name__ == "__main__":
    main()
