# -*- coding: utf-8 -*-
"""A/B evaluation: congestion-aware PIBT vs plain PIBT on the same scenario.

Runs ``plan_pibt_repeated_tasks`` with congestion_weight = 0 (the plain-PIBT
baseline) and with one or more positive weights that switch on the trained SimVP
predictor, then reports throughput and congestion so you can see whether steering
agents off predicted congestion helps.

All runs share the identical map / starts / task RNG (PIBT's own seed), so the only
difference is the congestion penalty. Congestion in the report is the GROUND-TRUTH
additive field recomputed from the resulting positions (not the model's forecast).

Run (OpenSTL conda env, from this folder):
    python ab_eval.py                                  # 300 AMRs, 150 s, weights 0/1/3
    python ab_eval.py --agents 500 --seconds 300 --weights 0 0.5 1 2 4
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
PRED_DIR = os.path.join(REPO_ROOT, "calm", "congestion_prediction")   # predict.py lives here
for p in (HERE, PRED_DIR, REPO_ROOT):       # this dir + predict.py's dir + calm (repo root)
    if p not in sys.path:
        sys.path.insert(0, p)

from predict import CongestionPredictor                      # noqa: E402
from calm import PiBT as mapf                                  # noqa: E402
from calm.PiBT import factory_map_generator as fmg     # noqa: E402


def evaluate(paths, summary, walkable, config, wall):
    """Throughput + ground-truth congestion stats from the resulting paths."""
    H, W = walkable.shape[:2]
    ap = mapf.paths_to_agent_positions(paths, config.max_time)            # (T, N, 2)
    cong = mapf.build_additive_congestion_label_sequence(
        ap, H, W, config.congestion_center_value, config.congestion_step_value)  # (T, H, W)
    T = ap.shape[0]
    tidx = np.arange(T)[:, None]
    agent_cong = cong[tidx, ap[..., 1], ap[..., 0]]          # (T, N) congestion at each robot
    return {
        "deliveries": int(summary["total_completed_deliveries"]),
        "mean_robot_cong": float(agent_cong.mean()),         # crowding where robots actually are
        "p99_cong": float(np.percentile(cong, 99.0)),
        "peak_cong": float(cong.max()),
        "collisions": int(mapf.compute_collision_count(ap)),
        "preds": int(summary["congestion_prediction_count"]),
        "wall_s": float(wall),
    }


def run_one(weight, predictor, env, config, predict_every):
    walkable = np.asarray(env["walkable_map"]).astype(bool)
    starts, _ = mapf.select_start_goal_pairs(env, walkable, config)
    pickup = [p for p in mapf.normalize_points(env.get("pickup_points")) if mapf.is_walkable(*p, walkable)]
    delivery = [p for p in mapf.normalize_points(env.get("delivery_points")) if mapf.is_walkable(*p, walkable)]
    t0 = time.perf_counter()
    paths, summary = mapf.plan_pibt_repeated_tasks(
        starts, pickup, delivery, walkable, config,
        pickup_point_groups=mapf.normalize_point_groups(env.get("pickup_point_groups")),
        delivery_point_groups=mapf.normalize_point_groups(env.get("delivery_point_groups")),
        congestion_predictor=(predictor if weight > 0 else None),
        congestion_weight=weight,
        predict_every=predict_every,
    )
    return evaluate(paths, summary, walkable, config, time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", type=int, default=300)
    ap.add_argument("--seconds", type=int, default=150, help="episode length (steps)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frac", type=float, default=0.0, help="distributed-spawn fraction (0..1)")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 1.0, 3.0],
                    help="congestion_weight values; 0 = plain-PIBT baseline")
    ap.add_argument("--predict-every", type=int, default=10)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    env = fmg.build_factory_map()
    base = mapf.load_config()
    config = base.replace(num_agents=args.agents, seed=args.seed,
                          max_time=args.seconds, distributed_fraction=args.frac,
                          show_planning_progress=False)

    predictor = None
    if any(w > 0 for w in args.weights):
        predictor = CongestionPredictor(device=args.device)
        print(f"predictor: best.ckpt on {predictor.device} | y_scale={predictor.y_scale:g}")

    print(f"scenario: {args.agents} AMRs | {args.seconds} s | seed {args.seed} | "
          f"frac {args.frac} | predict_every {args.predict_every}\n")
    header = (f"{'weight':>7} | {'deliveries':>10} | {'mean@robot':>10} | {'p99':>8} | "
              f"{'peak':>7} | {'coll':>4} | {'preds':>5} | {'wall_s':>7}")
    print(header)
    print("-" * len(header))
    rows = []
    for w in args.weights:
        m = run_one(w, predictor, env, config, args.predict_every)
        rows.append((w, m))
        print(f"{w:>7.2f} | {m['deliveries']:>10} | {m['mean_robot_cong']:>10.1f} | "
              f"{m['p99_cong']:>8.1f} | {m['peak_cong']:>7.0f} | {m['collisions']:>4} | "
              f"{m['preds']:>5} | {m['wall_s']:>7.2f}")

    # headline deltas vs the weight-0 baseline
    base_row = next((m for w, m in rows if w == 0.0), None)
    if base_row is not None:
        print("\nvs baseline (weight 0):")
        for w, m in rows:
            if w == 0.0:
                continue
            dd = m["deliveries"] - base_row["deliveries"]
            dc = 100.0 * (m["mean_robot_cong"] - base_row["mean_robot_cong"]) / base_row["mean_robot_cong"]
            dp = 100.0 * (m["p99_cong"] - base_row["p99_cong"]) / base_row["p99_cong"]
            print(f"  weight {w:>4.2f}: deliveries {dd:+d}  |  mean@robot {dc:+.1f}%  |  p99 {dp:+.1f}%")


if __name__ == "__main__":
    main()
