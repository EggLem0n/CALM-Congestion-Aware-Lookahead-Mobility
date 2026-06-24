"""Run paired PIBT baseline-vs-AI sweeps over multiple seeds."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

from macpf.config import PROJ_ROOT


METRIC_KEYS = [
    "total_completed_deliveries",
    "total_completed_targets",
    "average_deliveries_per_agent",
    "total_waiting_time",
    "waiting_ratio",
    "max_consecutive_wait_per_agent",
    "mean_max_consecutive_wait_per_agent",
    "agents_consecutive_wait_ge_5",
    "agents_consecutive_wait_ge_10",
    "actual_moved_cells",
    "deliveries_per_moved_cell",
    "mean_observed_speed_mps",
    "proximity_slowdown_events",
    "car_following_slowdown_events",
    "collision_count",
    "interpolated_safe_gap_violation_count",
    "pibt_inherited_count",
    "pibt_backtrack_count",
    "pibt_forced_wait_count",
    "pibt_candidate_reject_vertex",
    "pibt_candidate_reject_swap",
    "pibt_candidate_reject_continuous",
    "congestion_peak",
    "congestion_overlap_cell_count",
    "road_peak_cell_occupancy",
    "road_mean_used_cell_occupancy",
]


def parse_seeds(raw: str) -> list[int]:
    raw = raw.strip()
    if "-" in raw and "," not in raw:
        start, end = raw.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def numeric(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def summarize(values: Iterable[float]) -> dict[str, float]:
    vals = list(values)
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(mean(vals)), "std": float(pstdev(vals)) if len(vals) > 1 else 0.0}


def run_one(args, mode: str, seed: int, metrics_out: Path) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "throughput_tuning.runner_tuned",
        "--mode",
        mode,
        "--config",
        args.config,
        "--max-time",
        str(args.max_time),
        "--num-agents",
        str(args.num_agents),
        "--seed",
        str(seed),
        "--metrics-out",
        str(metrics_out),
        "--no-figures",
    ]
    if args.kinodynamic:
        cmd.append("--kinodynamic")
    if args.amr_safety:
        cmd.append("--amr-safety")
    if args.continuous_safe_gap is not None:
        cmd.extend(["--continuous-safe-gap", str(args.continuous_safe_gap)])
    if args.common_wait_priority_weight is not None:
        cmd.extend(["--common-wait-priority-weight", str(args.common_wait_priority_weight)])
    if args.stalled_wait_threshold is not None:
        cmd.extend(["--stalled-wait-threshold", str(args.stalled_wait_threshold)])
    if mode == "ai":
        cmd.extend(["--model", args.model, "--device", args.device])
        if args.ai_cost_weight is not None:
            cmd.extend(["--ai-cost-weight", str(args.ai_cost_weight)])
        if args.ai_cost_threshold is not None:
            cmd.extend(["--ai-cost-threshold", str(args.ai_cost_threshold)])
        if args.ai_cost_cap is not None:
            cmd.extend(["--ai-cost-cap", str(args.ai_cost_cap)])
        if args.ai_cost_mode is not None:
            cmd.extend(["--ai-cost-mode", str(args.ai_cost_mode)])
        if args.ai_priority_weight is not None:
            cmd.extend(["--ai-priority-weight", str(args.ai_priority_weight)])
        if args.pickup_ai_multiplier is not None:
            cmd.extend(["--pickup-ai-multiplier", str(args.pickup_ai_multiplier)])
        if args.delivery_ai_multiplier is not None:
            cmd.extend(["--delivery-ai-multiplier", str(args.delivery_ai_multiplier)])
        if args.skip_ai_fraction is not None:
            cmd.extend(["--skip-ai-fraction", str(args.skip_ai_fraction)])
        if args.throughput_profile is not None:
            cmd.extend(["--throughput-profile", str(args.throughput_profile)])
        if args.goal_near_radius is not None:
            cmd.extend(["--goal-near-radius", str(args.goal_near_radius)])
        if args.goal_far_radius is not None:
            cmd.extend(["--goal-far-radius", str(args.goal_far_radius)])
        if args.near_congestion_multiplier is not None:
            cmd.extend(["--near-congestion-multiplier", str(args.near_congestion_multiplier)])
        if args.goal_progress_bonus is not None:
            cmd.extend(["--goal-progress-bonus", str(args.goal_progress_bonus)])
        if args.target_entry_bonus is not None:
            cmd.extend(["--target-entry-bonus", str(args.target_entry_bonus)])
        if args.wait_penalty is not None:
            cmd.extend(["--wait-penalty", str(args.wait_penalty)])
        if args.completion_priority_weight is not None:
            cmd.extend(["--completion-priority-weight", str(args.completion_priority_weight)])
        if args.delivery_priority_multiplier is not None:
            cmd.extend(["--delivery-priority-multiplier", str(args.delivery_priority_multiplier)])
        if args.stalled_ai_cost_multiplier is not None:
            cmd.extend(["--stalled-ai-cost-multiplier", str(args.stalled_ai_cost_multiplier)])
        if args.stalled_priority_weight is not None:
            cmd.extend(["--stalled-priority-weight", str(args.stalled_priority_weight)])
    print(f"[sweep] seed={seed} mode={mode}")
    subprocess.run(cmd, cwd=PROJ_ROOT, check=True)
    return json.loads(metrics_out.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="PIBT baseline-vs-ConvLSTM-AI paired sweep.")
    parser.add_argument("--seeds", default="42-46", help='e.g. "42-46" or "42,43,44"')
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="models/congestion_convlstm.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-time", type=int, default=900)
    parser.add_argument("--num-agents", type=int, default=30)
    parser.add_argument("--kinodynamic", action="store_true")
    parser.add_argument("--amr-safety", action="store_true")
    parser.add_argument("--continuous-safe-gap", type=float, default=None)
    parser.add_argument("--ai-cost-weight", type=float, default=None)
    parser.add_argument("--ai-cost-threshold", type=float, default=None)
    parser.add_argument("--ai-cost-cap", type=float, default=None)
    parser.add_argument("--ai-cost-mode", choices=["additive", "tiebreak"], default=None)
    parser.add_argument("--ai-priority-weight", type=float, default=None)
    parser.add_argument("--pickup-ai-multiplier", type=float, default=None)
    parser.add_argument("--delivery-ai-multiplier", type=float, default=None)
    parser.add_argument("--skip-ai-fraction", type=float, default=None)
    parser.add_argument("--throughput-profile", choices=["off", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--goal-near-radius", type=float, default=7.0)
    parser.add_argument("--goal-far-radius", type=float, default=22.0)
    parser.add_argument("--near-congestion-multiplier", type=float, default=0.10)
    parser.add_argument("--goal-progress-bonus", type=float, default=0.75)
    parser.add_argument("--target-entry-bonus", type=float, default=3.0)
    parser.add_argument("--wait-penalty", type=float, default=0.65)
    parser.add_argument("--completion-priority-weight", type=float, default=18.0)
    parser.add_argument("--delivery-priority-multiplier", type=float, default=2.0)
    parser.add_argument("--stalled-wait-threshold", type=int, default=5)
    parser.add_argument("--stalled-ai-cost-multiplier", type=float, default=0.20)
    parser.add_argument("--stalled-priority-weight", type=float, default=8.0)
    parser.add_argument("--common-wait-priority-weight", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.out) if args.out else PROJ_ROOT / "reports" / "pibt_sweeps" / datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    rows = []
    for seed in seeds:
        paired = {}
        for mode in ("baseline", "ai"):
            metrics_path = metrics_dir / f"{mode}_seed{seed}.json"
            paired[mode] = run_one(args, mode, seed, metrics_path)

        row = {"seed": seed}
        for mode, metrics in paired.items():
            for key in METRIC_KEYS:
                row[f"{mode}_{key}"] = metrics.get(key, 0)
        row["delta_deliveries_ai_minus_baseline"] = (
            numeric(paired["ai"].get("total_completed_deliveries"))
            - numeric(paired["baseline"].get("total_completed_deliveries"))
        )
        row["delta_wait_ai_minus_baseline"] = (
            numeric(paired["ai"].get("total_waiting_time"))
            - numeric(paired["baseline"].get("total_waiting_time"))
        )
        rows.append(row)

    csv_path = out_dir / "pibt_sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "seeds": seeds,
        "max_time": args.max_time,
        "num_agents": args.num_agents,
        "kinodynamic": bool(args.kinodynamic),
        "amr_safety": bool(args.amr_safety),
        "continuous_safe_gap": args.continuous_safe_gap,
        "ai_tuning": {
            "ai_cost_weight": args.ai_cost_weight,
            "ai_cost_threshold": args.ai_cost_threshold,
            "ai_cost_cap": args.ai_cost_cap,
            "ai_cost_mode": args.ai_cost_mode,
            "ai_priority_weight": args.ai_priority_weight,
            "pickup_ai_multiplier": args.pickup_ai_multiplier,
            "delivery_ai_multiplier": args.delivery_ai_multiplier,
            "skip_ai_fraction": args.skip_ai_fraction,
            "throughput_profile": args.throughput_profile,
            "goal_near_radius": args.goal_near_radius,
            "goal_far_radius": args.goal_far_radius,
            "near_congestion_multiplier": args.near_congestion_multiplier,
            "goal_progress_bonus": args.goal_progress_bonus,
            "target_entry_bonus": args.target_entry_bonus,
            "wait_penalty": args.wait_penalty,
            "completion_priority_weight": args.completion_priority_weight,
            "delivery_priority_multiplier": args.delivery_priority_multiplier,
            "stalled_wait_threshold": args.stalled_wait_threshold,
            "stalled_ai_cost_multiplier": args.stalled_ai_cost_multiplier,
            "stalled_priority_weight": args.stalled_priority_weight,
            "common_wait_priority_weight": args.common_wait_priority_weight,
        },
        "metrics": {},
    }
    for mode in ("baseline", "ai"):
        for key in METRIC_KEYS:
            summary["metrics"][f"{mode}_{key}"] = summarize(numeric(row[f"{mode}_{key}"]) for row in rows)
    summary["metrics"]["delta_deliveries_ai_minus_baseline"] = summarize(
        numeric(row["delta_deliveries_ai_minus_baseline"]) for row in rows
    )
    summary["metrics"]["delta_wait_ai_minus_baseline"] = summarize(
        numeric(row["delta_wait_ai_minus_baseline"]) for row in rows
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[sweep] results csv: {csv_path.resolve()}")
    print(f"[sweep] summary json: {summary_path.resolve()}")
    print("[sweep] key summary:")
    print(json.dumps({
        "baseline_deliveries": summary["metrics"]["baseline_total_completed_deliveries"],
        "ai_deliveries": summary["metrics"]["ai_total_completed_deliveries"],
        "delta_deliveries_ai_minus_baseline": summary["metrics"]["delta_deliveries_ai_minus_baseline"],
        "baseline_wait": summary["metrics"]["baseline_total_waiting_time"],
        "ai_wait": summary["metrics"]["ai_total_waiting_time"],
        "delta_wait_ai_minus_baseline": summary["metrics"]["delta_wait_ai_minus_baseline"],
        "baseline_waiting_ratio": summary["metrics"]["baseline_waiting_ratio"],
        "ai_waiting_ratio": summary["metrics"]["ai_waiting_ratio"],
        "baseline_max_consecutive_wait": summary["metrics"]["baseline_max_consecutive_wait_per_agent"],
        "ai_max_consecutive_wait": summary["metrics"]["ai_max_consecutive_wait_per_agent"],
        "baseline_conflict_reject_vertex": summary["metrics"]["baseline_pibt_candidate_reject_vertex"],
        "ai_conflict_reject_vertex": summary["metrics"]["ai_pibt_candidate_reject_vertex"],
    }, indent=2))


if __name__ == "__main__":
    main()
