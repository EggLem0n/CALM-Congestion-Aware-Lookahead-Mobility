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
    "total_waiting_time",
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
        "macpf.pibt",
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
    if mode == "ai":
        cmd.extend(["--model", args.model, "--device", args.device])
        if args.ai_cost_weight is not None:
            cmd.extend(["--ai-cost-weight", str(args.ai_cost_weight)])
        if args.ai_cost_threshold is not None:
            cmd.extend(["--ai-cost-threshold", str(args.ai_cost_threshold)])
        if args.ai_cost_cap is not None:
            cmd.extend(["--ai-cost-cap", str(args.ai_cost_cap)])
        if args.pickup_ai_multiplier is not None:
            cmd.extend(["--pickup-ai-multiplier", str(args.pickup_ai_multiplier)])
        if args.delivery_ai_multiplier is not None:
            cmd.extend(["--delivery-ai-multiplier", str(args.delivery_ai_multiplier)])
        if args.skip_ai_fraction is not None:
            cmd.extend(["--skip-ai-fraction", str(args.skip_ai_fraction)])
        if args.ai_dispatch:
            cmd.append("--ai-dispatch")
            cmd.extend(["--dispatch-distance-weight", str(args.dispatch_distance_weight)])
            cmd.extend(["--dispatch-congestion-weight", str(args.dispatch_congestion_weight)])
            cmd.extend(["--dispatch-top-k", str(args.dispatch_top_k)])
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
    parser.add_argument("--ai-cost-weight", type=float, default=None)
    parser.add_argument("--ai-cost-threshold", type=float, default=None)
    parser.add_argument("--ai-cost-cap", type=float, default=None)
    parser.add_argument("--pickup-ai-multiplier", type=float, default=None)
    parser.add_argument("--delivery-ai-multiplier", type=float, default=None)
    parser.add_argument("--skip-ai-fraction", type=float, default=None)
    parser.add_argument("--ai-dispatch", action="store_true")
    parser.add_argument("--dispatch-distance-weight", type=float, default=1.0)
    parser.add_argument("--dispatch-congestion-weight", type=float, default=5.0)
    parser.add_argument("--dispatch-top-k", type=int, default=3)
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
        "ai_tuning": {
            "ai_cost_weight": args.ai_cost_weight,
            "ai_cost_threshold": args.ai_cost_threshold,
            "ai_cost_cap": args.ai_cost_cap,
            "pickup_ai_multiplier": args.pickup_ai_multiplier,
            "delivery_ai_multiplier": args.delivery_ai_multiplier,
            "skip_ai_fraction": args.skip_ai_fraction,
            "ai_dispatch": bool(args.ai_dispatch),
            "dispatch_distance_weight": args.dispatch_distance_weight,
            "dispatch_congestion_weight": args.dispatch_congestion_weight,
            "dispatch_top_k": args.dispatch_top_k,
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
    }, indent=2))


if __name__ == "__main__":
    main()
