from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List


METRICS = [
    ("ai_total_completed_deliveries", "up"),
    ("ai_total_completed_targets", "up"),
    ("ai_average_deliveries_per_agent", "up"),
    ("ai_total_waiting_time", "down"),
    ("ai_waiting_ratio", "down"),
    ("ai_max_consecutive_wait_per_agent", "down"),
    ("ai_agents_consecutive_wait_ge_5", "down"),
    ("ai_agents_consecutive_wait_ge_10", "down"),
    ("ai_actual_moved_cells", "neutral"),
    ("ai_deliveries_per_moved_cell", "up"),
    ("ai_pibt_candidate_reject_vertex", "down"),
    ("ai_pibt_candidate_reject_swap", "down"),
    ("ai_congestion_peak", "down"),
    ("ai_congestion_overlap_cell_count", "down"),
    ("ai_road_peak_cell_occupancy", "down"),
    ("ai_collision_count", "down"),
    ("ai_interpolated_safe_gap_violation_count", "down"),
]


def read_rows(run_dir: Path) -> List[Dict[str, str]]:
    csv_path = run_dir / "pibt_sweep_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing sweep result: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise KeyError(f"Missing metric column: {key}")
    return float(value)


def index_by_seed(rows: Iterable[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    return {int(row["seed"]): row for row in rows}


def better_label(delta: float, direction: str) -> str:
    if direction == "up":
        return "D better" if delta > 0 else ("B better" if delta < 0 else "same")
    if direction == "down":
        return "D better" if delta < 0 else ("B better" if delta > 0 else "same")
    return "neutral"


def fmt(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare B Rule-only vs D AI+Rule ablation results.")
    parser.add_argument("--rule-only", required=True, type=Path, help="B run directory.")
    parser.add_argument("--ai-rule", required=True, type=Path, help="D run directory.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for comparison files.")
    args = parser.parse_args()

    b_rows = index_by_seed(read_rows(args.rule_only))
    d_rows = index_by_seed(read_rows(args.ai_rule))
    seeds = sorted(set(b_rows) & set(d_rows))
    if not seeds:
        raise ValueError("No matching seeds between B and D results.")

    comparison_rows: List[Dict[str, object]] = []
    for seed in seeds:
        row: Dict[str, object] = {"seed": seed}
        for metric, direction in METRICS:
            b = as_float(b_rows[seed], metric)
            d = as_float(d_rows[seed], metric)
            delta = d - b
            row[f"{metric}_B_rule_only"] = b
            row[f"{metric}_D_ai_rule"] = d
            row[f"{metric}_delta_D_minus_B"] = delta
            row[f"{metric}_winner"] = better_label(delta, direction)
        comparison_rows.append(row)

    summary = {}
    for metric, direction in METRICS:
        b_values = [float(row[f"{metric}_B_rule_only"]) for row in comparison_rows]
        d_values = [float(row[f"{metric}_D_ai_rule"]) for row in comparison_rows]
        b_mean = mean(b_values)
        d_mean = mean(d_values)
        delta = d_mean - b_mean
        summary[metric] = {
            "direction": direction,
            "B_rule_only_mean": b_mean,
            "D_ai_rule_mean": d_mean,
            "delta_D_minus_B": delta,
            "delta_percent": None if b_mean == 0 else (delta / b_mean) * 100.0,
            "winner": better_label(delta, direction),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    seed_csv = args.out / "seed_comparison.csv"
    with seed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# B Rule-only vs D AI+Rule Comparison",
        "",
        f"- B Rule-only: `{args.rule_only}`",
        f"- D AI+Rule: `{args.ai_rule}`",
        f"- Seeds: `{seeds[0]}-{seeds[-1]}`" if len(seeds) > 1 else f"- Seed: `{seeds[0]}`",
        "",
        "| Metric | Better | B Rule-only | D AI+Rule | D-B | Winner |",
        "|---|:---:|---:|---:|---:|:---:|",
    ]
    for metric, direction in METRICS:
        item = summary[metric]
        arrow = "higher" if direction == "up" else ("lower" if direction == "down" else "-")
        md_lines.append(
            "| {metric} | {arrow} | {b} | {d} | {delta} | {winner} |".format(
                metric=metric,
                arrow=arrow,
                b=fmt(float(item["B_rule_only_mean"])),
                d=fmt(float(item["D_ai_rule_mean"])),
                delta=fmt(float(item["delta_D_minus_B"])),
                winner=item["winner"],
            )
        )
    (args.out / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("\n".join(md_lines))
    print(f"\nSaved: {seed_csv}")
    print(f"Saved: {args.out / 'summary.json'}")
    print(f"Saved: {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
