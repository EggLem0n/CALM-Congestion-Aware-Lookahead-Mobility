# -*- coding: utf-8 -*-
"""Build the by-lambda metrics table (console + PNG) from a grid_eval metrics.csv.

Read-only: it only loads an existing metrics.csv (e.g. from an interrupted run) and
writes metrics_table.png next to it. No solving, no GPU, no torch.

    python make_metrics_table.py                         # latest reports/CALM_comparison/* run
    python make_metrics_table.py reports/CALM_comparison/260624_1901
    python make_metrics_table.py path/to/metrics.csv
"""
from __future__ import annotations

import os
import sys
import csv
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

NUMERIC = ["deliveries", "energy", "energy_per_delivery", "density_uniformity",
           "occ_cv", "mean_robot_cong", "p99_cong", "peak_cong", "collisions"]


def resolve_csv(arg: str | None) -> str:
    if arg and arg.endswith(".csv"):
        return arg
    if arg and os.path.isdir(arg):
        return os.path.join(arg, "metrics.csv")
    runs = sorted(glob.glob(os.path.join(REPO_ROOT, "reports", "CALM_comparison", "*")))
    if not runs:
        sys.exit("no reports/CALM_comparison/* runs found; pass a metrics.csv or run dir.")
    return os.path.join(runs[-1], "metrics.csv")


def load_rows(csv_path: str):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row = {"weight": float(r["weight"]), "episode": int(r["episode"]),
                   "num_agents": int(r["num_agents"]), "frac": float(r["frac"])}
            for k in NUMERIC:
                try:
                    row[k] = float(r[k])
                except (KeyError, ValueError):
                    row[k] = float("nan")
            rows.append(row)
    return rows


def aggregate(rows):
    weights = sorted({r["weight"] for r in rows})
    n_cells = len({r["episode"] for r in rows})
    table = []
    for w in weights:
        sub = [r for r in rows if r["weight"] == w]
        m = lambda k: float(np.nanmean([r[k] for r in sub]))
        table.append((w, {
            "n": len(sub), "deliveries": m("deliveries"), "energy": m("energy"),
            "energy_per_delivery": m("energy_per_delivery"),
            "density_uniformity": m("density_uniformity"), "occ_cv": m("occ_cv"),
            "mean_robot_cong": m("mean_robot_cong"), "p99_cong": m("p99_cong"),
            "collisions": int(np.nansum([r["collisions"] for r in sub])),
        }))
    return weights, n_cells, table


def print_table(n_cells, table):
    print(f"\n=== mean over {n_cells} completed cells, by congestion weight ===")
    hdr = (f"{'lam':>5} | {'deliv':>7} {'energy':>9} {'e/deliv':>8} {'unifrm':>6} "
           f"{'occCV':>5} {'cong@r':>6} {'p99':>6} {'coll':>4}")
    print(hdr); print("-" * len(hdr))
    for w, d in table:
        print(f"{w:>5.2f} | {d['deliveries']:>7.1f} {d['energy']:>9.0f} "
              f"{d['energy_per_delivery']:>8.1f} {d['density_uniformity']:>6.3f} "
              f"{d['occ_cv']:>5.2f} {d['mean_robot_cong']:>6.1f} {d['p99_cong']:>6.0f} "
              f"{d['collisions']:>4}")


def save_png(n_cells, table, out_path):
    cols = ["lambda", "deliveries", "energy", "energy/deliv", "uniformity",
            "occ_cv", "cong@robot", "p99 cong", "collisions"]
    body = [[f"{w:g}", f"{d['deliveries']:.1f}", f"{d['energy']:.0f}",
             f"{d['energy_per_delivery']:.1f}", f"{d['density_uniformity']:.3f}",
             f"{d['occ_cv']:.2f}", f"{d['mean_robot_cong']:.1f}",
             f"{d['p99_cong']:.0f}", f"{d['collisions']}"] for w, d in table]
    fig, ax = plt.subplots(figsize=(1.35 * len(cols), 0.7 + 0.45 * (len(body) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=body, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    for j in range(len(cols)):
        c = tbl[0, j]; c.set_facecolor("#40466e"); c.set_text_props(color="white", weight="bold")
    for i in range(1, len(body) + 1):
        for j in range(len(cols)):
            tbl[i, j].set_facecolor("#f2f2f7" if i % 2 else "#ffffff")
    ax.set_title(f"Congestion-aware PIBT - metrics by lambda  (mean over {n_cells} cells)",
                 fontsize=12, pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    csv_path = resolve_csv(sys.argv[1] if len(sys.argv) > 1 else None)
    rows = load_rows(csv_path)
    if not rows:
        sys.exit(f"no rows in {csv_path}")
    weights, n_cells, table = aggregate(rows)
    print(f"loaded {len(rows)} rows from {csv_path}")
    print_table(n_cells, table)
    out_png = os.path.join(os.path.dirname(csv_path), "metrics_table.png")
    save_png(n_cells, table, out_png)
    print(f"\ntable image -> {out_png}")


if __name__ == "__main__":
    main()
