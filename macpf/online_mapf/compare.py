"""Compare a classical run vs an online run and save a side-by-side metrics report.

Both `classical_mapf` and `online_mapf` write a `classical_metrics.json` into their
run directory. This tool loads two of them, lines up the shared KPIs, computes the
delta + which planner wins per metric (given each metric's "better" direction), and
saves a Markdown table + a structured JSON under `reports/comparisons/<timestamp>/`.

    # explicit run dirs
    python -m macpf.online_mapf.compare --classical data/classical_runs/<ts> \
                                        --online data/online_runs/<ts>
    # or omit to use the latest run of each
    python -m macpf.online_mapf.compare
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJ_ROOT = Path(__file__).resolve().parents[2]

# (metric key, human label, better direction: "up" | "down" | None)
KPIS = [
    ("total_completed_deliveries", "deliveries (배송 완료)", "up"),
    ("total_completed_targets", "targets reached (타깃 도달)", "up"),
    ("average_deliveries_per_agent", "deliveries/agent", "up"),
    ("collision_count", "collisions (충돌)", "down"),
    ("congestion_overlap_cell_count", "congestion overlap (혼잡 겹침)", "down"),
    ("congestion_peak", "congestion peak (혼잡 peak)", "down"),
    ("total_waiting_time", "total waiting (총 대기)", "down"),
    ("mean_observed_speed_mps", "mean speed m/s (평균 속도)", "up"),
    ("proximity_slowdown_events", "proximity slowdowns (근접 감속)", "down"),
    ("proximity_emergency_stop_events", "emergency stops (비상정지)", "down"),
    ("car_following_slowdown_events", "car-following slowdowns (차간 감속)", "down"),
    ("mean_nearest_amr_distance_cells", "mean nearest dist (평균 최근접)", "up"),
    ("min_inter_amr_manhattan_distance_cells", "min spacing (최소 간격)", "up"),
    ("makespan", "makespan", None),
    ("total_path_length", "total path length", None),
]


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    f = run_dir / "classical_metrics.json"
    if not f.exists():
        raise FileNotFoundError(f"No classical_metrics.json in {run_dir}")
    return json.loads(f.read_text(encoding="utf-8"))


def latest_run(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir() and (p / "classical_metrics.json").exists()]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def build_rows(cm: Dict[str, Any], om: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for key, label, direction in KPIS:
        if key not in cm or key not in om:
            continue
        c, o = cm[key], om[key]
        if not isinstance(c, (int, float)) or not isinstance(o, (int, float)):
            continue
        delta = o - c
        pct = (delta / abs(c) * 100.0) if c not in (0, 0.0) else None
        winner = "="
        if direction and delta != 0:
            online_better = (delta > 0) if direction == "up" else (delta < 0)
            winner = "online" if online_better else "classical"
        rows.append({
            "key": key, "label": label, "direction": direction,
            "classical": c, "online": o, "delta": delta, "pct": pct, "winner": winner,
        })
    return rows


def make_charts(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    """Render comparison charts (PNG) into out_dir; return their filenames so the
    Markdown report can embed them.

    Chart axis labels use only the English part of each KPI label (the text before
    " (") so the figures never depend on a Korean-capable matplotlib font; the full
    bilingual labels stay in the Markdown table.
    """
    if not rows:
        return []
    import matplotlib

    matplotlib.use("Agg")  # headless: render straight to file, no display needed
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["axes.unicode_minus"] = False
    arrow = {"up": "↑", "down": "↓", None: ""}

    def clean(label: str) -> str:
        return label.split(" (")[0].strip()

    written: List[str] = []
    n = len(rows)
    classical = np.array([float(r["classical"]) for r in rows])
    online = np.array([float(r["online"]) for r in rows])
    ylabels = [f"{arrow[r['direction']]} {clean(r['label'])}".strip() for r in rows]
    y = np.arange(n)[::-1]  # first KPI at the top

    # Chart 1: per-KPI normalized grouped bars. Each row is scaled to its own max so
    # KPIs of wildly different magnitudes (deliveries ~tens vs path length ~thousands)
    # stay visually comparable; raw values are printed at the bar ends.
    denom = np.maximum.reduce([np.abs(classical), np.abs(online), np.full(n, 1e-9)])
    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.52 * n + 1.6)))
    bh = 0.38
    ax.barh(y + bh / 2, classical / denom, height=bh, color="#4C72B0", label="classical")
    ax.barh(y - bh / 2, online / denom, height=bh, color="#55A868", label="online")
    for yi, c, o, d in zip(y, classical, online, denom):
        ax.text(c / d + 0.02, yi + bh / 2, f"{c:g}", va="center", ha="left", fontsize=7, color="#333")
        ax.text(o / d + 0.02, yi - bh / 2, f"{o:g}", va="center", ha="left", fontsize=7, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("relative magnitude (each KPI scaled to its own max; raw values labelled)")
    ax.set_title("Classical vs Online — KPI comparison")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_kpis.png", dpi=150)
    plt.close(fig)
    written.append("comparison_kpis.png")

    # Chart 2: % change of online vs classical, one bar per directional KPI, coloured
    # by who wins (green = online, blue = classical).
    idx = [i for i, r in enumerate(rows) if r["pct"] is not None and r["direction"] is not None]
    if idx:
        pcts = np.array([float(rows[i]["pct"]) for i in idx])
        labs = [f"{arrow[rows[i]['direction']]} {clean(rows[i]['label'])}".strip() for i in idx]
        colors = [
            "#55A868" if rows[i]["winner"] == "online"
            else "#4C72B0" if rows[i]["winner"] == "classical"
            else "#999999"
            for i in idx
        ]
        y2 = np.arange(len(idx))[::-1]
        fig, ax = plt.subplots(figsize=(10.5, max(3.5, 0.52 * len(idx) + 1.6)))
        ax.barh(y2, pcts, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        for yi, p in zip(y2, pcts):
            ax.text(p + (1.5 if p >= 0 else -1.5), yi, f"{p:+.1f}%", va="center",
                    ha="left" if p >= 0 else "right", fontsize=7, color="#333")
        ax.set_yticks(y2)
        ax.set_yticklabels(labs, fontsize=9)
        ax.set_xlabel("Δ% (online vs classical)")
        ax.set_title("Online vs Classical — % change  (green = online wins, blue = classical wins)")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "comparison_delta.png", dpi=150)
        plt.close(fig)
        written.append("comparison_delta.png")

    return written


def to_markdown(rows, classical_dir, online_dir, stamp, chart_files: Optional[List[str]] = None) -> str:
    arrow = {"up": "↑", "down": "↓", None: "—"}
    flag = {"online": "🟢 online", "classical": "🔵 classical", "=": "="}
    lines = [
        "# Classical vs Online MAPF — 성능지표 비교",
        "",
        f"- classical: `{classical_dir}`",
        f"- online:    `{online_dir}`",
        f"- generated: {stamp}",
        "",
        "| 지표 | better | Classical | Online | Δ (online−classical) | 우세 |",
        "|---|:---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        d = r["delta"]
        pct = f" ({r['pct']:+.1f}%)" if r["pct"] is not None else ""
        dtxt = (f"{d:+.3f}" if isinstance(d, float) else f"{d:+d}") + pct
        lines.append(
            f"| {r['label']} | {arrow[r['direction']]} | {_fmt(r['classical'])} | "
            f"{_fmt(r['online'])} | {dtxt} | {flag[r['winner']]} |"
        )
    wins_o = sum(1 for r in rows if r["winner"] == "online")
    wins_c = sum(1 for r in rows if r["winner"] == "classical")
    lines += ["", f"**요약**: 방향성 있는 지표 중 online 우세 {wins_o} · classical 우세 {wins_c}."]
    if chart_files:
        lines += ["", "## 차트"]
        for fn in chart_files:
            lines += ["", f"![{fn}]({fn})"]
    return "\n".join(lines) + "\n"


def main() -> None:
    # The report uses Unicode (arrows, em-dash); the Windows console may be cp949.
    # Write UTF-8 to stdout so printing the table never crashes.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(description="Compare classical vs online run metrics.")
    parser.add_argument("--classical", default=None, help="Classical run dir (default: latest).")
    parser.add_argument("--online", default=None, help="Online run dir (default: latest).")
    parser.add_argument("--out", default=None, help="Output dir (default: reports/comparisons/<ts>).")
    args = parser.parse_args()

    classical_dir = Path(args.classical) if args.classical else latest_run(PROJ_ROOT / "data" / "classical_runs")
    online_dir = Path(args.online) if args.online else latest_run(PROJ_ROOT / "data" / "online_runs")
    if classical_dir is None or online_dir is None:
        raise SystemExit("Could not find classical and/or online runs (give --classical/--online).")

    cm, om = load_metrics(classical_dir), load_metrics(online_dir)
    rows = build_rows(cm, om)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    out_dir = Path(args.out) if args.out else (PROJ_ROOT / "reports" / "comparisons" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_files = make_charts(rows, out_dir)
    md = to_markdown(rows, classical_dir, online_dir, stamp, chart_files)
    (out_dir / "comparison.md").write_text(md, encoding="utf-8")
    (out_dir / "comparison.json").write_text(json.dumps({
        "classical_run": str(classical_dir),
        "online_run": str(online_dir),
        "generated": stamp,
        "charts": chart_files,
        "rows": rows,
        "classical_metrics": cm,
        "online_metrics": om,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(md)
    print(f"Saved comparison -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
