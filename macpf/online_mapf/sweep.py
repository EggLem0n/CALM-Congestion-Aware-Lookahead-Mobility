"""Multi-seed sweep: run classical + online across seeds in parallel (multiprocessing),
then aggregate the metrics (mean ± std) into a saved comparison report.

A single MAPF run is sequential and can't be split across cores, but independent runs
*can* — so this is where multiprocessing pays off (same idea as generate_heatmap's
parallel episodes). Each (planner, seed) is its own subprocess using the runners'
lightweight `--metrics-out` mode (metrics only; no per-run save or figures).

    python -m macpf.online_mapf.sweep --seeds 10-19 --num_of_process 8
    python -m macpf.online_mapf.sweep --seeds 10,12,14 --max-time 200 --num-agents 20
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

from macpf.online_mapf.compare import KPIS  # reuse the metric spec

PROJ_ROOT = Path(__file__).resolve().parents[2]
PLANNER_MODULE = {
    "classical": "macpf.classical_mapf.classical_mapf",
    "online": "macpf.online_mapf",
}


def parse_seeds(spec: str) -> List[int]:
    seeds: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_job(planner, seed, config, max_time, num_agents, device, out_file) -> Dict:
    cmd = [sys.executable, "-m", PLANNER_MODULE[planner], "--config", config,
           "--seed", str(seed), "--metrics-out", str(out_file)]
    if max_time is not None:
        cmd += ["--max-time", str(max_time)]
    if num_agents is not None:
        cmd += ["--num-agents", str(num_agents)]
    if planner == "online":
        cmd += ["--device", device]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJ_ROOT))
    if proc.returncode != 0 or not Path(out_file).exists():
        tail = (proc.stderr or proc.stdout or "no output").strip()[-400:]
        return {"planner": planner, "seed": seed, "ok": False, "error": tail}
    return {"planner": planner, "seed": seed, "ok": True,
            "metrics": json.loads(Path(out_file).read_text(encoding="utf-8"))}


def aggregate(results: List[Dict]) -> Dict:
    by = {"classical": [], "online": []}
    for r in results:
        if r["ok"]:
            by[r["planner"]].append(r["metrics"])
    agg = {}
    for key, label, direction in KPIS:
        c = [m[key] for m in by["classical"] if isinstance(m.get(key), (int, float))]
        o = [m[key] for m in by["online"] if isinstance(m.get(key), (int, float))]
        if not c or not o:
            continue
        agg[key] = {
            "label": label, "direction": direction,
            "classical_mean": mean(c), "classical_std": pstdev(c) if len(c) > 1 else 0.0,
            "online_mean": mean(o), "online_std": pstdev(o) if len(o) > 1 else 0.0,
        }
    return agg


def to_markdown(agg, seeds, n_ok, n_total, stamp) -> str:
    arrow = {"up": "↑", "down": "↓", None: "—"}
    lines = [
        "# Classical vs Online — 멀티시드 스윕 (mean ± std)",
        "",
        f"- seeds: {seeds} ({len(seeds)}개)",
        f"- 성공 run: {n_ok}/{n_total}",
        f"- generated: {stamp}",
        "",
        "| 지표 | better | Classical | Online | Δmean | 우세 |",
        "|---|:---:|---:|---:|---:|:---:|",
    ]
    wins_o = wins_c = 0
    for a in agg.values():
        cm, cs, om, osd = a["classical_mean"], a["classical_std"], a["online_mean"], a["online_std"]
        d = om - cm
        winner = "="
        if a["direction"] and abs(d) > 1e-9:
            better = (d > 0) if a["direction"] == "up" else (d < 0)
            winner = "🟢 online" if better else "🔵 classical"
            wins_o += int(better)
            wins_c += int(not better)
        lines.append(
            f"| {a['label']} | {arrow[a['direction']]} | {cm:.2f}±{cs:.2f} | "
            f"{om:.2f}±{osd:.2f} | {d:+.2f} | {winner} |"
        )
    lines += ["", f"**요약**: 방향성 지표 중 online 우세 {wins_o} · classical 우세 {wins_c} "
              f"(seed {len(seeds)}개 평균)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(description="Parallel multi-seed classical-vs-online sweep.")
    parser.add_argument("--seeds", default="10-19", help='e.g. "10-19" or "10,12,14".')
    parser.add_argument("--config", default="configs/compare.yaml",
                        help="Base config (distributed_starts/max_time/...); seed is overridden per job.")
    parser.add_argument("--num_of_process", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="Parallel processes (default: half the logical cores).")
    parser.add_argument("--max-time", type=int, default=None, help="Override config.max_time.")
    parser.add_argument("--num-agents", type=int, default=None, help="Override config.num_agents.")
    parser.add_argument("--device", default="auto", help="online inference device (auto|cpu|cuda).")
    parser.add_argument("--planners", default="classical,online")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    planners = [p.strip() for p in args.planners.split(",") if p.strip()]
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = PROJ_ROOT / "reports" / "comparisons" / f"sweep_{stamp}"
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(p, s) for s in seeds for p in planners]
    print(f"sweep: {len(jobs)} jobs ({planners} × {len(seeds)} seeds), "
          f"{args.num_of_process} processes, config={args.config}", flush=True)

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.num_of_process)) as ex:
        futs = {
            ex.submit(run_job, p, s, args.config, args.max_time, args.num_agents,
                      args.device, runs_dir / f"{p}_seed{s}.json"): (p, s)
            for (p, s) in jobs
        }
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            status = "ok" if r["ok"] else f"FAIL: {r.get('error', '')[:120]}"
            print(f"  [{len(results)}/{len(jobs)}] {r['planner']} seed{r['seed']}: {status}", flush=True)

    n_ok = sum(1 for r in results if r["ok"])
    agg = aggregate(results)
    md = to_markdown(agg, seeds, n_ok, len(jobs), stamp)
    (out_dir / "sweep.md").write_text(md, encoding="utf-8")
    (out_dir / "sweep.json").write_text(json.dumps({
        "seeds": seeds, "planners": planners, "config": args.config,
        "max_time": args.max_time, "num_agents": args.num_agents,
        "n_ok": n_ok, "n_total": len(jobs), "generated": stamp,
        "aggregate": agg, "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(md)
    print(f"Saved sweep -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
