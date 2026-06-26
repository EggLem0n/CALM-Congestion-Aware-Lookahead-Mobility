"""Generate ``analysis_summary.md`` (+ a copy of ``metrics_glossary.md``) for a grid_eval run.

A grid_eval run drops ``metrics.csv`` (one row per planner run) and ``metadata.json`` into a
dated report folder. This reads that ``metrics.csv`` and writes, INTO THE SAME FOLDER:

  * ``analysis_summary.md`` -- a per-run statistical write-up of the congestion-aware effect:
    6-cell **paired** differences vs each cell's own baseline, 95% CI + paired t-test p-values,
    total-vs-per-delivery energy, the argmax/winner's-curse inflation, a density regression, and
    cost. All numbers are computed from THIS run's csv (no scipy needed -- the Student-t and
    incomplete-beta are implemented here so it runs in the pure-numpy grid_eval env).
  * ``metrics_glossary.md`` -- the static column glossary, so each run folder is self-contained.

grid_eval calls :func:`write_run_reports` when a sweep finishes; you can also regenerate for an
existing run:  ``python make_analysis_summary.py <run_dir> [<run_dir> ...]``.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Student-t / Pearson p-values without scipy (regularized incomplete beta)
# ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value P(|T| >= |t|) for Student-t with `df` degrees of freedom."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def t_crit_975(df: int) -> float:
    """Critical t for a two-sided 95% interval (= t.ppf(0.975, df)), via bisection."""
    if df <= 0:
        return float("nan")
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mean_ci_p(vals: Sequence[float]) -> Tuple[float, float, float, int]:
    """One-sample test of `vals` vs 0 -> (mean, 95% half-width, two-sided p, n)."""
    a = np.asarray([v for v in vals if v is not None and math.isfinite(v)], float)
    n = a.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(a.mean())
    if n == 1:
        return mean, float("nan"), float("nan"), 1
    sd = float(a.std(ddof=1))
    se = sd / math.sqrt(n)
    if se == 0:
        return mean, 0.0, (0.0 if mean == 0 else 0.0), n
    ci = t_crit_975(n - 1) * se
    p = t_two_sided_p(mean / se, n - 1)
    return mean, ci, p, n


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, int]:
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    n = x.size
    if n < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan"), n
    r = float(np.corrcoef(x, y)[0, 1])
    r = max(-0.999999, min(0.999999, r))
    t = r * math.sqrt((n - 2) / (1.0 - r * r))
    return r, t_two_sided_p(t, n - 2), n


# ---------------------------------------------------------------------------
# load + group
# ---------------------------------------------------------------------------
_INT = ("episode", "num_agents", "horizon", "seed", "deliveries", "energy", "collisions", "preds")
_FLT = ("frac", "gamma", "weight", "energy_per_delivery", "density_uniformity", "occ_cv",
        "mean_robot_cong", "p99_cong", "peak_cong", "wall_s")


def _num(v, cast):
    if v is None or v == "":
        return None
    try:
        return cast(float(v))
    except (TypeError, ValueError):
        return None


def load_rows(csv_path: Path) -> List[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d = dict(r)
            for k in _INT:
                d[k] = _num(d.get(k), int)
            for k in _FLT:
                d[k] = _num(d.get(k), float)
            d["min_depth"] = _num(d.get("min_depth"), int)
            rows.append(d)
    return rows


def baselines(rows: List[dict]) -> Dict[int, dict]:
    return {r["episode"]: r for r in rows if r["depth_mode"] == "baseline"}


def treat_rows(rows: List[dict]) -> List[dict]:
    return [r for r in rows if r["depth_mode"] != "baseline"]


def _sorted_set(rows, key, pred=lambda r: True):
    return sorted({r[key] for r in rows if pred(r) and r[key] is not None})


def gamma_avg(rows, ep, mode, md, lam, key):
    """Mean of `key` over gammas for one (cell, mode, md, lambda)."""
    vals = [r[key] for r in rows
            if r["episode"] == ep and r["depth_mode"] == mode and r["min_depth"] == md
            and r["weight"] == lam and r[key] is not None]
    return float(np.mean(vals)) if vals else None


def paired_pct(rows, base, mode, md, lam, key) -> List[float]:
    """Per-cell percent change of `key` vs that cell's baseline (gamma-averaged treatment)."""
    out = []
    for ep, b in base.items():
        t = gamma_avg(rows, ep, mode, md, lam, key)
        bv = b.get(key)
        if t is None or bv in (None, 0):
            continue
        out.append(100.0 * (t - bv) / bv)
    return out


def paired_abs(rows, base, mode, md, lam, key) -> List[float]:
    out = []
    for ep, b in base.items():
        t = gamma_avg(rows, ep, mode, md, lam, key)
        bv = b.get(key)
        if t is None or bv is None:
            continue
        out.append(t - bv)
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "<.001"
    if p >= 0.10:
        return f"{p:.2f} (ns)"
    return f"{p:.3f}"


def _meta(run_dir: Path) -> dict:
    mp = run_dir / "metadata.json"
    if mp.exists():
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def build_analysis_summary(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    rows = load_rows(run_dir / "metrics.csv")
    base = baselines(rows)
    treats = treat_rows(rows)
    meta = _meta(run_dir)
    run_id = run_dir.name

    cells = sorted(base.keys())
    n_cells = len(cells)
    modes = _sorted_set(treats, "depth_mode")
    mds = _sorted_set(treats, "min_depth")
    gammas = _sorted_set(treats, "gamma")
    lams = _sorted_set(treats, "weight", lambda r: r["weight"] and r["weight"] > 0)

    L: List[str] = []
    add = L.append

    add(f"# 분석 요약 — Congestion-aware PIBT (run `{run_id}`)")
    add("")
    add("학습된 혼잡 예측(SimVP)을 PIBT 이동 점수에 `λ × 예측혼잡`으로 넣었을 때의 효과를, "
        "`metrics.csv`를 직접 읽어 **셀별 paired·95% CI·argmax 편향 제거** 기준으로 자동 정리한 문서.")
    add("지표 정의는 같은 폴더 [`metrics_glossary.md`](./metrics_glossary.md) 참조. "
        "*(이 문서는 `make_analysis_summary.py`가 자동 생성한다.)*")
    add("")
    add("---")
    add("")

    # ---- best fixed setting (data-chosen, gamma-averaged) drives the TL;DR ----
    combos = [(mode, md, lam) for mode in modes for md in mds for lam in lams]
    combo_eff = {}
    for c in combos:
        d = paired_abs(rows, base, *c, "deliveries")
        if d:
            combo_eff[c] = float(np.mean(d))
    best = max(combo_eff, key=combo_eff.get) if combo_eff else None

    if best is not None:
        bm, bmd, bl = best
        dv_pct = paired_pct(rows, base, *best, "deliveries")
        best_pct, best_ci, best_p, _ = mean_ci_p(dv_pct)
        cong_pct = float(np.mean(paired_pct(rows, base, *best, "mean_robot_cong") or [float("nan")]))
        edl_pct = float(np.mean(paired_pct(rows, base, *best, "energy_per_delivery") or [float("nan")]))
        # cost
        bw = [b["wall_s"] for b in base.values() if b.get("wall_s") is not None]
        tw = [r["wall_s"] for r in treats if r.get("wall_s") is not None]
        ratio = (np.mean(tw) / np.mean(bw)) if bw and tw and np.mean(bw) > 0 else float("nan")
        add("## TL;DR")
        add("")
        add(f"> 데이터가 고른 최선 고정설정 **{bm} md{bmd} λ{bl:g}** (gamma {len(gammas)}개 평균, paired n={n_cells}):")
        add(f"> **처리량 {best_pct:+.1f}%** (95% CI ±{best_ci:.1f}, p {_fmt_p(best_p)}), "
            f"혼잡(cong@r) **{cong_pct:+.1f}%**, 배송당 에너지 **{edl_pct:+.1f}%**.")
        add(f"> 비용은 baseline 대비 계획시간 **≈{ratio:.0f}배** (§6). "
            f"효과크기는 argmax가 아닌 고정설정+CI(§2) 기준으로 보고할 것.")
        add("")
        add("---")
        add("")

    # ---- §1 config + baseline table ----
    add("## 1. 실험 구성")
    add("")
    counts = meta.get("counts") or _sorted_set(rows, "num_agents")
    fracs = meta.get("fracs") or sorted({b["frac"] for b in base.values() if b.get("frac") is not None})
    cong_per_cell = len(modes) * len(mds) * len(gammas) * len(lams)
    add(f"- **{n_cells} 셀** = agent {list(counts)} × frac {list(fracs)}")
    add(f"- 셀당: baseline 1 + 혼잡 설정 {cong_per_cell} "
        f"(= mode{list(modes)} × min_depth{list(mds)} × gamma{[float(g) for g in gammas]} × λ{[float(x) for x in lams]})")
    if meta:
        add(f"- horizon {meta.get('horizon','?')}, predict_every {meta.get('predict_every','?')}, "
            f"base_seed {meta.get('base_seed','?')} (셀 내 baseline과 공유 → paired 비교)")
    add(f"- 총 {len(rows)} 런 (= {n_cells} × {cong_per_cell + 1})"
        + (f", 완료 {meta['cells_completed']}/{meta.get('cells','?')} 셀" if "cells_completed" in meta else ""))
    add("")
    add("| 셀 | agents | frac | baseline 배송 | baseline cong@r |")
    add("|----|--------|------|--------------|-----------------|")
    for ep in cells:
        b = base[ep]
        add(f"| {ep} | {b['num_agents']} | {b['frac']:g} | {b['deliveries']} | "
            f"{(b['mean_robot_cong'] or 0):.0f} |")
    add("")
    add("---")
    add("")

    # ---- §2 effect sizes per min_depth ----
    add("## 2. 핵심 효과크기 (gamma 평균, paired, 95% CI)")
    add("")
    add(f"각 (mode, λ)에 대해 셀별 baseline 대비 차이를 gamma {len(gammas)}개 평균 후 {n_cells}셀 통계. "
        "argmax/체리픽 없는 정직한 추정치.")
    add("")
    for md in mds:
        add(f"**min_depth = {md}**")
        add("")
        add("| 설정 | Δ배송 | %배송 (95% CI) | p | %cong@r | %e/dlv | %p99 |")
        add("|------|------:|:--------------:|:--:|:------:|:-----:|:----:|")
        for mode in modes:
            for lam in lams:
                dabs = paired_abs(rows, base, mode, md, lam, "deliveries")
                if not dabs:
                    continue
                dpct = paired_pct(rows, base, mode, md, lam, "deliveries")
                mean_pct, ci, p, _ = mean_ci_p(dpct)
                cong = np.mean(paired_pct(rows, base, mode, md, lam, "mean_robot_cong") or [float("nan")])
                edl = np.mean(paired_pct(rows, base, mode, md, lam, "energy_per_delivery") or [float("nan")])
                p99 = np.mean(paired_pct(rows, base, mode, md, lam, "p99_cong") or [float("nan")])
                tag = f"**{mode} λ{lam:g}**" if best == (mode, md, lam) else f"{mode} λ{lam:g}"
                add(f"| {tag} | {np.mean(dabs):+.0f} | {mean_pct:+.1f}% (±{ci:.1f}) | {_fmt_p(p)} "
                    f"| {cong:+.1f}% | {edl:+.1f}% | {p99:+.1f}% |")
        add("")
    add("> **읽는 법:** %배송이 양수이고 p<0.05면 처리량이 유의하게 늘었다는 뜻. "
        "cong@r·p99가 음수면 혼잡이 줄어든 것. e/dlv 음수 = 배송당 에너지 효율 개선. "
        "굵게 표시된 행이 이 런에서 Δ배송이 가장 큰 고정설정.")
    add("")
    add("---")
    add("")

    # ---- §3 energy total vs efficiency ----
    add("## 3. 에너지 — \"효율\"이지 \"총량\"이 아님 ⚠️")
    add("")
    add("| 설정 | %처리량 | %총 에너지 | %배송당 에너지(효율) |")
    add("|------|:-------:|:----------:|:--------------------:|")
    for c in sorted(combo_eff, key=combo_eff.get, reverse=True)[:4]:
        mode, md, lam = c
        dv = np.mean(paired_pct(rows, base, *c, "deliveries") or [float("nan")])
        en = np.mean(paired_pct(rows, base, *c, "energy") or [float("nan")])
        edl = np.mean(paired_pct(rows, base, *c, "energy_per_delivery") or [float("nan")])
        add(f"| {mode} md{md} λ{lam:g} | {dv:+.1f}% | {en:+.1f}% | {edl:+.1f}% |")
    add("")
    add("우회 때문에 **총 이동거리(`energy`)는 보통 늘어난다**. 효율 결론은 항상 *배송당* 에너지"
        "(`energy_per_delivery`)로: \"총 이동량은 약간 더 쓰지만 배송이 더 늘어 배송당 에너지가 개선\". "
        "총에너지 감소라고 쓰면 틀림.")
    add("")
    add("---")
    add("")

    # ---- §4 argmax bias ----
    add("## 4. argmax 편향 (winner's curse)")
    add("")
    # best-lambda-per-gamma (grid_eval headline), averaged
    blg = []
    for mode in modes:
        for md in mds:
            for g in gammas:
                best_g = None
                for lam in lams:
                    diffs = []
                    for ep, b in base.items():
                        rs = [r for r in rows if r["episode"] == ep and r["depth_mode"] == mode
                              and r["min_depth"] == md and r["gamma"] == g and r["weight"] == lam]
                        if rs and b.get("deliveries") is not None and rs[0].get("deliveries") is not None:
                            diffs.append(rs[0]["deliveries"] - b["deliveries"])
                    if diffs:
                        m = float(np.mean(diffs))
                        best_g = m if best_g is None else max(best_g, m)
                if best_g is not None:
                    blg.append(best_g)
    # single global max cell
    gmax, gmax_lbl = float("-inf"), ""
    for r in treats:
        b = base.get(r["episode"])
        if not b or r.get("deliveries") is None or b.get("deliveries") is None:
            continue
        d = r["deliveries"] - b["deliveries"]
        if d > gmax:
            gmax = d
            gmax_lbl = f"{r['depth_mode']} md{r['min_depth']} γ{r['gamma']:g} λ{r['weight']:g} (cell {r['episode']})"
    best_fixed_val = combo_eff[best] if best else float("nan")
    add("| 추정 방식 | Δ배송 |")
    add("|-----------|------:|")
    if best:
        add(f"| 정직한 고정설정 ({best[0]} md{best[1]} λ{best[2]:g}, gamma 평균) | **{best_fixed_val:+.0f}** |")
    if blg:
        add(f"| best-λ per gamma 평균 (grid_eval 헤드라인) | {np.mean(blg):+.0f} |")
    if math.isfinite(gmax):
        add(f"| 단일 전역 최댓값 셀 (`{gmax_lbl}`) | **{gmax:+.0f}** |")
    add("")
    if best and math.isfinite(gmax):
        infl = gmax - best_fixed_val
        pct = (100.0 * infl / best_fixed_val) if best_fixed_val else float("nan")
        add(f"전역 최댓값을 집으면 고정설정 대비 **약 {infl:+.0f} ({pct:+.0f}%)** 부풀려진다. "
            "→ **보고 수치는 고정설정+CI(§2)** 로 쓸 것.")
    add("")
    add("---")
    add("")

    # ---- §5 density regression ----
    if best:
        add("## 5. 밀도(에이전트 수) vs 상대 이득")
        add("")
        xs, ys, labels = [], [], []
        for ep, b in base.items():
            t = gamma_avg(rows, ep, *best, "deliveries")
            if t is None or not b.get("deliveries"):
                continue
            pct = 100.0 * (t - b["deliveries"]) / b["deliveries"]
            xs.append(b["num_agents"])
            ys.append(pct)
            labels.append((ep, b["num_agents"], b["frac"], b["mean_robot_cong"], pct))
        add(f"`{best[0]} md{best[1]} λ{best[2]:g}` 셀별 %개선:")
        add("")
        add("| 셀 | agents | frac | baseline cong@r | %개선 |")
        add("|----|--------|------|-----------------|------:|")
        for ep, na, fr, cg, pct in sorted(labels):
            add(f"| {ep} | {na} | {fr:g} | {(cg or 0):.0f} | {pct:+.1f}% |")
        add("")
        r, pr, n = pearson(xs, ys)
        if math.isfinite(r):
            direction = "감소" if r < 0 else "증가"
            sig = "유의" if (pr == pr and pr < 0.05) else "비유의"
            add(f"- **%개선 vs num_agents: r = {r:.2f}, p = {_fmt_p(pr)} ({sig})** "
                f"— 밀도가 높을수록 *상대* 이득은 {direction} 경향.")
            add("- 절대 배송 수는 더 늘 수 있어도 % 이득은 다를 수 있으니 \"고밀도에서 더 빛난다\"는 서사는 회귀로 검증 후에만.")
        else:
            add(f"- 셀 수가 적어 회귀가 불안정(n={n}). 셀/seed를 늘리면 신뢰도가 올라간다.")
        add("")
        add("---")
        add("")

    # ---- §6 cost ----
    add("## 6. 비용")
    add("")
    bw = [b["wall_s"] for b in base.values() if b.get("wall_s") is not None]
    tw = [r["wall_s"] for r in treats if r.get("wall_s") is not None and r.get("weight")]
    preds = [r["preds"] for r in treats if r.get("preds") is not None]
    add("| | baseline | 예측 ON | 배수 |")
    add("|--|---------:|--------:|-----:|")
    if bw and tw:
        ratio = np.mean(tw) / np.mean(bw) if np.mean(bw) > 0 else float("nan")
        add(f"| wall_s (에피소드당 계획시간) | {np.mean(bw):.1f}s | {np.mean(tw):.1f}s | **≈{ratio:.0f}×** |")
    if preds:
        add(f"| 신경망 추론 횟수 | 0 | {int(np.median(preds))} | — |")
    add("")
    add("신경망 추론이 계획시간을 지배한다. **\"처리량 향상을 N배 계산비로 산다\"** 가 솔직한 손익 — "
        "실시간 배치라면 `predict_every`를 키워 비용을 깎을 수 있는지가 1순위 질문.")
    add("")
    add("---")
    add("")

    # ---- §7 recommendations (auto + general) ----
    add("## 7. 권고 & 다음 스텝")
    add("")
    if best:
        add(f"- **이 런 기준 최선 고정설정:** `{best[0]} md{best[1]} λ{best[2]:g}` "
            f"(처리량 {best_pct:+.1f}%, p {_fmt_p(best_p)}).")
    add("- **효과크기는 argmax 말고 고정설정 + 95% CI(§2)** 로 보고.")
    add("- **에너지는 \"배송당\"(§3)** 으로 — 총량 감소라고 쓰지 말 것.")
    add("- **p99(혼잡 꼬리)** 강조 — 시스템 안정성과 직결.")
    add("- 다음 런: `predict_every` 스윕(비용 대비 효과 곡선), λ 범위 확장(정점 확인), agent/seed 추가(CI 축소).")
    add("")
    add("---")
    add("")
    add("### 재현 (수치 산출 방법)")
    add(f"- 출처: `metrics.csv` ({len(rows)}행). 생성기: `calm/evaluation/make_analysis_summary.py`.")
    add("- 효과크기: 셀별 paired 차이 `treat − baseline`(같은 seed)를 gamma 평균한 뒤, "
        f"{n_cells}셀에 대해 평균 ± `t.ppf(0.975, df={max(n_cells-1,0)})·SE`, 1-sample paired t-test.")
    add("- argmax 행은 (mode,md,gamma)별 λ 최댓값 / 전역 단일 최댓값. 회귀는 Pearson r (n=셀 수).")
    add("- t-분포·incomplete-beta는 본 스크립트에 직접 구현(scipy 불필요).")
    add("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# static glossary (copied into each run folder so it is self-contained)
# ---------------------------------------------------------------------------
METRICS_GLOSSARY = """# 지표 용어집 (metrics glossary)

`metrics.csv`의 각 컬럼이 **무엇을 재는가 → 어느 방향이 좋은가 → 현장에서 무슨 이득인가**.
이 런의 실제 변화량은 같은 폴더 [`analysis_summary.md`](./analysis_summary.md) 참조
(아래 괄호 안 숫자는 참고용 예시 런 기준).

---

## ① 처리량·효율 (핵심 목표)

| 컬럼 | 잰 것 | 좋은 방향 | 이득 |
|------|-------|:--------:|------|
| `deliveries` | 시간 내 완료한 배송 수 = 처리량 | **↑** | 같은 로봇 수로 더 많은 주문 처리 → 산출↑, 또는 같은 작업에 더 적은 로봇 → 투자↓ |
| `energy` | 전 로봇 이동거리 합(맨해튼) | **↓** *(부차적)* | 배터리 소모·충전 다운타임·기계 마모↓. **단독 판단 금지** — 일을 적게 해도 낮아짐 |
| `energy_per_delivery` | 총 에너지 ÷ 배송 수 = **진짜 효율** | **↓** | 산출 1단위당 비용↓. 작업량과 무관하게 효율만 떼어 보는 정직한 지표. 운영비 직결 |

> **함정 주의:** 혼잡을 피해 우회하면 `energy`(총량)는 ↑ 한다.
> 그러나 배송이 더 늘어 `energy_per_delivery`는 ↓ 할 수 있다(효율 개선). 효율 결론은 항상 `energy_per_delivery`로.

---

## ② 혼잡·분산 (흐름의 질)

| 컬럼 | 잰 것 | 좋은 방향 | 이득 |
|------|-------|:--------:|------|
| `mean_robot_cong` | 로봇이 실제로 겪은 평균 혼잡도 | **↓** | 멈춤·서행(stop-and-go)↓ → 부드러운 주행, 대기·큐잉↓, 교착 위험↓ |
| `p99_cong` | 혼잡 99분위 = 꼬리(최악급) | **↓** | **시스템 안정성의 핵심.** 연쇄 지연·준교착을 일으키는 "가장 심한 정체" 완화. 평균보다 중요할 때 많음 |
| `peak_cong` | 혼잡 절대 최댓값 | **↓** | 그리드락(전면 정지) 상한선↓ → 안전 마진·예측가능성 |
| `density_uniformity` | 로봇이 공간에 얼마나 고르게 퍼졌나(엔트로피, 0~1) | **↑** | 핫스팟 없음 → 병목 분산, 바닥·통로 국소 마모 분산, 트래픽 예측 쉬움 |
| `occ_cv` | 칸별 점유의 편차/평균(변동계수) | **↓** | `density_uniformity`의 동전 뒷면 — 낮을수록 특정 칸 과밀 없이 통로를 골고루 사용 |

> `density_uniformity ↑` 와 `occ_cv ↓` 는 **같은 말**(둘 다 "고르게 분산").
> 혼잡은 평균(`mean_robot_cong`) → 꼬리(`p99_cong`) → 최댓값(`peak_cong`) 으로 단계적으로 본다.

---

## ③ 안전

| 컬럼 | 잰 것 | 좋은 방향 | 이득 |
|------|-------|:--------:|------|
| `collisions` | 로봇 간 충돌 횟수 | **0** | 장비 파손·수동 복구·라인 정지 없음. PIBT가 0을 *보장* — 혼잡 가중치는 경로 모양만 바꿀 뿐 안전엔 영향 없음 |

---

## ④ 비용 (품질이 아니라 대가)

| 컬럼 | 잰 것 | 좋은 방향 | 의미 |
|------|-------|:--------:|------|
| `preds` | 에피소드당 신경망 추론 횟수 | **↓** | 계산비 프록시. 0 = vanilla PIBT, 클수록 매 스텝 예측 |
| `wall_s` | 계획에 걸린 실제 벽시계 시간 | **↓** | 실시간 배치 가능성. 예측 ON이면 수십 배로 뛴다 = 이 방법의 최대 약점 |

---

## 묶어서 보는 법 (지표 간 관계)

- **혼잡(↓) ↔ 처리량(↑) 은 보통 상충.** λ를 키우면 혼잡 지표는 계속 좋아지지만(`cong`/`p99`/`occ_cv` ↓) 배송은 어느 지점부터 무너진다. **둘 다 좋아지는 구간이 스윗스폿.**
- **`energy` 단독은 함정** — 우회로 ↑ 한다. 반드시 `energy_per_delivery`로 환산해 판단.
- **평균보다 꼬리(`p99`/`peak`)가 더 의미 있을 때가 많다** — 시스템을 멈추게 하는 건 평균이 아니라 최악의 정체. 보고 시 `p99` 강조가 설득력 큼.
- **모드 선택:** `frontload` = 처리량 위주(혼잡 소폭↓, λ에 강건) / `peaked` = 혼잡 위주(같은 처리량에 혼잡 더↓, 단 λ 작은 구간만 안전).

---

## 컬럼 한눈 요약

| 컬럼 | 좋은 방향 | 한 줄 |
|------|:--------:|------|
| `deliveries` | ↑ | 처리량 |
| `energy` | ↓* | 총 이동거리(부차적) |
| `energy_per_delivery` | ↓ | 효율(핵심) |
| `mean_robot_cong` | ↓ | 평균 혼잡 |
| `p99_cong` | ↓ | 혼잡 꼬리(안정성) |
| `peak_cong` | ↓ | 혼잡 최댓값 |
| `density_uniformity` | ↑ | 고른 분산 |
| `occ_cv` | ↓ | 고른 분산(역지표) |
| `collisions` | 0 | 안전(보장) |
| `preds` | ↓ | 추론 횟수(비용) |
| `wall_s` | ↓ | 계획 시간(비용) |

\\* `energy`는 단독이 아니라 `deliveries`와 함께 `energy_per_delivery`로 환산해 해석.
"""


def write_run_reports(run_dir) -> List[Path]:
    """Write analysis_summary.md + metrics_glossary.md into `run_dir`. Returns the paths."""
    run_dir = Path(run_dir)
    written = []
    summary = build_analysis_summary(run_dir)
    sp = run_dir / "analysis_summary.md"
    sp.write_text(summary, encoding="utf-8")
    written.append(sp)
    gp = run_dir / "metrics_glossary.md"
    gp.write_text(METRICS_GLOSSARY, encoding="utf-8")
    written.append(gp)
    return written


def main() -> None:
    import sys
    args = sys.argv[1:]
    if not args:
        raise SystemExit("usage: python make_analysis_summary.py <run_dir> [<run_dir> ...]")
    for d in args:
        for p in write_run_reports(d):
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
