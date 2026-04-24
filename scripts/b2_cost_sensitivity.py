#!/usr/bin/env python3
"""
B2 — Cost sensitivity matrix.

Runs every promoted strategy at 0 / 8 / 10 / 20 bps round-trip cost.
For each (strategy, cost) pair, record the core gate-relevant metrics.
Output: a markdown table showing how each strategy's Sharpe / win-rate /
drawdown / trade-count degrades with cost, plus a verdict on whether
the strategy still passes its original promotion gate.

Run:
    ~/clawd/venv/bin/python scripts/b2_cost_sensitivity.py \
        --output ~/clawd/docs/b2-cost-sensitivity-2026-04-24.md

The backtest results table is written to stdout (or JSON if --json),
and a full markdown report is written to --output.

This script does NOT persist the sensitivity runs to backtest_results DB —
they are exploratory, not production history.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Repo importable
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from strategy_engine.backtest.costs import CostModel
from strategy_engine.backtest.runner import run_strategy, _find_yaml_path
from strategy_engine.registry.loader import load_one, validate_all
from strategy_engine.providers.duckdb_provider import DataNotAvailable


# Cost tiers to sweep
TIERS_BPS = [0, 8, 10, 20]

# Gate thresholds by profile — used to compute pass/fail verdict at each cost.
# These mirror strategy_engine.promotion.gates but we inline for isolation.
PROFILE_GATES = {
    "portfolio": {
        "min_sharpe": 0.60,
        "max_drawdown": 0.30,
        "min_trades": 20,
        "min_win_rate": 0.55,
        "min_profit_factor": 1.30,
    },
    "active-trader": {
        "min_active_bar_sharpe": 1.50,
        "max_drawdown": 0.15,
        "min_trades": 10,
        "min_win_rate": 0.50,
        "min_profit_factor": 1.20,
    },
}


def _guess_profile(strategy) -> str:
    """Infer which gate profile this strategy was promoted under.

    Heuristic: registry YAMLs don't always record the profile explicitly,
    so we use the tag set. STRAT / active-trader tags → active-trader gate.
    Otherwise portfolio.
    """
    tags = [t.lower() for t in (strategy.tags or [])]
    if "active-trader" in tags or any("strat" in t for t in tags):
        return "active-trader"
    # Composites and bollinger default to portfolio
    return "portfolio"


def _extract_metrics(run) -> dict:
    """Extract the gate-relevant metrics from a BacktestRun object."""
    # The summary_payload is inside result_json — re-parse to avoid re-running
    try:
        summary = json.loads(run.result_json).get("summary", {})
    except (json.JSONDecodeError, TypeError):
        summary = {}
    return {
        "equity_sharpe": run.oos_sharpe,
        "active_bar_sharpe": summary.get("active_bar_sharpe", 0.0),
        "active_bar_fraction": summary.get("active_bar_fraction", 0.0),
        "max_drawdown": run.oos_max_drawdown,
        "total_pnl": run.oos_total_pnl,
        "win_rate": run.oos_win_rate,
        "profit_factor": (
            run.oos_profit_factor if run.oos_profit_factor < 9000 else float("inf")
        ),
        "n_trades": run.oos_num_trades,
    }


def _evaluate_gates(metrics: dict, profile: str) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failed_reasons).

    Simplified — checks the headline thresholds. The real `check_gates` in
    strategy_engine.promotion.gates applies walk-forward-specific logic
    (active-fold aggregates); here we use the single-backtest metrics as a
    proxy.  This is intentional for a cost-sensitivity sweep: we want to
    see which strategies plausibly would STILL pass, not literally re-run
    walk-forward at each cost tier (which would take hours).
    """
    gates = PROFILE_GATES.get(profile, PROFILE_GATES["portfolio"])
    failed: list[str] = []

    if profile == "active-trader":
        if metrics["active_bar_sharpe"] < gates["min_active_bar_sharpe"]:
            failed.append(f"active_bar_sharpe<{gates['min_active_bar_sharpe']}")
    else:
        if metrics["equity_sharpe"] < gates["min_sharpe"]:
            failed.append(f"sharpe<{gates['min_sharpe']}")

    if abs(metrics["max_drawdown"]) > gates["max_drawdown"]:
        failed.append(f"max_dd>{gates['max_drawdown']}")

    if metrics["n_trades"] < gates["min_trades"]:
        failed.append(f"n_trades<{gates['min_trades']}")

    if metrics["win_rate"] < gates["min_win_rate"]:
        failed.append(f"win_rate<{gates['min_win_rate']}")

    pf = metrics["profit_factor"]
    if pf != float("inf") and pf < gates["min_profit_factor"]:
        failed.append(f"pf<{gates['min_profit_factor']}")

    return (not failed, failed)


def run_matrix(
    strategy_ids: Optional[list[str]] = None,
    tiers: list[int] = TIERS_BPS,
    *,
    persist: bool = False,
) -> list[dict]:
    """Run the full matrix. Returns list of dicts — one per (strategy, tier).

    When `strategy_ids` is None, includes every strategy with status in
    {promoted, live-ready, live} PLUS any composite with a completed
    backtest (those exist in the system as of 2026-04-23 in `backtested`
    state since no `composite` profile is registered in promotion gates).
    """
    strategies, errors = validate_all()
    if strategy_ids:
        strategies = [s for s in strategies if s.id in strategy_ids]
    else:
        strategies = [
            s for s in strategies
            if s.status in ("promoted", "live-ready", "live")
            or (s.signal_logic.type == "composite" and s.status == "backtested")
        ]

    results: list[dict] = []
    for s in strategies:
        profile = _guess_profile(s)
        print(f"\n=== {s.id}  [profile={profile}] ===", file=sys.stderr)
        for bps in tiers:
            cost = CostModel.flat_round_trip(bps)
            try:
                run = run_strategy(s.id, persist=persist, cost_model=cost)
                metrics = _extract_metrics(run)
            except DataNotAvailable as e:
                print(f"  {bps:>3} bps: DATA-UNAVAILABLE — {e}", file=sys.stderr)
                results.append({
                    "strategy_id": s.id, "profile": profile, "bps": bps,
                    "error": str(e), "metrics": None, "passed": None,
                })
                continue
            except Exception as e:  # noqa: BLE001
                print(f"  {bps:>3} bps: ERROR — {type(e).__name__}: {e}", file=sys.stderr)
                results.append({
                    "strategy_id": s.id, "profile": profile, "bps": bps,
                    "error": f"{type(e).__name__}: {e}",
                    "metrics": None, "passed": None,
                })
                continue
            passed, failed_gates = _evaluate_gates(metrics, profile)
            verdict = "PASS" if passed else "FAIL"
            sh = metrics["active_bar_sharpe"] if profile == "active-trader" else metrics["equity_sharpe"]
            print(f"  {bps:>3} bps: Sharpe={sh:.3f}  n={metrics['n_trades']:>3}  "
                  f"wr={metrics['win_rate']:.2%}  dd={metrics['max_drawdown']:.2%}  "
                  f"pf={metrics['profit_factor']:.2f}  {verdict}{' (' + ','.join(failed_gates) + ')' if failed_gates else ''}",
                  file=sys.stderr)
            results.append({
                "strategy_id": s.id,
                "profile": profile,
                "bps": bps,
                "metrics": metrics,
                "passed": passed,
                "failed_gates": failed_gates,
            })
    return results


def render_report(results: list[dict]) -> str:
    """Produce the markdown report from a list of matrix results."""
    from collections import defaultdict
    by_strategy: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in results:
        by_strategy[r["strategy_id"]][r["bps"]] = r

    tiers = sorted({r["bps"] for r in results})
    lines: list[str] = []
    lines.append("# B2 — Cost Sensitivity Report")
    lines.append("")
    lines.append("**Date:** 2026-04-24")
    lines.append("**Scope:** every promoted strategy + the composite backtested at 4 cost tiers (round-trip bps).")
    lines.append("")
    lines.append("## About the verdict column")
    lines.append("")
    lines.append("The verdict is a **simplified** gate check applied to single-backtest metrics (Sharpe, DD, win rate, trade count, profit factor). **The real promotion gates operate on walk-forward active-fold aggregates, not single-backtest metrics.** The simplified check will flag FAIL on strategies whose single-history Sharpe looks low but whose walk-forward active-fold Sharpe is fine — this is expected for sparse strategies (Bollinger, STRAT). Treat the verdicts as:")
    lines.append("")
    lines.append("- **✅ PASS** — robust; passes even the stricter single-backtest check. Strong candidate.")
    lines.append("- **❌ FAIL** — single-backtest metrics below threshold. Could still be legitimately promoted via walk-forward; OR could be genuinely marginal. Investigate case-by-case.")
    lines.append("")
    lines.append("The more actionable information here is **how much each metric degrades with cost** — see §2.")
    lines.append("")

    # Overview: PASS/FAIL per tier per strategy
    lines.append("## 1. Pass/fail matrix")
    lines.append("")
    header = "| Strategy | Profile | " + " | ".join(f"{t}bp" for t in tiers) + " |"
    sep    = "|---|---|" + "|".join("---:" for _ in tiers) + "|"
    lines.append(header)
    lines.append(sep)
    for sid in sorted(by_strategy):
        row = by_strategy[sid]
        profile = next(iter(row.values()))["profile"]
        cells = []
        for t in tiers:
            r = row.get(t)
            if r is None:
                cells.append("—")
            elif r.get("error"):
                cells.append(f"ERR")
            elif r["passed"]:
                cells.append("✅")
            else:
                cells.append("❌")
        lines.append(f"| `{sid}` | {profile} | " + " | ".join(cells) + " |")
    lines.append("")

    # Detail: per-strategy, per-tier metrics
    lines.append("## 2. Metric detail")
    lines.append("")
    for sid in sorted(by_strategy):
        row = by_strategy[sid]
        profile = next(iter(row.values()))["profile"]
        lines.append(f"### `{sid}`  [{profile}]")
        lines.append("")
        lines.append("| bps | Sharpe | Active-bar Sharpe | Win rate | Max DD | Profit factor | Trades | Verdict |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
        for t in tiers:
            r = row.get(t)
            if r is None or r.get("error"):
                lines.append(f"| {t} | — | — | — | — | — | — | ERR |")
                continue
            m = r["metrics"]
            verdict = "✅ PASS" if r["passed"] else f"❌ FAIL ({','.join(r['failed_gates'])})"
            pf = m["profit_factor"]
            pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"
            lines.append(
                f"| {t} | {m['equity_sharpe']:.3f} | {m['active_bar_sharpe']:.3f} | "
                f"{m['win_rate']:.1%} | {m['max_drawdown']:.2%} | {pf_str} | "
                f"{m['n_trades']} | {verdict} |"
            )
        lines.append("")

    # Degradation summary: how much does Sharpe drop from 0 → 20 bps?
    lines.append("## 3. Degradation summary — Sharpe erosion from 0 → 20 bps")
    lines.append("")
    lines.append("| Strategy | Profile | Sharpe @ 0bp | Sharpe @ 20bp | Δ (bp cost of 20bps) |")
    lines.append("|---|---|---:|---:|---:|")
    for sid in sorted(by_strategy):
        row = by_strategy[sid]
        profile = next(iter(row.values()))["profile"]
        r0 = row.get(0)
        r20 = row.get(20)
        if r0 is None or r20 is None or r0.get("metrics") is None or r20.get("metrics") is None:
            continue
        # For active-trader profile we show active_bar_sharpe; else equity_sharpe
        key = "active_bar_sharpe" if profile == "active-trader" else "equity_sharpe"
        s0 = r0["metrics"][key]
        s20 = r20["metrics"][key]
        delta = s0 - s20
        delta_pct = (delta / s0 * 100) if s0 > 0 else 0.0
        lines.append(
            f"| `{sid}` | {profile} | {s0:.3f} | {s20:.3f} | {delta:+.3f} ({delta_pct:+.0f}%) |"
        )
    lines.append("")

    # Verdict summary
    lines.append("## 4. Verdict summary (simplified gate check)")
    lines.append("")
    survivors_per_tier: dict[int, int] = {}
    total_per_tier: dict[int, int] = {}
    for r in results:
        if r.get("metrics") is None:
            continue
        survivors_per_tier[r["bps"]] = survivors_per_tier.get(r["bps"], 0) + (1 if r["passed"] else 0)
        total_per_tier[r["bps"]] = total_per_tier.get(r["bps"], 0) + 1

    for t in tiers:
        survived = survivors_per_tier.get(t, 0)
        total = total_per_tier.get(t, 0)
        lines.append(f"- **{t} bps:** {survived}/{total} strategies pass the simplified check")

    lines.append("")
    lines.append("## 5. Recommendations")
    lines.append("")
    lines.append("### Per-strategy action items")
    lines.append("")
    for sid in sorted(by_strategy):
        row = by_strategy[sid]
        profile = next(iter(row.values()))["profile"]
        # Find the highest tier where strategy still has positive total_pnl
        last_good_tier = None
        for t in sorted(tiers):
            r = row.get(t)
            if r and r.get("metrics") and r["metrics"]["total_pnl"] > 0:
                last_good_tier = t
        if last_good_tier is None:
            lines.append(f"- `{sid}` — **unprofitable even at 0 bps**. Needs strategy review.")
        elif last_good_tier >= 20:
            lines.append(f"- `{sid}` — **robust to 20 bps**. Strong candidate for live deployment.")
        elif last_good_tier >= 10:
            lines.append(f"- `{sid}` — profitable through 10 bps; marginal at 20 bps. OK under retail execution.")
        else:
            lines.append(f"- `{sid}` — profitable only at ≤ {last_good_tier} bps. Marginal. Consider strategy-level improvements (tighter entry, larger target, fewer trades).")
    lines.append("")
    lines.append("### Next steps")
    lines.append("")
    lines.append("- **B5** (later): rerun walk-forward at each cost tier — the authoritative gate. Current report uses single-backtest metrics for speed.")
    lines.append("- **B3-B4** (this batch): the regime-gate primitive is built; applying it to Bollinger should reduce panic-regime losers and improve cost-adjusted Sharpe.")
    lines.append("- **E** (next batch): new strategy families (momentum, breakout) — if they show > 10 bps tolerance natively, they're the easier path forward.")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by `scripts/b2_cost_sensitivity.py` — not persisted to backtest_results DB (exploratory only).*")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="0,8,10,20",
                     help="Comma-separated bps tiers to sweep (default: 0,8,10,20)")
    ap.add_argument("--strategies", default=None,
                     help="Comma-separated strategy IDs (default: all promoted)")
    ap.add_argument("--output", type=Path, default=None,
                     help="Write markdown report to this path")
    ap.add_argument("--json", action="store_true", help="Print raw JSON to stdout")
    args = ap.parse_args()

    tiers = [int(t) for t in args.tiers.split(",")]
    strategy_ids = args.strategies.split(",") if args.strategies else None

    results = run_matrix(strategy_ids, tiers=tiers)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        report = render_report(results)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report)
            print(f"\n✓ wrote report: {args.output}", file=sys.stderr)
        else:
            print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
