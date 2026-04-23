"""`strategy-engine` CLI."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import click

from .config import REGISTRY_DIR
from .registry.loader import load_one, validate_all, RegistryError


@click.group()
def cli() -> None:
    """Trading-strategy registry + backtest + promotion + live-signal engine."""


# ── registry ─────────────────────────────────────────────────────────────────


@cli.group()
def registry() -> None:
    """Operate on the registry (`~/clawd/research/strategies/`)."""


@registry.command("list")
@click.option("--status", default=None)
@click.option("--category", default=None, help="Filter by top-level subdir (e.g. bollinger, strat)")
@click.option("--json", "as_json", is_flag=True)
def registry_list(status: str | None, category: str | None, as_json: bool) -> None:
    strategies, errors = validate_all()
    if category:
        strategies = [s for s in strategies if (REGISTRY_DIR / category) in s_paths_for(s, REGISTRY_DIR)]
    if status:
        strategies = [s for s in strategies if s.status == status]
    if as_json:
        click.echo(json.dumps([s.model_dump() for s in strategies], indent=2))
        if errors:
            click.echo("ERRORS:", err=True)
            for p, e in errors:
                click.echo(f"  {p}: {e}", err=True)
        return

    if not strategies and not errors:
        click.echo("No strategies in registry.")
        return

    for s in strategies:
        marker = {
            "draft": "·",
            "backtested": "◐",
            "promoted": "●",
            "live-ready": "▲",
            "live": "★",
            "retired": "✕",
            "archived": "□",
        }.get(s.status, "?")
        click.echo(f"{marker} [{s.status:<11}] {s.id:<45} {s.asset_class}/{s.timeframe}")

    if errors:
        click.echo(f"\n{len(errors)} error(s):", err=True)
        for p, e in errors:
            click.echo(f"  {p}: {e}", err=True)


def s_paths_for(strategy, root: Path) -> list[Path]:
    """Helper to determine which subdir a strategy was loaded from."""
    for p in root.rglob(f"{strategy.id}.yaml"):
        return [p.parent, *list(p.parents)]
    return []


@registry.command("show")
@click.argument("strategy_id")
def registry_show(strategy_id: str) -> None:
    matches = list(REGISTRY_DIR.rglob(f"{strategy_id}.yaml"))
    if not matches:
        click.echo(f"No strategy found with id {strategy_id!r}", err=True)
        sys.exit(1)
    if len(matches) > 1:
        click.echo(f"Multiple matches: {matches}", err=True)
        sys.exit(1)
    path = matches[0]
    try:
        strat = load_one(path)
    except RegistryError as e:
        click.echo(f"load failed: {e}", err=True)
        sys.exit(1)
    click.echo(json.dumps(strat.model_dump(), indent=2))


@registry.command("validate")
@click.option("--json", "as_json", is_flag=True)
def registry_validate(as_json: bool) -> None:
    strategies, errors = validate_all()
    if as_json:
        click.echo(json.dumps({
            "valid": len(strategies),
            "errors": [{"path": str(p), "message": e} for p, e in errors],
        }, indent=2))
    else:
        click.echo(f"Validated {len(strategies)} strategies.")
        if errors:
            click.echo(f"{len(errors)} error(s):")
            for p, e in errors:
                click.echo(f"  {p}: {e}")
    sys.exit(1 if errors else 0)


@registry.command("count")
def registry_count() -> None:
    strategies, _ = validate_all()
    counts: dict[str, int] = {}
    for s in strategies:
        counts[s.status] = counts.get(s.status, 0) + 1
    click.echo(f"Total: {len(strategies)}")
    for status, n in sorted(counts.items()):
        click.echo(f"  {status}: {n}")


# ── backtest ─────────────────────────────────────────────────────────────────


@cli.command("backtest")
@click.argument("strategy_id")
@click.option("--start", default=None, help="Start date YYYY-MM-DD (overrides YAML backtest_window)")
@click.option("--end", default=None, help="End date YYYY-MM-DD (overrides YAML backtest_window)")
@click.option("--no-persist", is_flag=True, help="Do not write to backtest-results.duckdb")
@click.option("--no-yaml-update", is_flag=True, help="Do not append run_id to strategy YAML")
@click.option("--json", "as_json", is_flag=True)
def backtest_cmd(
    strategy_id: str,
    start: str | None,
    end: str | None,
    no_persist: bool,
    no_yaml_update: bool,
    as_json: bool,
) -> None:
    """Run a backtest for a single strategy and persist the result.

    Precedence for date window: --start/--end flags > YAML backtest_window > full history.
    """
    from .backtest.runner import run_strategy, append_run_to_yaml, BacktestError
    from .providers.duckdb_provider import DataNotAvailable

    try:
        run = run_strategy(strategy_id, persist=not no_persist, start=start, end=end)
    except (BacktestError, DataNotAvailable) as e:
        click.echo(f"backtest failed: {e}", err=True)
        sys.exit(1)

    if not no_yaml_update and not no_persist:
        append_run_to_yaml(strategy_id, run.run_id, run.oos_sharpe)

    if as_json:
        click.echo(json.dumps({
            "run_id": run.run_id,
            "strategy_id": run.strategy_id,
            "oos_sharpe": run.oos_sharpe,
            "oos_win_rate": run.oos_win_rate,
            "oos_num_trades": run.oos_num_trades,
            "oos_max_drawdown": run.oos_max_drawdown,
            "oos_total_pnl": run.oos_total_pnl,
            "oos_profit_factor": run.oos_profit_factor,
            "start_date": run.start_date,
            "end_date": run.end_date,
            "persisted": not no_persist,
            "yaml_updated": not no_yaml_update and not no_persist,
        }, indent=2))
    else:
        click.echo(f"✓ backtest complete for {run.strategy_id}")
        click.echo(f"  run_id:          {run.run_id}")
        click.echo(f"  period:          {run.start_date} → {run.end_date}")
        click.echo(f"  Sharpe:          {run.oos_sharpe:.3f}")
        click.echo(f"  Win rate:        {run.oos_win_rate:.1%}")
        click.echo(f"  Trades:          {run.oos_num_trades}")
        click.echo(f"  Total P&L:       {run.oos_total_pnl:.2%}")
        click.echo(f"  Max DD:          {run.oos_max_drawdown:.2%}")
        click.echo(f"  Profit factor:   {run.oos_profit_factor:.2f}")
        if not no_persist:
            click.echo(f"  persisted to:    ~/clawd/data/backtest-results.duckdb")
        if not no_yaml_update and not no_persist:
            click.echo(f"  registry updated (backtest_runs +1, status: backtested)")


# ── walkforward ──────────────────────────────────────────────────────────────


@cli.command("walkforward")
@click.argument("strategy_id")
@click.option("--train-years", default=3, type=int)
@click.option("--test-years", default=1, type=int)
@click.option("--step-years", default=1, type=int)
@click.option("--start", default=None)
@click.option("--end", default=None)
@click.option("--json", "as_json", is_flag=True)
def walkforward_cmd(
    strategy_id: str,
    train_years: int,
    test_years: int,
    step_years: int,
    start: str | None,
    end: str | None,
    as_json: bool,
) -> None:
    """Run walk-forward cross-validation on a strategy."""
    from .backtest.walkforward import run_walkforward
    from .providers.duckdb_provider import load_ohlcv, DataNotAvailable
    from .registry.loader import load_one
    from .backtest.runner import _find_yaml_path, _resolve_window

    yaml_path = _find_yaml_path(strategy_id)
    if not yaml_path:
        click.echo(f"No YAML for strategy {strategy_id!r}", err=True)
        sys.exit(1)
    strategy = load_one(yaml_path)

    eff_start, eff_end = _resolve_window(strategy, start, end)
    try:
        bars = load_ohlcv(
            symbol=strategy.instruments[0],
            timeframe=strategy.timeframe,
            start=eff_start,
            end=eff_end,
        )
    except DataNotAvailable as e:
        click.echo(f"Data not available: {e}", err=True)
        sys.exit(1)

    # For STRAT, preload higher-timeframe bars once (used by FTFC across all folds)
    higher_tfs = None
    if strategy.signal_logic.type == "strat-pattern":
        from .providers.duckdb_provider import load_multi_timeframe
        sl = strategy.signal_logic
        if bool(getattr(sl, "require_ftfc", True)):
            ftfc_tfs = list(getattr(sl, "ftfc_timeframes", ["1mo", "1w", "1d", "1h"]))
            all_tfs = list(dict.fromkeys([*ftfc_tfs, strategy.timeframe]))
            higher_tfs = load_multi_timeframe(
                strategy.instruments[0], all_tfs, start=eff_start, end=eff_end,
            )

    wf = run_walkforward(
        strategy, bars,
        train_years=train_years, test_years=test_years, step_years=step_years,
        higher_timeframes=higher_tfs,
    )

    if as_json:
        click.echo(json.dumps({
            "strategy_id": strategy_id,
            "n_folds": wf.n_folds,
            "n_active_folds": wf.n_active_folds,
            "activation_rate": wf.activation_rate,
            "oos_all_mean_sharpe": wf.oos_all_mean_sharpe,
            "oos_active_mean_sharpe": wf.oos_active_mean_sharpe,
            "oos_active_std_sharpe": wf.oos_active_std_sharpe,
            "oos_active_min_sharpe": wf.oos_active_min_sharpe,
            "oos_worst_dd": wf.oos_worst_dd,
            "oos_total_trades": wf.oos_total_trades,
            "oos_active_mean_win_rate": wf.oos_active_mean_win_rate,
            "folds": [
                {
                    "fold": f.fold_index,
                    "train": f"{f.train_start.date()} → {f.train_end.date()}",
                    "test": f"{f.test_start.date()} → {f.test_end.date()}",
                    "test_trades": f.test_n_trades,
                    "test_sharpe": round(f.test_equity_sharpe, 3),
                    "test_dd": round(f.test_max_drawdown, 4),
                    "test_win_rate": round(f.test_win_rate, 3),
                }
                for f in wf.folds
            ],
        }, indent=2))
    else:
        click.echo(f"Walk-forward for {strategy_id}")
        click.echo(f"  folds:                 {wf.n_folds} ({train_years}y train / {test_years}y test)")
        click.echo(f"  active folds:          {wf.n_active_folds} ({wf.activation_rate:.0%} activation rate)")
        click.echo(f"  all-fold mean Sharpe:  {wf.oos_all_mean_sharpe:.3f}  (includes 0-trade folds)")
        click.echo(f"  ACTIVE mean Sharpe:    {wf.oos_active_mean_sharpe:.3f}  (quality when firing)")
        click.echo(f"  ACTIVE Sharpe std:     {wf.oos_active_std_sharpe:.3f}  (stability when firing)")
        click.echo(f"  ACTIVE min Sharpe:     {wf.oos_active_min_sharpe:.3f}  (worst active fold)")
        click.echo(f"  worst DD any fold:     {wf.oos_worst_dd:.2%}")
        click.echo(f"  total OOS trades:      {wf.oos_total_trades}")
        click.echo(f"  ACTIVE mean win rate:  {wf.oos_active_mean_win_rate:.1%}")
        click.echo()
        click.echo(f"  {'#':>3} {'train window':<25} {'test window':<25} {'trades':>7} {'Sharpe':>7} {'DD':>8} {'Win%':>6}")
        for f in wf.folds:
            click.echo(
                f"  {f.fold_index:>3} "
                f"{str(f.train_start.date())} → {str(f.train_end.date())}  "
                f"{str(f.test_start.date())} → {str(f.test_end.date())}  "
                f"{f.test_n_trades:>7} "
                f"{f.test_equity_sharpe:>7.3f} "
                f"{f.test_max_drawdown:>7.2%} "
                f"{f.test_win_rate:>6.1%}"
            )


# ── promote ──────────────────────────────────────────────────────────────────


@cli.command("promote")
@click.argument("strategy_id")
@click.option("--profile", default="portfolio", type=click.Choice(["portfolio", "active-trader"]),
              help="Gate profile: portfolio (full-bar Sharpe) or active-trader (active-bar Sharpe)")
@click.option("--train-years", default=3, type=int)
@click.option("--test-years", default=1, type=int)
@click.option("--step-years", default=1, type=int)
@click.option("--start", default=None)
@click.option("--end", default=None)
@click.option("--min-sharpe", default=None, type=float, help="Override Sharpe threshold for active profile")
@click.option("--max-dd", default=None, type=float)
@click.option("--min-trades", default=None, type=int)
@click.option("--update-yaml/--no-update-yaml", default=True)
@click.option("--json", "as_json", is_flag=True)
def promote_cmd(
    strategy_id: str,
    profile: str,
    train_years: int,
    test_years: int,
    step_years: int,
    start: str | None,
    end: str | None,
    min_sharpe: float | None,
    max_dd: float | None,
    min_trades: int | None,
    update_yaml: bool,
    as_json: bool,
) -> None:
    """Run walk-forward, apply promotion gates, optionally update YAML status."""
    from .backtest.walkforward import run_walkforward
    from .providers.duckdb_provider import load_ohlcv, DataNotAvailable
    from .registry.loader import load_one
    from .backtest.runner import _find_yaml_path, _resolve_window
    from .promotion.gates import check_gates, DEFAULT_THRESHOLDS
    import yaml

    yaml_path = _find_yaml_path(strategy_id)
    if not yaml_path:
        click.echo(f"No YAML for strategy {strategy_id!r}", err=True)
        sys.exit(1)
    strategy = load_one(yaml_path)

    eff_start, eff_end = _resolve_window(strategy, start, end)
    try:
        bars = load_ohlcv(
            symbol=strategy.instruments[0],
            timeframe=strategy.timeframe,
            start=eff_start, end=eff_end,
        )
    except DataNotAvailable as e:
        click.echo(f"Data not available: {e}", err=True)
        sys.exit(1)

    # STRAT needs higher timeframes for FTFC
    higher_tfs = None
    if strategy.signal_logic.type == "strat-pattern":
        from .providers.duckdb_provider import load_multi_timeframe
        sl = strategy.signal_logic
        if bool(getattr(sl, "require_ftfc", True)):
            ftfc_tfs = list(getattr(sl, "ftfc_timeframes", ["1mo", "1w", "1d", "1h"]))
            all_tfs = list(dict.fromkeys([*ftfc_tfs, strategy.timeframe]))
            higher_tfs = load_multi_timeframe(
                strategy.instruments[0], all_tfs, start=eff_start, end=eff_end,
            )

    wf = run_walkforward(
        strategy, bars,
        train_years=train_years, test_years=test_years, step_years=step_years,
        higher_timeframes=higher_tfs,
    )

    # CLI overrides populate into the thresholds dict (profile base applied in check_gates)
    overrides = {}
    if min_sharpe is not None:
        # Route to whichever Sharpe key matches the profile
        key = "min_active_bar_sharpe" if profile == "active-trader" else "min_active_sharpe"
        overrides[key] = min_sharpe
    if max_dd is not None:
        overrides["max_drawdown"] = max_dd
    if min_trades is not None:
        overrides["min_trades"] = min_trades

    decision = check_gates(strategy_id, wf, thresholds=overrides, profile=profile)

    if as_json:
        click.echo(json.dumps({
            "strategy_id": decision.strategy_id,
            "passed": decision.passed,
            "summary": decision.summary,
            "gates": [
                {"name": g.name, "passed": g.passed, "value": g.value,
                 "threshold": g.threshold, "reason": g.reason}
                for g in decision.gates
            ],
            "walkforward": {
                "n_folds": wf.n_folds,
                "oos_mean_sharpe": wf.oos_mean_sharpe,
                "oos_std_sharpe": wf.oos_std_sharpe,
                "oos_total_trades": wf.oos_total_trades,
            },
        }, indent=2))
    else:
        click.echo(f"Promotion check for {strategy_id}")
        click.echo(f"  walk-forward folds: {wf.n_folds}")
        click.echo()
        for g in decision.gates:
            marker = "✓" if g.passed else "✗"
            click.echo(f"  {marker} {g.name:<24} {g.reason}")
        click.echo()
        click.echo(f"  {decision.summary}")

    # Update YAML status
    if update_yaml:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if decision.passed:
            data["status"] = "promoted"
            data.setdefault("promotion", {})["decision_date"] = (
                __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds")
            )
            with yaml_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=100)
            click.echo(f"\n  → YAML updated: status = promoted")
        else:
            # Record failure annotation; don't change status (stays backtested/draft)
            data.setdefault("promotion", {})["last_check"] = {
                "date": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
                "passed": False,
                "failed_gates": [g.name for g in decision.gates if not g.passed],
            }
            with yaml_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=100)
            click.echo(f"\n  → YAML updated: promotion.last_check annotated (status unchanged)")

    sys.exit(0 if decision.passed else 1)


# ── detect (live signal detection) ───────────────────────────────────────────


@cli.command("detect")
@click.option("--strategy", "strategy_id", default=None, help="Check a single strategy by id")
@click.option("--all-promoted", is_flag=True, help="Check all strategies with status='promoted'")
@click.option("--force", is_flag=True, help="Ignore timeframe schedule (check every strategy regardless of bar-close alignment)")
@click.option("--no-persist", is_flag=True, help="Don't write to live-signals.duckdb")
@click.option("--notify/--no-notify", default=True, help="Send telegram/stdout notification on fire")
@click.option("--channel", default="auto", type=click.Choice(["auto", "telegram", "stdout"]))
def detect_cmd(
    strategy_id: str | None,
    all_promoted: bool,
    force: bool,
    no_persist: bool,
    notify: bool,
    channel: str,
) -> None:
    """Check for fired signals on latest bars. Writes to live-signals.duckdb."""
    from .live import (
        detect_signals_for_strategy,
        detect_all_promoted,
    )
    from .live.detector import persist_signal
    from .live.notification import notify_signal, format_signal

    if strategy_id:
        try:
            sig = detect_signals_for_strategy(strategy_id)
        except Exception as e:
            click.echo(f"detect failed: {e}", err=True)
            sys.exit(1)
        if sig is None:
            click.echo(f"  — no signal on latest bar for {strategy_id}")
            return
        click.echo("\n" + format_signal(sig) + "\n")
        if not no_persist:
            persist_signal(sig)
            click.echo(f"  persisted: signal_id={sig.signal_id}")
        if notify:
            used = notify_signal(sig, channel=channel)
            click.echo(f"  notified via: {used}")
        return

    if all_promoted:
        fired = detect_all_promoted(persist=not no_persist, respect_schedule=not force)
        if not fired:
            click.echo("  — no signals fired across promoted strategies")
            return
        click.echo(f"✓ {len(fired)} signal(s) fired:\n")
        for sig in fired:
            click.echo(format_signal(sig))
            click.echo()
            if notify:
                used = notify_signal(sig, channel=channel)
                click.echo(f"  notified via: {used}\n")
        return

    click.echo("specify --strategy <id> or --all-promoted", err=True)
    sys.exit(2)


if __name__ == "__main__":
    cli()
