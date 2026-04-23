"""
Backtest runner — glue between registry → data → strategy logic → results DB.

For v1 (Day 4), supports only signal_logic.type == 'bollinger-mean-reversion'.
STRAT / other types will be added later; runner will dispatch by type.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import hashlib
import json
import uuid

import duckdb
import pandas as pd

from .. import __version__ as ENGINE_VERSION
from ..config import BACKTEST_DB
from ..providers.duckdb_provider import load_ohlcv, load_multi_timeframe, DataNotAvailable
from ..registry.loader import load_one
from ..registry.schema import Strategy
from . import bollinger as bol
from .strat import (
    classify_bars,
    detect_patterns,
    compute_ftfc,
    StratParams,
    simulate_trades as strat_simulate,
    summarize as strat_summarize,
)


class BacktestError(Exception):
    pass


@dataclass
class BacktestRun:
    run_id: str
    strategy_id: str
    config_hash: str
    run_date: str
    start_date: str
    end_date: str
    oos_sharpe: float
    oos_max_drawdown: float
    oos_total_pnl: float
    oos_win_rate: float
    oos_profit_factor: float
    oos_num_trades: int
    num_windows: int
    cost_model: str
    symbols: str
    result_json: str
    engine_version: str
    registry_yaml_path: Optional[str] = None


def _config_hash(strategy: Strategy) -> str:
    payload = json.dumps(strategy.model_dump(), sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _find_yaml_path(strategy_id: str) -> Optional[Path]:
    from ..config import REGISTRY_DIR
    matches = list(REGISTRY_DIR.rglob(f"{strategy_id}.yaml"))
    return matches[0] if matches else None


def _resample_for_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Some registry timeframes don't map directly to a firstrate source
    (e.g. '1w' and '4h'). Provider returns us the correct bars, but
    the Bollinger logic needs the bars at the strategy's native timeframe.
    Provider already handles resampling (via _RESAMPLE_FROM), so this is a
    safety net / no-op for supported timeframes.
    """
    return df


def _resolve_window(
    strategy: Strategy,
    start_override: Optional[str | pd.Timestamp] = None,
    end_override: Optional[str | pd.Timestamp] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve the effective [start, end] window. Precedence:
      CLI override  >  strategy.backtest_window  >  None (full history)
    """
    start = start_override
    end = end_override
    if start is None and strategy.backtest_window and strategy.backtest_window.start:
        start = strategy.backtest_window.start
    if end is None and strategy.backtest_window and strategy.backtest_window.end:
        end = strategy.backtest_window.end
    return (str(start) if start else None, str(end) if end else None)


def _run_strat(
    strategy: Strategy,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Run a STRAT-pattern strategy. Returns (StratResult, bars)."""
    symbol = strategy.instruments[0]
    tf = strategy.timeframe
    sl = strategy.signal_logic

    # Load trade-timeframe bars
    trade_bars = load_ohlcv(symbol=symbol, timeframe=tf, start=start, end=end)
    if trade_bars.empty:
        raise DataNotAvailable(f"No bars for {symbol} {tf}")

    # Classify + detect
    classified = classify_bars(trade_bars)
    classified = detect_patterns(classified)

    # Filter to only the pattern this strategy cares about
    wanted_pattern = getattr(sl, "pattern", None)
    if wanted_pattern:
        classified["strat_pattern"] = classified["strat_pattern"].where(
            classified["strat_pattern"] == wanted_pattern, None
        )

    # Load higher-tf bars for FTFC (skip if disabled)
    require_ftfc = bool(getattr(sl, "require_ftfc", True))
    ftfc_timeframes = list(getattr(sl, "ftfc_timeframes", ["1mo", "1w", "1d", "1h"]))
    ftfc_threshold = float(getattr(sl, "ftfc_threshold", 0.75))

    ftfc_df = None
    if require_ftfc:
        # Include the trade timeframe itself in FTFC check
        all_tfs = list(dict.fromkeys([*ftfc_timeframes, tf]))  # dedupe, keep order
        htfs = load_multi_timeframe(symbol, all_tfs, start=start, end=end)
        ftfc_df = compute_ftfc(trade_bars, htfs, threshold=ftfc_threshold)

    params = StratParams(
        require_ftfc=require_ftfc,
        ftfc_threshold=ftfc_threshold,
        min_risk_reward=float(getattr(strategy.exit, "min_risk_reward", 1.5)),
        max_holding_bars=int(getattr(sl, "max_holding_bars", 20)),
    )

    trades, n_raw, n_ftfc = strat_simulate(classified, ftfc_df, params)
    result = strat_summarize(
        trades, n_raw, n_ftfc, trade_bars, strategy.capital_allocation, tf,
    )
    return result, trade_bars


def _run_bollinger(
    strategy: Strategy,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> tuple[bol.BollingerResult, pd.DataFrame]:
    symbol = strategy.instruments[0]  # v1 — single instrument strategies only
    tf = strategy.timeframe
    bars = load_ohlcv(symbol=symbol, timeframe=tf, start=start, end=end)
    if bars.empty:
        raise DataNotAvailable(f"No bars for {symbol} {tf} in [{start}, {end}]")
    params = bol.BollingerParams.from_strategy(strategy)
    result = bol.run_bollinger(
        bars,
        params,
        capital_allocation=strategy.capital_allocation,
        timeframe=tf,
    )
    return result, bars


def run_strategy(
    strategy_id: str,
    *,
    persist: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> BacktestRun:
    """
    Top-level entry point. Loads strategy YAML, dispatches by signal type,
    computes metrics, persists to backtest-results.duckdb.

    `start` / `end` override any `backtest_window` in the YAML.

    Returns a BacktestRun record.
    """
    yaml_path = _find_yaml_path(strategy_id)
    if not yaml_path:
        raise BacktestError(f"No YAML for strategy {strategy_id!r}")
    strategy = load_one(yaml_path)

    signal_type = strategy.signal_logic.type

    # Resolve effective window: CLI override > YAML > full history
    eff_start, eff_end = _resolve_window(strategy, start, end)

    if signal_type == "bollinger-mean-reversion":
        result, bars = _run_bollinger(strategy, start=eff_start, end=eff_end)
    elif signal_type == "strat-pattern":
        result, bars = _run_strat(strategy, start=eff_start, end=eff_end)
    else:
        raise BacktestError(
            f"Unsupported signal_type {signal_type!r} for {strategy_id}. "
            f"Supported: bollinger-mean-reversion, strat-pattern."
        )

    # Build BacktestRun record
    run_id = f"bt-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    start_date = bars.index.min().date().isoformat() if not bars.empty else ""
    end_date = bars.index.max().date().isoformat() if not bars.empty else ""

    # Per-trade detail + summary serialization — shape differs per signal_type
    if signal_type == "bollinger-mean-reversion":
        trades_payload = [
            {
                "signal_date": t.signal_date.isoformat(),
                "signal_price": t.signal_price,
                "second_half_hit": t.second_half_entry_price is not None,
                "second_half_entry_price": t.second_half_entry_price,
                "second_half_entry_date": t.second_half_entry_date.isoformat()
                if t.second_half_entry_date is not None else None,
                "avg_entry_price": t.avg_entry_price,
                "exit_date": t.exit_date.isoformat(),
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "holding_bars": t.holding_bars,
                "pct_return": t.pct_return,
                "target_hit": t.target_hit,
                "trough_pct": t.trough_pct,
            }
            for t in result.trades
        ]
        summary_payload = {
            "n_signals": result.n_signals,
            "avg_return": result.avg_return,
            "median_return": result.median_return,
            "win_rate": result.win_rate,
            "equity_sharpe": result.equity_sharpe,
            "equity_max_drawdown": result.equity_max_drawdown,
            "equity_total_return": result.equity_total_return,
            "equity_ann_return": result.equity_ann_return,
            "active_bar_sharpe": result.active_bar_sharpe,
            "active_bars": result.active_bars,
            "active_bar_fraction": result.active_bar_fraction,
            "worst_trade_trough": result.worst_trade_trough,
            "profit_factor": (
                result.profit_factor if result.profit_factor != float("inf") else "inf"
            ),
            "total_pnl_pct": result.total_pnl_pct,
            "second_half_hit_rate": result.second_half_hit_rate,
        }
    else:  # strat-pattern
        trades_payload = [
            {
                "pattern": t.pattern,
                "direction": t.direction,
                "signal_date": t.signal_date.isoformat(),
                "entry_date": t.entry_date.isoformat(),
                "entry_price": t.entry_price,
                "target_price": t.target_price,
                "stop_price": t.stop_price,
                "exit_date": t.exit_date.isoformat(),
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "holding_bars": t.holding_bars,
                "pct_return": t.pct_return,
                "risk_reward_planned": t.risk_reward_planned,
                "ftfc_score": t.ftfc_score,
            }
            for t in result.trades
        ]
        summary_payload = {
            "n_signals_raw": result.n_signals_raw,
            "n_signals_ftfc": result.n_signals_ftfc,
            "n_trades": result.n_trades,
            "win_rate": result.win_rate,
            "avg_return": result.avg_return,
            "equity_sharpe": result.equity_sharpe,
            "equity_max_drawdown": result.equity_max_drawdown,
            "equity_total_return": result.equity_total_return,
            "equity_ann_return": result.equity_ann_return,
            "active_bar_sharpe": result.active_bar_sharpe,
            "active_bars": result.active_bars,
            "active_bar_fraction": result.active_bar_fraction,
            "profit_factor": (
                result.profit_factor if result.profit_factor != float("inf") else "inf"
            ),
            "total_pnl_pct": result.total_pnl_pct,
            "target_hit_rate": result.target_hit_rate,
            "stop_hit_rate": result.stop_hit_rate,
        }
    result_json = json.dumps(
        {"summary": summary_payload, "trades": trades_payload},
        default=str,
    )

    # For the top-level `oos_num_trades`, use n_trades (both strategy types expose this)
    n_trades = getattr(result, "n_trades", 0)
    win_rate = getattr(result, "win_rate", 0.0)
    pf = getattr(result, "profit_factor", 0.0)

    run = BacktestRun(
        run_id=run_id,
        strategy_id=strategy.id,
        config_hash=_config_hash(strategy),
        run_date=datetime.now().astimezone().isoformat(timespec="seconds"),
        start_date=start_date,
        end_date=end_date,
        oos_sharpe=result.equity_sharpe,
        oos_max_drawdown=result.equity_max_drawdown,
        oos_total_pnl=result.equity_total_return,
        oos_win_rate=win_rate,
        oos_profit_factor=(pf if pf != float("inf") else 9999.0),
        oos_num_trades=n_trades,
        num_windows=1,
        cost_model="zero-cost",
        symbols=",".join(strategy.instruments),
        result_json=result_json,
        engine_version=ENGINE_VERSION,
        registry_yaml_path=str(yaml_path.relative_to(Path.home())) if yaml_path else None,
    )

    if persist:
        _persist(run)

    return run


def _persist(run: BacktestRun) -> None:
    con = duckdb.connect(str(BACKTEST_DB))
    try:
        # Get next id (auto-increment via max+1)
        next_id = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM backtest_results").fetchone()[0]
        con.execute(
            """
            INSERT INTO backtest_results (
                id, strategy_id, config_hash, run_date, start_date, end_date,
                oos_sharpe, oos_max_drawdown, oos_total_pnl, oos_win_rate,
                oos_profit_factor, oos_num_trades, num_windows, cost_model,
                symbols, result_json, engine_version, registry_yaml_path
            ) VALUES (
                ?, ?, ?, ?::TIMESTAMP, ?::DATE, ?::DATE,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?
            )
            """,
            [
                next_id,
                run.strategy_id,
                run.config_hash,
                run.run_date,
                run.start_date,
                run.end_date,
                run.oos_sharpe,
                run.oos_max_drawdown,
                run.oos_total_pnl,
                run.oos_win_rate,
                run.oos_profit_factor,
                run.oos_num_trades,
                run.num_windows,
                run.cost_model,
                run.symbols,
                run.result_json,
                run.engine_version,
                run.registry_yaml_path,
            ],
        )
        # Store our run_id label in config_hash? No — config_hash is already used.
        # Just track via strategy_id + run_date; run_id lives in the YAML link-back.
    finally:
        con.close()


def append_run_to_yaml(strategy_id: str, run_id: str, oos_sharpe: float) -> None:
    """
    After a successful run, append its run_id to the strategy's YAML
    `backtest_runs` list. Keeps the registry in sync with the results DB.
    """
    yaml_path = _find_yaml_path(strategy_id)
    if not yaml_path:
        return
    import yaml
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    runs = data.get("backtest_runs") or []
    if run_id in runs:
        return
    runs.append(run_id)
    data["backtest_runs"] = runs
    # Also update status if draft → backtested
    if data.get("status") == "draft":
        data["status"] = "backtested"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=100)
