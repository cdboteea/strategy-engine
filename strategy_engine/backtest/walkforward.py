"""
Walk-forward cross-validation.

Rolling [train_years] / [test_years] folds. Each fold trains params on train
window, tests on out-of-sample test window, records OOS metrics.

For param-free strategies (or fixed-param strategies), train window is not
used for tuning — we still run on train to compute in-sample baseline, then
test on OOS. This makes stability metrics (cross-fold Sharpe std) meaningful.

v1 handles: fixed-param bollinger-mean-reversion (no grid-search on train).
v2 (later): grid-search train → pick best params → apply to test.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from ..config import BARS_PER_YEAR
from ..registry.schema import Strategy
from .bollinger import (
    BollingerParams,
    run_bollinger,
    BollingerResult,
)
from .strat import (
    classify_bars,
    detect_patterns,
    compute_ftfc,
    StratParams,
    simulate_trades as strat_simulate,
    summarize as strat_summarize,
    StratResult,
)


@dataclass
class WalkForwardFold:
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    # Train-window metrics (for reference)
    train_n_trades: int = 0
    train_equity_sharpe: float = 0.0
    train_total_return: float = 0.0
    # OOS test-window metrics
    test_n_trades: int = 0
    test_equity_sharpe: float = 0.0              # full-bar Sharpe
    test_active_bar_sharpe: float = 0.0          # Sharpe restricted to invested bars
    test_active_bar_fraction: float = 0.0
    test_win_rate: float = 0.0
    test_max_drawdown: float = 0.0
    test_total_return: float = 0.0
    test_profit_factor: float = 0.0


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold] = field(default_factory=list)
    n_folds: int = 0
    # ALL-FOLD aggregates (empty folds count as 0) — honest "full lifecycle"
    oos_all_mean_sharpe: float = 0.0
    oos_worst_dd: float = 0.0               # deepest DD across any fold
    oos_total_trades: int = 0
    # ACTIVE-FOLD aggregates (folds with ≥1 trade) — meaningful "strategy when it fires"
    n_active_folds: int = 0
    activation_rate: float = 0.0            # n_active / n_folds
    oos_active_mean_sharpe: float = 0.0     # full-bar Sharpe, mean across active folds
    oos_active_std_sharpe: float = 0.0
    oos_active_min_sharpe: float = 0.0
    oos_active_mean_win_rate: float = 0.0
    # ACTIVE-BAR-SHARPE aggregates (per-fold) — the "trade quality when invested" signal
    oos_active_bar_mean_sharpe: float = 0.0  # mean of per-fold active-bar Sharpes
    oos_active_bar_std_sharpe: float = 0.0
    oos_active_bar_min_sharpe: float = 0.0
    oos_total_active_bars: int = 0
    oos_mean_active_bar_fraction: float = 0.0


def _slice_bars(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return bars.loc[(bars.index >= start) & (bars.index <= end)]


def _run_fold_bollinger(
    bars: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    params: BollingerParams,
    capital_allocation: float,
    timeframe: str,
) -> BollingerResult:
    slc = _slice_bars(bars, start, end)
    if len(slc) < params.lookback + 5:
        return BollingerResult()
    return run_bollinger(slc, params, capital_allocation=capital_allocation, timeframe=timeframe)


def _run_fold_strat(
    bars: pd.DataFrame,
    classified_full: pd.DataFrame,
    ftfc_full: Optional[pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    wanted_pattern: Optional[str],
    params: StratParams,
    capital_allocation: float,
    timeframe: str,
):
    """Run STRAT on a slice of pre-classified bars + pre-computed FTFC."""
    slc_bars = _slice_bars(bars, start, end)
    if len(slc_bars) < 10:
        return StratResult()
    # Slice the classified + FTFC frames to the window
    slc_classified = classified_full.loc[(classified_full.index >= start) & (classified_full.index <= end)].copy()
    slc_ftfc = None
    if ftfc_full is not None:
        slc_ftfc = ftfc_full.loc[(ftfc_full.index >= start) & (ftfc_full.index <= end)].copy()

    # Re-apply pattern filter since slice may be fresh
    if wanted_pattern:
        slc_classified["strat_pattern"] = slc_classified["strat_pattern"].where(
            slc_classified["strat_pattern"] == wanted_pattern, None
        )

    trades, n_raw, n_ftfc = strat_simulate(slc_classified, slc_ftfc, params)
    return strat_summarize(trades, n_raw, n_ftfc, slc_bars, capital_allocation, timeframe)


def _result_metrics(res) -> dict:
    """Pull common fields from either Bollinger or STRAT result."""
    pf = getattr(res, "profit_factor", 0.0)
    if pf == float("inf"):
        pf = 9999.0
    return {
        "n_trades": getattr(res, "n_trades", 0),
        "equity_sharpe": getattr(res, "equity_sharpe", 0.0),
        "equity_max_drawdown": getattr(res, "equity_max_drawdown", 0.0),
        "equity_total_return": getattr(res, "equity_total_return", 0.0),
        "win_rate": getattr(res, "win_rate", 0.0),
        "profit_factor": pf,
        "active_bar_sharpe": getattr(res, "active_bar_sharpe", 0.0),
        "active_bars": getattr(res, "active_bars", 0),
        "active_bar_fraction": getattr(res, "active_bar_fraction", 0.0),
    }


def run_walkforward(
    strategy: Strategy,
    bars: pd.DataFrame,
    *,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
    higher_timeframes: Optional[dict[str, pd.DataFrame]] = None,
) -> WalkForwardResult:
    """
    Generate rolling [train_years] → [test_years] folds, advancing by step_years.

    Example for 2006-2026 with train=3, test=1, step=1:
      Fold 1: train 2006-2008, test 2009
      Fold 2: train 2007-2009, test 2010
      ...
      Fold N: train 2022-2024, test 2025
    """
    signal_type = strategy.signal_logic.type
    if signal_type not in ("bollinger-mean-reversion", "strat-pattern"):
        raise NotImplementedError(
            f"Walk-forward doesn't support signal type {signal_type!r}"
        )

    cap = strategy.capital_allocation
    tf = strategy.timeframe

    if bars.empty:
        return WalkForwardResult()

    # Pre-compute for STRAT (avoids re-classifying each fold)
    classified_full = None
    ftfc_full = None
    wanted_pattern = None
    strat_params = None
    if signal_type == "strat-pattern":
        sl = strategy.signal_logic
        classified_full = detect_patterns(classify_bars(bars))
        wanted_pattern = getattr(sl, "pattern", None)
        require_ftfc = bool(getattr(sl, "require_ftfc", True))
        if require_ftfc and higher_timeframes:
            ftfc_full = compute_ftfc(
                bars, higher_timeframes,
                threshold=float(getattr(sl, "ftfc_threshold", 0.75)),
            )
        strat_params = StratParams(
            require_ftfc=require_ftfc,
            ftfc_threshold=float(getattr(sl, "ftfc_threshold", 0.75)),
            min_risk_reward=float(getattr(strategy.exit, "min_risk_reward", 1.5)),
            max_holding_bars=int(getattr(sl, "max_holding_bars", 20)),
        )
    else:
        bol_params = BollingerParams.from_strategy(strategy)

    # Generate fold date boundaries
    history_start = bars.index.min()
    history_end = bars.index.max()

    folds: list[WalkForwardFold] = []
    fold_train_start = history_start

    while True:
        train_start = fold_train_start
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_years)

        if test_end > history_end:
            break

        if signal_type == "bollinger-mean-reversion":
            train_res = _run_fold_bollinger(bars, train_start, train_end, bol_params, cap, tf)
            test_res = _run_fold_bollinger(bars, test_start, test_end, bol_params, cap, tf)
        else:
            train_res = _run_fold_strat(
                bars, classified_full, ftfc_full, train_start, train_end,
                wanted_pattern, strat_params, cap, tf,
            )
            test_res = _run_fold_strat(
                bars, classified_full, ftfc_full, test_start, test_end,
                wanted_pattern, strat_params, cap, tf,
            )

        tr = _result_metrics(train_res)
        te = _result_metrics(test_res)

        fold = WalkForwardFold(
            fold_index=len(folds) + 1,
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            train_n_trades=tr["n_trades"],
            train_equity_sharpe=tr["equity_sharpe"],
            train_total_return=tr["equity_total_return"],
            test_n_trades=te["n_trades"],
            test_equity_sharpe=te["equity_sharpe"],
            test_active_bar_sharpe=te["active_bar_sharpe"],
            test_active_bar_fraction=te["active_bar_fraction"],
            test_win_rate=te["win_rate"],
            test_max_drawdown=te["equity_max_drawdown"],
            test_total_return=te["equity_total_return"],
            test_profit_factor=te["profit_factor"],
        )
        folds.append(fold)

        fold_train_start = fold_train_start + pd.DateOffset(years=step_years)

    if not folds:
        return WalkForwardResult()

    all_sharpes = np.array([f.test_equity_sharpe for f in folds])
    all_dds = np.array([f.test_max_drawdown for f in folds])
    total_trades = sum(f.test_n_trades for f in folds)

    active_folds = [f for f in folds if f.test_n_trades >= 1]
    if active_folds:
        active_sharpes = np.array([f.test_equity_sharpe for f in active_folds])
        active_bar_sharpes = np.array([f.test_active_bar_sharpe for f in active_folds])
        active_wrs = np.array([f.test_win_rate for f in active_folds])
        active_fracs = np.array([f.test_active_bar_fraction for f in active_folds])
        active_mean_sharpe = float(active_sharpes.mean())
        active_std_sharpe = float(active_sharpes.std(ddof=0))
        active_min_sharpe = float(active_sharpes.min())
        active_mean_wr = float(active_wrs.mean())
        active_bar_mean = float(active_bar_sharpes.mean())
        active_bar_std = float(active_bar_sharpes.std(ddof=0))
        active_bar_min = float(active_bar_sharpes.min())
        total_active_bars = int(sum(
            (f.test_active_bar_fraction * (f.test_end - f.test_start).days) for f in active_folds
        ))  # approximate; used for stats only
        mean_active_frac = float(active_fracs.mean())
    else:
        active_mean_sharpe = 0.0
        active_std_sharpe = 0.0
        active_min_sharpe = 0.0
        active_mean_wr = 0.0
        active_bar_mean = 0.0
        active_bar_std = 0.0
        active_bar_min = 0.0
        total_active_bars = 0
        mean_active_frac = 0.0

    return WalkForwardResult(
        folds=folds,
        n_folds=len(folds),
        oos_all_mean_sharpe=float(all_sharpes.mean()),
        oos_worst_dd=float(all_dds.min()),
        oos_total_trades=int(total_trades),
        n_active_folds=len(active_folds),
        activation_rate=len(active_folds) / len(folds) if folds else 0.0,
        oos_active_mean_sharpe=active_mean_sharpe,
        oos_active_std_sharpe=active_std_sharpe,
        oos_active_min_sharpe=active_min_sharpe,
        oos_active_mean_win_rate=active_mean_wr,
        oos_active_bar_mean_sharpe=active_bar_mean,
        oos_active_bar_std_sharpe=active_bar_std,
        oos_active_bar_min_sharpe=active_bar_min,
        oos_total_active_bars=total_active_bars,
        oos_mean_active_bar_fraction=mean_active_frac,
    )
