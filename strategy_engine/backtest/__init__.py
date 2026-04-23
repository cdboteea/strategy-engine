"""Backtest module — v1: single-history backtest for bollinger-mean-reversion.

Day 5+ additions:
- Walk-forward cross-validation
- STRAT pattern-detection runner
- Cost-model integration (slippage, commissions)
- Regime-breakdown analysis
"""
from .runner import run_strategy, BacktestRun, BacktestError, append_run_to_yaml
from . import bollinger

__all__ = ["run_strategy", "BacktestRun", "BacktestError", "append_run_to_yaml", "bollinger"]
