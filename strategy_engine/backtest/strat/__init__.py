"""STRAT methodology implementation (bar classification, FTFC, patterns, simulator)."""
from .classification import classify_bars, TYPE_INSIDE, TYPE_DIRECTIONAL, TYPE_OUTSIDE
from .ftfc import compute_ftfc, align_higher_tf_to_trade_tf
from .patterns import detect_patterns, setup_bars_for_pattern
from .simulator import (
    StratParams,
    StratTrade,
    StratResult,
    simulate_trades,
    summarize,
    build_equity_curve,
)

__all__ = [
    "classify_bars", "TYPE_INSIDE", "TYPE_DIRECTIONAL", "TYPE_OUTSIDE",
    "compute_ftfc", "align_higher_tf_to_trade_tf",
    "detect_patterns", "setup_bars_for_pattern",
    "StratParams", "StratTrade", "StratResult",
    "simulate_trades", "summarize", "build_equity_curve",
]
