"""Paths and constants. Single source of truth."""
from __future__ import annotations
from pathlib import Path

# Registry (YAML data) — separate from code
REGISTRY_DIR = Path.home() / "clawd" / "research" / "strategies"

# Backtest results
BACKTEST_DB = Path.home() / "clawd" / "data" / "backtest-results.duckdb"

# Data sources
FIRSTRATE_DB = Path.home() / "clawd" / "data" / "firstrate.duckdb"
FMP_DB = Path.home() / "clawd" / "data" / "fmp.duckdb"

# Live state — intraday splice (fresh 1h / 5m / 15m / 30m bars from EODHD)
LIVE_TICKS_DB = Path.home() / "clawd" / "data" / "live-ticks.duckdb"

# Enum constants — must match schema.py
STATUSES = [
    "draft",        # not backtested yet
    "backtested",   # has at least one backtest run
    "promoted",     # passed promotion gates
    "live-ready",   # promoted AND deployed to signal detector
    "live",         # actively trading
    "retired",      # was in use, no longer
    "archived",     # historical reference
]

ASSET_CLASSES = [
    "equity",
    "equity-index",
    "crypto",
    "futures",
    "fx",
    "options",
    "prediction-market",
    "multi-asset",
]

TIMEFRAMES = [
    "1m", "5m", "15m", "30m", "1h", "4h",
    "1d", "1w", "1mo",
]

# Bars per year, for annualizing returns at each timeframe.
# Based on US equity market: 252 trading days/year, 6.5 hours/day.
BARS_PER_YEAR = {
    "1m": 252 * 6 * 60 + 252 * 30,   # ~98280
    "5m": 252 * 78,                   # 19656
    "15m": 252 * 26,                  # 6552
    "30m": 252 * 13,                  # 3276
    "1h": 252 * 7,                    # ~1764 (including some after-hours)
    "4h": 252 * 2,                    # 504
    "1d": 252,
    "1w": 52,
    "1mo": 12,
}

SIGNAL_TYPES = [
    "bollinger-mean-reversion",
    "strat-pattern",
    "composite",
    "sma-crossover",
    "rsi-mean-reversion",
    "macd-crossover",
    "donchian-breakout",
    "momentum",
    "pairs",
    "funding-rate",
    "custom",
]

DATA_SOURCES = [
    "firstrate",
    "eodhd-eod",
    "eodhd-intraday",
    "fmp",
    "unusual-whales",
    "yahoo",
    "hyperliquid",
    "polymarket",
    "tradingview",
]
