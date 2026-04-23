# strategy-engine

Trading-strategy registry, backtest engine, walk-forward validation, promotion
gates, and live-signal detection — all built around a YAML-defined strategy
registry and DuckDB persistence.

## Components

- **Registry** (`strategy_engine/registry/`) — pydantic-validated strategy YAMLs
- **Backtest** (`strategy_engine/backtest/`) — Bollinger + STRAT (bar classification, FTFC, patterns)
- **Walk-forward** (`strategy_engine/backtest/walkforward.py`) — rolling train/test folds
- **Promotion gates** (`strategy_engine/promotion/`) — portfolio + active-trader profiles
- **Live detection** (`strategy_engine/live/`) — checks promoted strategies for new-bar signals
- **Providers** (`strategy_engine/providers/`) — DuckDB OHLCV loader, multi-timeframe, FMP splice

## Registry location

Strategy YAMLs live at `~/clawd/research/strategies/` (not in this repo). The
path is configured in `strategy_engine/config.py`.

## CLI

```
strategy-engine registry list
strategy-engine registry validate
strategy-engine backtest <id>
strategy-engine walkforward <id>
strategy-engine promote <id> [--profile portfolio|active-trader]
strategy-engine detect --strategy <id>
strategy-engine detect --all-promoted [--force]
```

## Status as of 2026-04-22

- 10 promoted strategies (4 Bollinger portfolio + 6 STRAT active-trader)
- 151 strategies in registry
- 37/37 tests pass
- launchd jobs: daily (weekdays 16:30) + weekly (Sat 09:00)
