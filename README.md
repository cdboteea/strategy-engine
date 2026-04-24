# strategy-engine

Trading-strategy registry, backtest engine, walk-forward validation,
transaction-cost model, promotion gates, live-signal detection, paper-trading
book, and system-health observability — all built around a YAML-defined
strategy registry and DuckDB persistence.

**Design reference:** `~/clawd/docs/guides/quant-trading-system-guide-2026-04-23.md`
**Daily operations:** `~/clawd/docs/runbook.md`

## Components

- **Registry** (`strategy_engine/registry/`) — pydantic-validated strategy YAMLs (supports Bollinger, STRAT, composite signal types + optional `cost_model` + optional `backtest_window` + optional `composite` block)
- **Backtest** (`strategy_engine/backtest/`) — Bollinger mean-reversion, STRAT pattern (with FTFC), composite (primary × confirmation), runner + cost model
- **Walk-forward** (`strategy_engine/backtest/walkforward.py`) — rolling train/test folds, active-fold + active-bar Sharpe aggregations
- **Promotion gates** (`strategy_engine/promotion/`) — portfolio + active-trader profiles
- **Live detection** (`strategy_engine/live/detector.py`) — checks promoted strategies for new-bar signals; deterministic signal IDs for replay-safety; per-strategy error isolation with DB-persisted tracebacks
- **Notifications** (`strategy_engine/live/notification.py`) — Telegram via OpenClaw gateway with 3-attempt exponential backoff + stdout fallback; every attempt logged
- **Intraday poller** (`strategy_engine/live/intraday_poller.py`) — EODHD hourly bars → `live-ticks.duckdb` for promoted intraday strategies
- **Health CLI** (`strategy_engine/live/health.py`) — 19-check system health report (DBs, launchd, errors, paper book, registry, dispatch)
- **Paper book** (`strategy_engine/paper/`) — auto-open on signal fire, MTM + auto-close cron, risk metrics (Sortino / Calmar / max-DD), per-strategy equity-curve PNGs
- **Providers** (`strategy_engine/providers/`) — DuckDB OHLCV loader with firstrate + fmp (daily) + live-ticks (intraday) splice

## Data locations

| Purpose | Path |
|---|---|
| Strategy YAMLs | `~/clawd/research/strategies/` (152 entries) |
| Historical bars | `~/clawd/data/firstrate.duckdb` (14.18B rows, monthly refresh) |
| Fresh daily bars | `~/clawd/data/fmp.duckdb` |
| Fresh intraday (1h/5m/15m/30m) | `~/clawd/data/live-ticks.duckdb` |
| Backtest runs | `~/clawd/data/backtest-results.duckdb` |
| Live signals + paper book + errors | `~/clawd/data/live-signals.duckdb` |

## CLI

```bash
# Registry
strategy-engine registry list | show <id> | validate | count

# Backtest + walk-forward
strategy-engine backtest <id> [--cost-profile zero|retail-equity|institutional-equity] [--start/--end]
strategy-engine walkforward <id> [--train-years 3] [--test-years 1] [--step-years 1] [--json]

# Promotion
strategy-engine promote <id> [--profile portfolio|active-trader]

# Live
strategy-engine detect --strategy <id>
strategy-engine detect --all-promoted [--force] [--paper]
strategy-engine intraday poll [--lookback-days 3]
strategy-engine intraday status

# Paper book
strategy-engine paper report              # NAV + per-strategy P&L + Sortino/Calmar/max-DD
strategy-engine paper positions [--status open|closed-target|all]
strategy-engine paper mtm                 # manual mark-to-market
strategy-engine paper equity-curves       # per-strategy PNGs → ~/clawd/reports/paper/
strategy-engine paper close <id> --price <p> --reason <r>

# Health
strategy-engine health [-v] [--json]
```

## Cost model

Every backtest applies a transaction-cost model by default. Named profiles:

| Profile | Round-trip |
|---|---:|
| `zero` | 0 bp |
| `retail-equity` (default) | 8 bp |
| `institutional-equity` | 5.2 bp |

Configure per-strategy via YAML `cost_model:` block, or override at CLI with `--cost-profile`.

## Launchd agents (installed via `scripts/launchagents/`)

- `com.matias.strategy-engine.weekly-detect` (Sat 09:00)
- `com.matias.strategy-engine.daily-detect` (weekdays 16:30)
- `com.matias.strategy-engine.intraday-poll` (weekdays 10:05–17:05 hourly)
- `com.matias.strategy-engine.paper-mtm` (weekdays 17:00)

## Status as of 2026-04-23

- **105 behavioral tests pass** (up from 37 at v0.1 of the guide)
- **11 promoted strategies** — 4 Bollinger (portfolio profile) + 6 STRAT (active-trader profile) + 1 composite (SPY Bollinger × STRAT confirmation)
- **152 strategy YAMLs** in registry
- **222 backtest runs** persisted in `backtest-results.duckdb`
- **Paper book NAV: $101,149.98** (+1.15% from 3 closed positions, 100% win rate)
- **19/19 system health checks green**
- **Cost model live** — every backtest nets 8 bps round-trip by default

## Repo layout

```
strategy_engine/
├── backtest/         bollinger, strat, composite, walkforward, costs, runner
├── live/             detector, intraday_poller, notification, health
├── paper/            book (open/MTM/close), reporting (NAV metrics, equity curves)
├── promotion/        gates (portfolio + active-trader profiles)
├── providers/        duckdb_provider (firstrate + fmp + live-ticks splice)
├── registry/         schema.py (pydantic), loader.py
└── cli.py
scripts/
├── init_live_signals_db.py
├── init_live_ticks_db.py
├── init_paper_trading_db.py    # paper_positions + paper_nav_snapshots
├── migrate_*.py                # incremental schema upgrades
├── seed_strat_registry.py      # generate 144 STRAT YAMLs
└── launchagents/               # plist mirrors (source of truth is ~/Library/LaunchAgents)
tests/                          # 105 behavioral tests, zero source-inspection
```

## Links

- Comprehensive guide (MD+PDF): `~/clawd/docs/guides/quant-trading-system-guide-2026-04-23.md`
- Operations runbook: `~/clawd/docs/runbook.md`
- Documentation & orphan audit: `~/clawd/docs/audit-2026-04-23.md`
- Companion repo (ideas pipeline): `cdboteea/ideas-pipeline`
