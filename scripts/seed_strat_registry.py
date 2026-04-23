"""
Generate STRAT strategy YAML entries.

v2 (2026-04-22, Day 7):
  - FTFC threshold lowered from 0.75 → 0.5 (2 of 4 higher tfs aligned)
  - Universe expanded from SPY-only → SPY + AAPL, AMZN, MSFT, NVDA, TSLA
  - 6 patterns × 4 timeframes × 6 tickers = 144 strategies

Re-running is idempotent; existing files are overwritten with current template.
"""
from __future__ import annotations
from pathlib import Path
import yaml


# 6 reversal patterns from the-strat-implementation-plan.md §2.2
PATTERNS = [
    {
        "code": "2d-1-2u",
        "label": "2d-1-2u Bullish Reversal",
        "direction": "bullish",
        "confidence": "medium",
        "description": "Down directional bar → inside bar → up directional bar. Classic reversal.",
    },
    {
        "code": "2u-1-2d",
        "label": "2u-1-2d Bearish Reversal",
        "direction": "bearish",
        "confidence": "medium",
        "description": "Up directional bar → inside bar → down directional bar. Classic reversal.",
    },
    {
        "code": "3-1-2u",
        "label": "3-1-2u Bullish Reversal (off outside bar)",
        "direction": "bullish",
        "confidence": "high",
        "description": "Outside bar → inside bar → up directional bar. Highest-confidence bullish.",
    },
    {
        "code": "3-1-2d",
        "label": "3-1-2d Bearish Reversal (off outside bar)",
        "direction": "bearish",
        "confidence": "high",
        "description": "Outside bar → inside bar → down directional bar. Highest-confidence bearish.",
    },
    {
        "code": "2d-2u",
        "label": "2d-2u Quick Bullish Reversal",
        "direction": "bullish",
        "confidence": "medium",
        "description": "Down directional → up directional (no inside bar in between). Fast reversal.",
    },
    {
        "code": "2u-2d",
        "label": "2u-2d Quick Bearish Reversal",
        "direction": "bearish",
        "confidence": "medium",
        "description": "Up directional → down directional (no inside bar in between). Fast reversal.",
    },
]

# 4 timeframes. 4H and 1D are the sweet spots on SPY; 1H gives many signals; 1W for swing.
TIMEFRAMES = ["1h", "4h", "1d", "1w"]

# Universe: SPY (equity index) + 5 volatile single names.
INSTRUMENTS = [
    ("SPY",  "equity-index", 0.05),
    ("AAPL", "equity",       0.03),
    ("AMZN", "equity",       0.03),
    ("MSFT", "equity",       0.03),
    ("NVDA", "equity",       0.03),
    ("TSLA", "equity",       0.03),
]

# FTFC threshold (v2 default) — 0.5 = 2 of 4 higher timeframes aligned.
# v1 used 0.75 (3 of 4) which over-filtered on SPY.
FTFC_THRESHOLD = 0.5


def make_entry(pattern: dict, timeframe: str, instrument: str, asset_class: str, capital_alloc: float) -> dict:
    pattern_code = pattern["code"]
    direction = pattern["direction"]
    strat_id = f"strat-{pattern_code}-{timeframe}-{instrument.lower()}-v1"

    entry = {
        "id": strat_id,
        "name": f"STRAT {pattern['label']} ({timeframe}, {instrument})",
        "status": "draft",
        "asset_class": asset_class,
        "instruments": [instrument],
        "timeframe": timeframe,
        "signal_logic": {
            "type": "strat-pattern",
            "pattern": pattern_code,
            "direction": direction,
            "confidence": pattern["confidence"],
            "description": pattern["description"],
            "require_ftfc": True,
            "ftfc_timeframes": ["1mo", "1w", "1d", "1h"],
            "ftfc_threshold": FTFC_THRESHOLD,
            "ftfc_notes": f"{int(FTFC_THRESHOLD * 4)} of 4 higher timeframes aligned.",
        },
        "entry": {
            "mode": "breakout",
            "trigger": f"break-of-{'inside-bar-high' if direction == 'bullish' else 'inside-bar-low'}",
        },
        "exit": {
            "mode": "pattern-targets",
            "target": f"prior-setup-bar-{'high' if direction == 'bullish' else 'low'}",
            "stop_loss": f"inside-bar-{'low' if direction == 'bullish' else 'high'}",
            "min_risk_reward": 1.5,
        },
        "capital_allocation": capital_alloc,
        "data_sources": ["firstrate"],
        "backtest_runs": [],
        "live_run": None,
        "tags": [
            "strat",
            "reversal",
            asset_class,
            timeframe,
            direction,
            pattern["confidence"] + "-confidence",
            instrument.lower(),
        ] + (["outside-bar"] if pattern_code.startswith("3") else []),
        "notes_doc": "research/the-strat-implementation-plan.md#pattern-detection-logic",
    }

    return entry


def main() -> int:
    out_dir = Path.home() / "clawd" / "research" / "strategies" / "strat"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for pattern in PATTERNS:
        for tf in TIMEFRAMES:
            for instrument, asset_class, cap in INSTRUMENTS:
                entry = make_entry(pattern, tf, instrument, asset_class, cap)
                path = out_dir / f"{entry['id']}.yaml"
                with path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(entry, f, sort_keys=False, allow_unicode=True, width=100)
                count += 1

    print(f"Generated {count} STRAT strategy YAML files at {out_dir}")
    print(f"  Tickers:    {[i[0] for i in INSTRUMENTS]}")
    print(f"  Timeframes: {TIMEFRAMES}")
    print(f"  Patterns:   {[p['code'] for p in PATTERNS]}")
    print(f"  FTFC:       {FTFC_THRESHOLD} ({int(FTFC_THRESHOLD * 4)} of 4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
