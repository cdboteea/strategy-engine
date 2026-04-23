"""Behavioral tests for the registry loader + schema."""
from __future__ import annotations
from pathlib import Path
import textwrap
import pytest

from strategy_engine.registry.loader import load_one, validate_all, RegistryError
from strategy_engine.registry.schema import Strategy


VALID_YAML = textwrap.dedent("""
    id: test-spy-bollinger-v1
    name: "Test SPY Bollinger"
    status: backtested
    asset_class: equity-index
    instruments: [SPY]
    timeframe: 1w
    signal_logic:
      type: bollinger-mean-reversion
      lookback: 20
      std_dev: 2.0
    entry:
      mode: hybrid-50-50
    exit:
      mode: profit-target
      target: 0.05
    capital_allocation: 0.10
    data_sources: [firstrate]
    tags: [bollinger]
""")


def _write(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(content)
    return p


def test_load_one_valid(tmp_path):
    path = _write(tmp_path, "test-spy-bollinger-v1.yaml", VALID_YAML)
    strat = load_one(path)
    assert strat.id == "test-spy-bollinger-v1"
    assert strat.asset_class == "equity-index"
    assert strat.capital_allocation == 0.10


def test_load_rejects_invalid_status(tmp_path):
    bad = VALID_YAML.replace("status: backtested", "status: not-a-real-status")
    path = _write(tmp_path, "test-spy-bollinger-v1.yaml", bad)
    with pytest.raises(RegistryError):
        load_one(path)


def test_load_rejects_invalid_timeframe(tmp_path):
    bad = VALID_YAML.replace("timeframe: 1w", "timeframe: weekly")
    path = _write(tmp_path, "test-spy-bollinger-v1.yaml", bad)
    with pytest.raises(RegistryError):
        load_one(path)


def test_load_rejects_bad_id_format(tmp_path):
    bad = VALID_YAML.replace("id: test-spy-bollinger-v1", "id: Test_SPY_Bollinger")
    path = _write(tmp_path, "Test_SPY_Bollinger.yaml", bad)
    with pytest.raises(RegistryError):
        load_one(path)


def test_load_rejects_filename_mismatch(tmp_path):
    path = _write(tmp_path, "wrong-name.yaml", VALID_YAML)
    with pytest.raises(RegistryError, match="filename mismatch"):
        load_one(path)


def test_load_rejects_unknown_signal_type(tmp_path):
    bad = VALID_YAML.replace("type: bollinger-mean-reversion", "type: made-up-strategy")
    path = _write(tmp_path, "test-spy-bollinger-v1.yaml", bad)
    with pytest.raises(RegistryError):
        load_one(path)


def test_load_rejects_unknown_data_source(tmp_path):
    bad = VALID_YAML.replace("data_sources: [firstrate]", "data_sources: [firstrate, made-up-source]")
    path = _write(tmp_path, "test-spy-bollinger-v1.yaml", bad)
    with pytest.raises(RegistryError, match="unknown data_sources"):
        load_one(path)


def test_load_rejects_capital_allocation_out_of_range(tmp_path):
    bad = VALID_YAML.replace("capital_allocation: 0.10", "capital_allocation: 1.5")
    path = _write(tmp_path, "test-spy-bollinger-v1.yaml", bad)
    with pytest.raises(RegistryError):
        load_one(path)


def test_validate_all_catches_duplicate_ids(tmp_path):
    _write(tmp_path, "test-spy-bollinger-v1.yaml", VALID_YAML)
    dup_yaml = VALID_YAML.replace("instruments: [SPY]", "instruments: [SPY, QQQ]")
    # Same id, different file — will fail filename check first so rename to keep id
    (tmp_path / "sub").mkdir()
    _write(tmp_path / "sub", "test-spy-bollinger-v1.yaml", dup_yaml)
    ok, errors = validate_all(root=tmp_path)
    assert len(ok) == 1
    assert any("duplicate id" in e for _, e in errors)


def test_validate_all_reports_multiple_errors(tmp_path):
    _write(tmp_path, "valid-one.yaml", VALID_YAML.replace("test-spy-bollinger-v1", "valid-one"))
    _write(tmp_path, "broken-one.yaml",
           VALID_YAML.replace("test-spy-bollinger-v1", "broken-one").replace("status: backtested", "status: invalid"))
    _write(tmp_path, "broken-two.yaml",
           VALID_YAML.replace("test-spy-bollinger-v1", "broken-two").replace("timeframe: 1w", "timeframe: wrong"))
    ok, errors = validate_all(root=tmp_path)
    assert len(ok) == 1
    assert len(errors) == 2


# ─── backtest_window field ──────────────────────────────────────────────────


def test_backtest_window_optional_and_parses(tmp_path):
    """backtest_window is optional; when provided, dates parse correctly."""
    yaml_with_window = VALID_YAML.strip() + textwrap.dedent("""
        backtest_window:
          start: 2006-01-01
          end: 2026-03-31
    """)
    path = _write(tmp_path, "test-spy-bollinger-v1.yaml", yaml_with_window)
    strat = load_one(path)
    assert strat.backtest_window is not None
    assert str(strat.backtest_window.start) == "2006-01-01"
    assert str(strat.backtest_window.end) == "2026-03-31"


def test_backtest_window_unset_is_full_history(tmp_path):
    """Unset means None — engine treats as 'full history'."""
    path = _write(tmp_path, "test-spy-bollinger-v1.yaml", VALID_YAML)
    strat = load_one(path)
    assert strat.backtest_window is None


def test_backtest_window_rejects_start_after_end(tmp_path):
    yaml_bad = VALID_YAML.strip() + textwrap.dedent("""
        backtest_window:
          start: 2026-01-01
          end: 2020-01-01
    """)
    path = _write(tmp_path, "test-spy-bollinger-v1.yaml", yaml_bad)
    with pytest.raises(RegistryError):
        load_one(path)


def test_backtest_window_allows_one_bound_only(tmp_path):
    """start-only or end-only is valid."""
    yaml_start_only = VALID_YAML.strip() + textwrap.dedent("""
        backtest_window:
          start: 2006-01-01
    """)
    path = _write(tmp_path, "test-spy-bollinger-v1.yaml", yaml_start_only)
    strat = load_one(path)
    assert str(strat.backtest_window.start) == "2006-01-01"
    assert strat.backtest_window.end is None
