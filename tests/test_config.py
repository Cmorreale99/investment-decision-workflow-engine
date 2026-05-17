"""Each of the four YAML configs loads and has the required top-level keys."""

from __future__ import annotations

from ai_investment_workflow.utils import (
    load_risk_rules,
    load_settings,
    load_strategies,
    load_universe,
)


def test_settings_loads_with_required_keys() -> None:
    settings = load_settings()
    for key in (
        "rebalance_frequency",
        "benchmark",
        "evaluation_window_days",
        "top_n_candidates",
        "require_human_approval",
    ):
        assert key in settings, f"settings.yaml missing key: {key}"
    assert isinstance(settings["evaluation_window_days"], int)
    assert isinstance(settings["top_n_candidates"], int)
    assert isinstance(settings["require_human_approval"], bool)


def test_universe_loads_and_meets_mvp_size() -> None:
    tickers = load_universe()
    assert isinstance(tickers, list)
    assert 25 <= len(tickers) <= 50, (
        f"universe must hold 25-50 tickers per MVP scope, got {len(tickers)}"
    )
    assert all(isinstance(t, str) and t.strip() for t in tickers)
    assert len(tickers) == len(set(tickers)), "universe contains duplicate tickers"


def test_strategies_loads_with_value_and_momentum() -> None:
    cfg = load_strategies()
    strategies = cfg["strategies"]
    assert "value" in strategies
    assert "momentum" in strategies
    for name in ("value", "momentum"):
        assert "enabled" in strategies[name]
        assert "weight" in strategies[name]
        weight = strategies[name]["weight"]
        assert isinstance(weight, (int, float))
        assert 0.0 <= weight <= 1.0


def test_risk_rules_load_with_required_keys() -> None:
    rules = load_risk_rules()
    for key in (
        "max_single_position_weight",
        "max_sector_weight",
        "max_volatility_threshold",
        "min_liquidity_threshold",
    ):
        assert key in rules, f"risk_rules.yaml missing key: {key}"
