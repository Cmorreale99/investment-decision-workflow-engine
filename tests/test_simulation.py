"""Phase 5 simulation + benchmarking tests."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ai_investment_workflow.features import compute_feature_frame
from ai_investment_workflow.ingestion import FixtureProvider
from ai_investment_workflow.simulation import (
    Portfolio,
    annualized_return,
    apply_rebalance,
    benchmark_returns,
    build_rankings_history,
    clip_position_weights,
    clip_sector_weights,
    equal_weight_topn,
    hit_rate,
    max_drawdown,
    rebalance_weekly,
    run_paper_portfolio,
    run_simulation,
    total_return,
    volatility,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _business_days(start: date, n: int) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < n:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _flat_prices(tickers: list[str], n_days: int = 120) -> pd.DataFrame:
    dates = _business_days(date(2025, 1, 2), n_days)
    rows = []
    for t in tickers:
        base = 100.0 + abs(hash(t)) % 50
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "asset_id": t,
                    "open": base,
                    "high": base,
                    "low": base,
                    "close": base,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def _trending_prices(
    tickers: dict[str, float], n_days: int = 120
) -> pd.DataFrame:
    """Each ticker grows by its daily-drift rate. Deterministic."""
    dates = _business_days(date(2025, 1, 2), n_days)
    rows = []
    for t, drift in tickers.items():
        price = 100.0
        for d in dates:
            price *= 1.0 + drift
            rows.append(
                {
                    "date": d,
                    "asset_id": t,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def _rankings_history_two_assets(prices: pd.DataFrame) -> pd.DataFrame:
    """Synthetic rankings: TOP always ranked 1, OTHER always ranked 2."""
    rows = []
    for d in sorted(prices["date"].unique()):
        rows.append(
            {
                "date": d,
                "asset_id": "TOP",
                "composite_score": 0.5,
                "rank": 1,
                "strategy_conflict": False,
            }
        )
        rows.append(
            {
                "date": d,
                "asset_id": "OTHER",
                "composite_score": 0.1,
                "rank": 2,
                "strategy_conflict": False,
            }
        )
    return pd.DataFrame(rows)


def _fixture_features_and_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    provider = FixtureProvider()
    prices = provider.fetch(
        ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "SPY"],
        date(2025, 1, 1),
        date(2025, 6, 1),
    )
    features = compute_feature_frame(prices)
    return features, prices


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_equal_weight_topn_assigns_uniform_weights() -> None:
    rankings = pd.DataFrame(
        {
            "asset_id": ["A", "B", "C", "D"],
            "rank": [1, 2, 3, 4],
            "strategy_conflict": [False, False, False, False],
        }
    )
    weights = equal_weight_topn(rankings, top_n=3)
    assert set(weights.keys()) == {"A", "B", "C"}
    assert all(w == pytest.approx(1 / 3) for w in weights.values())


def test_equal_weight_topn_tiebreak_by_asset_id() -> None:
    rankings = pd.DataFrame(
        {
            "asset_id": ["B", "A", "C"],
            "rank": [1, 1, 1],
            "strategy_conflict": [False, False, False],
        }
    )
    weights = equal_weight_topn(rankings, top_n=2)
    assert set(weights.keys()) == {"A", "B"}


def test_equal_weight_topn_exclude_conflicts() -> None:
    rankings = pd.DataFrame(
        {
            "asset_id": ["A", "B", "C"],
            "rank": [1, 2, 3],
            "strategy_conflict": [True, False, False],
        }
    )
    weights = equal_weight_topn(rankings, top_n=2, exclude_conflicts=True)
    assert set(weights.keys()) == {"B", "C"}


def test_clip_position_weights_caps_each_position() -> None:
    weights = {f"A{i}": 0.10 for i in range(10)}
    clipped = clip_position_weights(weights, max_weight=0.05)
    assert all(w == pytest.approx(0.05) for w in clipped.values())
    # Residual is left to cash, not redistributed.
    assert sum(clipped.values()) == pytest.approx(0.50)


def test_clip_position_weights_noop_when_under_cap() -> None:
    weights = {"A": 0.10, "B": 0.20}
    assert clip_position_weights(weights, max_weight=0.50) == weights


def test_clip_sector_weights_scales_overcap_sectors() -> None:
    weights = {"A": 0.20, "B": 0.20, "C": 0.10}
    sectors = {"A": "Tech", "B": "Tech", "C": "Health"}
    out = clip_sector_weights(weights, sectors, max_sector_weight=0.30)
    # Tech total was 0.40 → scaled by 0.30/0.40 = 0.75 → A=0.15, B=0.15.
    assert out["A"] == pytest.approx(0.15)
    assert out["B"] == pytest.approx(0.15)
    assert out["C"] == pytest.approx(0.10)


def test_clip_sector_weights_noop_without_sectors() -> None:
    weights = {"A": 0.50, "B": 0.50}
    assert clip_sector_weights(weights, None, max_sector_weight=0.30) == weights


def test_apply_rebalance_uses_nav_and_target_weights() -> None:
    shares, cash = apply_rebalance(
        prev_shares={},
        cash=1_000.0,
        target_weights={"A": 0.5, "B": 0.5},
        prices_at_date={"A": 10.0, "B": 20.0},
    )
    # 500 / 10 = 50 shares of A, 500 / 20 = 25 of B.
    assert shares == pytest.approx({"A": 50.0, "B": 25.0})
    assert cash == pytest.approx(0.0)


def test_apply_rebalance_with_partial_weights_keeps_residual_cash() -> None:
    shares, cash = apply_rebalance(
        prev_shares={},
        cash=1_000.0,
        target_weights={"A": 0.3},
        prices_at_date={"A": 10.0},
    )
    assert shares["A"] == pytest.approx(30.0)
    assert cash == pytest.approx(700.0)


# ---------------------------------------------------------------------------
# Rebalance schedule
# ---------------------------------------------------------------------------


def test_rebalance_weekly_one_portfolio_per_week() -> None:
    prices = _flat_prices(["TOP", "OTHER"], n_days=60)
    rankings = _rankings_history_two_assets(prices)
    portfolios = rebalance_weekly(rankings, top_n=1)
    # 60 business days ≈ 12 weeks → ~12 portfolios.
    assert 10 <= len(portfolios) <= 13
    # All rebalance dates are distinct.
    assert len({p.as_of for p in portfolios}) == len(portfolios)


def test_rebalance_weekly_picks_top_n() -> None:
    prices = _flat_prices(["TOP", "OTHER"], n_days=20)
    rankings = _rankings_history_two_assets(prices)
    portfolios = rebalance_weekly(rankings, top_n=1)
    for p in portfolios:
        assert set(p.weights.keys()) == {"TOP"}
        assert p.weights["TOP"] == pytest.approx(1.0)
        assert p.cash == pytest.approx(0.0)


def test_rebalance_weekly_exclude_conflicts() -> None:
    prices = _flat_prices(["TOP", "OTHER"], n_days=20)
    rankings = _rankings_history_two_assets(prices)
    # Flag TOP as conflicted on every date.
    rankings.loc[rankings["asset_id"] == "TOP", "strategy_conflict"] = True

    p_with = rebalance_weekly(rankings, top_n=1, exclude_conflicts=False)
    p_without = rebalance_weekly(rankings, top_n=1, exclude_conflicts=True)

    assert all(set(p.weights.keys()) == {"TOP"} for p in p_with)
    assert all(set(p.weights.keys()) == {"OTHER"} for p in p_without)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def test_engine_value_flat_under_flat_prices() -> None:
    prices = _flat_prices(["TOP", "OTHER"], n_days=40)
    rankings = _rankings_history_two_assets(prices)
    portfolios = rebalance_weekly(rankings, top_n=2)
    state = run_paper_portfolio(portfolios, prices, starting_cash=100_000.0)
    # Constant prices → constant NAV → daily returns are zero.
    assert (state.returns.abs() < 1e-9).all()
    assert state.total_value.iloc[-1] == pytest.approx(100_000.0)


def test_engine_grows_with_trending_prices() -> None:
    prices = _trending_prices({"TOP": 0.01, "OTHER": 0.01}, n_days=40)
    rankings = _rankings_history_two_assets(prices)
    portfolios = rebalance_weekly(rankings, top_n=2)
    state = run_paper_portfolio(portfolios, prices, starting_cash=100_000.0)
    assert state.total_value.iloc[-1] > 100_000.0


def test_engine_history_columns_and_weights_sum_to_invested_fraction() -> None:
    prices = _flat_prices(["TOP", "OTHER"], n_days=20)
    rankings = _rankings_history_two_assets(prices)
    portfolios = rebalance_weekly(rankings, top_n=2)
    state = run_paper_portfolio(portfolios, prices, starting_cash=100_000.0)
    assert list(state.history.columns) == ["date", "asset_id", "weight", "value"]
    weights_by_date = state.history.groupby("date")["weight"].sum()
    # Two equal-weight names → invested weight ≈ 1.0; cash ≈ 0.
    assert (weights_by_date > 0.99).all()


def test_engine_deterministic() -> None:
    prices = _flat_prices(["TOP", "OTHER"], n_days=30)
    rankings = _rankings_history_two_assets(prices)
    portfolios = rebalance_weekly(rankings, top_n=2)
    a = run_paper_portfolio(portfolios, prices, starting_cash=100_000.0)
    b = run_paper_portfolio(portfolios, prices, starting_cash=100_000.0)
    pd.testing.assert_frame_equal(a.history, b.history)
    pd.testing.assert_series_equal(a.returns, b.returns)


# ---------------------------------------------------------------------------
# Metrics + benchmark
# ---------------------------------------------------------------------------


def test_max_drawdown_on_known_dip() -> None:
    returns = pd.Series([0.0, 0.10, -0.20, 0.0])
    # Cumulative: 1.0 → 1.10 → 0.88 → 0.88; drawdown at row 2 = -0.20.
    assert max_drawdown(returns) == pytest.approx(-0.20, abs=1e-6)


def test_total_and_annualized_return() -> None:
    # 252 days at +0.1% per day → ~28.6% total, annualized ~28.6%.
    returns = pd.Series([0.001] * 252)
    assert total_return(returns) == pytest.approx(1.001**252 - 1.0)
    assert annualized_return(returns) == pytest.approx(1.001**252 - 1.0, rel=1e-6)


def test_volatility_zero_for_constant_returns() -> None:
    assert volatility(pd.Series([0.01] * 50)) == pytest.approx(0.0)


def test_hit_rate_basic() -> None:
    assert hit_rate(pd.Series([0.0, 0.1, -0.1, 0.2])) == pytest.approx(0.5)
    assert hit_rate(pd.Series([])) == pytest.approx(0.0)


def test_benchmark_returns_from_prices() -> None:
    prices = _trending_prices({"SPY": 0.005}, n_days=30)
    bench = benchmark_returns(prices, benchmark="SPY")
    assert isinstance(bench, pd.Series)
    # First day return is 0 (no prior close); subsequent are ~0.005 each.
    assert bench.iloc[0] == pytest.approx(0.0)
    assert (bench.iloc[1:] - 0.005).abs().max() < 1e-9


def test_benchmark_returns_raises_when_missing() -> None:
    prices = _flat_prices(["AAPL"], n_days=5)
    with pytest.raises(ValueError, match="not present"):
        benchmark_returns(prices, benchmark="SPY")


# ---------------------------------------------------------------------------
# Full pipeline + parquet I/O
# ---------------------------------------------------------------------------


def test_run_simulation_against_fixture(tmp_path) -> None:
    features, prices = _fixture_features_and_prices()
    rankings = build_rankings_history(features)
    assert not rankings.empty

    settings = {
        "top_n_candidates": 3,
        "benchmark": "SPY",
        "rebalance_frequency": "weekly",
        "starting_cash": 100_000.0,
    }
    result = run_simulation(rankings, prices, settings, risk_clipping=False)
    state = result["portfolio_state"]
    assert not state.history.empty
    assert "total_return" in result["metrics"]
    assert "benchmark_total_return" in result["metrics"]

    # Round-trip the three persistence frames.
    state.history.to_parquet(tmp_path / "portfolio_history.parquet", index=False)
    hist = pd.read_parquet(tmp_path / "portfolio_history.parquet")
    assert list(hist.columns) == ["date", "asset_id", "weight", "value"]

    returns_df = pd.DataFrame(
        {
            "date": [
                ts.date() if hasattr(ts, "date") else ts
                for ts in state.returns.index
            ],
            "portfolio_return": state.returns.to_numpy(),
            "benchmark_return": result["benchmark_returns"].to_numpy(),
            "excess_return": result["excess_returns"].to_numpy(),
        }
    )
    returns_df.to_parquet(tmp_path / "portfolio_returns.parquet", index=False)
    rr = pd.read_parquet(tmp_path / "portfolio_returns.parquet")
    assert list(rr.columns) == [
        "date",
        "portfolio_return",
        "benchmark_return",
        "excess_return",
    ]

    summary_df = pd.DataFrame([result["metrics"]])
    summary_df.to_parquet(tmp_path / "portfolio_summary.parquet", index=False)
    rs = pd.read_parquet(tmp_path / "portfolio_summary.parquet")
    assert len(rs) == 1


def test_run_simulation_is_deterministic() -> None:
    features, prices = _fixture_features_and_prices()
    rankings = build_rankings_history(features)
    settings = {
        "top_n_candidates": 3,
        "benchmark": "SPY",
        "rebalance_frequency": "weekly",
        "starting_cash": 100_000.0,
    }
    a = run_simulation(rankings, prices, settings, risk_clipping=False)
    b = run_simulation(rankings, prices, settings, risk_clipping=False)
    pd.testing.assert_series_equal(
        a["portfolio_state"].returns, b["portfolio_state"].returns
    )
    assert a["metrics"] == b["metrics"]


def test_run_simulation_risk_clipping_leaves_residual_cash() -> None:
    """With max_single_position_weight=0.05 and top_n=10 (config default),
    every weight gets clipped to 0.05 → invested ≈ 50%, cash ≈ 50%."""
    features, prices = _fixture_features_and_prices()
    rankings = build_rankings_history(features)
    settings = {
        "top_n_candidates": 10,
        "benchmark": "SPY",
        "rebalance_frequency": "weekly",
        "starting_cash": 100_000.0,
    }
    result = run_simulation(rankings, prices, settings, risk_clipping=True)
    state = result["portfolio_state"]
    # Some rebalance dates may carry fewer than 10 names depending on coverage;
    # invested weight must still be ≤ N * 0.05 ≤ 0.50.
    invested = state.history.groupby("date")["weight"].sum()
    assert (invested <= 0.50 + 1e-9).all()


def test_run_simulation_exclude_conflicts_toggle() -> None:
    """A synthetic rankings frame where one asset is *always* conflicted
    and *always* rank 1 — the toggle must materially change selection."""
    prices = _trending_prices(
        {"FLAGGED": 0.01, "CLEAN_A": 0.005, "CLEAN_B": 0.005, "SPY": 0.0},
        n_days=40,
    )
    rows = []
    for d in sorted(prices["date"].unique()):
        rows.append(
            {
                "date": d,
                "asset_id": "FLAGGED",
                "composite_score": 0.9,
                "rank": 1,
                "strategy_conflict": True,
            }
        )
        rows.append(
            {
                "date": d,
                "asset_id": "CLEAN_A",
                "composite_score": 0.5,
                "rank": 2,
                "strategy_conflict": False,
            }
        )
        rows.append(
            {
                "date": d,
                "asset_id": "CLEAN_B",
                "composite_score": 0.4,
                "rank": 3,
                "strategy_conflict": False,
            }
        )
    rankings = pd.DataFrame(rows)

    settings = {
        "top_n_candidates": 1,
        "benchmark": "SPY",
        "rebalance_frequency": "weekly",
        "starting_cash": 100_000.0,
    }
    included = run_simulation(
        rankings, prices, settings, risk_clipping=False, exclude_conflicts=False
    )
    excluded = run_simulation(
        rankings, prices, settings, risk_clipping=False, exclude_conflicts=True
    )

    included_assets = set(included["portfolio_state"].history["asset_id"])
    excluded_assets = set(excluded["portfolio_state"].history["asset_id"])
    assert included_assets == {"FLAGGED"}
    assert excluded_assets == {"CLEAN_A"}
