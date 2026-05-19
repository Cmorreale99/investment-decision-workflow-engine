"""Phase 6 evaluation diagnostics tests."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ai_investment_workflow.evaluation import (
    build_signals_history,
    calendar_to_trading_days,
    evaluate_asset_decisions,
    evaluate_portfolio,
    forward_window_max_drawdown,
    forward_window_return,
    outcome_label,
    performance_by_strategy,
    performance_records_to_frame,
    run_evaluation,
)
from ai_investment_workflow.features import compute_feature_frame
from ai_investment_workflow.ingestion import FixtureProvider
from ai_investment_workflow.schemas import OutcomeLabel, PerformanceRecord
from ai_investment_workflow.simulation import (
    build_rankings_history,
    run_simulation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _business_days(start: date, n: int) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < n:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _linear_prices(asset_id: str, n_days: int = 40, start_close: float = 100.0) -> pd.DataFrame:
    dates = _business_days(date(2025, 1, 1), n_days)
    closes = [start_close + i for i in range(n_days)]
    return pd.DataFrame(
        {
            "date": dates,
            "asset_id": asset_id,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1_000_000,
        }
    )


def _two_asset_prices(
    a_growth: float = 0.01, b_growth: float = 0.005, n_days: int = 40
) -> pd.DataFrame:
    """Compounding prices for two tickers + a SPY benchmark."""
    dates = _business_days(date(2025, 1, 1), n_days)
    rows = []
    for ticker, growth in [("A", a_growth), ("B", b_growth), ("SPY", 0.002)]:
        price = 100.0
        for d in dates:
            price *= 1.0 + growth
            rows.append(
                {
                    "date": d,
                    "asset_id": ticker,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def _fixture_data():
    provider = FixtureProvider()
    prices = provider.fetch(
        ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "SPY"],
        date(2025, 1, 1),
        date(2025, 6, 1),
    )
    features = compute_feature_frame(prices)
    return prices, features


# ---------------------------------------------------------------------------
# windows.py
# ---------------------------------------------------------------------------


def test_calendar_to_trading_days_default_30() -> None:
    assert calendar_to_trading_days(30) == 21
    assert calendar_to_trading_days(0) == 0
    assert calendar_to_trading_days(7) == 5


def test_forward_window_return_basic_math() -> None:
    prices = _linear_prices("TST", n_days=30, start_close=100.0)
    start = prices["date"].iloc[5]  # close = 105
    ret = forward_window_return(prices, "TST", start, trading_days=10)
    # close[15] = 115, close[5] = 105 → 115/105 - 1 ≈ 0.0952
    assert ret == pytest.approx(115 / 105 - 1)


def test_forward_window_return_returns_none_when_truncated() -> None:
    prices = _linear_prices("TST", n_days=10)
    last_date = prices["date"].iloc[-1]
    assert forward_window_return(prices, "TST", last_date, trading_days=1) is None


def test_forward_window_return_returns_none_when_asset_missing() -> None:
    prices = _linear_prices("TST", n_days=10)
    assert (
        forward_window_return(prices, "MISSING", prices["date"].iloc[0], trading_days=2)
        is None
    )


def test_forward_window_max_drawdown_zero_on_monotone_prices() -> None:
    prices = _linear_prices("TST", n_days=30)
    dd = forward_window_max_drawdown(prices, "TST", prices["date"].iloc[0], 10)
    assert dd == pytest.approx(0.0)


def test_forward_window_max_drawdown_on_known_dip() -> None:
    dates = _business_days(date(2025, 1, 1), 5)
    closes = [100.0, 110.0, 88.0, 95.0, 100.0]  # dips from 110 → 88
    df = pd.DataFrame(
        {
            "date": dates,
            "asset_id": "TST",
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1_000_000,
        }
    )
    dd = forward_window_max_drawdown(df, "TST", dates[0], 10)
    # peak at index 1 (1.10), trough at index 2 (0.88) → (0.88 - 1.10)/1.10
    assert dd == pytest.approx((0.88 - 1.10) / 1.10, abs=1e-9)


def test_outcome_label_thresholds() -> None:
    assert outcome_label(0.05) == OutcomeLabel.OUTPERFORMED
    assert outcome_label(-0.05) == OutcomeLabel.UNDERPERFORMED
    assert outcome_label(0.0005, neutral_band=0.001) == OutcomeLabel.NEUTRAL
    assert outcome_label(-0.0005, neutral_band=0.001) == OutcomeLabel.NEUTRAL


def test_outcome_label_pending_on_nan_and_none() -> None:
    assert outcome_label(None) == OutcomeLabel.PENDING
    assert outcome_label(float("nan")) == OutcomeLabel.PENDING


# ---------------------------------------------------------------------------
# evaluate_asset_decisions
# ---------------------------------------------------------------------------


def _make_rankings(prices: pd.DataFrame, asset_ranks: list[tuple[str, int]]) -> pd.DataFrame:
    rows = []
    for d in sorted(prices["date"].unique()):
        for asset_id, rank in asset_ranks:
            rows.append(
                {
                    "date": d,
                    "asset_id": asset_id,
                    "composite_score": 1.0 / rank,
                    "rank": rank,
                    "strategy_conflict": False,
                }
            )
    return pd.DataFrame(rows)


def test_evaluate_asset_decisions_produces_valid_records() -> None:
    prices = _two_asset_prices()
    rankings = _make_rankings(prices, [("A", 1), ("B", 2)])
    records = evaluate_asset_decisions(
        rankings, prices, evaluation_window_days=14, benchmark="SPY", top_n=1
    )
    assert records
    for r in records:
        assert isinstance(r, PerformanceRecord)
        assert r.asset_id == "A"
        assert r.decision_id.startswith("dec_")
        assert r.decision_id.endswith("_A")
        assert r.evaluation_end >= r.evaluation_start
        assert r.max_drawdown <= 0.0


def test_evaluate_asset_decisions_outperformed_when_pick_beats_benchmark() -> None:
    # Asset A grows at 1%/day; SPY at 0.2%/day → A outperforms.
    prices = _two_asset_prices(a_growth=0.01, b_growth=0.005)
    rankings = _make_rankings(prices, [("A", 1)])
    records = evaluate_asset_decisions(
        rankings, prices, evaluation_window_days=14, top_n=1
    )
    completed = [r for r in records if r.outcome_label != OutcomeLabel.PENDING]
    assert completed
    assert all(r.outcome_label == OutcomeLabel.OUTPERFORMED for r in completed)


def test_evaluate_asset_decisions_pending_at_end_of_history() -> None:
    prices = _two_asset_prices(n_days=30)
    rankings = _make_rankings(prices, [("A", 1)])
    records = evaluate_asset_decisions(
        rankings, prices, evaluation_window_days=30, top_n=1
    )
    # The last rebalance dates won't have a full forward window → some pending.
    pending = [r for r in records if r.outcome_label == OutcomeLabel.PENDING]
    assert pending, "expected some PENDING records near end of history"
    for r in pending:
        # Schema invariants must still hold for pending rows.
        assert r.asset_return == 0.0
        assert r.benchmark_return == 0.0
        assert r.excess_return == 0.0
        assert r.max_drawdown <= 0.0
        assert r.evaluation_end >= r.evaluation_start


def test_performance_record_round_trip_json() -> None:
    prices = _two_asset_prices()
    rankings = _make_rankings(prices, [("A", 1)])
    records = evaluate_asset_decisions(
        rankings, prices, evaluation_window_days=14, top_n=1
    )
    assert records
    for r in records:
        restored = PerformanceRecord.model_validate_json(r.model_dump_json())
        assert restored == r


def test_decision_id_uses_yyyymmdd_format() -> None:
    prices = _two_asset_prices()
    rankings = _make_rankings(prices, [("A", 1)])
    records = evaluate_asset_decisions(
        rankings, prices, evaluation_window_days=14, top_n=1
    )
    for r in records:
        # dec_YYYYMMDD_ASSET — middle token must be 8 digits.
        parts = r.decision_id.split("_")
        assert parts[0] == "dec"
        assert len(parts[1]) == 8 and parts[1].isdigit()


# ---------------------------------------------------------------------------
# by_strategy
# ---------------------------------------------------------------------------


def _make_record(
    asset_id: str,
    start: date,
    excess: float,
    label: OutcomeLabel = OutcomeLabel.OUTPERFORMED,
) -> PerformanceRecord:
    return PerformanceRecord(
        decision_id=f"dec_{start.strftime('%Y%m%d')}_{asset_id}",
        asset_id=asset_id,
        evaluation_start=start,
        evaluation_end=start + timedelta(days=30),
        asset_return=excess + 0.01,
        benchmark_return=0.01,
        excess_return=excess,
        max_drawdown=-0.02,
        outcome_label=label,
    )


def test_performance_by_strategy_without_signals_buckets_as_composite() -> None:
    records = [
        _make_record("AAA", date(2025, 1, 3), 0.02),
        _make_record("BBB", date(2025, 1, 3), -0.01, OutcomeLabel.UNDERPERFORMED),
    ]
    df = performance_by_strategy(pd.DataFrame(), records, signals_history=None)
    assert len(df) == 1
    assert df.loc[0, "strategy"] == "composite"
    assert df.loc[0, "n_decisions"] == 2
    assert df.loc[0, "hit_rate"] == pytest.approx(0.5)


def test_performance_by_strategy_attribution_with_signals() -> None:
    d = date(2025, 1, 3)
    records = [
        _make_record("AAA", d, 0.03),
        _make_record("BBB", d, -0.02, OutcomeLabel.UNDERPERFORMED),
    ]
    signals = pd.DataFrame(
        [
            {"date": d, "asset_id": "AAA", "strategy": "momentum", "score": 0.8},
            {"date": d, "asset_id": "AAA", "strategy": "value", "score": 0.1},
            {"date": d, "asset_id": "BBB", "strategy": "momentum", "score": 0.0},
            {"date": d, "asset_id": "BBB", "strategy": "value", "score": 0.6},
        ]
    )
    df = performance_by_strategy(pd.DataFrame(), records, signals_history=signals)
    by_strat = dict(zip(df["strategy"], df["mean_excess_return"]))
    assert by_strat["momentum"] == pytest.approx(0.03)
    assert by_strat["value"] == pytest.approx(-0.02)


def test_performance_by_strategy_excludes_pending() -> None:
    d = date(2025, 1, 3)
    records = [
        _make_record("AAA", d, 0.02),
        _make_record("BBB", d, 0.0, OutcomeLabel.PENDING),
    ]
    df = performance_by_strategy(pd.DataFrame(), records, signals_history=None)
    assert df.loc[0, "n_decisions"] == 1


# ---------------------------------------------------------------------------
# portfolio_eval
# ---------------------------------------------------------------------------


def test_evaluate_portfolio_empty_state() -> None:
    from ai_investment_workflow.simulation import PortfolioState

    empty_state = PortfolioState(
        history=pd.DataFrame(columns=["date", "asset_id", "weight", "value"]),
        cash_series=pd.Series(dtype=float, name="cash"),
        total_value=pd.Series(dtype=float, name="total_value"),
        returns=pd.Series(dtype=float, name="portfolio_return"),
    )
    diag = evaluate_portfolio(empty_state, pd.Series(dtype=float), pd.Series(dtype=float))
    assert diag["n_trading_days"] == 0
    assert diag["longest_losing_streak_days"] == 0


# ---------------------------------------------------------------------------
# pipeline + parquet I/O
# ---------------------------------------------------------------------------


def test_run_evaluation_end_to_end_against_fixture(tmp_path) -> None:
    prices, features = _fixture_data()
    rankings = build_rankings_history(features)
    settings = {
        "top_n_candidates": 3,
        "benchmark": "SPY",
        "rebalance_frequency": "weekly",
        "starting_cash": 100_000.0,
        "evaluation_window_days": 14,
    }
    simulation_result = run_simulation(rankings, prices, settings, risk_clipping=False)
    result = run_evaluation(
        simulation_result, rankings, prices, settings, features=features
    )
    assert "performance_records" in result
    assert "portfolio_diagnostics" in result
    assert "by_strategy_table" in result
    assert result["performance_records"], "expected at least one PerformanceRecord"

    # Persist + read back all three.
    records_df = performance_records_to_frame(result["performance_records"])
    records_path = tmp_path / "performance_records.parquet"
    records_df.to_parquet(records_path, index=False)
    rdf = pd.read_parquet(records_path)
    assert set(rdf.columns) >= {
        "decision_id",
        "asset_id",
        "evaluation_start",
        "evaluation_end",
        "asset_return",
        "benchmark_return",
        "excess_return",
        "max_drawdown",
        "outcome_label",
    }

    diag_path = tmp_path / "portfolio_diagnostics.parquet"
    pd.DataFrame([result["portfolio_diagnostics"]]).to_parquet(diag_path, index=False)
    ddf = pd.read_parquet(diag_path)
    assert len(ddf) == 1
    assert "n_trading_days" in ddf.columns

    by_strat_path = tmp_path / "performance_by_strategy.parquet"
    result["by_strategy_table"].to_parquet(by_strat_path, index=False)
    bdf = pd.read_parquet(by_strat_path)
    assert set(bdf.columns) >= {
        "strategy",
        "n_decisions",
        "hit_rate",
        "mean_excess_return",
        "median_excess_return",
    }


def test_run_evaluation_is_deterministic() -> None:
    prices, features = _fixture_data()
    rankings = build_rankings_history(features)
    settings = {
        "top_n_candidates": 3,
        "benchmark": "SPY",
        "rebalance_frequency": "weekly",
        "starting_cash": 100_000.0,
        "evaluation_window_days": 14,
    }
    sim_a = run_simulation(rankings, prices, settings, risk_clipping=False)
    sim_b = run_simulation(rankings, prices, settings, risk_clipping=False)

    a = run_evaluation(sim_a, rankings, prices, settings, features=features)
    b = run_evaluation(sim_b, rankings, prices, settings, features=features)

    assert len(a["performance_records"]) == len(b["performance_records"])
    for ra, rb in zip(a["performance_records"], b["performance_records"]):
        assert ra == rb
    pd.testing.assert_frame_equal(a["by_strategy_table"], b["by_strategy_table"])
    assert a["portfolio_diagnostics"] == b["portfolio_diagnostics"]


def test_build_signals_history_emits_long_format() -> None:
    _, features = _fixture_data()
    signals = build_signals_history(features)
    assert set(signals.columns) == {"date", "asset_id", "strategy", "score"}
    assert (signals["score"].between(-1.0, 1.0)).all()
    assert set(signals["strategy"].unique()) <= {"value", "momentum"}
