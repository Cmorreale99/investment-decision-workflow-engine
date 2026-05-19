"""Phase 4 strategy + composite + ranking + conflict tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ai_investment_workflow.features import compute_feature_frame
from ai_investment_workflow.ingestion import FixtureProvider
from ai_investment_workflow.schemas import StrategySignal
from ai_investment_workflow.strategies import (
    MomentumStrategy,
    Strategy,
    ValueStrategy,
    build_strategy_signals,
    composite_score,
    detect_conflicts,
    load_enabled_strategies,
    rank_candidates,
    score_strategies,
)


AS_OF = date(2025, 6, 1)


def _hand_built_features() -> pd.DataFrame:
    """Three assets at a single date with hand-chosen feature values."""
    rows = [
        {
            "date": AS_OF,
            "asset_id": "A",
            "distance_from_52w_high": -0.30,
            "distance_from_52w_low": 0.05,
            "momentum_3m": 0.20,
            "momentum_6m": 0.40,
            "return_21d": 0.05,
            "return_63d": 0.10,
        },
        {
            "date": AS_OF,
            "asset_id": "B",
            "distance_from_52w_high": 0.00,
            "distance_from_52w_low": 0.50,
            "momentum_3m": 0.10,
            "momentum_6m": 0.15,
            "return_21d": 0.02,
            "return_63d": 0.05,
        },
        {
            "date": AS_OF,
            "asset_id": "C",
            "distance_from_52w_high": -0.50,
            "distance_from_52w_low": 0.00,
            "momentum_3m": -0.10,
            "momentum_6m": -0.20,
            "return_21d": -0.05,
            "return_63d": -0.10,
        },
    ]
    return pd.DataFrame(rows)


def _fixture_feature_frame() -> pd.DataFrame:
    provider = FixtureProvider()
    prices = provider.fetch(
        ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "SPY"],
        date(2025, 1, 1),
        date(2025, 6, 1),
    )
    return compute_feature_frame(prices)


# ---------------------------------------------------------------------------
# Protocol contract
# ---------------------------------------------------------------------------


def test_strategies_satisfy_protocol() -> None:
    for strat in (ValueStrategy(), MomentumStrategy()):
        assert isinstance(strat, Strategy)
        assert strat.name in ("value", "momentum")
        assert isinstance(strat.required_features, tuple)
        assert len(strat.required_features) > 0


def test_load_enabled_strategies_from_config() -> None:
    enabled = load_enabled_strategies()
    names = {s.name for s in enabled}
    # config/strategies.yaml has both enabled.
    assert names == {"value", "momentum"}


def test_strategy_missing_feature_raises() -> None:
    bad = pd.DataFrame({"date": [AS_OF], "asset_id": ["X"]})
    with pytest.raises(ValueError, match="missing required feature columns"):
        ValueStrategy().score(bad)
    with pytest.raises(ValueError, match="missing required feature columns"):
        MomentumStrategy().score(bad)


# ---------------------------------------------------------------------------
# Value strategy math
# ---------------------------------------------------------------------------


def test_value_strategy_ranks_cheapest_highest() -> None:
    features = _hand_built_features()
    out = ValueStrategy().score(features).set_index("asset_id")["score"]
    # C is closest to 52w low AND farthest below 52w high → highest value proxy.
    # B is at the 52w high and far from the low → lowest value proxy.
    assert out["C"] > out["A"] > out["B"]


def test_value_strategy_score_in_range() -> None:
    features = _hand_built_features()
    out = ValueStrategy().score(features)["score"]
    assert (out.dropna() >= -1.0).all() and (out.dropna() <= 1.0).all()


# ---------------------------------------------------------------------------
# Momentum strategy math
# ---------------------------------------------------------------------------


def test_momentum_strategy_orders_by_momentum() -> None:
    features = _hand_built_features()
    out = MomentumStrategy().score(features).set_index("asset_id")["score"]
    # A has the largest momentum/returns; C has the most negative.
    assert out["A"] > out["B"] > out["C"]


def test_momentum_strategy_score_in_range() -> None:
    features = _fixture_feature_frame()
    out = MomentumStrategy().score(features)["score"]
    finite = out.dropna()
    assert (finite >= -1.0).all() and (finite <= 1.0).all()


def test_momentum_uses_optional_features_when_present() -> None:
    features = _hand_built_features()
    out_with = MomentumStrategy().score(features).set_index("asset_id")["score"]
    # Drop the optional columns and re-score: result must still produce
    # finite, in-range scores using only the two primary features.
    primary_only = features.drop(columns=["return_21d", "return_63d"])
    out_without = MomentumStrategy().score(primary_only).set_index("asset_id")["score"]
    # Both should rank A > B > C, but values typically differ.
    assert out_with["A"] > out_with["C"]
    assert out_without["A"] > out_without["C"]


# ---------------------------------------------------------------------------
# Composite + ranking
# ---------------------------------------------------------------------------


def test_composite_score_with_equal_weights() -> None:
    scores = {
        "value": {"A": 0.2, "B": -0.4, "C": 0.6},
        "momentum": {"A": 0.4, "B": 0.0, "C": -0.6},
    }
    weights = {"value": 0.5, "momentum": 0.5}
    out = composite_score(scores, weights, as_of=AS_OF)
    assert out["A"] == pytest.approx(0.3)
    assert out["B"] == pytest.approx(-0.2)
    assert out["C"] == pytest.approx(0.0)


def test_composite_score_renormalizes_unequal_weights() -> None:
    scores = {
        "value": {"A": 0.4},
        "momentum": {"A": 0.0},
    }
    # 0.7/0.3 → 0.7*0.4 + 0.3*0.0 = 0.28
    out = composite_score(scores, {"value": 0.7, "momentum": 0.3}, as_of=AS_OF)
    assert out["A"] == pytest.approx(0.28)


def test_composite_score_excludes_asset_with_no_valid_scores() -> None:
    scores = {"value": {"A": 0.2}, "momentum": {"B": 0.3}}
    out = composite_score(scores, {"value": 0.5, "momentum": 0.5}, as_of=AS_OF)
    # A has only value (renormalized to 1.0), B has only momentum.
    assert set(out.index) == {"A", "B"}
    assert out["A"] == pytest.approx(0.2)
    assert out["B"] == pytest.approx(0.3)


def test_composite_score_renormalizes_per_asset_when_partial_coverage() -> None:
    scores = {
        "value": {"A": 0.5, "B": -0.5},  # B missing from momentum below
        "momentum": {"A": 0.1},
    }
    out = composite_score(scores, {"value": 0.5, "momentum": 0.5}, as_of=AS_OF)
    # A blends both: 0.5*0.5 + 0.5*0.1 = 0.30
    assert out["A"] == pytest.approx(0.30)
    # B uses only value strategy → -0.5
    assert out["B"] == pytest.approx(-0.5)


def test_composite_score_clipped_to_range() -> None:
    # Even with weights that could blow it out, score stays in [-1, 1].
    scores = {"value": {"A": 1.0}, "momentum": {"A": 1.0}}
    out = composite_score(scores, {"value": 0.5, "momentum": 0.5})
    assert -1.0 <= out["A"] <= 1.0


def test_rank_candidates_orders_descending_and_starts_at_one() -> None:
    composite = pd.Series({"A": 0.1, "B": -0.5, "C": 0.7}, name=AS_OF)
    ranked = rank_candidates(composite)
    assert list(ranked["asset_id"]) == ["C", "A", "B"]
    assert list(ranked["rank"]) == [1, 2, 3]
    assert (ranked["date"] == AS_OF).all()


def test_rank_candidates_tiebreaks_by_asset_id_ascending() -> None:
    composite = pd.Series({"B": 0.5, "A": 0.5, "C": 0.5}, name=AS_OF)
    ranked = rank_candidates(composite)
    # All tied → asset_id ascending.
    assert list(ranked["asset_id"]) == ["A", "B", "C"]
    assert list(ranked["rank"]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


def test_conflict_sign_disagreement_above_threshold() -> None:
    scores = {
        "value": {"A": 0.5, "B": -0.5, "C": 0.5},
        "momentum": {"A": -0.5, "B": -0.5, "C": 0.5},
    }
    flags = detect_conflicts(scores, threshold=0.4, require_sign_disagreement=True)
    assert flags["A"] is np.True_ or flags["A"] == True  # noqa: E712 - dtype is bool
    assert flags["B"] == False  # noqa: E712 - same sign
    assert flags["C"] == False  # noqa: E712 - same sign


def test_conflict_below_threshold_not_flagged() -> None:
    scores = {
        "value": {"A": 0.2},
        "momentum": {"A": -0.2},
    }
    flags = detect_conflicts(scores, threshold=0.4, require_sign_disagreement=True)
    assert flags["A"] == False  # noqa: E712


def test_conflict_single_strategy_never_flags() -> None:
    scores = {"value": {"A": 0.9}}
    flags = detect_conflicts(scores, threshold=0.4, require_sign_disagreement=True)
    assert flags["A"] == False  # noqa: E712


def test_conflict_spread_mode_flags_when_spread_exceeds_double_threshold() -> None:
    scores = {
        "value": {"A": 0.7, "B": 0.4},
        "momentum": {"A": 0.0, "B": 0.35},
    }
    flags = detect_conflicts(scores, threshold=0.3, require_sign_disagreement=False)
    # A spread = 0.7 → >= 0.6 → conflict
    assert flags["A"] == True  # noqa: E712
    # B spread = 0.05 → no conflict
    assert flags["B"] == False  # noqa: E712


# ---------------------------------------------------------------------------
# StrategySignal builder + persistence
# ---------------------------------------------------------------------------


def test_build_strategy_signals_against_fixture_frame() -> None:
    features = _fixture_feature_frame()
    signals = build_strategy_signals(features)
    assert signals, "expected at least one signal from fixture data"
    for asset_id, sig in signals.items():
        assert isinstance(sig, StrategySignal)
        assert sig.asset_id == asset_id
        assert 1 <= sig.rank <= len(signals)
        assert -1.0 <= sig.composite_score <= 1.0
        for k, v in sig.strategy_scores.items():
            assert -1.0 <= v <= 1.0
            assert k in {"value", "momentum"}


def test_build_strategy_signals_round_trip_json() -> None:
    features = _fixture_feature_frame()
    signals = build_strategy_signals(features)
    for sig in signals.values():
        restored = StrategySignal.model_validate_json(sig.model_dump_json())
        assert restored == sig


def test_signals_and_rankings_parquet_roundtrip(tmp_path) -> None:
    features = _fixture_feature_frame()
    signals = build_strategy_signals(features)
    assert signals
    as_of = next(iter(signals.values())).timestamp

    signal_rows = [
        {"date": as_of, "asset_id": a, "strategy": s, "score": v}
        for a, sig in signals.items()
        for s, v in sig.strategy_scores.items()
    ]
    ranking_rows = [
        {
            "date": as_of,
            "asset_id": sig.asset_id,
            "composite_score": sig.composite_score,
            "rank": sig.rank,
            "strategy_conflict": sig.strategy_conflict,
        }
        for sig in signals.values()
    ]
    signals_df = pd.DataFrame(signal_rows)
    rankings_df = pd.DataFrame(ranking_rows).sort_values("rank")

    signals_path = tmp_path / "signals.parquet"
    rankings_path = tmp_path / "rankings.parquet"
    signals_df.to_parquet(signals_path, index=False)
    rankings_df.to_parquet(rankings_path, index=False)

    rs = pd.read_parquet(signals_path)
    rr = pd.read_parquet(rankings_path)
    assert list(rs.columns) == ["date", "asset_id", "strategy", "score"]
    assert list(rr.columns) == [
        "date",
        "asset_id",
        "composite_score",
        "rank",
        "strategy_conflict",
    ]
    assert len(rs) == len(signals_df)
    assert len(rr) == len(rankings_df)


# ---------------------------------------------------------------------------
# Determinism + NaN + disabled
# ---------------------------------------------------------------------------


def test_score_strategies_is_deterministic() -> None:
    features = _fixture_feature_frame()
    first = score_strategies(features)
    second = score_strategies(features)
    assert first == second


def test_build_strategy_signals_is_deterministic() -> None:
    features = _fixture_feature_frame()
    a = build_strategy_signals(features)
    b = build_strategy_signals(features)
    assert a == b


def test_nan_inputs_excluded_from_strategy_scores() -> None:
    features = _hand_built_features().copy()
    # Wipe every momentum/return input for asset A — its momentum score
    # becomes NaN because there is nothing left to average.
    momentum_cols = ["momentum_3m", "momentum_6m", "return_21d", "return_63d"]
    features.loc[features["asset_id"] == "A", momentum_cols] = np.nan
    scores = score_strategies(
        features, as_of=AS_OF, strategies=[ValueStrategy(), MomentumStrategy()]
    )
    assert "A" in scores["value"]
    assert "A" not in scores["momentum"]
    # Composite still ranks A using only the value contribution.
    out = composite_score(
        scores, {"value": 0.5, "momentum": 0.5}, as_of=AS_OF
    )
    assert "A" in out.index


def test_disabled_strategy_via_explicit_strategies_arg() -> None:
    features = _hand_built_features()
    only_momentum = score_strategies(
        features, as_of=AS_OF, strategies=[MomentumStrategy()]
    )
    assert set(only_momentum.keys()) == {"momentum"}
