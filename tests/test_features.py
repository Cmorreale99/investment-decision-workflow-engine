"""Phase 3 feature engineering tests."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ai_investment_workflow.features import (
    FeatureBuilder,
    KEY_COLUMNS,
    MomentumBuilder,
    RecentReturnsBuilder,
    ValuationBuilder,
    VolatilityBuilder,
    build_company_snapshots,
    build_features,
    compute_feature_frame,
    default_builders,
    validate_price_frame,
)
from ai_investment_workflow.ingestion import FixtureProvider
from ai_investment_workflow.schemas import CompanySnapshot, FeatureSet


FIXTURE_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "SPY"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_frame(asset_id: str, n_days: int, start_close: float = 100.0) -> pd.DataFrame:
    """One row per business day, close = start_close + i."""
    dates: list[date] = []
    cursor = date(2025, 1, 1)
    while len(dates) < n_days:
        if cursor.weekday() < 5:
            dates.append(cursor)
        cursor += timedelta(days=1)
    closes = [start_close + i for i in range(n_days)]
    return pd.DataFrame(
        {
            "date": dates,
            "asset_id": asset_id,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000] * n_days,
        }
    )


def _fixture_prices() -> pd.DataFrame:
    provider = FixtureProvider()
    return provider.fetch(FIXTURE_TICKERS, date(2025, 1, 1), date(2025, 6, 1))


# ---------------------------------------------------------------------------
# Protocol contract
# ---------------------------------------------------------------------------


def test_all_default_builders_satisfy_protocol() -> None:
    for builder in default_builders():
        assert isinstance(builder, FeatureBuilder)


def test_validate_price_frame_rejects_missing_columns() -> None:
    bad = pd.DataFrame({"date": [], "asset_id": []})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_price_frame(bad)


# ---------------------------------------------------------------------------
# Momentum / recent returns math
# ---------------------------------------------------------------------------


def test_momentum_1m_value_on_linear_frame() -> None:
    prices = _linear_frame("TST", 25)
    out = MomentumBuilder().compute(prices)
    # close[21] = 121, close[0] = 100  →  0.21
    row_21 = out.iloc[21]
    assert row_21["momentum_1m"] == pytest.approx(0.21)
    # Rows before index 21 must be NaN.
    assert out.loc[:20, "momentum_1m"].isna().all()


def test_recent_returns_5d_value_on_linear_frame() -> None:
    prices = _linear_frame("TST", 10)
    out = RecentReturnsBuilder().compute(prices)
    # close[5] / close[0] - 1 = 105/100 - 1 = 0.05
    assert out.iloc[5]["return_5d"] == pytest.approx(0.05)
    assert out.loc[:4, "return_5d"].isna().all()


def test_short_history_yields_nan_for_long_windows() -> None:
    prices = _linear_frame("TST", 10)
    out = compute_feature_frame(prices)
    # 10 rows < every momentum/recent_returns window except return_5d
    assert out["momentum_1m"].isna().all()
    assert out["momentum_3m"].isna().all()
    assert out["momentum_6m"].isna().all()
    assert out["momentum_12m"].isna().all()
    assert out["return_21d"].isna().all()
    assert out["return_63d"].isna().all()
    # return_5d defined for rows 5..9
    assert out["return_5d"].notna().sum() == 5


# ---------------------------------------------------------------------------
# Volatility math
# ---------------------------------------------------------------------------


def test_volatility_21d_against_hand_calculation() -> None:
    prices = _linear_frame("TST", 30, start_close=100.0)
    out = VolatilityBuilder().compute(prices)
    # Compute expected: 21-day rolling std of log returns at row 21, annualized.
    closes = prices["close"].to_numpy(dtype=float)
    log_rets = np.log(closes[1:] / closes[:-1])  # length 29; log_rets[k] = log(c[k+1]/c[k])
    # In the builder, _log_ret[i] = log(close[i] / close[i-1]) with _log_ret[0] = NaN.
    # rolling(21) at output row 21 uses _log_ret[1..21], which corresponds to
    # log_rets[0..20] in this test's indexing.
    window = log_rets[0:21]
    expected = window.std(ddof=1) * np.sqrt(252)
    assert out.iloc[21]["volatility_21d"] == pytest.approx(expected, rel=1e-9)


def test_volatility_nan_on_insufficient_history() -> None:
    prices = _linear_frame("TST", 10)
    out = VolatilityBuilder().compute(prices)
    assert out["volatility_21d"].isna().all()
    assert out["volatility_63d"].isna().all()


def test_volatility_is_nonnegative_when_defined() -> None:
    prices = _fixture_prices()
    out = VolatilityBuilder().compute(prices)
    defined = out["volatility_21d"].dropna()
    assert not defined.empty
    assert (defined >= 0).all()


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------


def test_valuation_distance_signs_on_linear_frame() -> None:
    prices = _linear_frame("TST", 30, start_close=100.0)
    out = ValuationBuilder().compute(prices)
    # With monotonically rising prices, close == high → distance_from_high = 0,
    # and close > low → distance_from_low > 0 after the first row.
    assert (out["distance_from_52w_high"] <= 0).all()
    assert (out["distance_from_52w_low"] >= 0).all()
    # Final row: close=129, low=100 → (129-100)/100 = 0.29
    last = out.iloc[-1]
    assert last["distance_from_52w_low"] == pytest.approx(0.29)
    assert last["distance_from_52w_high"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Pipeline: columns, ordering, FeatureSet/Snapshot round-trips
# ---------------------------------------------------------------------------


EXPECTED_FEATURE_COLUMNS = {
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m",
    "return_5d",
    "return_21d",
    "return_63d",
    "volatility_21d",
    "volatility_63d",
    "distance_from_52w_high",
    "distance_from_52w_low",
}


def test_compute_feature_frame_contains_all_expected_columns() -> None:
    frame = compute_feature_frame(_fixture_prices())
    assert set(KEY_COLUMNS).issubset(frame.columns)
    assert EXPECTED_FEATURE_COLUMNS.issubset(frame.columns)


def test_compute_feature_frame_preserves_keys_from_input() -> None:
    prices = _fixture_prices()
    frame = compute_feature_frame(prices)
    src_keys = prices[["date", "asset_id"]].drop_duplicates()
    out_keys = frame[["date", "asset_id"]].drop_duplicates()
    assert len(src_keys) == len(out_keys)


def test_build_features_returns_one_feature_set_per_asset() -> None:
    prices = _fixture_prices()
    result = build_features(prices)
    assert set(result.keys()) == set(FIXTURE_TICKERS)
    for asset_id, fs in result.items():
        assert isinstance(fs, FeatureSet)
        assert fs.asset_id == asset_id
        # Every value should be finite when present.
        for name, value in fs.features.items():
            assert np.isfinite(value), f"non-finite feature {asset_id}.{name}={value}"


def test_build_features_feature_set_round_trip_json() -> None:
    result = build_features(_fixture_prices())
    for fs in result.values():
        restored = FeatureSet.model_validate_json(fs.model_dump_json())
        assert restored == fs


def test_build_features_uses_max_date_when_as_of_none() -> None:
    prices = _fixture_prices()
    result = build_features(prices, as_of=None)
    expected = prices["date"].max()
    for fs in result.values():
        assert fs.timestamp == expected


def test_build_features_honors_explicit_as_of() -> None:
    prices = _fixture_prices()
    chosen = sorted(prices["date"].unique())[60]
    result = build_features(prices, as_of=chosen)
    for fs in result.values():
        assert fs.timestamp == chosen


# ---------------------------------------------------------------------------
# CompanySnapshot round-trip
# ---------------------------------------------------------------------------


def test_build_company_snapshots_round_trip() -> None:
    prices = _fixture_prices()
    sectors = {"AAPL": "Technology", "MSFT": "Technology", "SPY": "ETF"}
    snaps = build_company_snapshots(prices, sectors=sectors)
    assert set(snaps.keys()) == set(FIXTURE_TICKERS)
    for asset_id, snap in snaps.items():
        assert isinstance(snap, CompanySnapshot)
        assert snap.asset_id == asset_id
        # Unknown sector falls back to DEFAULT_SECTOR.
        if asset_id not in sectors:
            assert snap.sector == "Unknown"
        else:
            assert snap.sector == sectors[asset_id]
        restored = CompanySnapshot.model_validate_json(snap.model_dump_json())
        assert restored == snap


# ---------------------------------------------------------------------------
# Parquet I/O + determinism
# ---------------------------------------------------------------------------


def test_pipeline_parquet_roundtrip(tmp_path) -> None:
    prices = _fixture_prices()
    frame = compute_feature_frame(prices)
    path = tmp_path / "features.parquet"
    frame.to_parquet(path, index=False)
    reloaded = pd.read_parquet(path)
    assert list(reloaded.columns) == list(frame.columns)
    assert len(reloaded) == len(frame)


def test_compute_feature_frame_is_deterministic() -> None:
    prices = _fixture_prices()
    first = compute_feature_frame(prices)
    second = compute_feature_frame(prices)
    pd.testing.assert_frame_equal(first, second)
