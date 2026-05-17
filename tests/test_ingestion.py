"""Phase 2 ingestion tests. Default provider = ``FixtureProvider``."""

from __future__ import annotations

import importlib.util
from datetime import date

import pandas as pd
import pytest

from ai_investment_workflow.ingestion import (
    PRICE_COLUMNS,
    FixtureProvider,
    MarketDataProvider,
    YFinanceProvider,
    default_fixture_path,
    get_provider,
    run_ingestion,
    validate_prices,
)

FIXTURE_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
FIXTURE_BENCHMARK = "SPY"
FIXTURE_START = date(2025, 1, 1)
FIXTURE_END = date(2025, 6, 1)


# ---------------------------------------------------------------------------
# Protocol contract
# ---------------------------------------------------------------------------


def test_fixture_provider_satisfies_protocol() -> None:
    assert isinstance(FixtureProvider(), MarketDataProvider)


def test_yfinance_provider_module_imports_without_yfinance() -> None:
    # The module file itself must import cleanly even if yfinance is missing.
    from ai_investment_workflow.ingestion.providers import yfinance_provider

    assert hasattr(yfinance_provider, "YFinanceProvider")


def test_yfinance_provider_lazy_import_raises_when_missing() -> None:
    if importlib.util.find_spec("yfinance") is not None:
        pytest.skip("yfinance is installed in this environment")
    with pytest.raises(ImportError):
        YFinanceProvider()


# ---------------------------------------------------------------------------
# FixtureProvider
# ---------------------------------------------------------------------------


def test_fixture_path_exists() -> None:
    assert default_fixture_path().is_file()


def test_fixture_provider_roundtrip() -> None:
    provider = FixtureProvider()
    df = provider.fetch(FIXTURE_TICKERS + [FIXTURE_BENCHMARK], FIXTURE_START, FIXTURE_END)
    assert list(df.columns) == PRICE_COLUMNS
    assert not df.empty
    assert set(df["asset_id"].unique()) == set(FIXTURE_TICKERS + [FIXTURE_BENCHMARK])
    for ticker, group in df.groupby("asset_id"):
        assert group["date"].is_monotonic_increasing, f"{ticker} dates not monotonic"


def test_fixture_provider_filters_unknown_tickers() -> None:
    provider = FixtureProvider()
    df = provider.fetch(["AAPL", "DOES_NOT_EXIST"], FIXTURE_START, FIXTURE_END)
    assert set(df["asset_id"].unique()) == {"AAPL"}


def test_fixture_provider_filters_date_window() -> None:
    provider = FixtureProvider()
    df = provider.fetch(["AAPL"], date(2025, 2, 1), date(2025, 2, 28))
    assert not df.empty
    assert df["date"].min() >= date(2025, 2, 1)
    assert df["date"].max() <= date(2025, 2, 28)


# ---------------------------------------------------------------------------
# validate_prices
# ---------------------------------------------------------------------------


def _good_frame() -> pd.DataFrame:
    return FixtureProvider().fetch(["AAPL", "MSFT"], FIXTURE_START, FIXTURE_END)


def test_validate_prices_accepts_clean_frame() -> None:
    validate_prices(_good_frame())


def test_validate_prices_rejects_missing_column() -> None:
    df = _good_frame().drop(columns=["close"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_prices(df)


def test_validate_prices_rejects_empty_frame() -> None:
    empty = pd.DataFrame(columns=PRICE_COLUMNS)
    with pytest.raises(ValueError, match="empty"):
        validate_prices(empty)


def test_validate_prices_rejects_nan_close() -> None:
    df = _good_frame()
    df.loc[df.index[0], "close"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_prices(df)


def test_validate_prices_rejects_duplicate_rows() -> None:
    df = _good_frame()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_prices(df)


def test_validate_prices_rejects_nonmonotonic_dates() -> None:
    df = _good_frame()
    # Reverse one ticker's rows so dates are not monotonic increasing.
    aapl_mask = df["asset_id"] == "AAPL"
    aapl = df.loc[aapl_mask].iloc[::-1]
    df = pd.concat([aapl, df.loc[~aapl_mask]], ignore_index=True)
    with pytest.raises(ValueError, match="monotonically increasing"):
        validate_prices(df)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_pipeline_writes_raw_and_processed_parquet(tmp_path) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    df = run_ingestion(
        provider=FixtureProvider(),
        tickers=["AAPL", "MSFT"],
        benchmark="SPY",
        start=FIXTURE_START,
        end=FIXTURE_END,
        raw_dir=raw,
        processed_dir=processed,
    )
    assert list(df.columns) == PRICE_COLUMNS
    assert set(df["asset_id"].unique()) == {"AAPL", "MSFT", "SPY"}

    for ticker in ("AAPL", "MSFT", "SPY"):
        per_ticker = raw / f"{ticker}.parquet"
        assert per_ticker.is_file(), f"missing raw parquet: {per_ticker}"
        loaded = pd.read_parquet(per_ticker)
        assert set(loaded["asset_id"].unique()) == {ticker}

    merged_path = processed / "prices.parquet"
    assert merged_path.is_file()
    merged = pd.read_parquet(merged_path)
    assert list(merged.columns) == PRICE_COLUMNS
    assert len(merged) == len(df)


def test_pipeline_includes_benchmark_even_when_not_in_universe(tmp_path) -> None:
    df = run_ingestion(
        provider=FixtureProvider(),
        tickers=["AAPL"],  # SPY intentionally absent
        benchmark="SPY",
        start=FIXTURE_START,
        end=FIXTURE_END,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )
    assert "SPY" in set(df["asset_id"].unique())


def test_pipeline_deduplicates_benchmark_in_universe(tmp_path) -> None:
    # If SPY is already in tickers, it must not appear twice.
    df = run_ingestion(
        provider=FixtureProvider(),
        tickers=["AAPL", "SPY"],
        benchmark="SPY",
        start=FIXTURE_START,
        end=FIXTURE_END,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )
    spy_files = list((tmp_path / "raw").glob("SPY.parquet"))
    assert len(spy_files) == 1
    assert not df.duplicated(subset=["date", "asset_id"]).any()


# ---------------------------------------------------------------------------
# get_provider
# ---------------------------------------------------------------------------


def test_get_provider_default_is_fixture() -> None:
    assert isinstance(get_provider(None), FixtureProvider)
    assert isinstance(get_provider("fixture"), FixtureProvider)


def test_get_provider_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown data_provider"):
        get_provider("alpha_vantage")
