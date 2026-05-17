"""Regenerate ``tests/fixtures/prices.parquet`` deterministically.

Run from the repo root::

    python tests/fixtures/generate_prices.py

The fixture covers a small ticker set (AAPL, MSFT, NVDA, AMZN, GOOGL, SPY)
across business days in 2025-01-02 → 2025-04-30 with seeded synthetic OHLCV.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "SPY"]
START = date(2025, 1, 2)
END = date(2025, 4, 30)
SEED = 42

OUT_PATH = Path(__file__).resolve().parent / "prices.parquet"


def _business_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def build_frame() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    days = _business_days(START, END)
    rows: list[dict] = []
    for ticker in TICKERS:
        price = 100.0 + (abs(hash(ticker)) % 200)
        for d in days:
            ret = rng.normal(0.0005, 0.012)
            new_close = price * (1.0 + ret)
            open_ = price * (1.0 + rng.normal(0.0, 0.002))
            high = max(open_, new_close) * (1.0 + abs(rng.normal(0.0, 0.003)))
            low = min(open_, new_close) * (1.0 - abs(rng.normal(0.0, 0.003)))
            volume = int(rng.integers(1_000_000, 10_000_000))
            rows.append(
                {
                    "date": d,
                    "asset_id": ticker,
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(new_close),
                    "volume": volume,
                }
            )
            price = new_close
    df = pd.DataFrame(rows)
    return df.sort_values(["asset_id", "date"]).reset_index(drop=True)


def main() -> None:
    df = build_frame()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(df)} rows across {df['asset_id'].nunique()} tickers -> {OUT_PATH}")


if __name__ == "__main__":
    main()
