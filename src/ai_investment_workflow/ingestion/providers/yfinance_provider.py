"""yfinance-backed provider.

``yfinance`` is imported lazily inside ``__init__`` so the rest of the
package — and the Phase 1 deterministic baseline — can be used without it.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from ..base import PRICE_COLUMNS


class YFinanceProvider:
    """Pull daily OHLCV from Yahoo Finance via ``yfinance``."""

    def __init__(self) -> None:
        try:
            import yfinance as _yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is not installed. Install it with "
                "`pip install yfinance` to use YFinanceProvider."
            ) from exc
        self._yf = _yf

    def fetch(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        raw = self._yf.download(
            tickers=tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        frames: list[pd.DataFrame] = []
        if isinstance(raw.columns, pd.MultiIndex):
            top_level = set(raw.columns.get_level_values(0))
            for ticker in tickers:
                if ticker not in top_level:
                    continue
                frames.append(self._normalize(raw[ticker].reset_index(), ticker))
        elif len(tickers) == 1:
            frames.append(self._normalize(raw.reset_index(), tickers[0]))

        if not frames:
            return pd.DataFrame(columns=PRICE_COLUMNS)

        df = pd.concat(frames, ignore_index=True)
        return df.sort_values(["asset_id", "date"]).reset_index(drop=True)

    @staticmethod
    def _normalize(sub: pd.DataFrame, ticker: str) -> pd.DataFrame:
        sub = sub.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        sub["asset_id"] = ticker
        sub["date"] = pd.to_datetime(sub["date"]).dt.date
        sub = sub.dropna(subset=["close"])
        return sub[PRICE_COLUMNS]
