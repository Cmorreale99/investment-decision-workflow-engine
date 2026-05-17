"""Build per-asset ``CompanySnapshot`` objects at a point in time."""

from __future__ import annotations

from datetime import date
from typing import Mapping

import pandas as pd

from ..schemas import CompanySnapshot

DEFAULT_SECTOR: str = "Unknown"


def build_company_snapshots(
    prices: pd.DataFrame,
    sectors: Mapping[str, str] | None = None,
    as_of: date | None = None,
) -> dict[str, CompanySnapshot]:
    """One ``CompanySnapshot`` per asset present in ``prices`` at ``as_of``.

    Sectors are looked up from ``sectors``; missing assets fall back to
    ``DEFAULT_SECTOR``. Phase 2 ingestion does not yet pull sector
    classifications, so sectors must be injected by the caller.
    """
    if {"date", "asset_id", "close"}.difference(prices.columns):
        raise ValueError("prices must contain date, asset_id, close")

    sectors = sectors or {}
    if as_of is None:
        as_of = prices["date"].max()

    latest = (
        prices.loc[prices["date"] == as_of, ["asset_id", "close"]]
        .drop_duplicates(subset="asset_id", keep="last")
    )

    snapshots: dict[str, CompanySnapshot] = {}
    for row in latest.itertuples(index=False):
        snapshots[row.asset_id] = CompanySnapshot(
            asset_id=row.asset_id,
            timestamp=as_of,
            sector=sectors.get(row.asset_id, DEFAULT_SECTOR),
            price=float(row.close),
        )
    return snapshots
