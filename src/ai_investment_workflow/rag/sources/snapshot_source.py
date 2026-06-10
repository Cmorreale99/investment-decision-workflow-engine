"""CompanySnapshot source backed by normalized prices (Phase 2/3)."""

from __future__ import annotations

from datetime import date
from typing import Mapping

import pandas as pd

from ...features import build_company_snapshots
from ...schemas import CompanySnapshot


class SnapshotSource:
    """Builds ``CompanySnapshot`` from a prices frame + optional sector map.

    Wraps the Phase 3 ``build_company_snapshots`` builder so the snapshot
    embedded in a ContextPacket is byte-identical to the one the risk
    layer sees. Assets with no price row at ``as_of`` yield ``{}``.
    """

    name = "snapshot_source"

    def __init__(
        self,
        prices: pd.DataFrame,
        sectors: Mapping[str, str] | None = None,
    ) -> None:
        self._prices = prices
        self._sectors = dict(sectors or {})
        self._cache: dict[date, dict[str, CompanySnapshot]] = {}

    def _snapshots_for(self, as_of: date) -> dict[str, CompanySnapshot]:
        cached = self._cache.get(as_of)
        if cached is None:
            cached = build_company_snapshots(
                self._prices, sectors=self._sectors, as_of=as_of
            )
            self._cache[as_of] = cached
        return cached

    def fetch(self, asset_id: str, as_of: date) -> dict:
        snapshot = self._snapshots_for(as_of).get(asset_id)
        if snapshot is None:
            return {}
        return {"company_snapshot": snapshot}
