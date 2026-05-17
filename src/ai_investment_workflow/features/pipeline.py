"""Feature pipeline: orchestrate all builders and emit ``FeatureSet`` objects."""

from __future__ import annotations

from datetime import date
from typing import Sequence

import pandas as pd

from ..schemas import FeatureSet
from .base import KEY_COLUMNS, FeatureBuilder
from .momentum import MomentumBuilder
from .recent_returns import RecentReturnsBuilder
from .valuation import ValuationBuilder
from .volatility import VolatilityBuilder


def default_builders() -> list[FeatureBuilder]:
    """Fresh list of the canonical Phase 3 builders."""
    return [
        MomentumBuilder(),
        RecentReturnsBuilder(),
        VolatilityBuilder(),
        ValuationBuilder(),
    ]


def compute_feature_frame(
    prices: pd.DataFrame,
    builders: Sequence[FeatureBuilder] | None = None,
) -> pd.DataFrame:
    """Run every builder and merge the results on ``(date, asset_id)``."""
    builder_list = list(builders) if builders is not None else default_builders()
    if not builder_list:
        raise ValueError("at least one FeatureBuilder must be provided")

    frames = [b.compute(prices) for b in builder_list]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=KEY_COLUMNS, how="outer")

    return merged.sort_values(["asset_id", "date"]).reset_index(drop=True)


def build_features(
    prices: pd.DataFrame,
    as_of: date | None = None,
    builders: Sequence[FeatureBuilder] | None = None,
) -> dict[str, FeatureSet]:
    """One ``FeatureSet`` per ``asset_id`` snapshotted at ``as_of``.

    ``as_of=None`` uses the most recent date in ``prices``. NaN features
    are omitted from the resulting ``FeatureSet.features`` mapping.
    """
    frame = compute_feature_frame(prices, builders=builders)
    if as_of is None:
        as_of = frame["date"].max()

    snapshot = frame.loc[frame["date"] == as_of]
    feature_cols = [c for c in frame.columns if c not in KEY_COLUMNS]

    out: dict[str, FeatureSet] = {}
    for row in snapshot.itertuples(index=False):
        features: dict[str, float] = {}
        for col in feature_cols:
            value = getattr(row, col)
            if pd.notna(value):
                features[col] = float(value)
        out[row.asset_id] = FeatureSet(
            asset_id=row.asset_id,
            timestamp=as_of,
            features=features,
        )
    return out
