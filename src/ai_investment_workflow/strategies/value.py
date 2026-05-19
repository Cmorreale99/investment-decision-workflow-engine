"""Value-proxy strategy using 52-week price-position features.

This is a **price-relative value proxy**, not a fundamental value strategy.
It does **not** use P/E, P/B, EV/EBITDA, FCF yield, or dividend yield —
those require fundamentals ingestion, which is not yet wired into Phase 2.

Mechanics
---------
Two inputs from ``ValuationBuilder`` (Phase 3):

- ``distance_from_52w_high`` ∈ (-∞, 0]
- ``distance_from_52w_low``  ∈ [0, ∞)

We invert both signs so that "higher = closer to the cheap end of the
52-week range," then take the per-date z-score of each and average. The
final score is clipped to ``[-1, 1]`` via ``base.to_score_range``.

# TODO: fundamentals
#   Augment with z-scored fundamental ratios when those become available.
#   Until then, this strategy is at best a coarse trend-following inverse.
"""

from __future__ import annotations

import pandas as pd

from .base import assemble_long_output, average_z, to_score_range


class ValueStrategy:
    name: str = "value"
    required_features: tuple[str, ...] = (
        "distance_from_52w_high",
        "distance_from_52w_low",
    )

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.required_features if c not in features.columns]
        if missing:
            raise ValueError(
                f"value strategy missing required feature columns: {missing}"
            )
        df = features[["date", "asset_id", *self.required_features]].copy()
        df["_inv_high"] = -df["distance_from_52w_high"]
        df["_inv_low"] = -df["distance_from_52w_low"]
        z = average_z(df, ["_inv_high", "_inv_low"])
        return assemble_long_output(features, to_score_range(z))
