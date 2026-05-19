"""Momentum strategy: cross-sectional z-score blend of trailing returns.

Primary inputs (required): ``momentum_3m``, ``momentum_6m``.
Optional inputs (used if present): ``return_21d``, ``return_63d``.

All inputs are z-scored per date, averaged across the available columns
(NaN-skipped per row), and clipped to ``[-1, 1]``.
"""

from __future__ import annotations

import pandas as pd

from .base import assemble_long_output, average_z, to_score_range

PRIMARY_FEATURES: tuple[str, ...] = ("momentum_3m", "momentum_6m")
OPTIONAL_FEATURES: tuple[str, ...] = ("return_21d", "return_63d")


class MomentumStrategy:
    name: str = "momentum"
    required_features: tuple[str, ...] = PRIMARY_FEATURES

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.required_features if c not in features.columns]
        if missing:
            raise ValueError(
                f"momentum strategy missing required feature columns: {missing}"
            )
        cols: list[str] = list(self.required_features) + [
            c for c in OPTIONAL_FEATURES if c in features.columns
        ]
        df = features[["date", "asset_id", *cols]].copy()
        z = average_z(df, cols)
        return assemble_long_output(features, to_score_range(z))
