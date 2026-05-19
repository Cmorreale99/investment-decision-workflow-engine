"""Strategy Protocol and shared cross-sectional scoring helpers.

All strategies emit a **long-format** DataFrame with columns
``date, asset_id, score`` and scores clipped to ``[-1, 1]``. NaN inputs
propagate to NaN scores; the pipeline drops NaN scores when forming
``StrategySignal`` objects.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

OUTPUT_COLUMNS: list[str] = ["date", "asset_id", "score"]
SCORE_CLIP: tuple[float, float] = (-1.0, 1.0)
Z_CLIP: tuple[float, float] = (-3.0, 3.0)


@runtime_checkable
class Strategy(Protocol):
    """A deterministic, cross-sectional scoring rule.

    Implementations must not call LLMs, must not approve/reject anything,
    and must not rank. Ranking lives in ``strategies.pipeline``.
    """

    name: str
    required_features: tuple[str, ...]

    def score(self, features: pd.DataFrame) -> pd.DataFrame: ...


def cross_sectional_zscore(df: pd.DataFrame, column: str) -> pd.Series:
    """Per-``date`` z-score of ``column``. NaN where stddev is 0 or input is NaN."""
    grouped = df.groupby("date", sort=False)[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    z = (df[column] - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def average_z(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Average of per-date z-scores across ``columns``.

    NaN inputs are skipped per row; an asset with all-NaN inputs in the
    requested columns receives NaN.
    """
    if not columns:
        raise ValueError("average_z requires at least one column")
    z_frame = pd.DataFrame(
        {c: cross_sectional_zscore(df, c).to_numpy() for c in columns},
        index=df.index,
    )
    return z_frame.mean(axis=1, skipna=True)


def to_score_range(z: pd.Series) -> pd.Series:
    """Map z-scores into ``[-1, 1]`` by clipping to ``[-3, 3]`` and dividing by 3."""
    return (z.clip(*Z_CLIP) / 3.0).clip(*SCORE_CLIP)


def assemble_long_output(
    features: pd.DataFrame, scores: pd.Series
) -> pd.DataFrame:
    """Stitch ``date``/``asset_id`` from features with ``scores`` into long format."""
    return pd.DataFrame(
        {
            "date": features["date"].to_numpy(),
            "asset_id": features["asset_id"].to_numpy(),
            "score": scores.to_numpy(),
        }
    )
