"""Strategy pipeline: score → composite → rank → conflict → StrategySignal."""

from __future__ import annotations

from datetime import date
from typing import Mapping, Sequence

import pandas as pd

from ..schemas import StrategySignal
from ..utils.config import load_strategies as _load_strategies_yaml
from .base import Strategy
from .momentum import MomentumStrategy
from .value import ValueStrategy

_STRATEGY_REGISTRY: dict[str, type] = {
    "value": ValueStrategy,
    "momentum": MomentumStrategy,
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_enabled_strategies() -> list[Strategy]:
    """Instantiate strategies marked ``enabled: true`` in ``strategies.yaml``."""
    cfg = _load_strategies_yaml().get("strategies", {})
    enabled: list[Strategy] = []
    for name, params in cfg.items():
        if not params.get("enabled", False):
            continue
        cls = _STRATEGY_REGISTRY.get(name)
        if cls is None:
            continue
        enabled.append(cls())
    return enabled


def load_strategy_weights() -> dict[str, float]:
    """Raw (un-normalized) weights for enabled strategies."""
    cfg = _load_strategies_yaml().get("strategies", {})
    return {
        name: float(params.get("weight", 1.0))
        for name, params in cfg.items()
        if params.get("enabled", False)
    }


def load_conflict_config() -> dict:
    cfg = _load_strategies_yaml()
    block = cfg.get("conflict_detection", {})
    return {
        "threshold": float(block.get("threshold", 0.4)),
        "require_sign_disagreement": bool(block.get("require_sign_disagreement", True)),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_strategies(
    features: pd.DataFrame,
    as_of: date | None = None,
    strategies: Sequence[Strategy] | None = None,
) -> dict[str, dict[str, float]]:
    """Return ``{strategy_name: {asset_id: score}}`` at ``as_of``.

    NaN scores are silently dropped so downstream composite logic only
    sees valid contributions.
    """
    strategy_list = (
        list(strategies) if strategies is not None else load_enabled_strategies()
    )
    if not strategy_list:
        return {}
    if as_of is None:
        as_of = features["date"].max()

    out: dict[str, dict[str, float]] = {}
    for strat in strategy_list:
        long_df = strat.score(features)
        slice_ = long_df.loc[long_df["date"] == as_of]
        per_asset: dict[str, float] = {}
        for row in slice_.itertuples(index=False):
            if pd.notna(row.score):
                per_asset[row.asset_id] = float(row.score)
        out[strat.name] = per_asset
    return out


# ---------------------------------------------------------------------------
# Composite + ranking
# ---------------------------------------------------------------------------


def composite_score(
    scores: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
    as_of: date | None = None,
) -> pd.Series:
    """Weighted average across strategies; per-asset weight renormalization.

    NaN handling:
      - Strategies with weight ≤ 0 or absent from ``scores`` are ignored.
      - An asset with no enabled scores is excluded from the output.
      - An asset with partial coverage gets weights renormalized over the
        subset of strategies that produced a score for it.
      - Final composite is clipped to ``[-1, 1]`` as a safety guard only.
    """
    enabled = {k: float(v) for k, v in weights.items() if k in scores and v > 0}
    total = sum(enabled.values())
    if total == 0:
        return pd.Series(dtype=float, name=as_of)
    enabled = {k: v / total for k, v in enabled.items()}

    all_assets = sorted({a for s in scores.values() for a in s.keys()})
    rows: dict[str, float] = {}
    for asset in all_assets:
        contrib = {k: enabled[k] for k in enabled if asset in scores[k]}
        if not contrib:
            continue
        norm = sum(contrib.values())
        rows[asset] = sum(
            scores[k][asset] * (w / norm) for k, w in contrib.items()
        )

    series = pd.Series(rows, dtype=float, name=as_of)
    return series.clip(-1.0, 1.0)


def rank_candidates(
    composite: pd.Series, as_of: date | None = None
) -> pd.DataFrame:
    """Sort by composite descending with deterministic ``asset_id`` tiebreak."""
    if as_of is None:
        as_of = composite.name
    df = composite.rename("composite_score").to_frame()
    df.index.name = "asset_id"
    df = df.reset_index()
    df["date"] = as_of
    df = df.sort_values(
        ["composite_score", "asset_id"], ascending=[False, True]
    ).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df[["date", "asset_id", "composite_score", "rank"]]


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


def detect_conflicts(
    scores: Mapping[str, Mapping[str, float]],
    threshold: float,
    require_sign_disagreement: bool = True,
) -> pd.Series:
    """Per-asset boolean Series flagging strategy disagreement.

    ``require_sign_disagreement=True`` (recommended): flag when at least
    one enabled score ≥ +threshold **and** at least one ≤ -threshold for
    the same asset.

    ``require_sign_disagreement=False``: flag when the spread
    ``max - min`` across strategies for that asset is ≥ ``2 * threshold``.
    """
    all_assets = sorted({a for s in scores.values() for a in s.keys()})
    flags: dict[str, bool] = {}
    for asset in all_assets:
        values = [scores[k][asset] for k in scores if asset in scores[k]]
        if len(values) < 2:
            flags[asset] = False
            continue
        has_pos = any(v >= threshold for v in values)
        has_neg = any(v <= -threshold for v in values)
        if require_sign_disagreement:
            flags[asset] = bool(has_pos and has_neg)
        else:
            flags[asset] = bool((max(values) - min(values)) >= 2 * threshold)
    return pd.Series(flags, dtype=bool, name="strategy_conflict")


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_strategy_signals(
    features: pd.DataFrame,
    as_of: date | None = None,
) -> dict[str, StrategySignal]:
    """One ``StrategySignal`` per asset present at ``as_of`` with ≥1 score."""
    if as_of is None:
        as_of = features["date"].max()
    strategies = load_enabled_strategies()
    weights = load_strategy_weights()
    conflict_cfg = load_conflict_config()

    scores = score_strategies(features, as_of=as_of, strategies=strategies)
    if not scores or not any(scores.values()):
        return {}

    composite = composite_score(scores, weights, as_of=as_of)
    if composite.empty:
        return {}
    ranking = rank_candidates(composite, as_of=as_of)
    conflicts = detect_conflicts(
        scores,
        threshold=conflict_cfg["threshold"],
        require_sign_disagreement=conflict_cfg["require_sign_disagreement"],
    )

    out: dict[str, StrategySignal] = {}
    for row in ranking.itertuples(index=False):
        asset_scores = {
            k: scores[k][row.asset_id]
            for k in scores
            if row.asset_id in scores[k]
        }
        if not asset_scores:
            continue
        out[row.asset_id] = StrategySignal(
            asset_id=row.asset_id,
            timestamp=as_of,
            strategy_scores=asset_scores,
            composite_score=float(row.composite_score),
            rank=int(row.rank),
            strategy_conflict=bool(conflicts.get(row.asset_id, False)),
        )
    return out
