"""StrategySignal (+ risk flags) source backed by Phase 4 outputs.

Resolution order for ``(asset_id, as_of)``:

1. **Load** from persisted Phase 4 artifacts when ``artifacts_dir`` is
   given and both ``signals.parquet`` and ``rankings.parquet`` contain
   rows at ``as_of`` (risk flags from ``risk_flags.parquet`` likewise).
2. **Recompute** via the Phase 4 primitives (``build_strategy_signals``;
   ``build_company_snapshots`` + ``evaluate_risk`` when ``prices`` were
   provided) — the exact code paths ``scripts/run_strategies.py`` uses,
   so loaded and recomputed values agree.
3. Missing asset / missing inputs degrade to ``{}`` — never fabricated.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Mapping

import pandas as pd

from ...features import build_company_snapshots
from ...risk import RiskConfig, evaluate_risk
from ...schemas import StrategySignal
from ...strategies import build_strategy_signals


def _load_dated_parquet(path: Path, as_of: date) -> pd.DataFrame | None:
    """Rows at ``as_of`` from a long-format parquet, or None if unusable."""
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    if "date" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    slice_ = frame.loc[frame["date"] == as_of]
    return slice_ if not slice_.empty else None


class SignalSource:
    """Pulls ``StrategySignal`` and Phase 4 risk flags for one asset."""

    name = "signal_source"

    def __init__(
        self,
        features: pd.DataFrame,
        prices: pd.DataFrame | None = None,
        sectors: Mapping[str, str] | None = None,
        artifacts_dir: str | Path | None = None,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self._features = features
        self._prices = prices
        self._sectors = dict(sectors or {})
        self._artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self._risk_config = risk_config
        self._signal_cache: dict[date, dict[str, StrategySignal]] = {}
        self._flags_cache: dict[date, dict[str, dict[str, bool]]] = {}

    # ------------------------------------------------------------------
    # Strategy signals
    # ------------------------------------------------------------------

    def _load_signals(self, as_of: date) -> dict[str, StrategySignal] | None:
        if self._artifacts_dir is None:
            return None
        scores = _load_dated_parquet(self._artifacts_dir / "signals.parquet", as_of)
        rankings = _load_dated_parquet(self._artifacts_dir / "rankings.parquet", as_of)
        if scores is None or rankings is None:
            return None

        per_asset_scores: dict[str, dict[str, float]] = {}
        for row in scores.itertuples(index=False):
            per_asset_scores.setdefault(row.asset_id, {})[row.strategy] = float(
                row.score
            )

        out: dict[str, StrategySignal] = {}
        for row in rankings.itertuples(index=False):
            asset_scores = per_asset_scores.get(row.asset_id)
            if not asset_scores:
                continue
            out[row.asset_id] = StrategySignal(
                asset_id=row.asset_id,
                timestamp=as_of,
                strategy_scores=asset_scores,
                composite_score=float(row.composite_score),
                rank=int(row.rank),
                strategy_conflict=bool(row.strategy_conflict),
            )
        return out or None

    def _signals_for(self, as_of: date) -> dict[str, StrategySignal]:
        cached = self._signal_cache.get(as_of)
        if cached is None:
            cached = self._load_signals(as_of)
            if cached is None:
                cached = build_strategy_signals(self._features, as_of=as_of)
            self._signal_cache[as_of] = cached
        return cached

    # ------------------------------------------------------------------
    # Risk flags
    # ------------------------------------------------------------------

    def _load_risk_flags(self, as_of: date) -> dict[str, dict[str, bool]] | None:
        if self._artifacts_dir is None:
            return None
        flags = _load_dated_parquet(self._artifacts_dir / "risk_flags.parquet", as_of)
        if flags is None:
            return None
        out: dict[str, dict[str, bool]] = {}
        for row in flags.itertuples(index=False):
            out.setdefault(row.asset_id, {})[row.risk_flag] = bool(row.value)
        return out or None

    def _recompute_risk_flags(self, as_of: date) -> dict[str, dict[str, bool]]:
        if self._prices is None:
            return {}
        snapshots = build_company_snapshots(
            self._prices, sectors=self._sectors, as_of=as_of
        )
        config = self._risk_config or RiskConfig.from_yaml()
        slice_ = self._features.loc[self._features["date"] == as_of]

        out: dict[str, dict[str, bool]] = {}
        for record in slice_.to_dict("records"):
            asset_id = record.pop("asset_id")
            record.pop("date", None)
            snapshot = snapshots.get(asset_id)
            if snapshot is None:
                continue
            feats = {k: float(v) for k, v in record.items() if pd.notna(v)}
            out[asset_id] = evaluate_risk(snapshot, feats, config=config)
        return out

    def _risk_flags_for(self, as_of: date) -> dict[str, dict[str, bool]]:
        cached = self._flags_cache.get(as_of)
        if cached is None:
            cached = self._load_risk_flags(as_of)
            if cached is None:
                cached = self._recompute_risk_flags(as_of)
            self._flags_cache[as_of] = cached
        return cached

    # ------------------------------------------------------------------
    # ContextSource API
    # ------------------------------------------------------------------

    def fetch(self, asset_id: str, as_of: date) -> dict:
        signal = self._signals_for(as_of).get(asset_id)
        if signal is None:
            return {}
        payload: dict = {"strategy_signal": signal}
        flags = self._risk_flags_for(as_of).get(asset_id)
        if flags is not None:
            payload["risk_flags"] = flags
        return payload
