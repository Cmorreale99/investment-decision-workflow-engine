"""Deterministic boolean risk-flag engine.

This module:
- emits boolean risk flags only
- does **not** score, rank, recommend, approve, reject, or watchlist
- does **not** call LLMs
- does **not** import agents, rag, human_review, simulation, or app

Available rules in Phase 4
--------------------------
- ``high_volatility``       : max(volatility_21d, volatility_63d) > threshold
- ``near_52w_high``         : distance_from_52w_high >= -threshold (≥ -2% by default)
- ``near_52w_low``          : distance_from_52w_low  <=  threshold (≤  5% by default)
- ``extended_from_52w_low`` : distance_from_52w_low  >=  threshold (≥ 50% by default)

Unsupported rules
-----------------
Liquidity rules require a volume / dollar-volume feature that Phase 3 does
not yet produce; ``liquidity_unavailable: True`` is emitted as a
deterministic placeholder so downstream code can detect the gap without
crashing.

Missing-input handling
----------------------
When a rule's required feature is absent, the rule's flag is set to
``False`` and a companion ``<rule>_missing_input: True`` flag is added so
the absence is observable in the output map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..schemas import CompanySnapshot, FeatureSet
from ..utils.config import load_risk_rules as _load_risk_rules_yaml


@dataclass(frozen=True)
class RiskConfig:
    max_volatility_threshold: float
    near_52w_high_threshold: float
    near_52w_low_threshold: float
    extended_from_52w_low_threshold: float

    @classmethod
    def from_yaml(cls) -> "RiskConfig":
        cfg = _load_risk_rules_yaml()
        return cls(
            max_volatility_threshold=float(cfg.get("max_volatility_threshold", 0.40)),
            near_52w_high_threshold=float(cfg.get("near_52w_high_threshold", 0.02)),
            near_52w_low_threshold=float(cfg.get("near_52w_low_threshold", 0.05)),
            extended_from_52w_low_threshold=float(
                cfg.get("extended_from_52w_low_threshold", 0.50)
            ),
        )


def _as_feature_dict(
    features: FeatureSet | Mapping[str, float]
) -> dict[str, float]:
    if isinstance(features, FeatureSet):
        return dict(features.features)
    return {k: float(v) for k, v in dict(features).items()}


def evaluate_risk(
    snapshot: CompanySnapshot,
    features: FeatureSet | Mapping[str, float],
    config: RiskConfig | None = None,
) -> dict[str, bool]:
    """Return ``{flag_name: bool}`` for a single asset at a point in time.

    ``snapshot`` carries identity / price; the current rule set does not
    consume snapshot fields directly but the parameter is preserved for
    forward-compatibility with rules that would (e.g. minimum price).
    """
    del snapshot  # reserved for future rules; not used in Phase 4.

    cfg = config or RiskConfig.from_yaml()
    feats = _as_feature_dict(features)
    flags: dict[str, bool] = {}

    # --- high_volatility ---------------------------------------------------
    vol_21 = feats.get("volatility_21d")
    vol_63 = feats.get("volatility_63d")
    vol_values = [v for v in (vol_21, vol_63) if v is not None]
    if vol_values:
        flags["high_volatility"] = bool(
            max(vol_values) > cfg.max_volatility_threshold
        )
    else:
        flags["high_volatility"] = False
        flags["high_volatility_missing_input"] = True

    # --- near_52w_high ----------------------------------------------------
    d_high = feats.get("distance_from_52w_high")
    if d_high is not None:
        flags["near_52w_high"] = bool(d_high >= -cfg.near_52w_high_threshold)
    else:
        flags["near_52w_high"] = False
        flags["near_52w_high_missing_input"] = True

    # --- near_52w_low / extended_from_52w_low -----------------------------
    d_low = feats.get("distance_from_52w_low")
    if d_low is not None:
        flags["near_52w_low"] = bool(d_low <= cfg.near_52w_low_threshold)
        flags["extended_from_52w_low"] = bool(
            d_low >= cfg.extended_from_52w_low_threshold
        )
    else:
        flags["near_52w_low"] = False
        flags["extended_from_52w_low"] = False
        flags["near_52w_low_missing_input"] = True

    # --- liquidity (not yet supported) ------------------------------------
    flags["liquidity_unavailable"] = True

    return flags
