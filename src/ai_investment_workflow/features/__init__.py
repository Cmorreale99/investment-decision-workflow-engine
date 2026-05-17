"""Feature layer: deterministic signal generation.

Phase 3. Pure pandas/numpy — no network calls, no LLM, no strategy logic.
"""

from .base import KEY_COLUMNS, REQUIRED_PRICE_COLUMNS, FeatureBuilder, validate_price_frame
from .company_snapshot_builder import DEFAULT_SECTOR, build_company_snapshots
from .momentum import MomentumBuilder
from .pipeline import build_features, compute_feature_frame, default_builders
from .recent_returns import RecentReturnsBuilder
from .valuation import ValuationBuilder
from .volatility import VolatilityBuilder

__all__ = [
    "DEFAULT_SECTOR",
    "FeatureBuilder",
    "KEY_COLUMNS",
    "MomentumBuilder",
    "REQUIRED_PRICE_COLUMNS",
    "RecentReturnsBuilder",
    "ValuationBuilder",
    "VolatilityBuilder",
    "build_company_snapshots",
    "build_features",
    "compute_feature_frame",
    "default_builders",
    "validate_price_frame",
]
