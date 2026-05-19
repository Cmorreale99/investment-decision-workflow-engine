"""Strategy layer: deterministic candidate ranking.

Phase 4. Must not approve decisions or call LLMs. Must not execute trades.
A high composite score means an asset ranks highly under deterministic
scoring rules — not that it should be bought.
"""

from .base import (
    OUTPUT_COLUMNS,
    SCORE_CLIP,
    Strategy,
    assemble_long_output,
    average_z,
    cross_sectional_zscore,
    to_score_range,
)
from .momentum import MomentumStrategy
from .pipeline import (
    build_strategy_signals,
    composite_score,
    detect_conflicts,
    load_conflict_config,
    load_enabled_strategies,
    load_strategy_weights,
    rank_candidates,
    score_strategies,
)
from .value import ValueStrategy

__all__ = [
    "MomentumStrategy",
    "OUTPUT_COLUMNS",
    "SCORE_CLIP",
    "Strategy",
    "ValueStrategy",
    "assemble_long_output",
    "average_z",
    "build_strategy_signals",
    "composite_score",
    "cross_sectional_zscore",
    "detect_conflicts",
    "load_conflict_config",
    "load_enabled_strategies",
    "load_strategy_weights",
    "rank_candidates",
    "score_strategies",
    "to_score_range",
]
