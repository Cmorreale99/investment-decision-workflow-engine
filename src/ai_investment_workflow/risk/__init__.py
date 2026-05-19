"""Risk layer: deterministic boolean rule engine.

Phase 4. Must not score, rank, recommend, or call LLMs.
"""

from .rules import RiskConfig, evaluate_risk

__all__ = ["RiskConfig", "evaluate_risk"]
