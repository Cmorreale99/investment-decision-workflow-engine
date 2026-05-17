"""ContextPacket: grounded, point-in-time input for AI reasoning.

Embeds CompanySnapshot and StrategySignal by value so the LLM input is
fully reproducible and auditable.
"""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from .company_snapshot import CompanySnapshot
from .strategy_signal import StrategySignal


class ContextPacket(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str = Field(..., min_length=1, max_length=16)
    timestamp: date
    company_snapshot: CompanySnapshot
    strategy_signal: StrategySignal
    recent_performance: dict[str, float] = Field(default_factory=dict)
    risk_flags: dict[str, bool] = Field(default_factory=dict)
    prior_decisions: list[str] = Field(default_factory=list)
    human_notes: list[str] = Field(default_factory=list)
    research_notes: list[str] = Field(default_factory=list)
