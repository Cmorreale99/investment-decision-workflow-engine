"""Recommendation: structured AI-generated recommendation."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from .enums import SystemAction


class Recommendation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recommendation_id: str = Field(..., min_length=1)
    asset_id: str = Field(..., min_length=1, max_length=16)
    timestamp: date
    action: SystemAction
    conviction: float = Field(..., ge=0.0, le=1.0)
    rationale: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    suggested_next_step: str | None = None
