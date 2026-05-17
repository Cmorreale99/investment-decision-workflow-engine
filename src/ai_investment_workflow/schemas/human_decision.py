"""HumanDecision: human review action recorded against a recommendation."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .enums import HumanAction


class HumanDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str = Field(..., min_length=1, max_length=16)
    recommendation_id: str = Field(..., min_length=1)
    human_action: HumanAction
    override: bool = False
    human_notes: str | None = None
    review_status: str = Field(default="completed", min_length=1)
    reviewed_at: datetime
