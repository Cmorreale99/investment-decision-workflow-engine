"""DecisionRecord: persisted final decision joining recommendation + human decision."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from .enums import HumanAction, SystemAction


class DecisionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str = Field(..., min_length=1)
    asset_id: str = Field(..., min_length=1, max_length=16)
    timestamp: date
    recommendation_id: str = Field(..., min_length=1)
    system_action: SystemAction
    system_conviction: float = Field(..., ge=0.0, le=1.0)
    human_action: HumanAction
    override: bool = False
    human_notes: str | None = None
    final_status: str = Field(..., min_length=1)
