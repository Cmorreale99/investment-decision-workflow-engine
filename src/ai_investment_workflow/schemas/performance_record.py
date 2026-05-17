"""PerformanceRecord: outcome tracking over the evaluation window."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import OutcomeLabel


class PerformanceRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str = Field(..., min_length=1)
    asset_id: str = Field(..., min_length=1, max_length=16)
    evaluation_start: date
    evaluation_end: date
    asset_return: float
    benchmark_return: float
    excess_return: float
    max_drawdown: float = Field(..., le=0.0)
    outcome_label: OutcomeLabel

    @model_validator(mode="after")
    def _window_ordered(self) -> "PerformanceRecord":
        if self.evaluation_end < self.evaluation_start:
            raise ValueError("evaluation_end must be on or after evaluation_start")
        return self
