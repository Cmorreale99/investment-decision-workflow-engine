"""StrategySignal: strategy scores, composite rank, and conflict flag."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrategySignal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str = Field(..., min_length=1, max_length=16)
    timestamp: date
    strategy_scores: dict[str, float] = Field(..., min_length=1)
    composite_score: float = Field(..., ge=-1.0, le=1.0)
    rank: int = Field(..., ge=1)
    strategy_conflict: bool = False

    @field_validator("strategy_scores")
    @classmethod
    def _scores_in_range(cls, value: dict[str, float]) -> dict[str, float]:
        for name, score in value.items():
            if not -1.0 <= score <= 1.0:
                raise ValueError(
                    f"strategy_scores[{name}] = {score} is outside [-1.0, 1.0]"
                )
        return value
