"""FeatureSet: computed features for an asset at a point in time."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class FeatureSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str = Field(..., min_length=1, max_length=16)
    timestamp: date
    features: dict[str, float] = Field(default_factory=dict)
