"""CompanySnapshot: compact view of an asset at a point in time."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class CompanySnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str = Field(..., min_length=1, max_length=16)
    timestamp: date
    sector: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)
    market_cap: float | None = Field(default=None, ge=0)
    industry: str | None = None
