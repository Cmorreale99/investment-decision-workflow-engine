"""Shared enumerations used across schema objects."""

from enum import Enum


class HumanAction(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    WATCHLIST = "WATCHLIST"
    OVERRIDE = "OVERRIDE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class SystemAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    WATCHLIST = "WATCHLIST"


class OutcomeLabel(str, Enum):
    OUTPERFORMED = "outperformed"
    UNDERPERFORMED = "underperformed"
    NEUTRAL = "neutral"
    PENDING = "pending"
