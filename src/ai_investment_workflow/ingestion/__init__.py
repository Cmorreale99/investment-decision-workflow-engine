"""Ingestion layer: pull and normalize public market data.

Phase 2. Must not contain strategy or AI logic.
"""

from .base import PRICE_COLUMNS, MarketDataProvider
from .pipeline import get_provider, run_ingestion
from .providers import FixtureProvider, YFinanceProvider, default_fixture_path
from .validation import validate_prices

__all__ = [
    "FixtureProvider",
    "MarketDataProvider",
    "PRICE_COLUMNS",
    "YFinanceProvider",
    "default_fixture_path",
    "get_provider",
    "run_ingestion",
    "validate_prices",
]
