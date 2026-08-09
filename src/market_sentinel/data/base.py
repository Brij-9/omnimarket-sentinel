"""Provider interfaces for normalized market-data snapshots."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from market_sentinel.domain import AssetClass, Horizon, Instrument, MarketSnapshot


@dataclass(frozen=True)
class ProviderCapabilities:
    """Declared market coverage and request limits for one provider."""

    asset_classes: frozenset[AssetClass]
    horizons: frozenset[Horizon]
    supports_historical: bool
    supports_live_quotes: bool
    max_requests_per_second: Decimal


class MarketDataProvider(Protocol):
    """Returns immutable snapshots valid at a caller-supplied cutoff time."""

    @property
    def capabilities(self) -> ProviderCapabilities: ...

    def fetch_snapshot(
        self, instrument: Instrument, horizon: Horizon, as_of: datetime
    ) -> MarketSnapshot: ...
