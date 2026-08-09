"""Shared fail-closed validation for strategy market-bar inputs."""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from market_sentinel.domain.models import Bar

_ZERO = Decimal("0")


def bars_are_strictly_valid(bars: Sequence[Bar]) -> bool:
    """Validate bypassed model fields before any strategy calls field methods."""
    previous_at: datetime | None = None
    for bar in bars:
        if not isinstance(bar, Bar) or not isinstance(bar.at, datetime):
            return False
        try:
            if bar.at.tzinfo is None or bar.at.utcoffset() is None:
                return False
        except (OverflowError, ValueError):
            return False
        prices = (bar.open, bar.high, bar.low, bar.close)
        if any(
            not isinstance(price, Decimal) or not price.is_finite() or price <= _ZERO
            for price in prices
        ):
            return False
        if (
            not isinstance(bar.volume, Decimal)
            or not bar.volume.is_finite()
            or bar.volume < _ZERO
            or bar.low > min(bar.open, bar.close)
            or bar.high < max(bar.open, bar.close)
            or bar.high < bar.low
            or (previous_at is not None and bar.at <= previous_at)
        ):
            return False
        previous_at = bar.at
    return True
