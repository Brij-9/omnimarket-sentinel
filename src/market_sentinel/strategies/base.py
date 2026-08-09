"""Immutable input contract shared by deterministic strategies."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable

from market_sentinel.domain.enums import Horizon
from market_sentinel.domain.models import Bar, Signal


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """The complete point-in-time data a strategy may inspect."""

    instrument_id: str
    bars: tuple[Bar, ...]
    horizon: Horizon
    spread_bps: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if not self.instrument_id.strip():
            raise ValueError("instrument_id must not be empty")
        if not isinstance(self.horizon, Horizon):
            raise ValueError("horizon must be a Horizon")
        if not all(isinstance(bar, Bar) for bar in self.bars):
            raise ValueError("bars must contain Bar instances")
        if (
            not isinstance(self.spread_bps, Decimal)
            or not self.spread_bps.is_finite()
            or self.spread_bps < Decimal("0")
        ):
            raise ValueError("spread_bps must be a finite nonnegative Decimal")


@runtime_checkable
class Strategy(Protocol):
    """A pure deterministic strategy evaluated from one immutable context."""

    def evaluate(self, context: StrategyContext) -> Signal | None:
        """Return a signal or no action for this point-in-time context."""
