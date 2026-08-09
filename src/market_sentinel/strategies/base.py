"""Immutable input contract shared by deterministic strategies."""

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Protocol, runtime_checkable

from market_sentinel.domain.enums import Horizon, SignalDirection
from market_sentinel.domain.models import Bar, Signal


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """The complete point-in-time data a strategy may inspect."""

    instrument_id: str
    bars: tuple[Bar, ...]
    horizon: Horizon
    spread_bps: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        object.__setattr__(self, "bars", tuple(self.bars))
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


@dataclass(frozen=True, slots=True)
class StrategyMetadata:
    """Versioned execution constraints that do not belong on a market signal."""

    strategy_id: str
    version: str
    allowed_horizons: tuple[Horizon, ...]
    allowed_directions: tuple[SignalDirection, ...]
    max_holding_bars: int | None = None
    mandatory_preclose_closeout: bool = False
    preclose_buffer: timedelta | None = None
    spot_only: bool = False
    leverage_allowed: bool = False

    def __post_init__(self) -> None:
        if not self.strategy_id.strip() or not self.version.strip():
            raise ValueError("strategy metadata identifiers must not be empty")
        if not self.allowed_horizons or not self.allowed_directions:
            raise ValueError("strategy metadata eligibility must not be empty")
        if self.max_holding_bars is not None and self.max_holding_bars <= 0:
            raise ValueError("max_holding_bars must be positive")
        if self.mandatory_preclose_closeout != (self.preclose_buffer is not None):
            raise ValueError("pre-close policy and buffer must be declared together")
        if self.preclose_buffer is not None and self.preclose_buffer <= timedelta(0):
            raise ValueError("preclose_buffer must be positive")


@runtime_checkable
class Strategy(Protocol):
    """A pure deterministic strategy evaluated from one immutable context."""

    def evaluate(self, context: StrategyContext) -> Signal | None:
        """Return a signal or no action for this point-in-time context."""
