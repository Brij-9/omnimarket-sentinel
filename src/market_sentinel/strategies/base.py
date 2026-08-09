"""Immutable input and canonical evidence contracts for deterministic strategies."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import time, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Protocol, runtime_checkable

from market_sentinel.domain.enums import Horizon, SignalDirection
from market_sentinel.domain.models import Bar, Signal

_MAX_PLAIN_DECIMAL_PADDING = 32


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


@dataclass(frozen=True, slots=True)
class CanonicalParameter:
    """One explicitly typed, stable strategy-configuration value."""

    name: str
    kind: str
    value: str


@dataclass(frozen=True, slots=True)
class StrategyConfiguration:
    """Canonical configuration derived only from the actual strategy instance."""

    strategy_id: str
    strategy_version: str
    parameters: tuple[CanonicalParameter, ...]


def canonical_strategy_configuration(
    *, metadata: StrategyMetadata, parameters: Mapping[str, object]
) -> StrategyConfiguration:
    """Canonicalize the small stable value set supported by strategy constructors."""
    canonical: list[CanonicalParameter] = []
    for name, value in sorted(parameters.items()):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("canonical parameter names must be nonempty strings")
        kind, text = _canonical_value(value)
        canonical.append(CanonicalParameter(name=name, kind=kind, value=text))
    return StrategyConfiguration(
        strategy_id=metadata.strategy_id,
        strategy_version=metadata.version,
        parameters=tuple(canonical),
    )


def _canonical_value(value: object) -> tuple[str, str]:
    if isinstance(value, bool):
        return "boolean", "true" if value else "false"
    if type(value) is int:
        return "integer", str(value)
    if isinstance(value, Decimal) and value.is_finite():
        return "decimal", _exact_decimal_text(value)
    if isinstance(value, StrEnum):
        return "enum", value.value
    if isinstance(value, str):
        return "string", value
    if isinstance(value, time) and value.tzinfo is None:
        return "time", value.isoformat()
    if isinstance(value, timedelta):
        return "timedelta", f"{value.days}:{value.seconds}:{value.microseconds}"
    if value is None:
        return "null", "null"
    raise ValueError("strategy configuration contains a noncanonical value")


def _exact_decimal_text(value: Decimal) -> str:
    """Encode a finite Decimal exactly without context rounding or exponent-sized padding."""
    if value.is_zero():
        return "0"
    decimal_tuple = value.as_tuple()
    exponent = decimal_tuple.exponent
    if not isinstance(exponent, int):
        raise ValueError("strategy configuration contains a noncanonical Decimal")
    digits = list(decimal_tuple.digits)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    coefficient = "".join(str(digit) for digit in digits)
    prefix = "-" if decimal_tuple.sign else ""
    if exponent == 0:
        return prefix + coefficient
    if exponent > 0:
        if exponent <= _MAX_PLAIN_DECIMAL_PADDING:
            return prefix + coefficient + "0" * exponent
        return f"{prefix}{coefficient}e{exponent}"
    decimal_position = len(coefficient) + exponent
    if decimal_position > 0:
        return prefix + coefficient[:decimal_position] + "." + coefficient[decimal_position:]
    leading_zeros = -decimal_position
    if leading_zeros <= _MAX_PLAIN_DECIMAL_PADDING:
        return prefix + "0." + "0" * leading_zeros + coefficient
    return f"{prefix}{coefficient}e{exponent}"


@runtime_checkable
class Strategy(Protocol):
    """A pure deterministic strategy evaluated from one immutable context."""

    def evaluate(self, context: StrategyContext) -> Signal | None:
        """Return a signal or no action for this point-in-time context."""
