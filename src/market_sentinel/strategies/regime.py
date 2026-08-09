"""Fail-closed market-regime classification from trailing Decimal indicators."""

from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from market_sentinel.domain.models import Bar
from market_sentinel.strategies.indicators import atr, sma

_ZERO = Decimal("0")
_ONE = Decimal("1")
_ATR_WINDOW = 14
_FAST_WINDOW = 20
_SLOW_WINDOW = 50
_SLOPE_WINDOW = 20
_MIN_TREND_FRACTION = Decimal("0.001")


class MarketRegime(StrEnum):
    """The strategy eligibility state inferred from current market structure."""

    TRENDING = "trending"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    UNTRADEABLE = "untradeable"


def classify_regime(
    bars: Sequence[Bar],
    max_spread_bps: Decimal | int,
    *,
    spread_bps: Decimal | int = Decimal("0"),
    max_atr_percentage: Decimal = Decimal("0.05"),
    min_average_volume: Decimal = Decimal("1"),
) -> MarketRegime:
    """Classify supplied history, returning ``UNTRADEABLE`` for every bad input."""
    try:
        spread_cap = _nonnegative_decimal(max_spread_bps)
        current_spread = _nonnegative_decimal(spread_bps)
        volatility_cap = _positive_decimal(max_atr_percentage)
        minimum_volume = _positive_decimal(min_average_volume)
        checked = tuple(bars)
        if not _bars_are_valid(checked):
            return MarketRegime.UNTRADEABLE
        if len(checked) < _SLOW_WINDOW + _SLOPE_WINDOW:
            return MarketRegime.UNTRADEABLE
        if spread_cap == _ZERO or current_spread >= spread_cap:
            return MarketRegime.UNTRADEABLE
        trailing_volume_bars = checked[-_FAST_WINDOW:]
        if trailing_volume_bars[-1].volume <= _ZERO:
            return MarketRegime.UNTRADEABLE
        average_volume = sum(
            (bar.volume for bar in trailing_volume_bars), _ZERO
        ) / Decimal(_FAST_WINDOW)
        if average_volume < minimum_volume:
            return MarketRegime.UNTRADEABLE

        closes = tuple(bar.close for bar in checked)
        average_true_range = atr(checked, window=_ATR_WINDOW)
        fast_average = sma(closes, window=_FAST_WINDOW)
        slow_average = sma(closes, window=_SLOW_WINDOW)
        previous_fast_average = sma(closes[:-_SLOPE_WINDOW], window=_FAST_WINDOW)
        if (
            average_true_range is None
            or fast_average is None
            or slow_average is None
            or previous_fast_average is None
            or slow_average <= _ZERO
            or previous_fast_average <= _ZERO
            or closes[-1] <= _ZERO
        ):
            return MarketRegime.UNTRADEABLE

        if average_true_range / closes[-1] > volatility_cap:
            return MarketRegime.HIGH_VOLATILITY
        separation = (fast_average - slow_average) / slow_average
        slope = (fast_average - previous_fast_average) / previous_fast_average
        if (
            separation >= _MIN_TREND_FRACTION and slope >= _MIN_TREND_FRACTION
        ) or (
            separation <= -_MIN_TREND_FRACTION and slope <= -_MIN_TREND_FRACTION
        ):
            return MarketRegime.TRENDING
        return MarketRegime.RANGE_BOUND
    except (InvalidOperation, TypeError, ValueError, ZeroDivisionError):
        return MarketRegime.UNTRADEABLE


def _bars_are_valid(bars: tuple[Bar, ...]) -> bool:
    for bar in bars:
        if not isinstance(bar, Bar):
            return False
        prices = (bar.open, bar.high, bar.low, bar.close)
        if (
            any(
                not isinstance(price, Decimal) or not price.is_finite() or price <= _ZERO
                for price in prices
            )
            or not isinstance(bar.volume, Decimal)
            or not bar.volume.is_finite()
            or bar.volume < _ZERO
            or bar.low > min(bar.open, bar.close)
            or bar.high < max(bar.open, bar.close)
            or bar.high < bar.low
        ):
            return False
    return True


def _nonnegative_decimal(value: Decimal | int) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, int)):
        raise ValueError("value must be a Decimal or integer")
    result = Decimal(value)
    if not result.is_finite() or result < _ZERO:
        raise ValueError("value must be finite and nonnegative")
    return result


def _positive_decimal(value: Decimal) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= _ZERO:
        raise ValueError("value must be a finite positive Decimal")
    return value
