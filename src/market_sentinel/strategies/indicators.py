"""Decimal-only technical indicators over explicitly supplied market history."""

from collections.abc import Sequence
from decimal import Decimal

from market_sentinel.domain.models import Bar

_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")


def sma(values: Sequence[Decimal], window: int) -> Decimal | None:
    """Return the arithmetic mean of the trailing Decimal values."""
    _require_window(window)
    checked = _decimal_values(values)
    if len(checked) < window:
        return None
    return sum(checked[-window:], _ZERO) / Decimal(window)


def ema(values: Sequence[Decimal], window: int) -> Decimal | None:
    """Return an EMA seeded by the first complete window in supplied history."""
    _require_window(window)
    checked = _decimal_values(values)
    if len(checked) < window:
        return None
    result = sum(checked[:window], _ZERO) / Decimal(window)
    alpha = Decimal("2") / Decimal(window + 1)
    for value in checked[window:]:
        result = (value - result) * alpha + result
    return result


def atr(bars: Sequence[Bar], window: int) -> Decimal | None:
    """Return the mean trailing true range, using each bar's previous close."""
    _require_window(window)
    checked = _valid_bars(bars)
    if len(checked) < window + 1:
        return None
    trailing = checked[-(window + 1) :]
    ranges = [
        max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        )
        for previous, current in zip(trailing[:-1], trailing[1:], strict=True)
    ]
    return sum(ranges, _ZERO) / Decimal(window)


def rsi(values: Sequence[Decimal], window: int) -> Decimal | None:
    """Return Wilder RSI, including deterministic values for zero gain/loss cases."""
    _require_window(window)
    checked = _decimal_values(values)
    if len(checked) < window + 1:
        return None
    changes = [
        current - previous
        for previous, current in zip(checked[:-1], checked[1:], strict=True)
    ]
    gains = [max(change, _ZERO) for change in changes]
    losses = [max(-change, _ZERO) for change in changes]
    average_gain = sum(gains[:window], _ZERO) / Decimal(window)
    average_loss = sum(losses[:window], _ZERO) / Decimal(window)
    for gain, loss in zip(gains[window:], losses[window:], strict=True):
        average_gain = (average_gain * Decimal(window - 1) + gain) / Decimal(window)
        average_loss = (average_loss * Decimal(window - 1) + loss) / Decimal(window)
    if average_gain == _ZERO and average_loss == _ZERO:
        return Decimal("50")
    if average_loss == _ZERO:
        return _HUNDRED
    if average_gain == _ZERO:
        return _ZERO
    relative_strength = average_gain / average_loss
    return _HUNDRED - (_HUNDRED / (_ONE + relative_strength))


def vwap(bars: Sequence[Bar], window: int) -> Decimal | None:
    """Return the volume-weighted typical price for the trailing bar window."""
    _require_window(window)
    checked = _valid_bars(bars)
    if len(checked) < window:
        return None
    trailing = checked[-window:]
    volume = sum((bar.volume for bar in trailing), _ZERO)
    if volume <= _ZERO:
        raise ValueError("cumulative volume must be positive")
    weighted_prices = sum(
        (((bar.high + bar.low + bar.close) / Decimal("3")) * bar.volume for bar in trailing),
        _ZERO,
    )
    return weighted_prices / volume


def _require_window(window: int) -> None:
    if type(window) is not int or window <= 0:
        raise ValueError("window must be a positive integer")


def _decimal_values(values: Sequence[Decimal]) -> tuple[Decimal, ...]:
    checked = tuple(values)
    if any(not isinstance(value, Decimal) for value in checked):
        raise ValueError("values must be Decimal instances")
    if any(not value.is_finite() for value in checked):
        raise ValueError("values must be finite")
    return checked


def _valid_bars(bars: Sequence[Bar]) -> tuple[Bar, ...]:
    checked = tuple(bars)
    for bar in checked:
        if not isinstance(bar, Bar):
            raise ValueError("bars must contain Bar instances")
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
            raise ValueError("bars must contain finite, internally consistent OHLCV values")
    return checked
