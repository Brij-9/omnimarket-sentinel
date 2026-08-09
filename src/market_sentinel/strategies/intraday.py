"""Deterministic opening-range and VWAP intraday breakout."""

from datetime import time, timedelta
from decimal import Decimal, InvalidOperation

from market_sentinel.domain.enums import Horizon, SignalDirection
from market_sentinel.domain.models import Bar, Signal
from market_sentinel.strategies.base import StrategyContext, StrategyMetadata
from market_sentinel.strategies.indicators import vwap

_ZERO = Decimal("0")
_BASIS_POINTS = Decimal("10000")


class OpeningRangeVwapStrategy:
    """Emit only liquid, in-session long breakouts above both range and VWAP."""

    def __init__(
        self,
        *,
        session_start: time = time(9, 30),
        session_end: time = time(16),
        opening_range_bars: int = 6,
        closeout_buffer: timedelta = timedelta(minutes=5),
        max_spread_bps: Decimal = Decimal("15"),
        min_average_volume: Decimal = Decimal("100"),
    ) -> None:
        if session_start >= session_end:
            raise ValueError("session_start must precede session_end")
        if type(opening_range_bars) is not int or opening_range_bars <= 0:
            raise ValueError("opening_range_bars must be positive")
        if closeout_buffer <= timedelta(0):
            raise ValueError("closeout_buffer must be positive")
        self.session_start = session_start
        self.session_end = session_end
        self.opening_range_bars = opening_range_bars
        self.closeout_buffer = closeout_buffer
        self.max_spread_bps = _positive_decimal(max_spread_bps)
        self.min_average_volume = _positive_decimal(min_average_volume)
        self.metadata = StrategyMetadata(
            strategy_id="opening-range-vwap",
            version="1.0.0",
            allowed_horizons=(Horizon.INTRADAY,),
            allowed_directions=(SignalDirection.LONG,),
            mandatory_preclose_closeout=True,
            preclose_buffer=closeout_buffer,
        )

    def evaluate(self, context: StrategyContext) -> Signal | None:
        """Evaluate the current session prefix, never prior or future session bars."""
        try:
            if context.horizon is not Horizon.INTRADAY or not _bars_are_valid(context.bars):
                return None
            if context.spread_bps >= self.max_spread_bps or not context.bars:
                return None
            current = context.bars[-1]
            current_clock = current.at.timetz().replace(tzinfo=None)
            cutoff = _subtract_time(self.session_end, self.closeout_buffer)
            if not self.session_start <= current_clock < cutoff:
                return None

            session_bars = tuple(
                bar
                for bar in context.bars
                if bar.at.date() == current.at.date()
                and self.session_start <= bar.at.timetz().replace(tzinfo=None) < self.session_end
            )
            if (
                len(session_bars) <= self.opening_range_bars
                or session_bars[-1] != current
                or session_bars[0].at.timetz().replace(tzinfo=None) != self.session_start
            ):
                return None
            opening_range = session_bars[: self.opening_range_bars]
            opening_high = max(bar.high for bar in opening_range)
            opening_low = min(bar.low for bar in opening_range)
            range_width = opening_high - opening_low
            session_vwap = vwap(session_bars, len(session_bars))
            recent = session_bars[-min(20, len(session_bars)) :]
            recent_average_volume = sum((bar.volume for bar in recent), _ZERO) / Decimal(
                len(recent)
            )
            if (
                session_vwap is None
                or range_width <= _ZERO
                or current.close <= opening_high
                or current.close <= session_vwap
                or recent_average_volume < self.min_average_volume
                or current.volume < self.min_average_volume
            ):
                return None

            spread_price = current.close * context.spread_bps / _BASIS_POINTS
            stop_buffer = max(range_width / Decimal("10"), spread_price)
            invalidation = opening_low - stop_buffer
            risk = current.close - invalidation
            if invalidation <= _ZERO or risk <= _ZERO:
                return None
            return Signal(
                strategy_id=self.metadata.strategy_id,
                strategy_version=self.metadata.version,
                instrument_id=context.instrument_id,
                direction=SignalDirection.LONG,
                strength=Decimal("0.65"),
                horizon=context.horizon,
                entry_price=current.close,
                invalidation_price=invalidation,
                take_profit=current.close + Decimal("2") * risk,
                research_required=False,
                evidence_uris=(),
            )
        except (InvalidOperation, TypeError, ValueError, ZeroDivisionError):
            return None


def _subtract_time(value: time, delta: timedelta) -> time:
    anchor_minutes = value.hour * 60 + value.minute
    delta_minutes = int(delta.total_seconds() // 60)
    result = anchor_minutes - delta_minutes
    if result < 0:
        raise ValueError("closeout buffer exceeds session day")
    return time(result // 60, result % 60)


def _positive_decimal(value: Decimal) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= _ZERO:
        raise ValueError("strategy thresholds must be finite positive Decimals")
    return value


def _bars_are_valid(bars: tuple[Bar, ...]) -> bool:
    previous_at = None
    for bar in bars:
        prices = (bar.open, bar.high, bar.low, bar.close)
        if (
            any(not price.is_finite() or price <= _ZERO for price in prices)
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
