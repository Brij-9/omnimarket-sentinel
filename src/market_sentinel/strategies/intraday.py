"""Deterministic opening-range and VWAP intraday breakout."""

from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from market_sentinel.domain.enums import Horizon, SignalDirection
from market_sentinel.domain.models import Signal
from market_sentinel.strategies.base import (
    StrategyConfiguration,
    StrategyContext,
    StrategyMetadata,
    canonical_strategy_configuration,
)
from market_sentinel.strategies.indicators import vwap
from market_sentinel.strategies.validation import bars_are_strictly_valid

_ZERO = Decimal("0")
_BASIS_POINTS = Decimal("10000")


class OpeningRangeVwapStrategy:
    """Emit only liquid, in-session long breakouts above both range and VWAP."""

    def __init__(
        self,
        *,
        session_start: time = time(9, 30),
        session_end: time = time(16),
        session_timezone: str = "America/New_York",
        opening_range_bars: int = 6,
        closeout_buffer: timedelta = timedelta(minutes=5),
        max_spread_bps: Decimal = Decimal("15"),
        min_average_volume: Decimal = Decimal("100"),
    ) -> None:
        if session_start.tzinfo is not None or session_end.tzinfo is not None:
            raise ValueError("session clock values must be timezone-naive")
        if session_start >= session_end:
            raise ValueError("session_start must precede session_end")
        if type(opening_range_bars) is not int or opening_range_bars <= 0:
            raise ValueError("opening_range_bars must be positive")
        if not isinstance(closeout_buffer, timedelta) or closeout_buffer <= timedelta(0):
            raise ValueError("closeout_buffer must be positive")
        session_duration = datetime.combine(date.min, session_end) - datetime.combine(
            date.min, session_start
        )
        if closeout_buffer >= session_duration:
            raise ValueError("closeout_buffer must be shorter than the session duration")
        if not isinstance(session_timezone, str) or not session_timezone.strip():
            raise ValueError("session_timezone must name a valid IANA timezone")
        try:
            session_zone = ZoneInfo(session_timezone)
        except (ValueError, ZoneInfoNotFoundError) as error:
            raise ValueError("session_timezone must name a valid IANA timezone") from error
        self.session_start = session_start
        self.session_end = session_end
        self.session_timezone = session_timezone
        self.session_zone = session_zone
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

    @property
    def configuration(self) -> StrategyConfiguration:
        return canonical_strategy_configuration(
            metadata=self.metadata,
            parameters={
                "closeout_buffer": self.closeout_buffer,
                "max_spread_bps": self.max_spread_bps,
                "min_average_volume": self.min_average_volume,
                "opening_range_bars": self.opening_range_bars,
                "session_end": self.session_end,
                "session_start": self.session_start,
                "session_timezone": self.session_timezone,
            },
        )

    def evaluate(self, context: StrategyContext) -> Signal | None:
        """Evaluate the current session prefix, never prior or future session bars."""
        try:
            if (
                context.horizon is not Horizon.INTRADAY
                or not bars_are_strictly_valid(context.bars)
            ):
                return None
            if context.spread_bps >= self.max_spread_bps or not context.bars:
                return None
            current = context.bars[-1]
            localized = tuple((bar, bar.at.astimezone(self.session_zone)) for bar in context.bars)
            current_local = localized[-1][1]
            session_date = current_local.date()
            session_open = datetime.combine(
                session_date, self.session_start, tzinfo=self.session_zone
            )
            session_close = datetime.combine(
                session_date, self.session_end, tzinfo=self.session_zone
            )
            cutoff = session_close - self.closeout_buffer
            if not session_open <= current_local < cutoff:
                return None

            localized_session = tuple(
                (bar, local_at)
                for bar, local_at in localized
                if local_at.date() == session_date
                and session_open <= local_at < session_close
            )
            if (
                len(localized_session) <= self.opening_range_bars
                or localized_session[-1][0] != current
                or localized_session[0][1] != session_open
            ):
                return None
            session_bars = tuple(bar for bar, _ in localized_session)
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
        except (AttributeError, InvalidOperation, TypeError, ValueError, ZeroDivisionError):
            return None


def _positive_decimal(value: Decimal) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= _ZERO:
        raise ValueError("strategy thresholds must be finite positive Decimals")
    return value
