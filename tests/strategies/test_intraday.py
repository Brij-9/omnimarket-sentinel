"""Behavioral contracts for the opening-range/VWAP strategy."""

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from market_sentinel.domain.enums import Horizon
from market_sentinel.domain.models import Bar
from market_sentinel.strategies.base import StrategyContext
from market_sentinel.strategies.intraday import OpeningRangeVwapStrategy


def _bar(
    hour: int,
    minute: int,
    *,
    open_: str,
    high: str,
    low: str,
    close: str,
    volume: str = "1000",
    second: int = 0,
    microsecond: int = 0,
    on: date = date(2026, 8, 10),
    session_timezone: str = "America/New_York",
) -> Bar:
    return Bar(
        at=datetime(
            on.year,
            on.month,
            on.day,
            hour,
            minute,
            second,
            microsecond,
            tzinfo=ZoneInfo(session_timezone),
        ),
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
    )


def _breakout_bars(
    *,
    current_hour: int = 9,
    current_minute: int = 45,
    current_second: int = 0,
    current_microsecond: int = 0,
    opening_hour: int = 9,
    opening_minute: int = 30,
    on: date = date(2026, 8, 10),
    session_timezone: str = "America/New_York",
) -> tuple[Bar, ...]:
    opening = datetime.combine(
        on,
        time(opening_hour, opening_minute),
        tzinfo=ZoneInfo(session_timezone),
    )

    def opening_bar(offset_minutes: int, **prices: str) -> Bar:
        at = opening + timedelta(minutes=offset_minutes)
        return _bar(
            at.hour,
            at.minute,
            on=on,
            session_timezone=session_timezone,
            **prices,
        )

    return (
        opening_bar(0, open_="100", high="101", low="99", close="100"),
        opening_bar(5, open_="100", high="100.8", low="99.5", close="100.2"),
        opening_bar(10, open_="100.2", high="100.9", low="99.8", close="100.5"),
        _bar(
            current_hour,
            current_minute,
            second=current_second,
            microsecond=current_microsecond,
            on=on,
            session_timezone=session_timezone,
            open_="100.8",
            high="102.5",
            low="100.5",
            close="102",
            volume="1500",
        ),
    )


def _context(
    bars: tuple[Bar, ...] | None = None,
    *,
    spread_bps: Decimal = Decimal("4"),
    horizon: Horizon = Horizon.INTRADAY,
) -> StrategyContext:
    return StrategyContext(
        instrument_id="AAPL@alpaca",
        bars=_breakout_bars() if bars is None else bars,
        horizon=horizon,
        spread_bps=spread_bps,
    )


def _strategy() -> OpeningRangeVwapStrategy:
    return OpeningRangeVwapStrategy(
        session_start=time(9, 30),
        session_end=time(16),
        opening_range_bars=3,
        closeout_buffer=timedelta(minutes=5),
    )


def test_opening_range_breakout_requires_vwap_and_has_two_to_one_reward() -> None:
    """Wrong range/VWAP or reward math would emit an invalid intraday setup."""
    signal = _strategy().evaluate(_context())

    assert signal is not None
    assert signal.entry_price == Decimal("102")
    assert signal.invalidation_price < Decimal("99")
    assert signal.take_profit - signal.entry_price == Decimal("2") * (
        signal.entry_price - signal.invalidation_price
    )


def test_intraday_metadata_requires_preclose_flattening() -> None:
    """Omitting the closeout contract could accidentally create overnight exposure."""
    strategy = _strategy()

    assert strategy.metadata.mandatory_preclose_closeout is True
    assert strategy.metadata.preclose_buffer == timedelta(minutes=5)
    assert strategy.metadata.max_holding_bars is None
    with pytest.raises(FrozenInstanceError):
        strategy.metadata.preclose_buffer = timedelta(0)  # type: ignore[misc]


def test_intraday_rejects_breakouts_at_or_after_the_closeout_cutoff() -> None:
    """A boundary error at the cutoff could open a position that cannot be closed safely."""
    strategy = _strategy()

    assert (
        strategy.evaluate(_context(_breakout_bars(current_hour=15, current_minute=54)))
        is not None
    )
    assert strategy.evaluate(_context(_breakout_bars(current_hour=15, current_minute=55))) is None


def test_intraday_closeout_uses_full_second_and_microsecond_precision() -> None:
    """Flooring a 30-second buffer to minutes would admit a too-late entry."""
    strategy = OpeningRangeVwapStrategy(
        session_start=time(9, 30),
        session_end=time(16),
        opening_range_bars=3,
        closeout_buffer=timedelta(seconds=30),
    )

    just_before = _breakout_bars(
        current_hour=15,
        current_minute=59,
        current_second=29,
        current_microsecond=999_999,
    )
    at_cutoff = _breakout_bars(current_hour=15, current_minute=59, current_second=30)
    too_late = _breakout_bars(current_hour=15, current_minute=59, current_second=45)

    assert strategy.evaluate(_context(just_before)) is not None
    assert strategy.evaluate(_context(at_cutoff)) is None
    assert strategy.evaluate(_context(too_late)) is None


def test_intraday_rejects_closeout_buffer_longer_than_session() -> None:
    """An impossible cutoff configuration must fail clearly during construction."""
    with pytest.raises(ValueError, match="session duration"):
        OpeningRangeVwapStrategy(
            session_start=time(9, 30),
            session_end=time(16),
            closeout_buffer=timedelta(hours=6, minutes=30, microseconds=1),
        )


def test_intraday_requires_completed_opening_range_breakout_and_vwap_confirmation() -> None:
    """Trading inside the range or below VWAP violates both entry confirmations."""
    inside_range = list(_breakout_bars())
    inside_range[-1] = inside_range[-1].model_copy(
        update={"high": Decimal("101"), "close": Decimal("100.8")}
    )
    incomplete_range = _breakout_bars()[:3]

    assert _strategy().evaluate(_context(tuple(inside_range))) is None
    assert _strategy().evaluate(_context(incomplete_range)) is None


def test_intraday_enforces_horizon_spread_and_recent_liquidity() -> None:
    """Ineligible horizon, expensive spread, or dry recent tape must fail closed."""
    no_volume = tuple(bar.model_copy(update={"volume": Decimal("0")}) for bar in _breakout_bars())

    assert _strategy().evaluate(_context(horizon=Horizon.SWING)) is None
    assert _strategy().evaluate(_context(spread_bps=Decimal("15"))) is None
    assert _strategy().evaluate(_context(no_volume)) is None


def test_intraday_uses_only_current_session_bars() -> None:
    """A previous session's extremes must not redefine today's opening range."""
    yesterday = _bar(
        9,
        30,
        open_="200",
        high="250",
        low="190",
        close="220",
        on=date(2026, 8, 9),
    )

    assert _strategy().evaluate(_context((yesterday, *_breakout_bars()))) is not None


def test_intraday_abstains_when_the_session_open_is_missing() -> None:
    """Later bars must not be silently reinterpreted as the configured opening range."""
    shifted = tuple(
        bar.model_copy(update={"at": bar.at + timedelta(minutes=5)})
        for bar in _breakout_bars()
    )

    assert _strategy().evaluate(_context(shifted)) is None


@pytest.mark.parametrize(
    ("on", "expected_utc_hour"),
    [(date(2026, 1, 12), 14), (date(2026, 8, 10), 13)],
)
def test_intraday_converts_utc_bars_to_new_york_across_dst(
    on: date, expected_utc_hour: int
) -> None:
    """UTC-normalized bars must map to the configured exchange clock in both DST states."""
    bars = _breakout_bars(on=on)

    assert bars[0].at.tzinfo is UTC
    assert bars[0].at.hour == expected_utc_hour
    assert _strategy().evaluate(_context(bars)) is not None


def test_intraday_supports_configurable_asia_kolkata_session() -> None:
    """Session eligibility must use the configured IANA zone, not a hard-coded US offset."""
    bars = _breakout_bars(
        opening_hour=9,
        opening_minute=15,
        current_hour=9,
        current_minute=30,
        session_timezone="Asia/Kolkata",
    )
    strategy = OpeningRangeVwapStrategy(
        session_start=time(9, 15),
        session_end=time(15, 30),
        opening_range_bars=3,
        session_timezone="Asia/Kolkata",
    )

    assert bars[0].at.hour == 3 and bars[0].at.minute == 45
    assert strategy.evaluate(_context(bars)) is not None


def test_intraday_rejects_invalid_session_timezone_clearly() -> None:
    """A misspelled IANA zone must not silently fall back to another clock."""
    with pytest.raises(ValueError, match="IANA"):
        OpeningRangeVwapStrategy(session_timezone="Mars/Olympus_Mons")


@pytest.mark.parametrize(
    "update",
    [
        {"close": 102.0},
        {"volume": "1500"},
        {"at": "2026-08-10T13:45:00Z"},
    ],
)
def test_intraday_abstains_for_bypassed_malformed_bar_fields(update: dict[str, object]) -> None:
    """Bypassed model validation must never make malformed fields raise from evaluation."""
    malformed = list(_breakout_bars())
    malformed[-1] = malformed[-1].model_copy(update=update)

    assert _strategy().evaluate(_context(tuple(malformed))) is None
