from datetime import UTC, datetime, timedelta, timezone, tzinfo
from decimal import Decimal

import pytest
from pydantic import ValidationError

from market_sentinel.domain.enums import AssetClass, Side
from market_sentinel.domain.models import (
    Bar,
    Instrument,
    MarketSnapshot,
    OrderIntent,
    ResearchPacket,
)


class UndefinedOffsetTimezone(tzinfo):
    def utcoffset(self, dt: datetime | None) -> None:
        return None

    def dst(self, dt: datetime | None) -> None:
        return None

    def tzname(self, dt: datetime | None) -> None:
        return None


def test_instrument_rejects_nonpositive_precision() -> None:
    with pytest.raises(ValidationError):
        Instrument(
            symbol="AAPL",
            venue="alpaca",
            asset_class=AssetClass.EQUITY,
            quote_currency="USD",
            timezone="America/New_York",
            price_tick=Decimal("0"),
            quantity_step=Decimal("0.000000001"),
            minimum_notional=Decimal("1"),
        )


def test_snapshot_staleness_uses_source_timestamp() -> None:
    now = datetime(2026, 8, 9, 12, tzinfo=UTC)
    bar = Bar(
        at=now - timedelta(minutes=2),
        open=Decimal("10"),
        high=Decimal("11"),
        low=Decimal("9"),
        close=Decimal("10.5"),
        volume=Decimal("1000"),
    )
    snapshot = MarketSnapshot(
        instrument_id="AAPL@alpaca",
        observed_at=now,
        source_at=bar.at,
        bars=(bar,),
        provider="fixture",
        max_age_seconds=60,
    )
    assert snapshot.is_stale(now) is True


def test_aware_datetimes_normalize_to_utc() -> None:
    india = timezone(timedelta(hours=5, minutes=30))
    source_at = datetime(2026, 8, 9, 10, tzinfo=india)
    bar = Bar(
        at=source_at,
        open=Decimal("10"),
        high=Decimal("11"),
        low=Decimal("9"),
        close=Decimal("10.5"),
        volume=Decimal("1000"),
    )
    snapshot = MarketSnapshot(
        instrument_id="AAPL@alpaca",
        observed_at=source_at,
        source_at=source_at,
        bars=(bar,),
        provider="fixture",
        max_age_seconds=60,
    )
    expected = datetime(2026, 8, 9, 4, 30, tzinfo=UTC)

    assert bar.at == expected
    assert bar.at.tzinfo is UTC
    assert snapshot.observed_at == expected
    assert snapshot.observed_at.tzinfo is UTC
    assert snapshot.source_at == expected
    assert snapshot.source_at.tzinfo is UTC


def test_datetime_with_undefined_offset_is_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        Bar(
            at=datetime(2026, 8, 9, 10, tzinfo=UndefinedOffsetTimezone()),
            open=Decimal("10"),
            high=Decimal("11"),
            low=Decimal("9"),
            close=Decimal("10.5"),
            volume=Decimal("1000"),
        )


def test_models_are_immutable() -> None:
    instrument = Instrument(
        symbol="AAPL",
        venue="alpaca",
        asset_class=AssetClass.EQUITY,
        quote_currency="USD",
        timezone="America/New_York",
        price_tick=Decimal("0.01"),
        quantity_step=Decimal("0.000000001"),
        minimum_notional=Decimal("1"),
    )

    with pytest.raises(ValidationError):
        instrument.symbol = "MSFT"  # type: ignore[misc]


def test_research_packet_rejects_out_of_range_confidence() -> None:
    now = datetime(2026, 8, 9, 12, tzinfo=UTC)
    with pytest.raises(ValidationError):
        ResearchPacket(
            instrument_id="AAPL@alpaca",
            as_of=now,
            thesis="thesis",
            bear_case="bear case",
            catalysts=(),
            risks=(),
            evidence=(),
            confidence=Decimal("1.01"),
            model_id="fixture",
            prompt_version="v1",
            configuration_hash="hash",
        )


def test_order_intent_requires_exactly_one_sizing_input() -> None:
    now = datetime(2026, 8, 9, 12, tzinfo=UTC)
    with pytest.raises(ValidationError):
        OrderIntent(
            intent_id="intent-1",
            instrument_id="AAPL@alpaca",
            side=Side.BUY,
            quantity=Decimal("1"),
            notional=Decimal("10"),
            order_type="market",
            limit_price=None,
            stop_loss=Decimal("9"),
            take_profit=Decimal("11"),
            time_in_force="day",
            product="cash",
            session="regular",
            snapshot_hash="hash",
            created_at=now,
            expires_at=now + timedelta(minutes=1),
        )
