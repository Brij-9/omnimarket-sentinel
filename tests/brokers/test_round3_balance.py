"""Unified CCXT balance and local valuation freshness regressions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from market_sentinel.brokers.ccxt_spot import CcxtSpotBroker
from tests.brokers.test_ccxt_spot import FakeCcxtExchange, _market
from tests.settings import ccxt_settings

NOW = datetime(2026, 8, 9, 10, tzinfo=UTC)


class UnifiedExchange(FakeCcxtExchange):
    def fetch_balance(self) -> dict[str, object]:
        self.calls.append("fetch_balance")
        return self.balance


class Valuation:
    def __init__(self, record: dict[str, object]) -> None:
        self.record = record
        self.calls: list[tuple[str, str, str, object]] = []

    def value_spot(
        self, exchange_id: str, currency: str, quote: str, quantity: object
    ) -> dict[str, object]:
        self.calls.append((exchange_id, currency, quote, quantity))
        return self.record


def _record(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "instrument_id": "BTC/USDT@ccxt-spot",
        "average_price": "10",
        "market_price": "11",
        "observed_at": NOW,
        "max_age_seconds": 60,
    }
    value.update(overrides)
    return value


def _broker(
    balance: dict[str, object], record: dict[str, object] | None = None
) -> tuple[CcxtSpotBroker, UnifiedExchange, Valuation]:
    exchange = UnifiedExchange(_market())
    exchange.balance = balance
    valuation = Valuation(_record() if record is None else record)
    return (
        CcxtSpotBroker(
            ccxt_settings(),
            exchange=exchange,
            valuation_provider=valuation,
            quote_currency="USDT",
            clock=lambda: NOW,
        ),
        exchange,
        valuation,
    )


def test_ccxt_uses_only_total_mapping_and_ignores_unified_metadata_and_mirrors() -> None:
    balance: dict[str, object] = {
        "free": {"BTC": "999", "USDT": "5"},
        "used": {"BTC": "999"},
        "total": {"BTC": "2", "USDT": "5", "ETH": "0"},
        "info": ["ordinary", {"nested": True}],
        "timestamp": 1786269600000,
        "datetime": "2026-08-09T10:00:00Z",
        "BTC": {"free": "999", "total": "999"},
        "USD": {"free": "123"},
    }
    broker, exchange, valuation = _broker(balance)

    positions = broker.positions()

    assert [position.instrument_id for position in positions] == ["BTC/USDT@ccxt-spot"]
    assert valuation.calls == [("testexchange", "BTC", "USDT", 2)]
    assert exchange.calls == ["sandbox", "markets", "fetch_balance"]


@pytest.mark.parametrize("total", [None, {"": "1"}, {"BTC": "NaN"}, {"BTC": "-1"}])
def test_ccxt_rejects_missing_or_malformed_total(total: object) -> None:
    broker, _, _ = _broker({"free": {}, "used": {}, "total": total, "info": "anything"})

    with pytest.raises(RuntimeError, match="ccxt client operation failed"):
        broker.positions()


@pytest.mark.parametrize(
    ("record", "allowed"),
    [
        (_record(observed_at=NOW - timedelta(seconds=60), max_age_seconds=60), True),
        (_record(observed_at=NOW - timedelta(seconds=300), max_age_seconds=300), True),
        (_record(observed_at=NOW, max_age_seconds=301), False),
        (
            _record(observed_at=NOW - timedelta(seconds=60, microseconds=1), max_age_seconds=60),
            False,
        ),
        (_record(observed_at=NOW - timedelta(seconds=61), max_age_seconds=60), False),
        (_record(observed_at=NOW + timedelta(seconds=1), max_age_seconds=60), False),
        (_record(observed_at=datetime(2026, 8, 9, 10), max_age_seconds=60), False),
        (_record(max_age_seconds=-1), False),
        (_record(max_age_seconds=10**9), False),
    ],
)
def test_ccxt_valuation_observation_is_bounded_and_clock_checked(
    record: dict[str, object], allowed: bool
) -> None:
    broker, _, _ = _broker({"free": {}, "used": {}, "total": {"BTC": "2"}, "info": {}}, record)

    if allowed:
        assert broker.positions()[0].quantity == 2
    else:
        with pytest.raises(RuntimeError, match="ccxt client operation failed"):
            broker.positions()


def test_ccxt_valuation_malformed_clock_is_sanitized() -> None:
    exchange = UnifiedExchange(_market())
    exchange.balance = {"free": {}, "used": {}, "total": {"BTC": "2"}, "info": {}}
    broker = CcxtSpotBroker(
        ccxt_settings(),
        exchange=exchange,
        valuation_provider=Valuation(_record()),
        clock=lambda: "now",  # type: ignore[return-value]
    )

    with pytest.raises(RuntimeError, match="ccxt client operation failed") as raised:
        broker.positions()

    assert raised.value.__context__ is None
