"""Second review-round regressions for broker lifecycle and spot preparation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from market_sentinel.brokers.ccxt_spot import CcxtSpotBroker
from market_sentinel.brokers.groww import GrowwBroker, GrowwSession
from tests.brokers.test_ccxt_spot import FakeCcxtExchange, _market
from tests.brokers.test_groww import FakeGrowwClient
from tests.factories import intent, snapshot
from tests.settings import ccxt_settings, groww_settings

NOW = datetime(2026, 8, 9, 10, tzinfo=UTC)


class SecretClient(FakeGrowwClient):
    def __init__(self) -> None:
        super().__init__()
        self.profile_calls = 0

    def profile(self) -> dict[str, object]:
        self.profile_calls += 1
        return super().profile()

    def __repr__(self) -> str:
        return "SecretClient(secret-token-123)"


class SessionProvider:
    def __init__(self, clock: list[datetime]) -> None:
        self.clock = clock
        self.calls = 0
        self.client = SecretClient()

    def authenticated_session(self, _: object, now: datetime) -> GrowwSession:
        self.calls += 1
        return GrowwSession(self.client, now + timedelta(minutes=1))


def test_groww_session_is_safe_and_cached_across_preflight_and_submit() -> None:
    clock = [NOW]
    provider = SessionProvider(clock)
    broker = GrowwBroker(groww_settings(), auth_provider=provider, clock=lambda: clock[0])

    first = broker.preflight()
    second = broker.preflight()
    order = broker.submit(
        intent(instrument_id="RELIANCE@groww", quantity="1", notional=None),
        snapshot(instrument_id="RELIANCE@groww"),
    )

    assert first.ready is second.ready is True
    assert order.client_order_id == "intent-1"
    assert provider.calls == 1
    assert "secret-token-123" not in repr(GrowwSession(provider.client, NOW))


def test_groww_cached_session_expires_without_profile_request() -> None:
    clock = [NOW]
    provider = SessionProvider(clock)
    broker = GrowwBroker(groww_settings(), auth_provider=provider, clock=lambda: clock[0])
    assert broker.preflight().ready is True
    profile_calls = 1
    clock[0] = NOW + timedelta(minutes=2)

    report = broker.preflight()

    assert report.ready is False
    assert provider.client.profile_calls == profile_calls
    assert provider.calls == 1


def test_groww_rejects_malformed_clock_and_provider_failure_safely() -> None:
    class BrokenProvider:
        def authenticated_session(self, _: object, __: datetime) -> GrowwSession:
            raise RuntimeError("secret-token-123")

    malformed_clock = GrowwBroker(groww_settings(), client=FakeGrowwClient(), clock=lambda: "now")
    broken_provider = GrowwBroker(
        groww_settings(), auth_provider=BrokenProvider(), clock=lambda: NOW
    )

    for broker in (malformed_clock, broken_provider):
        report = broker.preflight()
        assert report.ready is False
        assert "secret-token-123" not in repr(report)


class OrderedExchange(FakeCcxtExchange):
    def fetch_order(self, order_id: str) -> dict[str, Any]:
        self.calls.append("fetch_order")
        return dict(super().fetch_order(order_id))

    def fetch_order_by_client_id(self, client_order_id: str) -> dict[str, Any]:
        self.calls.append("fetch_client")
        return dict(super().fetch_order_by_client_id(client_order_id))

    def fetch_balance(self) -> dict[str, Any]:
        self.calls.append("fetch_balance")
        return dict(self.balance)


@pytest.mark.parametrize("operation", ["get", "client", "positions"])
def test_ccxt_prepares_sandbox_before_every_first_public_operation(operation: str) -> None:
    exchange = OrderedExchange(_market())
    exchange.balance = {"free": {}, "used": {}, "total": {}, "info": {}}
    broker = CcxtSpotBroker(ccxt_settings(), exchange=exchange, clock=lambda: NOW)

    if operation == "get":
        broker.get_order("server-1")
    elif operation == "client":
        broker.get_order_by_client_id("intent-1")
    else:
        broker.positions()

    assert exchange.calls[:2] == ["sandbox", "markets"]


def test_ccxt_market_cost_uses_matching_fresh_snapshot_not_market_fixture_data() -> None:
    market = dict(_market())
    market.pop("reference_price", None)
    exchange = OrderedExchange(market)
    broker = CcxtSpotBroker(ccxt_settings(), exchange=exchange, clock=lambda: NOW)
    matching = snapshot(
        instrument_id="BTC/USDT@ccxt-spot",
        observed_at=NOW,
        source_at=NOW,
    )

    order = broker.submit(
        intent(instrument_id="BTC/USDT@ccxt-spot", quantity="1", notional=None),
        matching,
    )

    assert order.order_id == "ccxt-order-1"


@pytest.mark.parametrize(
    "bad_snapshot",
    [
        snapshot(instrument_id="OTHER@ccxt-spot", observed_at=NOW, source_at=NOW),
        snapshot(
            instrument_id="BTC/USDT@ccxt-spot",
            observed_at=NOW,
            source_at=NOW - timedelta(minutes=2),
        ),
        snapshot(
            instrument_id="BTC/USDT@ccxt-spot",
            observed_at=NOW + timedelta(minutes=1),
            source_at=NOW + timedelta(minutes=1),
        ),
    ],
)
def test_ccxt_market_cost_rejects_invalid_snapshot_reference(bad_snapshot: object) -> None:
    exchange = OrderedExchange({**_market(), "reference_price": None})
    broker = CcxtSpotBroker(ccxt_settings(), exchange=exchange, clock=lambda: NOW)
    order_intent = intent(instrument_id="BTC/USDT@ccxt-spot", quantity="1", notional=None)

    with pytest.raises(ValueError, match="snapshot"):
        broker.submit(order_intent, bad_snapshot)  # type: ignore[arg-type]


def test_ccxt_standard_balance_skips_quote_cash_and_uses_local_valuation() -> None:
    class Valuation:
        def value_spot(
            self, exchange_id: str, currency: str, quote: str, quantity: object
        ) -> dict[str, object]:
            assert (exchange_id, quote, quantity) == ("testexchange", "USDT", 2)
            return {
                "instrument_id": f"{currency}/USDT@ccxt-spot",
                "average_price": "10",
                "market_price": "11",
                "observed_at": NOW,
                "max_age_seconds": 60,
            }

    exchange = OrderedExchange(_market())
    exchange.balance = {
        "free": {"BTC": "2"},
        "used": {},
        "total": {"BTC": "2", "USDT": "5"},
        "info": {},
    }
    broker = CcxtSpotBroker(
        ccxt_settings(),
        exchange=exchange,
        valuation_provider=Valuation(),
        quote_currency="USDT",
        clock=lambda: NOW,
    )

    positions = broker.positions()

    assert [position.instrument_id for position in positions] == ["BTC/USDT@ccxt-spot"]
    assert exchange.calls == ["sandbox", "markets", "fetch_balance"]
