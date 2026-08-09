"""Safety regression tests for broker preflight and injected boundaries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from market_sentinel.brokers._records import broker_order
from market_sentinel.brokers.alpaca import AlpacaBroker
from market_sentinel.brokers.ccxt_spot import CcxtConnectionConfig, CcxtSpotBroker
from market_sentinel.brokers.groww import GrowwBroker, GrowwSession, _is_public_ipv4
from tests.brokers.test_alpaca import FakeAlpacaClient, _account
from tests.brokers.test_ccxt_spot import FakeCcxtExchange, _market
from tests.brokers.test_groww import FakeGrowwClient
from tests.factories import intent, snapshot
from tests.settings import alpaca_settings, ccxt_settings, groww_settings

NOW = datetime(2026, 8, 9, 10, tzinfo=UTC)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("8.8.8.8", True),
        ("10.0.0.1", False),
        ("172.16.0.1", False),
        ("192.168.0.1", False),
        ("127.0.0.1", False),
        ("169.254.1.1", False),
        ("224.0.0.1", False),
        ("0.0.0.0", False),
        ("192.0.2.1", False),
        ("::1", False),
        ("not-an-ip", False),
    ],
)
def test_groww_static_ip_requires_global_ipv4(candidate: str, expected: bool) -> None:
    assert _is_public_ipv4(candidate) is expected


class CountingAuth:
    def __init__(self) -> None:
        self.calls = 0

    def authenticated_client(self, _: object) -> FakeGrowwClient:
        self.calls += 1
        raise RuntimeError("secret-token-123")


def test_groww_never_acquires_auth_before_local_gates_pass() -> None:
    provider = CountingAuth()
    broker = GrowwBroker(
        groww_settings(GROWW_STATIC_OUTBOUND_IP="10.0.0.1"),
        auth_provider=provider,
    )

    report = broker.preflight()

    assert provider.calls == 0
    assert "GROWW_STATIC_OUTBOUND_IPV4" in report.missing_gate_names
    assert "secret-token-123" not in repr(report)


@pytest.mark.parametrize(
    "expiry",
    [NOW - timedelta(seconds=1), datetime(2026, 8, 9, 10)],
)
def test_groww_direct_bearer_requires_aware_future_expiry(expiry: datetime) -> None:
    broker = GrowwBroker(
        groww_settings(),
        client=FakeGrowwClient(),
        access_token_expires_at=expiry,
        clock=lambda: NOW,
    )

    report = broker.preflight()

    assert report.ready is False
    assert "GROWW_AUTH_SESSION_FRESH" in report.missing_gate_names


def test_groww_api_key_session_provider_supplies_fresh_aware_expiry() -> None:
    class Provider:
        def authenticated_session(self, _: object, now: datetime) -> GrowwSession:
            return GrowwSession(FakeGrowwClient(), now + timedelta(minutes=1))

    settings = groww_settings(
        GROWW_ACCESS_TOKEN="",
        GROWW_API_KEY="test-key",
        GROWW_SECRET_KEY="test-secret",
    )
    report = GrowwBroker(settings, auth_provider=Provider(), clock=lambda: NOW).preflight()

    assert report.ready is True


@pytest.mark.parametrize(
    "account",
    [
        {**_account(), "account_blocked": "false"},
        {**_account(), "account_blocked": None},
        {**_account(), "buying_power": "NaN"},
        {**_account(), "buying_power": "Infinity"},
        {**_account(), "status": "active"},
    ],
)
def test_alpaca_preflight_rejects_malformed_account_primitives(account: dict[str, Any]) -> None:
    report = AlpacaBroker(alpaca_settings(), client=FakeAlpacaClient(account)).preflight()

    assert report.ready is False


def test_records_reject_missing_ids_unknown_status_and_naive_time() -> None:
    record = {
        "id": "",
        "client_order_id": "client-1",
        "symbol": "AAPL",
        "status": "emulated",
        "qty": "1",
        "filled_qty": "0",
        "submitted_at": "2026-08-09T10:00:00",
        "updated_at": "2026-08-09T10:00:00",
    }

    with pytest.raises(ValueError):
        broker_order(record, broker="alpaca")


def test_ccxt_preflight_rejects_string_capabilities_and_wrong_exchange_config() -> None:
    exchange = FakeCcxtExchange(_market())
    exchange.id = "wrong-venue"
    exchange.enableRateLimit = "true"
    exchange.options = {"defaultType": "emulated"}
    exchange.has = {"createOrder": "true"}

    report = CcxtSpotBroker(ccxt_settings(), exchange=exchange).preflight()

    assert report.ready is False
    assert "CCXT_EXCHANGE_CONFIGURED" in report.missing_gate_names


def test_ccxt_market_order_requires_exact_native_capability() -> None:
    exchange = FakeCcxtExchange(_market())
    exchange.has = {"createOrder": True, "createMarketOrder": "emulated"}
    broker = CcxtSpotBroker(ccxt_settings(), exchange=exchange)

    with pytest.raises(ValueError, match="native"):
        broker.submit(
            intent(instrument_id="BTC/USDT@ccxt", quantity="1", notional=None),
            snapshot(instrument_id="BTC/USDT@ccxt"),
        )


def test_ccxt_rejects_cost_below_minimum_using_limit_price() -> None:
    exchange = FakeCcxtExchange(_market())
    broker = CcxtSpotBroker(ccxt_settings(), exchange=exchange)

    with pytest.raises(ValueError, match="cost"):
        broker.submit(
            intent(
                instrument_id="BTC/USDT@ccxt",
                quantity="1",
                notional=None,
                limit_price="1",
            ),
            snapshot(instrument_id="BTC/USDT@ccxt"),
        )


def test_groww_rejects_fractional_quantity_before_submission() -> None:
    client = FakeGrowwClient()
    broker = GrowwBroker(
        groww_settings(),
        client=client,
        access_token_expires_at=NOW + timedelta(minutes=1),
        clock=lambda: NOW,
    )

    with pytest.raises(ValueError, match="integer"):
        broker.submit(
            intent(instrument_id="RELIANCE@groww", quantity="1.5", notional=None),
            snapshot(instrument_id="RELIANCE@groww"),
        )


def test_groww_rejects_capability_for_different_instrument() -> None:
    class WrongInstrument(FakeGrowwClient):
        def instrument_capabilities(self, symbol: str) -> dict[str, object]:
            payload = super().instrument_capabilities(symbol)
            payload["symbol"] = "OTHER"
            return payload

    broker = GrowwBroker(
        groww_settings(),
        client=WrongInstrument(),
        access_token_expires_at=NOW + timedelta(minutes=1),
        clock=lambda: NOW,
    )

    with pytest.raises(ValueError, match="capability"):
        broker.submit(
            intent(instrument_id="RELIANCE@groww", quantity="1", notional=None),
            snapshot(instrument_id="RELIANCE@groww"),
        )


def test_injected_exception_is_sanitized_without_context() -> None:
    class ExplodingAlpaca(FakeAlpacaClient):
        def get_asset(self, symbol: str) -> dict[str, Any]:
            del symbol
            raise RuntimeError("secret-token-123")

    broker = AlpacaBroker(alpaca_settings(), client=ExplodingAlpaca(_account()))
    with pytest.raises(RuntimeError) as raised:
        broker.submit(intent(), snapshot())

    assert str(raised.value) == "alpaca client operation failed"
    assert repr(raised.value) == "RuntimeError('alpaca client operation failed')"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_groww_client_reference_lookup_and_cancel_are_not_server_lookup() -> None:
    client = FakeGrowwClient()
    broker = GrowwBroker(
        groww_settings(),
        client=client,
        access_token_expires_at=NOW + timedelta(minutes=1),
        clock=lambda: NOW,
    )

    by_client = broker.get_order_by_client_id("intent-1")
    cancelled = broker.cancel("server-1", at=NOW)

    assert by_client.order_id == "server-from-client-reference"
    assert cancelled.status.value == "cancelled"


def test_alpaca_client_reference_lookup_and_cancel_are_not_server_lookup() -> None:
    client = FakeAlpacaClient(_account())
    broker = AlpacaBroker(alpaca_settings(), client=client)

    by_client = broker.get_order_by_client_id("intent-1")
    cancelled = broker.cancel("server-1", at=NOW)

    assert by_client.order_id == "server-from-client-reference"
    assert cancelled.status.value == "cancelled"


def test_ccxt_positions_use_spot_balance_not_derivatives_positions() -> None:
    class Valuation:
        def value_spot(self, _: str, currency: str, quote: str, __: object) -> dict[str, object]:
            return {
                "instrument_id": f"{currency}/{quote}@ccxt-spot",
                "average_price": "100",
                "market_price": "101",
                "observed_at": NOW,
                "max_age_seconds": 60,
            }

    exchange = FakeCcxtExchange(_market())
    exchange.balance = {"free": {}, "used": {}, "total": {"BTC": "1"}, "info": {}}
    broker = CcxtSpotBroker(
        ccxt_settings(),
        exchange=exchange,
        valuation_provider=Valuation(),
        clock=lambda: NOW,
    )

    positions = broker.positions()

    assert positions[0].instrument_id == "BTC/USDT@ccxt-spot"
    assert exchange.fetch_positions_called is False


def test_ccxt_factory_receives_redacted_local_auth_configuration() -> None:
    received: list[CcxtConnectionConfig] = []

    def factory(config: CcxtConnectionConfig) -> FakeCcxtExchange:
        received.append(config)
        return FakeCcxtExchange(_market())

    report = CcxtSpotBroker(ccxt_settings(), exchange_factory=factory).preflight()

    assert report.ready is True
    assert received[0].exchange_id == "testexchange"
    assert received[0].enable_rate_limit is True
    assert "test-secret-value" not in repr(received[0])
    assert "credentials_present=True" in repr(received[0])


def test_ccxt_client_reference_lookup_uses_distinct_boundary() -> None:
    order = CcxtSpotBroker(
        ccxt_settings(), exchange=FakeCcxtExchange(_market())
    ).get_order_by_client_id("intent-1")

    assert order.order_id == "server-from-client-reference"
