from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from market_sentinel.brokers.ccxt_spot import CcxtSpotBroker
from tests.factories import intent, snapshot
from tests.settings import ccxt_settings


class FakeCcxtExchange:
    def __init__(self, market: Mapping[str, Any]) -> None:
        self.market = market
        self.calls: list[str] = []
        self.id = "testexchange"
        self.enableRateLimit = True
        self.options = {"defaultType": "spot"}
        self.has = {"createOrder": True, "createMarketOrder": True}
        self.submissions: list[dict[str, Any]] = []
        self.fetch_positions_called = False

    def set_sandbox_mode(self, enabled: bool) -> None:
        assert enabled is True
        self.calls.append("sandbox")

    def load_markets(self) -> Mapping[str, Mapping[str, Any]]:
        self.calls.append("markets")
        return {str(self.market["symbol"]): self.market}

    def amount_to_precision(self, symbol: str, amount: str) -> str:
        del symbol
        return amount

    def price_to_precision(self, symbol: str, price: str) -> str:
        del symbol
        return price

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: str,
        price: str | None,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls.append("submit")
        self.submissions.append(
            {
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "amount": amount,
                "price": price,
                "params": dict(params),
            }
        )
        return {
            "id": "ccxt-order-1",
            "clientOrderId": params["clientOrderId"],
            "symbol": symbol,
            "status": "open",
            "amount": amount,
            "filled": "0",
            "timestamp": 1786269600000,
        }

    def fetch_order(self, order_id: str) -> Mapping[str, Any]:
        return {
            "id": order_id,
            "clientOrderId": "intent-1",
            "symbol": "BTC/USDT",
            "status": "open",
            "amount": "1",
            "filled": "0",
            "timestamp": 1786269600000,
        }

    def fetch_order_by_client_id(self, client_order_id: str) -> Mapping[str, Any]:
        order = dict(self.fetch_order("server-from-client-reference"))
        order["clientOrderId"] = client_order_id
        return order

    def fetch_balance(self) -> Mapping[str, Any]:
        return getattr(self, "balance", {})


def _market() -> Mapping[str, Any]:
    fixture = Path(__file__).parents[1] / "fixtures" / "ccxt_market.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


def test_ccxt_sandbox_is_enabled_before_loading_markets() -> None:
    exchange = FakeCcxtExchange(_market())
    broker = CcxtSpotBroker(ccxt_settings(), exchange=exchange)

    assert broker.preflight().ready is True
    assert exchange.calls == ["sandbox", "markets"]


def test_ccxt_rejects_leverage_and_market_order_emulation() -> None:
    exchange = FakeCcxtExchange(_market())
    broker = CcxtSpotBroker(ccxt_settings(), exchange=exchange)

    try:
        broker.submit(
            intent(instrument_id="BTC/USDT@ccxt", quantity="1", notional=None, product="margin"),
            snapshot(instrument_id="BTC/USDT@ccxt"),
        )
    except ValueError as error:
        assert "spot" in str(error).lower()
    else:
        raise AssertionError("non-spot product must be rejected")


def test_ccxt_without_sandbox_needs_explicit_acknowledgement() -> None:
    broker = CcxtSpotBroker(ccxt_settings(CCXT_SANDBOX=False), exchange=FakeCcxtExchange(_market()))

    report = broker.preflight()
    assert report.ready is False
    assert "CCXT_NO_SANDBOX_ACKNOWLEDGED" in report.missing_gate_names
