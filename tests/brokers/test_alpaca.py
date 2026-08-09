from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from market_sentinel.brokers.alpaca import AlpacaBroker
from tests.factories import intent, snapshot
from tests.settings import alpaca_settings


class FakeAlpacaClient:
    def __init__(self, account: Mapping[str, Any]) -> None:
        self.account = account
        self.submissions: list[dict[str, Any]] = []

    def get_account(self) -> Mapping[str, Any]:
        return self.account

    def get_asset(self, symbol: str) -> Mapping[str, Any]:
        return {"symbol": symbol, "tradable": True, "fractionable": True}

    def submit_order(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.submissions.append(dict(payload))
        return {
            "id": "alpaca-order-1",
            "client_order_id": payload["client_order_id"],
            "symbol": payload["symbol"],
            "status": "accepted",
            "qty": payload.get("qty", "0"),
            "filled_qty": "0",
            "submitted_at": "2026-08-09T10:00:00Z",
            "updated_at": "2026-08-09T10:00:00Z",
        }

    def get_order(self, order_id: str) -> Mapping[str, Any]:
        return {
            "id": order_id,
            "client_order_id": "intent-1",
            "symbol": "AAPL",
            "status": "accepted",
            "qty": "1",
            "filled_qty": "0",
            "submitted_at": "2026-08-09T10:00:00Z",
            "updated_at": "2026-08-09T10:00:00Z",
        }

    def get_positions(self) -> list[Mapping[str, Any]]:
        return []


def _account() -> Mapping[str, Any]:
    fixture = Path(__file__).parents[1] / "fixtures" / "alpaca_account.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


def test_alpaca_live_preflight_accepts_only_ready_account_and_endpoint() -> None:
    broker = AlpacaBroker(alpaca_settings(), client=FakeAlpacaClient(_account()))

    assert broker.preflight().ready is True


def test_alpaca_submit_maps_intent_id_to_client_order_id() -> None:
    client = FakeAlpacaClient(_account())
    broker = AlpacaBroker(alpaca_settings(), client=client)
    submitted = broker.submit(intent(), snapshot())

    assert submitted.client_order_id == "intent-1"
    assert client.submissions == [
        {
            "client_order_id": "intent-1",
            "notional": "10",
            "side": "buy",
            "symbol": "AAPL",
            "time_in_force": "day",
            "type": "market",
        }
    ]


def test_alpaca_submit_fails_closed_when_live_endpoint_is_wrong() -> None:
    broker = AlpacaBroker(
        alpaca_settings(ALPACA_TRADING_ENDPOINT="https://paper-api.alpaca.markets"),
        client=FakeAlpacaClient(_account()),
    )

    try:
        broker.submit(intent(), snapshot())
    except PermissionError as error:
        assert "preflight" in str(error).lower()
    else:
        raise AssertionError("submit must not run without a ready preflight")
