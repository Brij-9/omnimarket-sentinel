from datetime import UTC, datetime, timedelta

from market_sentinel.brokers.groww import GrowwBroker
from tests.factories import intent, snapshot
from tests.settings import groww_settings


class FakeGrowwClient:
    def __init__(self) -> None:
        self.submissions: list[dict[str, object]] = []

    def profile(self) -> dict[str, object]:
        return {"active": True}

    def capabilities(self) -> dict[str, object]:
        return {"regular_session": True, "protected_orders": True}

    def instrument_capabilities(self, symbol: str) -> dict[str, object]:
        return {
            "symbol": symbol,
            "tradable": True,
            "protected_orders": True,
            "products": ("cash",),
            "sessions": ("regular",),
            "order_types": ("market", "limit", "stop", "stop_limit"),
            "lot_size": "1",
        }

    def submit_order(self, payload: dict[str, object]) -> dict[str, object]:
        self.submissions.append(payload)
        return {
            "order_id": "groww-order-1",
            "status": "ACKNOWLEDGED",
            "symbol": "RELIANCE",
            "quantity": "1",
            "filled_quantity": "0",
            "created_at": "2026-08-09T10:00:00Z",
        }

    def get_order(self, order_id: str) -> dict[str, object]:
        return {
            "order_id": order_id,
            "status": "ACKNOWLEDGED",
            "symbol": "RELIANCE",
            "quantity": "1",
            "filled_quantity": "0",
            "created_at": "2026-08-09T10:00:00Z",
        }

    def get_order_by_client_id(self, client_order_id: str) -> dict[str, object]:
        order = self.get_order("server-from-client-reference")
        order["client_order_id"] = client_order_id
        return order

    def cancel_order(self, order_id: str) -> dict[str, object]:
        order = self.get_order(order_id)
        order["status"] = "cancelled"
        order["client_order_id"] = "intent-1"
        return order

    def positions(self) -> list[dict[str, object]]:
        return []


def test_groww_static_ip_gate_cannot_be_bypassed() -> None:
    report = GrowwBroker.from_settings(groww_settings(static_ip_allowlisted=False)).preflight()

    assert report.ready is False
    assert "GROWW_STATIC_IP_ALLOWLISTED" in report.missing_gate_names


def test_groww_submit_maps_intent_id_to_order_reference() -> None:
    client = FakeGrowwClient()
    now = datetime(2026, 8, 9, 10, tzinfo=UTC)
    broker = GrowwBroker(
        groww_settings(),
        client=client,
        access_token_expires_at=now + timedelta(minutes=1),
        clock=lambda: now,
    )
    order = broker.submit(
        intent(instrument_id="RELIANCE@groww", quantity="1", notional=None),
        snapshot(instrument_id="RELIANCE@groww"),
    )

    assert order.client_order_id == "intent-1"
    assert client.submissions[0]["order_reference_id"] == "intent-1"
