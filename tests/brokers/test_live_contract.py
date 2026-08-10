"""Task 14 additions to the strict broker acknowledgement conversion contract."""

from decimal import Decimal

from market_sentinel.brokers._records import broker_order


def test_records_preserve_exact_notional_echo_without_fabricating_quantity() -> None:
    """A notional-capable adapter retains the provider echo needed by live validation."""
    order = broker_order(
        {
            "id": "order-1",
            "client_order_id": "intent-1",
            "symbol": "AAPL",
            "status": "accepted",
            "qty": None,
            "notional": "10.00",
            "filled_qty": "0",
            "filled_avg_price": None,
            "submitted_at": "2026-08-09T10:00:00Z",
            "updated_at": "2026-08-09T10:00:00Z",
        },
        broker="alpaca",
    )
    assert order.requested_quantity is None
    assert order.requested_notional == Decimal("10.00")
