"""Private conversions from injected broker client records to domain models."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from market_sentinel.domain import BrokerOrder, OrderStatus, Position


def value(record: object, name: str, default: object = None) -> object:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def symbol_from_instrument(instrument_id: str) -> str:
    return instrument_id.split("@", maxsplit=1)[0]


def decimal(value_: object, default: str = "0") -> Decimal:
    if value_ is None:
        return Decimal(default)
    return Decimal(str(value_))


def timestamp(value_: object) -> datetime:
    if isinstance(value_, datetime):
        return value_.astimezone(UTC)
    if isinstance(value_, (int, float)):
        return datetime.fromtimestamp(value_ / 1000, tz=UTC)
    if isinstance(value_, str):
        return datetime.fromisoformat(value_.replace("Z", "+00:00")).astimezone(UTC)
    return datetime.now(UTC)


def order_status(status: object) -> OrderStatus:
    normalized = str(status).lower().replace("-", "_")
    aliases = {
        "accepted": OrderStatus.ACKNOWLEDGED,
        "new": OrderStatus.ACKNOWLEDGED,
        "open": OrderStatus.ACKNOWLEDGED,
        "pending_new": OrderStatus.ACKNOWLEDGED,
        "partially_filled": OrderStatus.PARTIALLY_FILLED,
        "partiallyfilled": OrderStatus.PARTIALLY_FILLED,
        "filled": OrderStatus.FILLED,
        "rejected": OrderStatus.REJECTED,
        "cancelled": OrderStatus.CANCELLED,
        "canceled": OrderStatus.CANCELLED,
        "expired": OrderStatus.EXPIRED,
    }
    return aliases.get(normalized, OrderStatus.UNKNOWN)


def broker_order(
    record: object,
    *,
    broker: str,
    default_client_order_id: str = "",
) -> BrokerOrder:
    submitted_at = value(record, "submitted_at")
    created_at = value(record, "created_at", value(record, "timestamp"))
    created = timestamp(submitted_at if submitted_at is not None else created_at)
    client_order_id = value(record, "client_order_id")
    if client_order_id is None:
        client_order_id = value(record, "clientOrderId", default_client_order_id)
    requested_quantity = value(record, "qty")
    if requested_quantity is None:
        requested_quantity = value(record, "quantity", value(record, "amount"))
    filled_quantity = value(record, "filled_qty")
    if filled_quantity is None:
        filled_quantity = value(record, "filled_quantity", value(record, "filled"))
    return BrokerOrder(
        order_id=str(value(record, "id", value(record, "order_id", "unknown"))),
        client_order_id=str(client_order_id),
        broker=broker,
        instrument_id=f"{value(record, 'symbol', '')}@{broker}",
        status=order_status(value(record, "status", "unknown")),
        requested_quantity=_optional_decimal(requested_quantity),
        filled_quantity=decimal(filled_quantity),
        average_fill_price=_optional_decimal(
            value(record, "filled_avg_price", value(record, "average"))
        ),
        submitted_at=created,
        updated_at=timestamp(value(record, "updated_at", value(record, "timestamp", created))),
    )


def position(record: object, *, broker: str) -> Position:
    symbol = str(value(record, "symbol", value(record, "asset", "")))
    quantity = value(record, "qty")
    if quantity is None:
        quantity = value(record, "quantity", value(record, "contracts"))
    return Position(
        instrument_id=f"{symbol}@{broker}",
        quantity=decimal(quantity),
        average_price=decimal(value(record, "avg_entry_price", value(record, "average_price"))),
        market_price=decimal(value(record, "current_price", value(record, "market_price"))),
        unrealized_pnl=decimal(value(record, "unrealized_pl", value(record, "unrealized_pnl"))),
    )


def _optional_decimal(value_: object) -> Decimal | None:
    return None if value_ is None else decimal(value_)


def mapping(value_: object) -> Mapping[str, Any]:
    if not isinstance(value_, Mapping):
        raise TypeError("injected client must return a mapping record")
    return value_
