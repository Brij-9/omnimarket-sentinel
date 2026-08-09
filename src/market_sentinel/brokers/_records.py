"""Strict conversion of injected broker records to immutable domain records."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any

from market_sentinel.domain import BrokerOrder, OrderStatus, Position


def value(record: object, name: str, default: object = None) -> object:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def mapping(value_: object) -> Mapping[str, Any]:
    if not isinstance(value_, Mapping):
        raise ValueError("broker record must be a mapping")
    return value_


def symbol_from_instrument(instrument_id: str) -> str:
    symbol = instrument_id.split("@", maxsplit=1)[0]
    return _identifier(symbol, "instrument symbol")


def decimal(value_: object, *, nonnegative: bool = False, positive: bool = False) -> Decimal:
    if isinstance(value_, bool) or value_ is None:
        raise ValueError("numeric broker field is invalid")
    try:
        parsed = Decimal(str(value_))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("numeric broker field is invalid") from error
    if not parsed.is_finite() or (nonnegative and parsed < 0) or (positive and parsed <= 0):
        raise ValueError("numeric broker field is invalid")
    return parsed


def timestamp(value_: object) -> datetime:
    if isinstance(value_, datetime):
        if value_.tzinfo is None or value_.utcoffset() is None:
            raise ValueError("broker timestamp must be timezone-aware")
        return value_.astimezone(UTC)
    if isinstance(value_, (int, float)) and not isinstance(value_, bool):
        if not isfinite(value_):
            raise ValueError("broker timestamp is invalid")
        return datetime.fromtimestamp(value_ / 1000, tz=UTC)
    if isinstance(value_, str):
        parsed = datetime.fromisoformat(value_.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("broker timestamp must be timezone-aware")
        return parsed.astimezone(UTC)
    raise ValueError("broker timestamp is invalid")


def order_status(status: object) -> OrderStatus:
    if not isinstance(status, str):
        raise ValueError("broker order status is invalid")
    aliases = {
        "accepted": OrderStatus.ACKNOWLEDGED,
        "new": OrderStatus.ACKNOWLEDGED,
        "open": OrderStatus.ACKNOWLEDGED,
        "pending_new": OrderStatus.ACKNOWLEDGED,
        "acknowledged": OrderStatus.ACKNOWLEDGED,
        "partially_filled": OrderStatus.PARTIALLY_FILLED,
        "filled": OrderStatus.FILLED,
        "rejected": OrderStatus.REJECTED,
        "cancelled": OrderStatus.CANCELLED,
        "canceled": OrderStatus.CANCELLED,
        "expired": OrderStatus.EXPIRED,
    }
    try:
        return aliases[status.lower().replace("-", "_")]
    except KeyError as error:
        raise ValueError("broker order status is invalid") from error


def broker_order(record: object, *, broker: str, default_client_order_id: str = "") -> BrokerOrder:
    order_id = _identifier(value(record, "id", value(record, "order_id")), "order id")
    client_id = value(
        record, "client_order_id", value(record, "clientOrderId", default_client_order_id)
    )
    requested = value(record, "qty", value(record, "quantity", value(record, "amount")))
    filled = value(record, "filled_qty", value(record, "filled_quantity", value(record, "filled")))
    submitted = value(
        record, "submitted_at", value(record, "created_at", value(record, "timestamp"))
    )
    updated = value(record, "updated_at", value(record, "timestamp", submitted))
    average = value(record, "filled_avg_price", value(record, "average"))
    return BrokerOrder(
        order_id=order_id,
        client_order_id=_identifier(client_id, "client order id"),
        broker=_identifier(broker, "broker"),
        instrument_id=f"{_identifier(value(record, 'symbol'), 'symbol')}@{broker}",
        status=order_status(value(record, "status")),
        requested_quantity=None if requested is None else decimal(requested, positive=True),
        filled_quantity=decimal(filled, nonnegative=True),
        average_fill_price=None if average is None else decimal(average, positive=True),
        submitted_at=timestamp(submitted),
        updated_at=timestamp(updated),
    )


def position(record: object, *, broker: str) -> Position:
    symbol = _identifier(value(record, "symbol", value(record, "asset")), "symbol")
    return Position(
        instrument_id=f"{symbol}@{broker}",
        quantity=decimal(
            value(record, "qty", value(record, "quantity", value(record, "contracts")))
        ),
        average_price=decimal(
            value(record, "avg_entry_price", value(record, "average_price")), nonnegative=True
        ),
        market_price=decimal(
            value(record, "current_price", value(record, "market_price")), nonnegative=True
        ),
        unrealized_pnl=decimal(value(record, "unrealized_pl", value(record, "unrealized_pnl"))),
    )


def _identifier(value_: object, name: str) -> str:
    if not isinstance(value_, str) or not value_ or value_ != value_.strip():
        raise ValueError(f"broker {name} is invalid")
    return value_
