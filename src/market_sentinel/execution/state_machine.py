"""Explicit immutable order-state transitions with mandatory audit emission."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import overload

from market_sentinel.domain.enums import OrderStatus
from market_sentinel.domain.models import BrokerOrder


class InvalidOrderTransition(ValueError):
    """Raised when an order attempts an unapproved or terminal transition."""


@dataclass(frozen=True, slots=True)
class OrderTransitionEvent:
    """Complete immutable identity evidence for one successful transition."""

    prior_status: OrderStatus
    new_status: OrderStatus
    client_intent_id: str
    broker_order_id: str
    occurred_at: datetime


_ALLOWED: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.PROPOSED: frozenset(
        {OrderStatus.RISK_APPROVED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
    ),
    OrderStatus.RISK_APPROVED: frozenset(
        {
            OrderStatus.CONFIRMED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        }
    ),
    OrderStatus.CONFIRMED: frozenset(
        {OrderStatus.SUBMITTING, OrderStatus.CANCELLED, OrderStatus.EXPIRED}
    ),
    OrderStatus.SUBMITTING: frozenset(
        {
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.UNKNOWN,
        }
    ),
    OrderStatus.ACKNOWLEDGED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.UNKNOWN,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.UNKNOWN,
        }
    ),
}


class OrderStateMachine:
    """Validate one edge, emit its audit event, then return a new immutable state."""

    @staticmethod
    @overload
    def transition(
        current: BrokerOrder,
        new_status: OrderStatus,
        *,
        at: datetime | None = None,
        emit: Callable[[OrderTransitionEvent], None] | None = None,
        client_intent_id: str | None = None,
        broker_order_id: str | None = None,
    ) -> BrokerOrder: ...

    @staticmethod
    @overload
    def transition(
        current: OrderStatus,
        new_status: OrderStatus,
        *,
        at: datetime | None = None,
        emit: Callable[[OrderTransitionEvent], None] | None = None,
        client_intent_id: str | None = None,
        broker_order_id: str | None = None,
    ) -> OrderStatus: ...

    @staticmethod
    def transition(
        current: BrokerOrder | OrderStatus,
        new_status: OrderStatus,
        *,
        at: datetime | None = None,
        emit: Callable[[OrderTransitionEvent], None] | None = None,
        client_intent_id: str | None = None,
        broker_order_id: str | None = None,
    ) -> BrokerOrder | OrderStatus:
        """Apply only an approved edge; invalid edges never emit or mutate."""
        prior_status = current.status if isinstance(current, BrokerOrder) else current
        if not isinstance(prior_status, OrderStatus) or not isinstance(new_status, OrderStatus):
            raise ValueError("order states must be OrderStatus values")
        if new_status not in _ALLOWED.get(prior_status, frozenset()):
            raise InvalidOrderTransition(
                f"invalid order transition: {prior_status.value} -> {new_status.value}"
            )

        inferred_client_id = (
            current.client_order_id if isinstance(current, BrokerOrder) else client_intent_id
        )
        inferred_broker_id = (
            current.order_id if isinstance(current, BrokerOrder) else broker_order_id
        )
        if (
            at is None
            or emit is None
            or not inferred_client_id
            or not inferred_broker_id
        ):
            raise ValueError("successful transition requires complete audit context")
        occurred_at = _utc_timestamp(at)
        if isinstance(current, BrokerOrder) and occurred_at < current.updated_at.astimezone(UTC):
            raise ValueError("transition timestamp must not precede the current order timestamp")

        event = OrderTransitionEvent(
            prior_status=prior_status,
            new_status=new_status,
            client_intent_id=inferred_client_id,
            broker_order_id=inferred_broker_id,
            occurred_at=occurred_at,
        )
        emit(event)
        if isinstance(current, BrokerOrder):
            return current.model_copy(
                update={"status": new_status, "updated_at": occurred_at}
            )
        return new_status


def _utc_timestamp(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("transition timestamp must be timezone-aware")
    return value.astimezone(UTC)
