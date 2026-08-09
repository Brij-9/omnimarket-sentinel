"""Behavioral coverage for the durable order transition contract."""

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from market_sentinel.domain.enums import OrderStatus
from market_sentinel.domain.models import BrokerOrder
from market_sentinel.execution.state_machine import (
    InvalidOrderTransition,
    OrderStateMachine,
    OrderTransitionEvent,
)

AT = datetime(2026, 8, 9, 10, tzinfo=UTC)


def _order(status: OrderStatus = OrderStatus.PROPOSED) -> BrokerOrder:
    return BrokerOrder(
        order_id="paper-order-1",
        client_order_id="intent-1",
        broker="paper",
        instrument_id="AAPL@alpaca",
        status=status,
        requested_quantity=Decimal("1"),
        filled_quantity=Decimal("0"),
        average_fill_price=None,
        submitted_at=AT,
        updated_at=AT,
    )


@pytest.mark.parametrize(
    ("prior", "new"),
    [
        (OrderStatus.PROPOSED, OrderStatus.RISK_APPROVED),
        (OrderStatus.PROPOSED, OrderStatus.REJECTED),
        (OrderStatus.PROPOSED, OrderStatus.EXPIRED),
        (OrderStatus.RISK_APPROVED, OrderStatus.CONFIRMED),
        (OrderStatus.RISK_APPROVED, OrderStatus.REJECTED),
        (OrderStatus.RISK_APPROVED, OrderStatus.CANCELLED),
        (OrderStatus.RISK_APPROVED, OrderStatus.EXPIRED),
        (OrderStatus.CONFIRMED, OrderStatus.SUBMITTING),
        (OrderStatus.CONFIRMED, OrderStatus.CANCELLED),
        (OrderStatus.CONFIRMED, OrderStatus.EXPIRED),
        (OrderStatus.SUBMITTING, OrderStatus.ACKNOWLEDGED),
        (OrderStatus.SUBMITTING, OrderStatus.REJECTED),
        (OrderStatus.SUBMITTING, OrderStatus.CANCELLED),
        (OrderStatus.SUBMITTING, OrderStatus.EXPIRED),
        (OrderStatus.SUBMITTING, OrderStatus.UNKNOWN),
        (OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED),
        (OrderStatus.ACKNOWLEDGED, OrderStatus.FILLED),
        (OrderStatus.ACKNOWLEDGED, OrderStatus.CANCELLED),
        (OrderStatus.ACKNOWLEDGED, OrderStatus.REJECTED),
        (OrderStatus.ACKNOWLEDGED, OrderStatus.EXPIRED),
        (OrderStatus.ACKNOWLEDGED, OrderStatus.UNKNOWN),
        (OrderStatus.PARTIALLY_FILLED, OrderStatus.PARTIALLY_FILLED),
        (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED),
        (OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELLED),
        (OrderStatus.PARTIALLY_FILLED, OrderStatus.REJECTED),
        (OrderStatus.PARTIALLY_FILLED, OrderStatus.EXPIRED),
        (OrderStatus.PARTIALLY_FILLED, OrderStatus.UNKNOWN),
    ],
)
def test_transition_table_allows_only_explicit_forward_and_resolution_paths(
    prior: OrderStatus,
    new: OrderStatus,
) -> None:
    """Removing an approved edge would strand a valid order lifecycle branch."""
    events: list[OrderTransitionEvent] = []
    source = _order(prior)

    result = OrderStateMachine.transition(source, new, at=AT, emit=events.append)

    assert isinstance(result, BrokerOrder)
    assert result.status is new
    assert source.status is prior
    assert len(events) == 1


@pytest.mark.parametrize(
    "terminal",
    [
        OrderStatus.FILLED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
        OrderStatus.EXPIRED,
        OrderStatus.UNKNOWN,
    ],
)
@pytest.mark.parametrize("target", list(OrderStatus))
def test_terminal_states_are_immutable(
    terminal: OrderStatus,
    target: OrderStatus,
) -> None:
    """Any transition out of a terminal state could duplicate or resurrect execution."""
    events: list[OrderTransitionEvent] = []

    with pytest.raises(InvalidOrderTransition):
        OrderStateMachine.transition(_order(terminal), target, at=AT, emit=events.append)

    assert events == []

@pytest.mark.parametrize(
    ("prior", "new"),
    [
        (OrderStatus.FILLED, OrderStatus.SUBMITTING),
        (OrderStatus.ACKNOWLEDGED, OrderStatus.CONFIRMED),
        (OrderStatus.PROPOSED, OrderStatus.FILLED),
        (OrderStatus.CONFIRMED, OrderStatus.ACKNOWLEDGED),
        (OrderStatus.ACKNOWLEDGED, OrderStatus.ACKNOWLEDGED),
    ],
)
def test_invalid_transition_has_no_mutation_or_audit(
    prior: OrderStatus,
    new: OrderStatus,
) -> None:
    """Validating after emission would leave a false durable transition record."""
    events: list[OrderTransitionEvent] = []
    source = _order(prior)

    with pytest.raises(InvalidOrderTransition):
        OrderStateMachine.transition(source, new, at=AT, emit=events.append)

    assert source.status is prior
    assert events == []


def test_status_only_invalid_transition_matches_the_public_plan_contract() -> None:
    """The documented validator form must reject a backward terminal transition."""
    with pytest.raises(InvalidOrderTransition):
        OrderStateMachine.transition(OrderStatus.FILLED, OrderStatus.SUBMITTING)


def test_successful_transition_emits_complete_utc_identity_event() -> None:
    """Dropping an identity or timestamp would make reconciliation evidence ambiguous."""
    events: list[OrderTransitionEvent] = []
    offset_at = datetime(2026, 8, 9, 15, 31, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    result = OrderStateMachine.transition(
        _order(OrderStatus.SUBMITTING),
        OrderStatus.ACKNOWLEDGED,
        at=offset_at,
        emit=events.append,
    )

    assert isinstance(result, BrokerOrder)
    assert result.updated_at == datetime(2026, 8, 9, 10, 1, tzinfo=UTC)
    assert events == [
        OrderTransitionEvent(
            prior_status=OrderStatus.SUBMITTING,
            new_status=OrderStatus.ACKNOWLEDGED,
            client_intent_id="intent-1",
            broker_order_id="paper-order-1",
            occurred_at=datetime(2026, 8, 9, 10, 1, tzinfo=UTC),
        )
    ]


def test_successful_status_only_transition_requires_audit_context() -> None:
    """A valid status edge without durable identity evidence must fail closed."""
    with pytest.raises(ValueError, match="audit context"):
        OrderStateMachine.transition(OrderStatus.PROPOSED, OrderStatus.RISK_APPROVED)


@pytest.mark.parametrize(
    "at",
    [datetime(2026, 8, 9, 10), AT - timedelta(microseconds=1)],
)
def test_transition_rejects_naive_or_backward_timestamp_without_audit(at: datetime) -> None:
    """A non-chronological event would corrupt the per-order audit timeline."""
    events: list[OrderTransitionEvent] = []

    with pytest.raises(ValueError, match="timestamp"):
        OrderStateMachine.transition(
            _order(OrderStatus.PROPOSED),
            OrderStatus.RISK_APPROVED,
            at=at,
            emit=events.append,
        )

    assert events == []
