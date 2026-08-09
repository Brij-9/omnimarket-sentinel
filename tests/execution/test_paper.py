"""Deterministic paper-broker tests over real FillModel behavior."""

from __future__ import annotations

import copy as copy_module
import hashlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import market_sentinel.execution.paper as paper_module
from market_sentinel.backtest.engine import CostModel, FillModel
from market_sentinel.domain.clock import FrozenClock
from market_sentinel.domain.enums import AssetClass, OrderStatus, OrderType, Side
from market_sentinel.domain.models import (
    Bar,
    Fill,
    Instrument,
    MarketSnapshot,
    OrderIntent,
)
from market_sentinel.execution.base import BrokerAdapter, BrokerCapabilities
from market_sentinel.execution.paper import (
    DuplicateIntentConflict,
    PaperBroker,
    SessionHead,
)
from market_sentinel.execution.state_machine import InvalidOrderTransition
from market_sentinel.operations.audit import AuditEvent, AuditLog
from market_sentinel.portfolio.ledger import PortfolioLedger, PortfolioLedgerState
from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventRecord, EventStore

AT = datetime(2026, 8, 9, 10, tzinfo=UTC)
_PAPER_CAPABILITY_VALUES: dict[str, object] = {
    "broker": "paper",
    "supported_asset_classes": frozenset({AssetClass.EQUITY}),
    "supported_order_types": frozenset({OrderType.MARKET}),
    "supports_fractional_quantity": True,
    "supports_notional_orders": True,
    "supports_partial_fills": True,
    "supports_shorting": False,
    "supports_leverage": False,
    "supports_derivatives": False,
    "supports_cancel": True,
    "is_paper": True,
}


class _FailingAudit:
    """Exercise broker atomicity when durable audit persistence is unavailable."""

    def __init__(self, failing_kind: str, *, occurrence: int = 1) -> None:
        self.failing_kind = failing_kind
        self.occurrence = occurrence
        self._seen = 0

    def record_many(self, events: tuple[AuditEvent, ...]) -> None:
        for event in events:
            if event.kind == self.failing_kind:
                self._seen += 1
            if event.kind == self.failing_kind and self._seen == self.occurrence:
                raise RuntimeError("audit unavailable")


class _DiscardAudit:
    """A structural callable is not proof that a durable batch was persisted."""

    def record_many(self, events: tuple[AuditEvent, ...]) -> None:
        del events


def _instrument(
    *,
    instrument_id: str = "AAPL@alpaca",
    quote_currency: str = "USD",
    quantity_step: Decimal = Decimal("0.1"),
    minimum_notional: Decimal = Decimal("1"),
) -> Instrument:
    symbol, venue = instrument_id.split("@", maxsplit=1)
    return Instrument(
        symbol=symbol,
        venue=venue,
        asset_class=AssetClass.EQUITY,
        quote_currency=quote_currency,
        timezone="UTC",
        price_tick=Decimal("0.01"),
        quantity_step=quantity_step,
        minimum_notional=minimum_notional,
        session_calendar=None,
    )


def _bar(
    minute: int,
    *,
    open_price: str,
    high: str | None = None,
    low: str | None = None,
    close: str | None = None,
    volume: str = "100",
) -> Bar:
    opened = Decimal(open_price)
    closed = opened if close is None else Decimal(close)
    return Bar(
        at=AT + timedelta(minutes=minute),
        open=opened,
        high=max(opened, closed) if high is None else Decimal(high),
        low=min(opened, closed) if low is None else Decimal(low),
        close=closed,
        volume=Decimal(volume),
    )


def _snapshot(
    *bars: Bar,
    instrument_id: str = "AAPL@alpaca",
    observed_at: datetime | None = None,
    source_at: datetime | None = None,
    max_age_seconds: int = 60,
    provider: str = "fixture",
) -> MarketSnapshot:
    latest = bars[-1].at
    return MarketSnapshot(
        instrument_id=instrument_id,
        observed_at=latest if observed_at is None else observed_at,
        source_at=latest if source_at is None else source_at,
        bars=tuple(bars),
        provider=provider,
        max_age_seconds=max_age_seconds,
    )


def _intent(
    *,
    intent_id: str = "intent-1",
    instrument_id: str = "AAPL@alpaca",
    side: Side = Side.BUY,
    quantity: Decimal | None = Decimal("1"),
    notional: Decimal | None = None,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Decimal | None = None,
    trigger_price: Decimal | None = None,
    stop_loss: Decimal | None = Decimal("90"),
    take_profit: Decimal | None = Decimal("120"),
    time_in_force: str = "day",
    product: str = "cash",
    session: str = "regular",
    snapshot_hash: str = "a" * 64,
    created_at: datetime = AT,
    expires_at: datetime = AT + timedelta(minutes=10),
) -> OrderIntent:
    return OrderIntent(
        intent_id=intent_id,
        instrument_id=instrument_id,
        side=side,
        quantity=quantity,
        notional=notional,
        order_type=order_type,
        limit_price=limit_price,
        trigger_price=trigger_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        time_in_force=time_in_force,
        product=product,
        session=session,
        snapshot_hash=snapshot_hash,
        created_at=created_at,
        expires_at=expires_at,
    )


def test_same_client_intent_is_idempotent_and_audited() -> None:
    """Retrying an identical request must not create a second broker order."""
    submitted = _snapshot(_bar(0, open_price="100"))
    broker = PaperBroker(starting_cash=Decimal("1000"))

    first = broker.submit(_intent(), submitted)
    second = broker.submit(_intent(), submitted)

    assert second == first
    assert broker.order_count == 1
    assert [event.kind for event in broker.audit_events].count("paper.order.duplicate") == 1


def test_same_client_id_with_different_canonical_intent_is_rejected() -> None:
    """Treating an ID collision as a retry could execute altered size or prices."""
    submitted = _snapshot(_bar(0, open_price="100"))
    broker = PaperBroker(starting_cash=Decimal("1000"))
    broker.submit(_intent(), submitted)
    before = broker.get_order_by_client_id("intent-1")

    with pytest.raises(DuplicateIntentConflict):
        broker.submit(_intent(quantity=Decimal("2")), submitted)

    assert broker.order_count == 1
    assert broker.get_order_by_client_id("intent-1") == before
    assert broker.audit_events[-1].kind == "paper.order.duplicate_conflict"


def test_paper_broker_satisfies_immutable_credential_free_contract() -> None:
    """Adding credential or live-only controls to paper mode would enlarge its safety surface."""
    broker = PaperBroker()
    capabilities = broker.capabilities()

    assert isinstance(broker, BrokerAdapter)
    assert capabilities.broker == "paper"
    assert capabilities.supported_order_types == frozenset(OrderType)
    assert capabilities.supports_notional_orders is True
    assert capabilities.supports_fractional_quantity is True
    assert capabilities.supports_partial_fills is True
    assert capabilities.supports_shorting is False
    assert capabilities.supports_leverage is False
    assert capabilities.supports_derivatives is False
    with pytest.raises(FrozenInstanceError):
        capabilities.supports_leverage = True  # type: ignore[misc]
    public_surface = " ".join(
        [
            str(inspect.signature(PaperBroker)),
            *vars(broker),
            *dir(PaperBroker),
            repr(broker),
        ]
    ).lower()
    for credential_name in ("api_key", "secret", "credential", "password", "token"):
        assert credential_name not in public_surface


def test_broker_capabilities_reject_whitespace_names_and_non_bool_flags() -> None:
    """Truthiness must not make a malformed live capability appear enabled."""
    values = _PAPER_CAPABILITY_VALUES
    with pytest.raises(ValueError, match="broker"):
        BrokerCapabilities(**(values | {"broker": " paper "}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bool"):
        BrokerCapabilities(
            **(values | {"supports_leverage": 1})  # type: ignore[arg-type]
        )


def test_market_order_never_fills_on_submission_snapshot_and_uses_next_event_costs() -> None:
    """Using the current bar would introduce look-ahead and bypass shared fill costs."""
    first_bar = _bar(0, open_price="100")
    submitted = _snapshot(first_bar)
    model = FillModel(
        costs=CostModel(
            fee_bps=Decimal("10"),
            spread_bps=Decimal("20"),
            slippage_bps=Decimal("10"),
        )
    )
    broker = PaperBroker(fill_model=model, starting_cash=Decimal("1000"))
    order = broker.submit(_intent(), submitted)

    assert order.status is OrderStatus.ACKNOWLEDGED
    assert broker.on_snapshot(submitted, _instrument()) == ()
    next_snapshot = _snapshot(first_bar, _bar(1, open_price="100"))
    [fill] = broker.on_snapshot(next_snapshot, _instrument())

    assert fill.price == Decimal("100.20")
    assert fill.fee == Decimal("0.10020")
    assert fill.filled_at == AT + timedelta(minutes=1)
    assert broker.get_order(order.order_id).status is OrderStatus.FILLED


def test_limit_order_waits_for_a_later_cross_and_never_worsens_limit() -> None:
    """A non-crossing or cost-worsened limit must remain pending."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(
        fill_model=FillModel(costs=CostModel(spread_bps=Decimal("20"))),
        starting_cash=Decimal("1000"),
    )
    order = broker.submit(
        _intent(order_type=OrderType.LIMIT, limit_price=Decimal("99")),
        _snapshot(bar0),
    )

    no_cross = _snapshot(bar0, _bar(1, open_price="100", low="99.50", high="101"))
    assert broker.on_snapshot(no_cross, _instrument()) == ()
    cost_worsened = _snapshot(
        *no_cross.bars,
        _bar(2, open_price="99", low="98.50", high="100"),
    )
    assert broker.on_snapshot(cost_worsened, _instrument()) == ()
    crosses_below = _snapshot(
        *cost_worsened.bars,
        _bar(3, open_price="98", low="97.50", high="99"),
    )
    [fill] = broker.on_snapshot(crosses_below, _instrument())

    assert fill.price == Decimal("98.10")
    assert fill.price <= Decimal("99")
    assert broker.get_order(order.order_id).status is OrderStatus.FILLED


def test_latency_uses_fill_model_and_duplicate_snapshot_never_refills() -> None:
    """Paper-specific timing or replaying one event would drift from backtest behavior."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(
        fill_model=FillModel(costs=CostModel(latency=timedelta(seconds=90))),
        starting_cash=Decimal("1000"),
    )
    order = broker.submit(_intent(), _snapshot(bar0))
    too_early = _snapshot(bar0, _bar(1, open_price="100"))

    assert broker.on_snapshot(too_early, _instrument()) == ()
    eligible = _snapshot(*too_early.bars, _bar(2, open_price="100"))
    [fill] = broker.on_snapshot(eligible, _instrument())
    events_before_duplicate = len(broker.audit_events)

    assert fill.filled_at == AT + timedelta(minutes=2)
    assert broker.on_snapshot(eligible, _instrument()) == ()
    assert len(broker.fills) == 1
    assert broker.get_order(order.order_id).filled_quantity == Decimal("1")
    assert len(broker.audit_events) == events_before_duplicate + 1
    assert broker.audit_events[-1].kind == "paper.snapshot.duplicate"


def test_partial_fills_accumulate_exactly_without_overfill() -> None:
    """Replacing remaining-quantity accounting could overfill a multi-event order."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        max_volume_participation=Decimal("1"),
    )
    order = broker.submit(_intent(quantity=Decimal("1")), _snapshot(bar0))
    partial_snapshot = _snapshot(bar0, _bar(1, open_price="100", volume="0.4"))
    [partial] = broker.on_snapshot(partial_snapshot, _instrument())

    assert partial.quantity == Decimal("0.4")
    assert broker.get_order(order.order_id).status is OrderStatus.PARTIALLY_FILLED
    final_snapshot = _snapshot(
        *partial_snapshot.bars,
        _bar(2, open_price="102", volume="2"),
    )
    [final] = broker.on_snapshot(final_snapshot, _instrument())
    finished = broker.get_order(order.order_id)

    assert final.quantity == Decimal("0.6")
    assert finished.status is OrderStatus.FILLED
    assert finished.filled_quantity == Decimal("1.0")
    assert finished.average_fill_price == Decimal("101.2")
    assert sum((fill.quantity for fill in broker.fills), Decimal("0")) == Decimal("1.0")


def test_stop_order_triggers_and_fills_only_on_a_subsequent_event() -> None:
    """Evaluating the submission bar high would look ahead into the stop trigger."""
    bar0 = _bar(0, open_price="100", high="110", low="99")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(
        _intent(
            order_type=OrderType.STOP,
            trigger_price=Decimal("105"),
            stop_loss=Decimal("90"),
            take_profit=Decimal("120"),
        ),
        _snapshot(bar0),
    )

    assert broker.on_snapshot(_snapshot(bar0), _instrument()) == ()
    no_trigger = _snapshot(bar0, _bar(1, open_price="103", high="104", low="102"))
    assert broker.on_snapshot(no_trigger, _instrument()) == ()
    triggered = _snapshot(*no_trigger.bars, _bar(2, open_price="106", high="108", low="105"))
    [fill] = broker.on_snapshot(triggered, _instrument())

    assert fill.price == Decimal("106")
    assert broker.get_order(order.order_id).status is OrderStatus.FILLED


def test_stop_limit_uses_separate_later_trigger_and_cross_events() -> None:
    """Filling on the trigger bar would assume an unknowable intrabar event order."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(
        _intent(
            order_type=OrderType.STOP_LIMIT,
            trigger_price=Decimal("105"),
            stop_loss=Decimal("90"),
            limit_price=Decimal("106"),
            take_profit=Decimal("120"),
        ),
        _snapshot(bar0),
    )
    trigger = _snapshot(bar0, _bar(1, open_price="107", high="108", low="104"))

    assert broker.on_snapshot(trigger, _instrument()) == ()
    assert broker.get_order(order.order_id).status is OrderStatus.ACKNOWLEDGED
    later_cross = _snapshot(*trigger.bars, _bar(2, open_price="105", high="107", low="104"))
    [fill] = broker.on_snapshot(later_cross, _instrument())

    assert fill.price == Decimal("105")
    assert fill.price <= Decimal("106")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda snap: snap.model_copy(update={"source_at": snap.observed_at + timedelta(seconds=1)}),
        lambda snap: snap.model_copy(
            update={"observed_at": snap.source_at + timedelta(seconds=61)}
        ),
        lambda snap: snap.model_copy(update={"instrument_id": "MSFT@alpaca"}),
        lambda snap: snap.model_copy(
            update={"bars": (snap.bars[0].model_copy(update={"open": Decimal("NaN")}),)}
        ),
    ],
)
def test_invalid_future_stale_mismatched_or_nonfinite_snapshot_is_atomic(mutate: object) -> None:
    """Boundary rejection must happen before an order or audit record is created."""
    broker = PaperBroker(starting_cash=Decimal("1000"))
    original = _snapshot(_bar(0, open_price="100"))
    invalid = mutate(original)  # type: ignore[operator]

    with pytest.raises(ValueError):
        broker.submit(_intent(), invalid)

    assert broker.order_count == 0
    assert broker.audit_events == ()


def test_revised_or_backward_snapshot_is_rejected_without_fill_or_state_change() -> None:
    """Accepting revised history would make current-session paper results non-reproducible."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(_intent(), _snapshot(bar0))
    revised = _snapshot(bar0.model_copy(update={"close": Decimal("101")}))

    with pytest.raises(ValueError, match="revision"):
        broker.on_snapshot(revised, _instrument())

    assert broker.fills == ()
    assert broker.get_order(order.order_id).status is OrderStatus.ACKNOWLEDGED


def test_multi_event_snapshot_gap_is_rejected_instead_of_backfilled() -> None:
    """Retroactively filling an earlier unseen bar would invent an unavailable paper price."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(_intent(), _snapshot(bar0))
    gap = _snapshot(
        bar0,
        _bar(1, open_price="90"),
        _bar(2, open_price="110"),
    )

    with pytest.raises(ValueError, match="one unseen"):
        broker.on_snapshot(gap, _instrument())

    assert broker.fills == ()
    assert broker.get_order(order.order_id).status is OrderStatus.ACKNOWLEDGED
    [fill] = broker.on_snapshot(
        _snapshot(bar0, _bar(1, open_price="100")),
        _instrument(),
    )
    assert fill.filled_at == AT + timedelta(minutes=1)


def test_cancel_reject_unknown_and_expire_paths_are_terminal_and_audited() -> None:
    """Missing resolution branches would leave unresolved paper orders reusable."""
    submitted = _snapshot(_bar(0, open_price="100"))
    broker = PaperBroker(starting_cash=Decimal("1000"))
    orders = [
        broker.submit(_intent(intent_id=f"intent-{index}"), submitted)
        for index in range(4)
    ]

    cancelled = broker.cancel(orders[0].order_id, at=AT + timedelta(seconds=1))
    rejected = broker.reject(
        orders[1].order_id,
        at=AT + timedelta(seconds=1),
        reason_code="BROKER_REJECTED",
    )
    unknown = broker.mark_unknown(orders[2].order_id, at=AT + timedelta(seconds=1))
    expired = broker.expire(orders[3].order_id, at=AT + timedelta(seconds=1))

    assert [cancelled.status, rejected.status, unknown.status, expired.status] == [
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.UNKNOWN,
        OrderStatus.EXPIRED,
    ]
    for terminal in (cancelled, rejected, expired):
        with pytest.raises(InvalidOrderTransition):
            broker.cancel(terminal.order_id, at=AT + timedelta(seconds=2))
    reconciled = broker.reconcile_unknown(
        unknown.order_id,
        OrderStatus.CANCELLED,
        at=AT + timedelta(seconds=2),
    )
    assert reconciled.status is OrderStatus.CANCELLED
    transitions = [event for event in broker.audit_events if event.kind == "paper.order.transition"]
    assert {event.new_status for event in transitions[-4:]} == {
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.UNKNOWN,
        OrderStatus.EXPIRED,
    }


def test_order_expires_before_a_later_event_can_fill_it() -> None:
    """Checking expiry after pricing could execute a stale order intent."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(
        _intent(expires_at=AT + timedelta(seconds=30)),
        _snapshot(bar0),
    )
    later = _snapshot(bar0, _bar(1, open_price="100"))

    assert broker.on_snapshot(later, _instrument()) == ()
    assert broker.get_order(order.order_id).status is OrderStatus.EXPIRED


@pytest.mark.parametrize(
    "bad_intent",
    [
        _intent().model_copy(update={"quantity": Decimal("NaN")}),
        _intent().model_copy(update={"quantity": Decimal("0")}),
        _intent().model_copy(update={"order_type": OrderType.LIMIT}),
        _intent().model_copy(update={"expires_at": AT - timedelta(seconds=1)}),
    ],
)
def test_invalid_decimal_price_quantity_or_expiry_is_rejected_atomically(
    bad_intent: OrderIntent,
) -> None:
    """Pydantic bypasses and incomplete order types must fail at the broker boundary."""
    broker = PaperBroker(starting_cash=Decimal("1000"))

    with pytest.raises(ValueError):
        broker.submit(bad_intent, _snapshot(_bar(0, open_price="100")))

    assert broker.order_count == 0
    assert broker.audit_events == ()


def test_instrument_identity_and_quantity_precision_are_enforced_before_fill() -> None:
    """Filling against a different venue or untradeable increment corrupts reconciliation."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(_intent(quantity=Decimal("0.15")), _snapshot(bar0))
    later = _snapshot(bar0, _bar(1, open_price="100"))

    with pytest.raises(ValueError, match="instrument"):
        broker.on_snapshot(later, _instrument(instrument_id="MSFT@alpaca"))
    with pytest.raises(ValueError, match="step"):
        broker.on_snapshot(later, _instrument(quantity_step=Decimal("0.1")))

    assert broker.fills == ()
    assert broker.get_order(order.order_id).filled_quantity == Decimal("0")


def test_instrument_quote_currency_must_match_paper_cash_ledger() -> None:
    """Mixing INR consideration into a USD ledger would fabricate cash and exposure."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"), currency="USD")
    order = broker.submit(_intent(), _snapshot(bar0))

    with pytest.raises(ValueError, match="currency"):
        broker.on_snapshot(
            _snapshot(bar0, _bar(1, open_price="100")),
            _instrument(quote_currency="INR"),
        )

    assert broker.cash == Decimal("1000")
    assert broker.fills == ()
    assert broker.get_order(order.order_id).status is OrderStatus.ACKNOWLEDGED


def test_cash_constraint_prevents_negative_balance_and_sell_cannot_create_short() -> None:
    """Paper accounting must not manufacture leverage or a negative long-only position."""
    bar0 = _bar(0, open_price="100")
    buy_broker = PaperBroker(
        fill_model=FillModel(costs=CostModel(fee_bps=Decimal("100"))),
        starting_cash=Decimal("100"),
    )
    buy_order = buy_broker.submit(_intent(quantity=Decimal("1")), _snapshot(bar0))
    assert buy_broker.on_snapshot(_snapshot(bar0, _bar(1, open_price="100")), _instrument()) == ()
    assert buy_broker.cash == Decimal("100")
    assert buy_broker.get_order(buy_order.order_id).filled_quantity == Decimal("0")

    sell_broker = PaperBroker(starting_cash=Decimal("1000"))
    sell_order = sell_broker.submit(
        _intent(side=Side.SELL, stop_loss=Decimal("110"), take_profit=Decimal("80")),
        _snapshot(bar0),
    )
    assert sell_broker.on_snapshot(
        _snapshot(bar0, _bar(1, open_price="100")),
        _instrument(),
    ) == ()
    assert sell_broker.positions() == ()
    assert sell_broker.get_order(sell_order.order_id).filled_quantity == Decimal("0")


def test_venue_minimum_notional_blocks_an_under_minimum_order() -> None:
    """Ignoring the instrument minimum would create a fill the venue cannot represent."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(_intent(quantity=Decimal("0.005")), _snapshot(bar0))

    fills = broker.on_snapshot(
        _snapshot(bar0, _bar(1, open_price="100")),
        _instrument(quantity_step=Decimal("0.001"), minimum_notional=Decimal("1")),
    )

    assert fills == ()
    assert broker.get_order(order.order_id).status is OrderStatus.ACKNOWLEDGED


def test_failed_fill_audit_is_atomic_for_order_cash_fills_and_internal_events() -> None:
    """Audit failure must not leave an unaudited fill or partial state mutation."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=_FailingAudit("paper.order.fill"),
    )
    order = broker.submit(_intent(), _snapshot(bar0))
    before_events = broker.audit_events

    with pytest.raises(RuntimeError, match="audit unavailable"):
        broker.on_snapshot(_snapshot(bar0, _bar(1, open_price="100")), _instrument())

    assert broker.get_order(order.order_id).status is OrderStatus.ACKNOWLEDGED
    assert broker.get_order(order.order_id).filled_quantity == Decimal("0")
    assert broker.cash == Decimal("1000")
    assert broker.fills == ()
    assert broker.audit_events == before_events


def test_failed_submit_audit_batch_creates_no_order_or_internal_event() -> None:
    """A lifecycle audit outage must not leave a partially acknowledged local order."""
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=_FailingAudit("paper.order.submitted"),
    )

    with pytest.raises(RuntimeError, match="audit unavailable"):
        broker.submit(_intent(), _snapshot(_bar(0, open_price="100")))

    assert broker.order_count == 0
    assert broker.audit_events == ()


def test_invalid_terminal_rejection_adds_no_reason_or_transition_audit() -> None:
    """A failed resolution must not persist a misleading rejection reason."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(_intent(), _snapshot(bar0))
    broker.cancel(order.order_id, at=AT + timedelta(seconds=1))
    before = broker.audit_events

    with pytest.raises(InvalidOrderTransition):
        broker.reject(
            order.order_id,
            at=AT + timedelta(seconds=2),
            reason_code="BROKER_REJECTED",
        )

    assert broker.audit_events == before


@pytest.mark.parametrize(
    "bad_intent",
    [
        _intent().model_copy(update={"side": "buy"}),
        _intent().model_copy(update={"order_type": "market"}),
    ],
)
def test_bypassed_non_enum_intent_values_fail_with_fixed_boundary_error(
    bad_intent: OrderIntent,
) -> None:
    """String lookalikes must not reach audit rendering or execution branches."""
    broker = PaperBroker(starting_cash=Decimal("1000"))

    with pytest.raises(ValueError, match="enum"):
        broker.submit(bad_intent, _snapshot(_bar(0, open_price="100")))

    assert broker.order_count == 0
    assert broker.audit_events == ()


def test_fill_and_transition_audit_payloads_are_safe_and_complete() -> None:
    """A fill without immutable identity and quantities cannot be safely reconciled."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(_intent(), _snapshot(bar0))
    broker.on_snapshot(_snapshot(bar0, _bar(1, open_price="100")), _instrument())

    fill_events = [event for event in broker.audit_events if event.kind == "paper.order.fill"]
    [fill_event] = fill_events
    assert fill_event.client_intent_id == "intent-1"
    assert fill_event.broker_order_id == order.order_id
    assert fill_event.prior_status is OrderStatus.ACKNOWLEDGED
    assert fill_event.new_status is OrderStatus.FILLED
    assert fill_event.occurred_at == AT + timedelta(minutes=1)
    assert fill_event.payload == {
        "fill_id": broker.fills[0].fill_id,
        "quantity": "1",
            "price": "100",
            "fee": "0",
            "filled_at": "2026-08-09T10:01:00+00:00",
            "observed_at": "2026-08-09T10:01:00+00:00",
            "instrument": {
                "symbol": "AAPL",
                "venue": "alpaca",
                "asset_class": "equity",
                "quote_currency": "USD",
                "timezone": "UTC",
                "price_tick": "0.01",
                "quantity_step": "0.1",
                "minimum_notional": "1",
                "session_calendar": None,
            },
            "cumulative_filled_quantity": "1",
        "remaining_quantity": "0",
        "requested_notional": None,
        "cumulative_filled_notional": "100",
        "remaining_notional": None,
        "cumulative_fees": "0",
    }
    assert "secret" not in repr(broker.audit_events).lower()


def test_notional_order_never_exceeds_requested_notional() -> None:
    """Rounding a notional order upward would silently exceed its approved capital."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(
        _intent(quantity=None, notional=Decimal("10.05")),
        _snapshot(bar0),
    )
    [fill] = broker.on_snapshot(
        _snapshot(bar0, _bar(1, open_price="100")),
        _instrument(quantity_step=Decimal("0.01")),
    )

    assert fill.quantity == Decimal("0.10")
    assert fill.quantity * fill.price <= Decimal("10.05")
    assert broker.get_order(order.order_id).status is OrderStatus.PARTIALLY_FILLED


def test_stop_partial_fill_latches_trigger_and_remainder_becomes_market() -> None:
    """A triggered stop must not require the trigger to cross again after a partial fill."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(
        _intent(
            order_type=OrderType.STOP,
            trigger_price=Decimal("105"),
            stop_loss=Decimal("90"),
            take_profit=Decimal("120"),
        ),
        _snapshot(bar0),
    )
    trigger = _snapshot(bar0, _bar(1, open_price="106", high="107", low="105", volume="0.4"))
    [first] = broker.on_snapshot(trigger, _instrument())
    later = _snapshot(
        *trigger.bars,
        _bar(2, open_price="104", high="104", low="103", volume="1"),
    )
    [second] = broker.on_snapshot(later, _instrument())

    assert (first.quantity, second.quantity) == (Decimal("0.4"), Decimal("0.6"))
    assert broker.get_order(order.order_id).status is OrderStatus.FILLED
    trigger_events = [
        event for event in broker.audit_events if event.kind == "paper.order.stop_triggered"
    ]
    assert len(trigger_events) == 1
    assert trigger_events[0].payload["trigger_price"] == "105"


def test_event_liquidity_budget_is_shared_across_all_sorted_orders() -> None:
    """Reusing full bar volume per order would create impossible aggregate liquidity."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    orders = [
        broker.submit(_intent(intent_id=f"liquidity-{index}"), _snapshot(bar0))
        for index in range(2)
    ]

    fills = broker.on_snapshot(
        _snapshot(bar0, _bar(1, open_price="100", volume="1")),
        _instrument(),
    )

    assert sum((fill.quantity for fill in fills), Decimal("0")) == Decimal("1")
    assert sum(
        (broker.get_order(order.order_id).filled_quantity for order in orders),
        Decimal("0"),
    ) == Decimal("1")


def test_notional_residual_is_retained_for_a_later_lower_price() -> None:
    """Current-price lot dust may become executable when a later price is lower."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(
        _intent(
            quantity=None,
            notional=Decimal("19"),
            stop_loss=Decimal("70"),
        ),
        _snapshot(bar0),
    )
    first_snapshot = _snapshot(bar0, _bar(1, open_price="100", volume="0.1"))
    [first] = broker.on_snapshot(first_snapshot, _instrument())
    assert first.quantity == Decimal("0.1")
    assert broker.get_order(order.order_id).status is OrderStatus.PARTIALLY_FILLED

    lower = _snapshot(*first_snapshot.bars, _bar(2, open_price="80", volume="0.1"))
    [second] = broker.on_snapshot(lower, _instrument())

    assert second.quantity == Decimal("0.1")
    assert broker.get_order(order.order_id).status is OrderStatus.PARTIALLY_FILLED
    fill_events = [event for event in broker.audit_events if event.kind == "paper.order.fill"]
    assert fill_events[-1].payload["requested_notional"] == "19"
    assert fill_events[-1].payload["cumulative_filled_notional"] == "18"
    assert fill_events[-1].payload["remaining_notional"] == "1"
    assert fill_events[-1].payload["cumulative_fees"] == "0"


def test_unknown_order_can_only_be_resolved_by_reconciliation() -> None:
    """UNKNOWN must reconcile to broker truth without any resubmission transition."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(_intent(), _snapshot(bar0))
    unknown = broker.mark_unknown(order.order_id, at=AT + timedelta(seconds=1))

    with pytest.raises(InvalidOrderTransition):
        broker.reconcile_unknown(
            unknown.order_id,
            OrderStatus.SUBMITTING,
            at=AT + timedelta(seconds=2),
        )
    resolved = broker.reconcile_unknown(
        unknown.order_id,
        OrderStatus.ACKNOWLEDGED,
        at=AT + timedelta(seconds=2),
    )

    assert resolved.status is OrderStatus.ACKNOWLEDGED


def test_protocol_exposes_client_lookup_and_deterministic_order_lists() -> None:
    """Task 14 reconciliation needs stable client lookup and ordered open-order discovery."""
    bar0 = _bar(0, open_price="100")
    broker: BrokerAdapter = PaperBroker(starting_cash=Decimal("1000"))
    first = broker.submit(_intent(intent_id="list-2"), _snapshot(bar0))
    second = broker.submit(_intent(intent_id="list-1"), _snapshot(bar0))

    assert broker.get_order_by_client_id("list-2") == first
    assert broker.list_orders() == tuple(sorted((first, second), key=lambda item: item.order_id))
    assert broker.open_orders() == broker.list_orders()


def test_observation_time_drives_expiry_before_source_bar_fill() -> None:
    """A source bar from before expiry cannot execute when observed after expiry."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(
        _intent(expires_at=AT + timedelta(minutes=1, seconds=30)),
        _snapshot(bar0),
    )
    source_bar = _bar(1, open_price="100")
    observed_after_expiry = _snapshot(
        bar0,
        source_bar,
        observed_at=AT + timedelta(minutes=2),
        max_age_seconds=60,
    )

    assert broker.on_snapshot(observed_after_expiry, _instrument()) == ()
    expired = broker.get_order(order.order_id)
    assert expired.status is OrderStatus.EXPIRED
    assert expired.updated_at == AT + timedelta(minutes=2)
    assert broker.audit_events[-1].occurred_at == AT + timedelta(minutes=2)


def test_fill_and_audit_time_are_observation_time_not_source_time() -> None:
    """Paper evidence must record when data became available, not its provider source stamp."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    broker.submit(_intent(), _snapshot(bar0))
    source_bar = _bar(1, open_price="100")
    observed = AT + timedelta(minutes=1, seconds=20)
    [fill] = broker.on_snapshot(
        _snapshot(bar0, source_bar, observed_at=observed, max_age_seconds=30),
        _instrument(),
    )

    assert fill.filled_at == observed
    assert broker.audit_events[-1].occurred_at == observed


def test_same_time_multi_instrument_batch_is_permutation_invariant() -> None:
    """Shared cash allocation must not depend on caller input order."""
    aapl = _instrument(instrument_id="AAPL@alpaca")
    msft = _instrument(instrument_id="MSFT@alpaca")
    aapl0 = _snapshot(_bar(0, open_price="100"), instrument_id="AAPL@alpaca")
    msft0 = _snapshot(_bar(0, open_price="100"), instrument_id="MSFT@alpaca")
    event_aapl = _snapshot(
        *aapl0.bars,
        _bar(1, open_price="100"),
        instrument_id="AAPL@alpaca",
    )
    event_msft = _snapshot(
        *msft0.bars,
        _bar(1, open_price="100"),
        instrument_id="MSFT@alpaca",
    )

    outcomes = []
    for batch in (
        ((event_aapl, aapl), (event_msft, msft)),
        ((event_msft, msft), (event_aapl, aapl)),
    ):
        broker = PaperBroker(starting_cash=Decimal("100"))
        broker.submit(_intent(intent_id="aapl", instrument_id="AAPL@alpaca"), aapl0)
        broker.submit(_intent(intent_id="msft", instrument_id="MSFT@alpaca"), msft0)
        fills = broker.on_snapshots(batch)
        outcomes.append(
            (
                tuple((fill.instrument_id, fill.quantity, fill.price) for fill in fills),
                tuple((order.client_order_id, order.status) for order in broker.list_orders()),
                broker.cash,
            )
        )

    assert outcomes[0] == outcomes[1]
    assert len(outcomes[0][0]) == 1


class _RecordManyFailure:
    def __init__(self, *, fail_on: int) -> None:
        self.fail_on = fail_on
        self.calls = 0

    def record_many(self, events: tuple[AuditEvent, ...]) -> None:
        del events
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("record_many failed")


def test_later_invalid_order_rolls_back_entire_snapshot_then_retry_is_exactly_once() -> None:
    """One later validation error must not retain an earlier order fill."""
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    valid = broker.submit(_intent(intent_id="valid-a", quantity=Decimal("0.1")), _snapshot(bar0))
    invalid = broker.submit(
        _intent(intent_id="bad-z", quantity=Decimal("0.15")),
        _snapshot(bar0),
    )
    assert valid.order_id < invalid.order_id
    event = _snapshot(bar0, _bar(1, open_price="100"))
    before = (broker.list_orders(), broker.fills, broker.cash, broker.audit_events)

    with pytest.raises(ValueError, match="step"):
        broker.on_snapshot(event, _instrument(quantity_step=Decimal("0.1")))

    assert (broker.list_orders(), broker.fills, broker.cash, broker.audit_events) == before
    fills = broker.on_snapshot(event, _instrument(quantity_step=Decimal("0.05")))
    assert len(fills) == 2
    assert len(broker.fills) == 2


def test_record_many_failure_rolls_back_snapshot_and_safe_retry_is_exactly_once() -> None:
    """A test recorder failure precedes every current-session in-memory commit."""
    bar0 = _bar(0, open_price="100")
    recorder = _RecordManyFailure(fail_on=2)
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=recorder,
        session_id="audit-failure-session",
    )
    order = broker.submit(_intent(), _snapshot(bar0))
    event = _snapshot(bar0, _bar(1, open_price="100"))
    before = (broker.list_orders(), broker.fills, broker.cash, broker.audit_events)

    with pytest.raises(RuntimeError, match="record_many"):
        broker.on_snapshot(event, _instrument())

    assert (broker.list_orders(), broker.fills, broker.cash, broker.audit_events) == before
    [fill] = broker.on_snapshot(event, _instrument())
    assert fill.order_id == order.order_id
    assert len(broker.fills) == 1


def test_durable_mode_persists_rows_and_rehydrates_idempotently(tmp_path: Path) -> None:
    """Crash durability requires first-class rows plus a replayable committed state."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'paper-events.db'}")
    )
    clock = FrozenClock(AT + timedelta(days=1))
    audit = AuditLog(store, clock)
    bar0 = _bar(0, open_price="100")
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=audit,
        durable=True,
        session_id="durable-session",
    )
    value = _intent()
    order = broker.submit(value, _snapshot(bar0))
    event = _snapshot(bar0, _bar(1, open_price="100"))
    broker.on_snapshot(event, _instrument())
    rows = tuple(store.stream("durable-session"))

    assert [(row.sequence, row.kind, row.occurred_at) for row in rows] == [
        (1, "paper.market.submission", AT),
        (2, "paper.order.submitted", AT),
        (3, "paper.order.transition", AT),
        (4, "paper.order.transition", AT),
        (5, "paper.order.transition", AT),
        (6, "paper.order.transition", AT),
        (7, "paper.state.committed", AT),
        (8, "paper.order.transition", AT + timedelta(minutes=1)),
        (9, "paper.order.fill", AT + timedelta(minutes=1)),
        (10, "paper.market.observed", AT + timedelta(minutes=1)),
        (11, "paper.state.committed", AT + timedelta(minutes=1)),
    ]
    assert all(row.occurred_at != clock.now() for row in rows)
    assert len({row.event_id for row in rows}) == len(rows)
    restored = PaperBroker.rehydrate(
        rows,
        audit_log=audit,
        expected_head=broker.session_head,
        starting_cash=Decimal("1000"),
        session_id="durable-session",
    )

    assert restored.get_order(order.order_id) == broker.get_order(order.order_id)
    assert restored.fills == broker.fills
    assert restored.cash == broker.cash
    assert restored.submit(value, event) == restored.get_order(order.order_id)
    assert restored.on_snapshot(event, _instrument()) == ()


def test_two_durable_brokers_never_collide_event_or_fill_ids(tmp_path: Path) -> None:
    """Per-instance counters alone collide when sessions share one durable store."""
    store = EventStore(create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'shared.db'}"))
    audit = AuditLog(store, FrozenClock(AT))
    bar0 = _bar(0, open_price="100")
    fill_ids: list[str] = []
    event_ids: list[str] = []
    for index in range(2):
        session_id = f"session-{index}"
        broker = PaperBroker(
            starting_cash=Decimal("1000"),
            audit_log=audit,
            durable=True,
            session_id=session_id,
        )
        broker.submit(_intent(), _snapshot(bar0))
        [fill] = broker.on_snapshot(
            _snapshot(bar0, _bar(1, open_price="100")),
            _instrument(),
        )
        fill_ids.append(fill.fill_id)
        event_ids.extend(row.event_id for row in store.stream(session_id))

    assert len(set(fill_ids)) == 2
    assert len(set(event_ids)) == len(event_ids)


def test_rehydrate_preserves_latest_committed_resolution(
    tmp_path: Path,
) -> None:
    """The final complete commit must remain authoritative after replay."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'late-update.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=audit,
        durable=True,
        session_id="late-update-session",
    )
    aapl0 = _snapshot(_bar(0, open_price="100"), instrument_id="AAPL@alpaca")
    msft0 = _snapshot(_bar(0, open_price="100"), instrument_id="MSFT@alpaca")
    aapl_order = broker.submit(
        _intent(
            intent_id="aapl",
            instrument_id="AAPL@alpaca",
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50"),
            stop_loss=Decimal("40"),
        ),
        aapl0,
    )
    broker.submit(_intent(intent_id="msft", instrument_id="MSFT@alpaca"), msft0)
    broker.on_snapshots(
        (
            (
                _snapshot(
                    *aapl0.bars,
                    _bar(2, open_price="100"),
                    instrument_id="AAPL@alpaca",
                ),
                _instrument(instrument_id="AAPL@alpaca"),
            ),
            (
                _snapshot(
                    *msft0.bars,
                    _bar(2, open_price="100"),
                    instrument_id="MSFT@alpaca",
                ),
                _instrument(instrument_id="MSFT@alpaca"),
            ),
        )
    )
    cancelled = broker.cancel(aapl_order.order_id, at=AT + timedelta(minutes=3))

    rows = tuple(store.stream("late-update-session"))
    restored = PaperBroker.rehydrate(
        rows,
        audit_log=audit,
        expected_head=broker.session_head,
        starting_cash=Decimal("1000"),
        session_id="late-update-session",
    )

    assert restored.get_order(aapl_order.order_id) == cancelled


def test_runtime_durability_semantics_are_explicit_and_validated() -> None:
    """A default in-memory broker must not be mislabeled crash-durable."""
    assert PaperBroker().durability_mode == "current_session"
    with pytest.raises(ValueError, match="durable"):
        PaperBroker(durable=True, session_id="runtime-session")
    with pytest.raises(ValueError, match="AuditLog"):
        PaperBroker(durable=True, audit_log=_RecordManyFailure(fail_on=99))
    with pytest.raises(ValueError, match="AuditLog"):
        PaperBroker(
            durable=True,
            audit_log=object(),  # type: ignore[arg-type]
            session_id="runtime-session",
        )


def _mutable_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _mutable_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_payload(item) for item in value]
    return value


def _test_canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _test_positions_digest(
    positions: list[dict[str, str]],
) -> str:
    return _test_canonical_digest(positions)


def _test_bar_digest(bars: list[object]) -> str:
    digest = hashlib.sha256(b"paper-market-bars-v1").hexdigest()
    for bar in bars:
        encoded = json.dumps(
            bar,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(
            bytes.fromhex(digest) + b"\0" + encoded
        ).hexdigest()
    return digest


def _rehash_forged_final_market_group(
    rows: tuple[EventRecord, ...],
    *,
    activities: list[EventRecord],
    ledger: dict[str, object],
    fill_sequence: int,
) -> tuple[tuple[EventRecord, ...], SessionHead]:
    """Model an attacker who can rewrite every public unkeyed envelope and hash."""
    first_commit_index = next(
        index for index, row in enumerate(rows) if row.kind == "paper.state.committed"
    )
    prefix = list(rows[: first_commit_index + 1])
    checkpoint_source = rows[-1]
    event_count = len(prefix) + len(activities) + 1
    activity_facts = [
        {
            "kind": row.kind,
            "occurred_at": row.occurred_at.astimezone(UTC).isoformat(),
            "payload": _mutable_payload(row.payload),
        }
        for row in activities
    ]
    observed = next(row for row in activities if row.kind == "paper.market.observed")
    observed_payload = _mutable_payload(observed.payload)
    assert isinstance(observed_payload, dict)
    observed_at = observed_payload["observed_at"]
    source_at = observed_payload["source_at"]
    instrument_id = observed_payload["instrument_id"]
    assert isinstance(observed_at, str)
    assert isinstance(source_at, str)
    assert isinstance(instrument_id, str)
    prior_checkpoint = _mutable_payload(prefix[-1].payload)
    assert isinstance(prior_checkpoint, dict)
    prior_digest = prior_checkpoint["state_digest"]
    assert isinstance(prior_digest, str)
    state_digest = _test_canonical_digest(
        {
            "previous_state_digest": prior_digest,
            "activities": activity_facts,
            "event_sequence": event_count,
            "operation_sequence": 2,
            "fill_sequence": fill_sequence,
            "order_count": 1,
            "instrument_count": 1,
            "snapshot_count": 1,
            "market_cursors": [
                {
                    "instrument_id": instrument_id,
                    "total_count": observed_payload["bar_count"],
                    "digest": observed_payload["bars_digest"],
                    "latest_at": source_at,
                }
            ],
            "latest_at": observed_at,
            "last_event_key": [observed_at, source_at, instrument_id],
            "ledger": ledger,
        }
    )
    checkpoint = _mutable_payload(checkpoint_source.payload)
    assert isinstance(checkpoint, dict)
    checkpoint.update(
        {
            "event_sequence": event_count,
            "operation_sequence": 2,
            "operation_id": f"{checkpoint_source.aggregate_id}:operation:{2:020d}",
            "operation_kind": "MARKET",
            "first_activity_sequence": len(prefix) + 1,
            "activity_count": len(activities),
            "previous_state_digest": prior_digest,
            "state_digest": state_digest,
            "ledger": ledger,
        }
    )
    forged = prefix + activities + [replace(checkpoint_source, payload=checkpoint)]
    renumbered = tuple(
        replace(
            row,
            event_id=f"{row.aggregate_id}:event:{index:020d}",
            sequence=index,
        )
        for index, row in enumerate(forged, start=1)
    )
    return renumbered, SessionHead(
        checkpoint_source.aggregate_id,
        event_count,
        2,
        state_digest,
    )


def _tamper_latest_state(
    rows: tuple[EventRecord, ...],
    mutate: object,
) -> tuple[EventRecord, ...]:
    latest = max(
        (row for row in rows if row.kind == "paper.state.committed"),
        key=lambda row: row.sequence,
    )
    payload = _mutable_payload(latest.payload)
    assert isinstance(payload, dict)
    mutate(payload)  # type: ignore[operator]
    return tuple(
        replace(row, payload=payload) if row.event_id == latest.event_id else row
        for row in rows
    )


def _durable_stream(
    tmp_path: Path,
    *,
    session_id: str = "round-two-session",
    starting_cash: Decimal = Decimal("1000"),
    costs: CostModel | None = None,
    participation: Decimal = Decimal("0.5"),
) -> tuple[tuple[EventRecord, ...], EventStore, AuditLog, PaperBroker]:
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / f'{session_id}.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=starting_cash,
        fill_model=FillModel(costs=costs),
        max_volume_participation=participation,
        audit_log=audit,
        durable=True,
        session_id=session_id,
    )
    bar0 = _bar(0, open_price="100")
    broker.submit(_intent(), _snapshot(bar0))
    broker.on_snapshot(
        _snapshot(bar0, _bar(1, open_price="100", volume="10")),
        _instrument(),
    )
    return tuple(store.stream(session_id)), store, audit, broker


def test_rehydrate_derives_bound_configuration_and_exact_historical_peak(
    tmp_path: Path,
) -> None:
    """Caller defaults and replaying only the latest mark must not alter durable accounting."""
    _, store, audit, broker = _durable_stream(
        tmp_path,
        participation=Decimal("0.25"),
    )
    broker.on_snapshot(
        _snapshot(
            _bar(0, open_price="100"),
            _bar(1, open_price="100", volume="10"),
            _bar(2, open_price="150"),
        ),
        _instrument(),
    )
    broker.on_snapshot(
        _snapshot(
            _bar(0, open_price="100"),
            _bar(1, open_price="100", volume="10"),
            _bar(2, open_price="150"),
            _bar(3, open_price="80"),
        ),
        _instrument(),
    )
    rows = tuple(store.stream("round-two-session"))

    restored = PaperBroker.rehydrate(
        rows,
        audit_log=audit,
        expected_head=broker.session_head,
    )
    portfolio = restored.portfolio_snapshot()

    assert restored.session_id == "round-two-session"
    assert portfolio.cash == Decimal("900")
    assert portfolio.equity == Decimal("980")
    assert portfolio.peak_equity == Decimal("1050")
    assert restored.cash == broker.cash


@pytest.mark.parametrize(
    "kwargs",
    [
        {"starting_cash": Decimal("999")},
        {"currency": "INR"},
        {"fill_model": FillModel(costs=CostModel(fee_bps=Decimal("1")))},
        {"max_volume_participation": Decimal("1")},
        {"session_id": "wrong-session"},
    ],
)
def test_rehydrate_rejects_every_mismatched_runtime_expectation(
    tmp_path: Path,
    kwargs: dict[str, Any],
) -> None:
    """Recovery must derive durable configuration rather than silently replacing it."""
    rows, _, audit, broker = _durable_stream(tmp_path, costs=CostModel())

    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(
            rows,
            audit_log=audit,
            expected_head=broker.session_head,
            **kwargs,
        )


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda rows: rows[-1:],
        lambda rows: rows[:2] + rows[3:],
        lambda rows: rows[:2] + (rows[1],) + rows[2:],
        lambda rows: tuple(reversed(rows)),
        lambda rows: (
            replace(rows[0], event_id="round-two-session:event:00000000000000000002"),
            *rows[1:],
        ),
        lambda rows: rows[:-1],
    ],
)
def test_rehydrate_rejects_incomplete_duplicate_or_reordered_streams(
    tmp_path: Path,
    corrupt: object,
) -> None:
    """A latest-only or gapped stream cannot prove a complete committed session."""
    rows, _, audit, broker = _durable_stream(tmp_path)

    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(
            corrupt(rows),  # type: ignore[operator]
            audit_log=audit,
            expected_head=broker.session_head,
        )


def test_rehydrate_rejects_renumbered_state_without_lifecycle_history(
    tmp_path: Path,
) -> None:
    """A self-consistent latest snapshot is not a complete durable event stream."""
    rows, _, audit, broker = _durable_stream(tmp_path)
    latest = rows[-1]
    payload = _mutable_payload(latest.payload)
    assert isinstance(payload, dict)
    payload["event_sequence"] = 1
    latest_only = (
        replace(
            latest,
            event_id="round-two-session:event:00000000000000000001",
            payload=payload,
        ),
    )

    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(
            latest_only,
            audit_log=audit,
            expected_head=broker.session_head,
        )


@pytest.mark.parametrize("mutation", ["identity", "extra_key"])
def test_rehydrate_rejects_tampered_lifecycle_payload(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Lifecycle evidence must be schema-strict and linked to committed order identity."""
    rows, _, audit, broker = _durable_stream(tmp_path)
    fill_index = next(
        index for index, row in enumerate(rows) if row.kind == "paper.order.fill"
    )
    payload = _mutable_payload(rows[fill_index].payload)
    assert isinstance(payload, dict)
    if mutation == "identity":
        payload["client_intent_id"] = "unrelated-intent"
    else:
        payload["unexpected"] = "safe-but-untrusted"
    corrupted = rows[:fill_index] + (
        replace(rows[fill_index], payload=payload),
    ) + rows[fill_index + 1 :]

    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(
            corrupted,
            audit_log=audit,
            expected_head=broker.session_head,
        )


def test_rehydrate_rejects_unknown_or_reordered_lifecycle_activities(
    tmp_path: Path,
) -> None:
    """Only the canonical activity vocabulary and reducer order can advance state."""
    rows, _, audit, broker = _durable_stream(tmp_path)
    fill_index = next(
        index for index, row in enumerate(rows) if row.kind == "paper.order.fill"
    )
    transition_index = fill_index - 1
    assert rows[transition_index].kind == "paper.order.transition"
    unknown = rows[:fill_index] + (
        replace(rows[fill_index], kind="paper.order.unknown_activity"),
    ) + rows[fill_index + 1 :]
    reordered = list(rows)
    transition = rows[transition_index]
    fill = rows[fill_index]
    reordered[transition_index] = replace(
        transition,
        kind=fill.kind,
        payload=fill.payload,
        occurred_at=fill.occurred_at,
    )
    reordered[fill_index] = replace(
        fill,
        kind=transition.kind,
        payload=transition.payload,
        occurred_at=transition.occurred_at,
    )

    for corrupted in (unknown, tuple(reordered)):
        with pytest.raises(ValueError, match="invalid durable paper stream"):
            PaperBroker.rehydrate(
                corrupted,
                audit_log=audit,
                expected_head=broker.session_head,
            )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state: state.update({"event_sequence": "9"}),
        lambda state: state.update({"operation_sequence": True}),
        lambda state: state.update({"activity_count": "3"}),
        lambda state: state.update({"previous_state_digest": "0" * 64}),
        lambda state: state.update({"state_digest": "0" * 64}),
        lambda state: state["ledger"].update({"cash": "NaN"}),
        lambda state: state["ledger"]["market_prices"].append(
            {"instrument_id": "AAPL@alpaca", "price": "100"}
        ),
        lambda state: state.update({"unexpected": "field"}),
        lambda state: state.update({"configuration": {}}),
        lambda state: state.update({"capabilities": {}}),
    ],
)
def test_rehydrate_strictly_rejects_tampered_state_payloads(
    tmp_path: Path,
    mutate: object,
) -> None:
    """Malformed checkpoint primitives, digests, valuation, or extra keys fail closed."""
    rows, _, audit, broker = _durable_stream(tmp_path)
    corrupted = _tamper_latest_state(rows, mutate)

    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(
            corrupted,
            audit_log=audit,
            expected_head=broker.session_head,
        )


def test_global_watermark_rejects_every_earlier_operation_without_mutation() -> None:
    """Broker actions and market observations share one availability chronology."""
    aapl0 = _snapshot(_bar(0, open_price="100"), instrument_id="AAPL@alpaca")
    msft2 = _snapshot(_bar(2, open_price="100"), instrument_id="MSFT@alpaca")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    aapl = broker.submit(_intent(intent_id="aapl"), aapl0)
    broker.submit(
        _intent(
            intent_id="msft",
            instrument_id="MSFT@alpaca",
            created_at=AT + timedelta(minutes=2),
        ),
        msft2,
    )
    before = (broker.list_orders(), broker.fills, broker.cash, broker.audit_events)

    earlier_aapl = _snapshot(
        *aapl0.bars,
        _bar(1, open_price="100"),
        instrument_id="AAPL@alpaca",
    )
    with pytest.raises(ValueError, match="chronology"):
        broker.on_snapshot(earlier_aapl, _instrument())
    with pytest.raises(ValueError, match="chronology"):
        broker.cancel(aapl.order_id, at=AT + timedelta(minutes=1))
    with pytest.raises(ValueError, match="chronology"):
        broker.submit(
            _intent(
                intent_id="older",
                instrument_id="GOOG@alpaca",
                created_at=AT + timedelta(minutes=1),
            ),
            _snapshot(_bar(1, open_price="100"), instrument_id="GOOG@alpaca"),
        )

    assert (broker.list_orders(), broker.fills, broker.cash, broker.audit_events) == before


def test_multi_instrument_singular_events_fail_regardless_of_symbol_order() -> None:
    """Same-time scarce cash cannot depend on whether A or Z is delivered first."""
    a = _instrument(instrument_id="AAA@alpaca")
    z = _instrument(instrument_id="ZZZ@alpaca")
    a0 = _snapshot(_bar(0, open_price="100"), instrument_id="AAA@alpaca")
    z0 = _snapshot(_bar(0, open_price="100"), instrument_id="ZZZ@alpaca")
    a1 = _snapshot(*a0.bars, _bar(1, open_price="100"), instrument_id="AAA@alpaca")
    z1 = _snapshot(*z0.bars, _bar(1, open_price="100"), instrument_id="ZZZ@alpaca")

    for event, venue in ((a1, a), (z1, z)):
        broker = PaperBroker(starting_cash=Decimal("100"))
        broker.submit(_intent(intent_id="a", instrument_id="AAA@alpaca"), a0)
        broker.submit(_intent(intent_id="z", instrument_id="ZZZ@alpaca"), z0)
        before = (broker.list_orders(), broker.fills, broker.cash, broker.audit_events)

        with pytest.raises(ValueError, match="on_snapshots"):
            broker.on_snapshot(event, venue)
        with pytest.raises(ValueError, match="cohort"):
            broker.on_snapshots(((event, venue),))

        assert (broker.list_orders(), broker.fills, broker.cash, broker.audit_events) == before


def test_status_only_unknown_resolution_cannot_invent_a_fill() -> None:
    """A status change without fill evidence must never alter quantity or cash."""
    broker = PaperBroker(starting_cash=Decimal("1000"))
    order = broker.submit(_intent(), _snapshot(_bar(0, open_price="100")))
    unknown = broker.mark_unknown(order.order_id, at=AT + timedelta(seconds=1))
    before = (broker.get_order(order.order_id), broker.fills, broker.cash, broker.audit_events)

    with pytest.raises(InvalidOrderTransition):
        broker.reconcile_unknown(
            unknown.order_id,
            OrderStatus.FILLED,
            at=AT + timedelta(seconds=2),
        )

    assert (
        broker.get_order(order.order_id),
        broker.fills,
        broker.cash,
        broker.audit_events,
    ) == before


def test_authoritative_unknown_fill_reconciliation_updates_all_state_atomically() -> None:
    """Partial and final broker truth must reconcile order, fills, cash, position, and audit."""
    broker = PaperBroker(starting_cash=Decimal("1000"))
    bars = [_bar(0, open_price="100")]
    order = broker.submit(_intent(), _snapshot(*bars))
    bars.append(_bar(1, open_price="100", volume="0"))
    broker.on_snapshot(_snapshot(*bars), _instrument())
    unknown = broker.mark_unknown(
        order.order_id,
        at=AT + timedelta(minutes=1, seconds=1),
    )
    partial_fill = Fill(
        fill_id="authoritative-fill-1",
        order_id=order.order_id,
        instrument_id=order.instrument_id,
        side=Side.BUY,
        quantity=Decimal("0.4"),
        price=Decimal("100"),
        fee=Decimal("1"),
        filled_at=AT + timedelta(minutes=1, seconds=1, microseconds=1),
    )
    partial_truth = unknown.model_copy(
        update={
            "status": OrderStatus.PARTIALLY_FILLED,
            "filled_quantity": Decimal("0.4"),
            "average_fill_price": Decimal("100"),
            "updated_at": AT + timedelta(minutes=1, seconds=2),
        }
    )

    partial = broker.reconcile_unknown_fills(
        partial_truth,
        (partial_fill,),
        instrument=_instrument(),
    )
    assert partial == partial_truth
    assert broker.cash == Decimal("959")
    assert broker.positions()[0].quantity == Decimal("0.4")

    unknown_again = broker.mark_unknown(
        order.order_id,
        at=AT + timedelta(minutes=1, seconds=3),
    )
    final_fill = Fill(
        fill_id="authoritative-fill-2",
        order_id=order.order_id,
        instrument_id=order.instrument_id,
        side=Side.BUY,
        quantity=Decimal("0.6"),
        price=Decimal("102"),
        fee=Decimal("0"),
        filled_at=AT + timedelta(minutes=1, seconds=3, microseconds=1),
    )
    final_truth = unknown_again.model_copy(
        update={
            "status": OrderStatus.FILLED,
            "filled_quantity": Decimal("1"),
            "average_fill_price": Decimal("101.2"),
            "updated_at": AT + timedelta(minutes=1, seconds=4),
        }
    )

    final = broker.reconcile_unknown_fills(
        final_truth,
        (final_fill,),
        instrument=_instrument(),
    )

    assert final == final_truth
    assert broker.cash == Decimal("897.8")
    assert broker.positions()[0].quantity == Decimal("1.0")
    assert broker.positions()[0].average_price == Decimal("101.2")
    assert broker.fills == (partial_fill, final_fill)
    assert [event.kind for event in broker.audit_events].count("paper.order.fill") == 2


def test_authoritative_partial_fill_uses_shared_kernel_and_rehydrates(
    tmp_path: Path,
) -> None:
    """A qualifying order may receive a sub-minimum partial through durable truth."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'auth-kernel.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=audit,
        durable=True,
        session_id="auth-kernel",
    )
    instrument = _instrument(minimum_notional=Decimal("50"))
    initial = _snapshot(_bar(0, open_price="100"))
    order = broker.submit(_intent(quantity=Decimal("1")), initial)
    broker.on_snapshot(
        _snapshot(*initial.bars, _bar(1, open_price="100", volume="0")),
        instrument,
    )
    unknown = broker.mark_unknown(
        order.order_id,
        at=AT + timedelta(minutes=1, seconds=1),
    )
    partial_fill = Fill(
        fill_id="authoritative-subminimum-partial",
        order_id=order.order_id,
        instrument_id=order.instrument_id,
        side=Side.BUY,
        quantity=Decimal("0.1"),
        price=Decimal("100"),
        fee=Decimal("0"),
        filled_at=AT + timedelta(minutes=1, seconds=1, microseconds=1),
    )
    expected = unknown.model_copy(
        update={
            "status": OrderStatus.PARTIALLY_FILLED,
            "filled_quantity": Decimal("0.1"),
            "average_fill_price": Decimal("100"),
            "updated_at": AT + timedelta(minutes=1, seconds=2),
        }
    )

    broker.reconcile_unknown_fills(expected, (partial_fill,), instrument=instrument)
    restored = PaperBroker.rehydrate(
        store.stream("auth-kernel"),
        audit_log=audit,
        expected_head=broker.session_head,
    )

    assert restored.get_order(order.order_id) == expected
    assert restored.fills == (partial_fill,)
    assert restored.cash == Decimal("990")
    assert restored.positions()[0].quantity == Decimal("0.1")


def test_malformed_authoritative_fill_reconciliation_rolls_back_every_field() -> None:
    """A quantity/evidence mismatch cannot partially apply a broker reconciliation."""
    broker = PaperBroker(starting_cash=Decimal("1000"))
    bar0 = _bar(0, open_price="100")
    order = broker.submit(_intent(), _snapshot(bar0))
    broker.on_snapshot(
        _snapshot(bar0, _bar(1, open_price="100", volume="0")),
        _instrument(),
    )
    unknown = broker.mark_unknown(
        order.order_id,
        at=AT + timedelta(minutes=1, seconds=1),
    )
    truth = unknown.model_copy(
        update={
            "status": OrderStatus.PARTIALLY_FILLED,
            "filled_quantity": Decimal("0.4"),
            "average_fill_price": Decimal("100"),
            "updated_at": AT + timedelta(minutes=1, seconds=2),
        }
    )
    mismatched = Fill(
        fill_id="authoritative-fill-bad",
        order_id=order.order_id,
        instrument_id=order.instrument_id,
        side=Side.BUY,
        quantity=Decimal("0.3"),
        price=Decimal("100"),
        fee=Decimal("0"),
        filled_at=AT + timedelta(minutes=1, seconds=1, microseconds=1),
    )
    before = (broker.get_order(order.order_id), broker.fills, broker.cash, broker.audit_events)

    with pytest.raises(ValueError, match="authoritative fill"):
        broker.reconcile_unknown_fills(
            truth,
            (mismatched,),
            instrument=_instrument(),
        )

    assert (
        broker.get_order(order.order_id),
        broker.fills,
        broker.cash,
        broker.audit_events,
    ) == before


@pytest.mark.parametrize(
    ("intent", "snapshot"),
    [
        (_intent(stop_loss=None), _snapshot(_bar(0, open_price="100"))),
        (_intent(take_profit=None), _snapshot(_bar(0, open_price="100"))),
        (
            _intent(stop_loss=Decimal("101")),
            _snapshot(_bar(0, open_price="100")),
        ),
        (
            _intent(product="cash_secret_token"),
            _snapshot(_bar(0, open_price="100")),
        ),
        (
            _intent(snapshot_hash="token-not-a-digest"),
            _snapshot(_bar(0, open_price="100")),
        ),
        (
            _intent(),
            _snapshot(_bar(0, open_price="100")).model_copy(
                update={"provider": "provider_token=secret"}
            ),
        ),
    ],
)
def test_unprotected_or_untrusted_intent_and_snapshot_strings_fail_before_audit(
    intent: OrderIntent,
    snapshot: MarketSnapshot,
) -> None:
    """Unprotected entries and arbitrary audit strings must be rejected pre-persistence."""
    broker = PaperBroker(starting_cash=Decimal("1000"))

    with pytest.raises(ValueError):
        broker.submit(intent, snapshot)

    assert broker.order_count == 0
    assert broker.audit_events == ()


@pytest.mark.parametrize(
    "values",
    [
        {"broker": "paper broker"},
        {"broker": "paper\nadmin"},
        {"broker": " paper"},
    ],
)
def test_broker_capability_name_must_be_one_safe_stable_identifier(
    values: dict[str, object],
) -> None:
    base = _PAPER_CAPABILITY_VALUES | values
    with pytest.raises(ValueError, match="broker"):
        BrokerCapabilities(**base)  # type: ignore[arg-type]


def test_rehydrate_rejects_removed_partial_fill_rows_even_when_commit_is_renumbered(
    tmp_path: Path,
) -> None:
    """A checkpoint cannot invent the fill suffix omitted from its activity group."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'omitted-fill.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        max_volume_participation=Decimal("0.5"),
        audit_log=audit,
        durable=True,
        session_id="omitted-fill-session",
    )
    bars = [_bar(0, open_price="100")]
    broker.submit(_intent(), _snapshot(*bars))
    bars.append(_bar(1, open_price="100", volume="1"))
    broker.on_snapshot(_snapshot(*bars), _instrument())
    bars.append(_bar(2, open_price="100", volume="0.6"))
    broker.on_snapshot(_snapshot(*bars), _instrument())
    assert broker.get_order_by_client_id("intent-1").filled_quantity == Decimal("0.8")
    assert broker.cash == Decimal("920.0")
    rows = tuple(store.stream("omitted-fill-session"))
    final_commit_index = max(
        index for index, row in enumerate(rows) if row.kind == "paper.state.committed"
    )
    prior_commit_index = max(
        index
        for index, row in enumerate(rows[:final_commit_index])
        if row.kind == "paper.state.committed"
    )
    removed = {
        index
        for index in range(prior_commit_index + 1, final_commit_index)
        if rows[index].kind in {"paper.order.transition", "paper.order.fill"}
    }
    corrupted: list[EventRecord] = []
    for row in (item for index, item in enumerate(rows) if index not in removed):
        local = len(corrupted) + 1
        payload = _mutable_payload(row.payload)
        assert isinstance(payload, dict)
        if row.kind == "paper.state.committed":
            payload["event_sequence"] = local
        corrupted.append(
            replace(
                row,
                event_id=f"omitted-fill-session:event:{local:020d}",
                payload=payload,
            )
        )

    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(
            tuple(corrupted),
            audit_log=audit,
            expected_head=broker.session_head,
        )


def test_rehydrate_recomputes_marks_instead_of_trusting_ledger_checkpoint(
    tmp_path: Path,
) -> None:
    """A self-consistent forged valuation must disagree with the audited market event."""
    _, store, audit, broker = _durable_stream(tmp_path, participation=Decimal("1"))
    broker.on_snapshot(
        _snapshot(
            _bar(0, open_price="100"),
            _bar(1, open_price="100", volume="10"),
            _bar(2, open_price="150"),
        ),
        _instrument(),
    )
    rows = tuple(store.stream("round-two-session"))

    def forge_valuation(state: dict[str, object]) -> None:
        if "market_prices" in state:
            market_prices = state["market_prices"]
            assert isinstance(market_prices, dict)
            market_prices["AAPL@alpaca"] = "110"
        ledger = state["ledger"]
        assert isinstance(ledger, dict)
        prices = ledger["market_prices"]
        assert isinstance(prices, list)
        assert isinstance(prices[0], dict)
        prices[0]["price"] = "110"
        ledger["equity"] = "1010"
        ledger["peak_equity"] = "1010"
        ledger["drawdown"] = "0"

    corrupted = _tamper_latest_state(rows, forge_valuation)

    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(
            corrupted,
            audit_log=audit,
            expected_head=broker.session_head,
        )


def test_market_cohort_includes_terminal_orders_with_open_positions() -> None:
    """A terminal ZZZ order still requires ZZZ marks while its position remains open."""
    broker = PaperBroker(starting_cash=Decimal("1000"))
    z_bars = [_bar(0, open_price="100")]
    broker.submit(
        _intent(intent_id="z", instrument_id="ZZZ@alpaca"),
        _snapshot(*z_bars, instrument_id="ZZZ@alpaca"),
    )
    z_bars.append(_bar(1, open_price="100"))
    broker.on_snapshot(
        _snapshot(*z_bars, instrument_id="ZZZ@alpaca"),
        _instrument(instrument_id="ZZZ@alpaca"),
    )
    a_bars = [_bar(1, open_price="100")]
    broker.submit(
        _intent(
            intent_id="a",
            instrument_id="AAA@alpaca",
            created_at=AT + timedelta(minutes=1),
        ),
        _snapshot(*a_bars, instrument_id="AAA@alpaca"),
    )
    a_bars.append(_bar(2, open_price="100"))
    before = (broker.list_orders(), broker.cash, broker.fills, broker.audit_events)

    with pytest.raises(ValueError, match="on_snapshots"):
        broker.on_snapshot(
            _snapshot(*a_bars, instrument_id="AAA@alpaca"),
            _instrument(instrument_id="AAA@alpaca"),
        )
    with pytest.raises(ValueError, match="cohort"):
        broker.on_snapshots(
            (
                (
                    _snapshot(*a_bars, instrument_id="AAA@alpaca"),
                    _instrument(instrument_id="AAA@alpaca"),
                ),
            )
        )

    assert (broker.list_orders(), broker.cash, broker.fills, broker.audit_events) == before


def test_market_cohort_rejects_mixed_observation_instants_atomically() -> None:
    """A shared-cash cohort must represent one availability instant."""
    broker = PaperBroker(starting_cash=Decimal("1000"))
    a0 = _snapshot(_bar(0, open_price="100"), instrument_id="AAA@alpaca")
    z0 = _snapshot(_bar(0, open_price="100"), instrument_id="ZZZ@alpaca")
    broker.submit(_intent(intent_id="a", instrument_id="AAA@alpaca"), a0)
    broker.submit(_intent(intent_id="z", instrument_id="ZZZ@alpaca"), z0)
    a1 = _snapshot(*a0.bars, _bar(1, open_price="100"), instrument_id="AAA@alpaca")
    z1 = _snapshot(
        *z0.bars,
        _bar(1, open_price="100"),
        instrument_id="ZZZ@alpaca",
        observed_at=AT + timedelta(minutes=1, seconds=1),
    )
    before = (broker.list_orders(), broker.cash, broker.fills, broker.audit_events)

    with pytest.raises(ValueError, match="observed_at"):
        broker.on_snapshots(
            (
                (a1, _instrument(instrument_id="AAA@alpaca")),
                (z1, _instrument(instrument_id="ZZZ@alpaca")),
            )
        )

    assert (broker.list_orders(), broker.cash, broker.fills, broker.audit_events) == before


@pytest.mark.parametrize(
    ("quantity", "price"),
    [
        (Decimal("0.35"), Decimal("100")),
        (Decimal("0.4"), Decimal("100.001")),
    ],
)
def test_authoritative_reconciliation_enforces_venue_precision(
    quantity: Decimal,
    price: Decimal,
) -> None:
    """Broker evidence cannot bypass the persisted venue's quantity or price steps."""
    broker = PaperBroker(starting_cash=Decimal("1000"))
    bars = [_bar(0, open_price="100")]
    order = broker.submit(_intent(), _snapshot(*bars))
    bars.append(_bar(1, open_price="100", volume="0"))
    broker.on_snapshot(_snapshot(*bars), _instrument())
    unknown = broker.mark_unknown(order.order_id, at=AT + timedelta(minutes=1, seconds=1))
    fill = Fill(
        fill_id="bad-venue-precision",
        order_id=order.order_id,
        instrument_id=order.instrument_id,
        side=Side.BUY,
        quantity=quantity,
        price=price,
        fee=Decimal("0"),
        filled_at=AT + timedelta(minutes=1, seconds=2),
    )
    truth = unknown.model_copy(
        update={
            "status": OrderStatus.PARTIALLY_FILLED,
            "filled_quantity": quantity,
            "average_fill_price": price,
            "updated_at": AT + timedelta(minutes=1, seconds=3),
        }
    )
    before = (broker.get_order(order.order_id), broker.cash, broker.fills, broker.audit_events)

    with pytest.raises(ValueError, match="authoritative fill"):
        broker.reconcile_unknown_fills(truth, (fill,), instrument=_instrument())

    assert (
        broker.get_order(order.order_id),
        broker.cash,
        broker.fills,
        broker.audit_events,
    ) == before


def test_provider_must_be_canonical_and_secret_like_value_never_reaches_audit() -> None:
    """A regex-safe credential-shaped provider is not a supported market source."""
    broker = PaperBroker(starting_cash=Decimal("1000"))
    snapshot = _snapshot(
        _bar(0, open_price="100"),
        provider="sk_live_SUPERSECRET123",
    )

    with pytest.raises(ValueError, match="provider"):
        broker.submit(_intent(), snapshot)

    assert broker.order_count == 0
    assert broker.audit_events == ()
    assert "SUPERSECRET" not in repr(broker.audit_events)


@pytest.mark.parametrize("quantity", [Decimal("1E+130"), Decimal("1E+999999999")])
def test_decimal_encoding_rejects_oversized_exponent_before_durable_audit(
    tmp_path: Path,
    quantity: Decimal,
) -> None:
    """Every accepted Decimal must have one bounded, replayable canonical encoding."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'decimal.db'}")
    )
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=AuditLog(store, FrozenClock(AT)),
        durable=True,
        session_id="decimal-session",
    )

    with pytest.raises(ValueError, match="Decimal"):
        broker.submit(
            _intent(quantity=quantity),
            _snapshot(_bar(0, open_price="100")),
        )

    assert broker.order_count == 0
    assert tuple(store.stream("decimal-session")) == ()


def test_decimal_boundary_roundtrips_through_durable_recovery(tmp_path: Path) -> None:
    """The largest supported exponent remains symmetric across encode and replay."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'decimal-boundary.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=audit,
        durable=True,
        session_id="decimal-boundary-session",
    )
    expected = broker.submit(
        _intent(quantity=Decimal("1E+64")),
        _snapshot(_bar(0, open_price="100")),
    )

    restored = PaperBroker.rehydrate(
        tuple(store.stream("decimal-boundary-session")),
        audit_log=audit,
        expected_head=broker.session_head,
    )

    assert restored.get_order(expected.order_id) == expected


def test_progressed_order_retry_uses_current_state_not_obsolete_submission_snapshot() -> None:
    """Idempotency must remain available after the original snapshot falls behind."""
    original = _snapshot(_bar(0, open_price="100"))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        max_volume_participation=Decimal("0.5"),
    )
    order = broker.submit(_intent(), original)
    broker.on_snapshot(
        _snapshot(*original.bars, _bar(1, open_price="100", volume="1")),
        _instrument(),
    )
    current = broker.get_order(order.order_id)

    assert broker.submit(_intent(), original) == current
    assert broker.audit_events[-1].kind == "paper.order.duplicate"
    assert broker.audit_events[-1].occurred_at == AT + timedelta(minutes=1)

    with pytest.raises(DuplicateIntentConflict):
        broker.submit(_intent(quantity=Decimal("2")), original)
    assert broker.audit_events[-1].kind == "paper.order.duplicate_conflict"
    assert broker.audit_events[-1].occurred_at == AT + timedelta(minutes=1)


def test_paper_audit_payloads_are_recursively_immutable_live_and_after_replay(
    tmp_path: Path,
) -> None:
    """Nested evidence cannot be mutated through an otherwise frozen audit event."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'freeze.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=audit,
        durable=True,
        session_id="freeze-session",
    )
    broker.submit(_intent(), _snapshot(_bar(0, open_price="100")))
    live = next(event for event in broker.audit_events if event.kind == "paper.order.submitted")
    live_intent = live.payload["intent"]
    assert isinstance(live_intent, Mapping)

    with pytest.raises(TypeError):
        live_intent["quantity"] = "9"  # type: ignore[index]

    restored = PaperBroker.rehydrate(
        tuple(store.stream("freeze-session")),
        audit_log=audit,
        expected_head=broker.session_head,
    )
    replayed = next(
        event for event in restored.audit_events if event.kind == "paper.order.submitted"
    )
    replayed_intent = replayed.payload["intent"]
    assert isinstance(replayed_intent, Mapping)
    with pytest.raises(TypeError):
        replayed_intent["quantity"] = "9"  # type: ignore[index]
    assert live.payload["intent"] == replayed.payload["intent"]


def test_durable_market_history_payload_growth_is_linear_and_replayable(
    tmp_path: Path,
) -> None:
    """Later marks persist one cursor delta, never the complete growing bar prefix."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'linear.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=audit,
        durable=True,
        session_id="linear-session",
    )
    bars = [_bar(0, open_price="100")]
    broker.submit(_intent(), _snapshot(*bars))
    for minute in range(1, 41):
        bars.append(_bar(minute, open_price=str(100 + minute), volume="10"))
        broker.on_snapshot(_snapshot(*bars), _instrument())
    rows = tuple(store.stream("linear-session"))
    commit_sizes = [
        len(json.dumps(_mutable_payload(row.payload), sort_keys=True))
        for row in rows
        if row.kind == "paper.state.committed"
    ]
    commits: list[dict[str, object]] = []
    for row in rows:
        if row.kind != "paper.state.committed":
            continue
        commit = _mutable_payload(row.payload)
        assert isinstance(commit, dict)
        commits.append(commit)

    assert len(commit_sizes) == 41
    assert max(commit_sizes) - min(commit_sizes) < 768
    assert sum(
        len(json.dumps(_mutable_payload(row.payload), sort_keys=True)) for row in rows
    ) < len(rows) * 4096
    assert all(
        commits[index]["previous_state_digest"]
        == commits[index - 1]["state_digest"]
        for index in range(1, len(commits))
    )
    assert all(
        isinstance(commit["state_digest"], str)
        and len(commit["state_digest"]) == 64
        for commit in commits
    )
    restored = PaperBroker.rehydrate(
        rows,
        audit_log=audit,
        expected_head=broker.session_head,
    )
    assert restored.portfolio_snapshot() == broker.portfolio_snapshot()
    assert restored.fills == broker.fills


def test_durable_second_order_references_existing_snapshot_without_repeating_bars(
    tmp_path: Path,
) -> None:
    """Same-instrument submissions remain replayable without duplicating bar history."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'reference.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=audit,
        durable=True,
        session_id="reference-session",
    )
    snapshot = _snapshot(_bar(0, open_price="100"))
    broker.submit(_intent(intent_id="first"), snapshot)
    broker.submit(_intent(intent_id="second"), snapshot)
    rows = tuple(store.stream("reference-session"))

    assert sum(row.kind == "paper.market.submission" for row in rows) == 1
    assert sum(row.kind == "paper.market.submission_reference" for row in rows) == 1
    restored = PaperBroker.rehydrate(
        rows,
        audit_log=audit,
        expected_head=broker.session_head,
    )
    assert restored.list_orders() == broker.list_orders()


def test_oversized_initial_bar_history_fails_before_audit() -> None:
    """The per-instrument history bound is enforced before creating durable evidence."""
    bars = tuple(_bar(minute, open_price="100") for minute in range(1025))
    at = bars[-1].at
    broker = PaperBroker(starting_cash=Decimal("1000"))

    with pytest.raises(ValueError, match="bar"):
        broker.submit(
            _intent(created_at=at, expires_at=at + timedelta(minutes=10)),
            _snapshot(*bars),
        )

    assert broker.order_count == 0
    assert broker.audit_events == ()


def test_trusted_session_head_is_required_and_roundtrips_real_sqlite(
    tmp_path: Path,
) -> None:
    """Recovery must bind the complete stream to a separately retained trusted head."""
    rows, _, audit, broker = _durable_stream(tmp_path, session_id="trusted-head")
    head = broker.session_head

    assert head == SessionHead(
        session_id="trusted-head",
        event_count=len(rows),
        operation_count=2,
        head_digest=head.head_digest,
    )
    with pytest.raises(ValueError, match="trusted session head"):
        PaperBroker.rehydrate(rows, audit_log=audit)
    restored = PaperBroker.rehydrate(
        rows,
        audit_log=audit,
        expected_head=head,
    )

    assert restored.session_head == head
    assert restored.fills == broker.fills
    assert restored.portfolio_snapshot() == broker.portfolio_snapshot()


@pytest.mark.parametrize(
    "head",
    [
        SessionHead("wrong-session", 11, 2, "0" * 64),
        SessionHead("trusted-head-errors", 10, 2, "0" * 64),
        SessionHead("trusted-head-errors", 11, 3, "0" * 64),
        SessionHead("trusted-head-errors", 11, 2, "f" * 64),
    ],
)
def test_rehydrate_rejects_every_wrong_trusted_head(
    tmp_path: Path,
    head: SessionHead,
) -> None:
    """Session, append count, operation count, and digest are all trust boundaries."""
    rows, _, audit, _ = _durable_stream(
        tmp_path,
        session_id="trusted-head-errors",
    )

    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(rows, audit_log=audit, expected_head=head)


def test_production_durable_mode_rejects_discard_only_structural_recorder() -> None:
    """A callable record_many surface cannot claim crash durability."""
    with pytest.raises(ValueError, match="AuditLog"):
        PaperBroker(
            audit_log=_DiscardAudit(),
            durable=True,
            session_id="discard-session",
        )


def test_market_operation_rejects_nonempty_input_when_relevant_cohort_is_empty() -> None:
    """A terminal no-position session cannot persist a market group replay must reject."""
    broker = PaperBroker(starting_cash=Decimal("1000"))
    first = _snapshot(_bar(0, open_price="100"))
    order = broker.submit(_intent(), first)
    broker.cancel(order.order_id, at=AT + timedelta(seconds=1))
    before = (broker.list_orders(), broker.fills, broker.cash, broker.audit_events)

    with pytest.raises(ValueError, match="cohort"):
        broker.on_snapshots(
            (
                (
                    _snapshot(*first.bars, _bar(1, open_price="101")),
                    _instrument(),
                ),
            )
        )

    assert (broker.list_orders(), broker.fills, broker.cash, broker.audit_events) == before


def test_malformed_duplicate_fails_full_shape_validation_without_audit() -> None:
    """Idempotency lookup cannot audit a side-inconsistent pre-lookup intent."""
    snapshot = _snapshot(_bar(0, open_price="100"))
    broker = PaperBroker(starting_cash=Decimal("1000"))
    valid = _intent()
    broker.submit(valid, snapshot)
    malformed = valid.model_copy(update={"stop_loss": Decimal("130")})
    before = broker.audit_events

    with pytest.raises(ValueError, match="protection"):
        broker.submit(malformed, snapshot)

    assert broker.audit_events == before


def test_authoritative_partial_fill_uses_original_order_minimum_notional() -> None:
    """Venue minimum applies to qualifying requested size, not each partial execution."""
    broker = PaperBroker(starting_cash=Decimal("1000"))
    bars = [_bar(0, open_price="100")]
    order = broker.submit(_intent(), _snapshot(*bars))
    venue = _instrument(minimum_notional=Decimal("50"))
    bars.append(_bar(1, open_price="100", volume="0"))
    broker.on_snapshot(_snapshot(*bars), venue)
    unknown = broker.mark_unknown(order.order_id, at=AT + timedelta(minutes=1, seconds=1))
    fill = Fill(
        fill_id="qualifying-order-small-partial",
        order_id=order.order_id,
        instrument_id=order.instrument_id,
        side=Side.BUY,
        quantity=Decimal("0.1"),
        price=Decimal("100"),
        fee=Decimal("0"),
        filled_at=AT + timedelta(minutes=1, seconds=1, microseconds=1),
    )
    truth = unknown.model_copy(
        update={
            "status": OrderStatus.PARTIALLY_FILLED,
            "filled_quantity": Decimal("0.1"),
            "average_fill_price": Decimal("100"),
            "updated_at": AT + timedelta(minutes=1, seconds=2),
        }
    )

    assert broker.reconcile_unknown_fills(truth, (fill,), instrument=venue) == truth
    assert broker.cash == Decimal("990")


def test_rehydrate_consumes_at_most_the_event_limit_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hostile generator is bounded before the durable stream is materialized."""
    monkeypatch.setattr(paper_module, "_MAX_SESSION_EVENTS", 3)
    store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
    audit = AuditLog(store, FrozenClock(AT))
    row = EventRecord(
        event_id="bounded:event:00000000000000000001",
        kind="paper.order.submitted",
        aggregate_id="bounded",
        payload={},
        occurred_at=AT,
        sequence=1,
    )

    def hostile_records() -> object:
        for index in range(4):
            yield replace(
                row,
                event_id=f"bounded:event:{index + 1:020d}",
                sequence=index + 1,
            )
        raise AssertionError("rehydrate consumed beyond max plus one")

    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(
            hostile_records(),  # type: ignore[arg-type]
            audit_log=audit,
            expected_head=SessionHead("bounded", 3, 1, "0" * 64),
        )


def test_rehydrate_rejects_cyclic_payload_without_recursion_escape() -> None:
    """Replay validates a hostile mapping iteratively before canonical JSON conversion."""
    store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
    audit = AuditLog(store, FrozenClock(AT))
    payload: dict[str, object] = {}
    payload["self"] = payload
    row = EventRecord(
        event_id="cyclic:event:00000000000000000001",
        kind="paper.order.submitted",
        aggregate_id="cyclic",
        payload=payload,
        occurred_at=AT,
        sequence=1,
    )

    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(
            (row,),
            audit_log=audit,
            expected_head=SessionHead("cyclic", 1, 1, "0" * 64),
        )


@pytest.mark.parametrize("encoded", ["1.0", "1.00", "0.0", "-0"])
def test_durable_decimal_parser_rejects_noncanonical_writer_forms(encoded: str) -> None:
    """Persisted Decimal text has exactly one writer representation."""
    with pytest.raises(ValueError):
        paper_module._parse_decimal(encoded, nonnegative=True)


@pytest.mark.parametrize(
    ("operation_kind", "kinds"),
    [
        (
            "SUBMIT",
            (
                "paper.market.submission",
                "paper.order.submitted",
                "paper.order.transition",
                "paper.order.transition",
                "paper.order.transition",
            ),
        ),
        (
            "SUBMIT",
            (
                "paper.market.submission",
                "paper.order.transition",
                "paper.order.submitted",
                "paper.order.transition",
                "paper.order.transition",
                "paper.order.transition",
            ),
        ),
        ("IDEMPOTENCY", ("paper.order.duplicate", "paper.order.transition")),
        (
            "RESOLUTION",
            ("paper.order.transition", "paper.order.transition"),
        ),
        (
            "RECONCILIATION",
            ("paper.order.fill", "paper.order.transition"),
        ),
        ("RESOLUTION", ("paper.order.unknown_activity",)),
    ],
)
def test_non_market_operation_grammar_rejects_omitted_reordered_or_extra_rows(
    operation_kind: str,
    kinds: tuple[str, ...],
) -> None:
    """A recomputed envelope cannot turn a malformed row sequence into authority."""
    records = tuple(
        EventRecord(
            event_id=f"grammar:event:{index:020d}",
            kind=kind,
            aggregate_id="grammar",
            payload={},
            occurred_at=AT,
            sequence=index,
        )
        for index, kind in enumerate(kinds, start=1)
    )

    with pytest.raises(ValueError):
        paper_module._validate_operation_grammar(records, operation_kind)


def test_semantic_replay_rejects_forged_fill_fee_with_every_hash_recomputed(
    tmp_path: Path,
) -> None:
    """A zero-cost FillModel, not a public digest, proves the exact fee is zero."""
    rows, _, audit, broker = _durable_stream(
        tmp_path,
        session_id="forged-fee",
        participation=Decimal("1"),
    )
    first_commit = next(
        index for index, row in enumerate(rows) if row.kind == "paper.state.committed"
    )
    activities = list(rows[first_commit + 1 : -1])
    fill_index = next(
        index for index, row in enumerate(activities) if row.kind == "paper.order.fill"
    )
    fill_payload = _mutable_payload(activities[fill_index].payload)
    assert isinstance(fill_payload, dict)
    fill_payload["fee"] = "5"
    fill_payload["cumulative_fees"] = "5"
    activities[fill_index] = replace(activities[fill_index], payload=fill_payload)
    checkpoint = _mutable_payload(rows[-1].payload)
    assert isinstance(checkpoint, dict)
    ledger = checkpoint["ledger"]
    assert isinstance(ledger, dict)
    ledger.update(
        {
            "cash": "895",
            "fees": "5",
            "equity": "995",
            "peak_equity": "1000",
            "drawdown": "0.005",
        }
    )
    forged, forged_head = _rehash_forged_final_market_group(
        rows,
        activities=activities,
        ledger=ledger,
        fill_sequence=1,
    )

    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(
            forged,
            audit_log=audit,
            expected_head=broker.session_head,
        )
    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(
            forged,
            audit_log=audit,
            expected_head=forged_head,
        )


def test_semantic_replay_rejects_fill_omission_with_transition_and_hashes_retained(
    tmp_path: Path,
) -> None:
    """ACK to partial requires the exact shared-kernel fill activity and ledger delta."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'omit-semantic.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        max_volume_participation=Decimal("0.5"),
        audit_log=audit,
        durable=True,
        session_id="omit-semantic",
    )
    bar0 = _bar(0, open_price="100")
    broker.submit(_intent(), _snapshot(bar0))
    broker.on_snapshot(
        _snapshot(bar0, _bar(1, open_price="100", volume="1")),
        _instrument(),
    )
    rows = tuple(store.stream("omit-semantic"))
    first_commit = next(
        index for index, row in enumerate(rows) if row.kind == "paper.state.committed"
    )
    activities = [
        row
        for row in rows[first_commit + 1 : -1]
        if row.kind != "paper.order.fill"
    ]
    checkpoint = _mutable_payload(rows[-1].payload)
    assert isinstance(checkpoint, dict)
    ledger = checkpoint["ledger"]
    assert isinstance(ledger, dict)
    ledger.update(
        {
            "cash": "1000",
            "gross_realized_pnl": "0",
            "fees": "0",
            "equity": "1000",
            "peak_equity": "1000",
            "drawdown": "0",
            "positions_digest": _test_positions_digest([]),
            "fill_ids_digest": hashlib.sha256(b"[]").hexdigest(),
        }
    )
    forged, forged_head = _rehash_forged_final_market_group(
        rows,
        activities=activities,
        ledger=ledger,
        fill_sequence=0,
    )

    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(
            forged,
            audit_log=audit,
            expected_head=forged_head,
        )


def test_semantic_replay_rejects_invented_stop_trigger_on_noncrossing_bar(
    tmp_path: Path,
) -> None:
    """A trigger/fill grammar cannot be invented when the canonical bar never crosses."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'stop-forge.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=audit,
        durable=True,
        session_id="stop-forge",
    )
    bar0 = _bar(0, open_price="100")
    broker.submit(
        _intent(order_type=OrderType.STOP, trigger_price=Decimal("110")),
        _snapshot(bar0),
    )
    broker.on_snapshot(
        _snapshot(bar0, _bar(1, open_price="100", high="115", low="100")),
        _instrument(),
    )
    rows = tuple(store.stream("stop-forge"))
    first_commit = next(
        index for index, row in enumerate(rows) if row.kind == "paper.state.committed"
    )
    activities = list(rows[first_commit + 1 : -1])
    fill_index = next(
        index for index, row in enumerate(activities) if row.kind == "paper.order.fill"
    )
    fill_payload = _mutable_payload(activities[fill_index].payload)
    assert isinstance(fill_payload, dict)
    fill_payload["price"] = "100"
    fill_payload["cumulative_filled_notional"] = "100"
    activities[fill_index] = replace(activities[fill_index], payload=fill_payload)
    observed_index = next(
        index for index, row in enumerate(activities) if row.kind == "paper.market.observed"
    )
    observed_payload = _mutable_payload(activities[observed_index].payload)
    assert isinstance(observed_payload, dict)
    bar_payload = observed_payload["bar"]
    assert isinstance(bar_payload, dict)
    bar_payload["high"] = "105"
    initial_payload = _mutable_payload(rows[0].payload)
    assert isinstance(initial_payload, dict)
    initial_snapshot = initial_payload["snapshot"]
    assert isinstance(initial_snapshot, dict)
    initial_bars = initial_snapshot["bars"]
    assert isinstance(initial_bars, list)
    observed_payload["bars_digest"] = _test_bar_digest(
        [initial_bars[0], bar_payload]
    )
    activities[observed_index] = replace(
        activities[observed_index],
        payload=observed_payload,
    )
    checkpoint = _mutable_payload(rows[-1].payload)
    assert isinstance(checkpoint, dict)
    ledger = checkpoint["ledger"]
    assert isinstance(ledger, dict)
    ledger.update(
        {
            "cash": "900",
            "fees": "0",
            "equity": "1000",
            "peak_equity": "1000",
            "drawdown": "0",
            "positions_digest": _test_positions_digest(
                [
                    {
                        "instrument_id": "AAPL@alpaca",
                        "quantity": "1",
                        "average_price": "100",
                    }
                ]
            ),
        }
    )
    forged, forged_head = _rehash_forged_final_market_group(
        rows,
        activities=activities,
        ledger=ledger,
        fill_sequence=1,
    )

    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(
            forged,
            audit_log=audit,
            expected_head=forged_head,
        )


def test_rolling_market_cursor_recovers_open_position_beyond_1024_bars(
    tmp_path: Path,
) -> None:
    """Long sessions persist one chained delta and a bounded 64-bar overlap window."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'rollover.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=audit,
        durable=True,
        session_id="rollover",
    )
    bar0 = _bar(0, open_price="100")
    broker.submit(_intent(), _snapshot(bar0))
    bar1 = _bar(1, open_price="100")
    broker.on_snapshot(_snapshot(bar0, bar1), _instrument())
    overlap: tuple[Bar, ...] = (bar0, bar1)
    prior_digest = paper_module._bars_digest(overlap)
    for minute in range(2, 1_030):
        new_bar = _bar(minute, open_price="100")
        rolling = paper_module.RollingMarketWindow(
            instrument_id="AAPL@alpaca",
            observed_at=new_bar.at,
            source_at=new_bar.at,
            prior_bar_count=minute,
            prior_bars_digest=prior_digest,
            overlap=overlap,
            new_bar=new_bar,
            provider="fixture",
            max_age_seconds=60,
        )
        broker.on_rolling_snapshot(rolling, _instrument())
        prior_digest = paper_module._next_bar_digest(prior_digest, new_bar)
        overlap = (overlap + (new_bar,))[-64:]

    rows = tuple(store.stream("rollover"))
    observed = [row for row in rows if row.kind == "paper.market.observed"]
    assert observed[-1].payload["bar_count"] == 1_030
    restored = PaperBroker.rehydrate_durable(audit, expected_head=broker.session_head)

    assert restored.session_head == broker.session_head
    assert restored.positions() == broker.positions()
    assert restored.portfolio_snapshot() == broker.portfolio_snapshot()


def test_instrument_cap_counts_orders_snapshots_cursors_and_positions_atomically() -> None:
    """A 129th identity cannot hide in a dictionary omitted from the session cap."""
    broker = PaperBroker(starting_cash=Decimal("1000"))
    for index in range(128):
        instrument_id = f"SYM{index}@alpaca"
        broker.submit(
            _intent(intent_id=f"intent-{index}", instrument_id=instrument_id),
            _snapshot(_bar(0, open_price="100"), instrument_id=instrument_id),
        )
    before = (broker.order_count, broker.audit_events)

    with pytest.raises(ValueError, match="activity|bounded durable limit"):
        broker.submit(
            _intent(intent_id="intent-128", instrument_id="SYM128@alpaca"),
            _snapshot(_bar(0, open_price="100"), instrument_id="SYM128@alpaca"),
        )

    assert (broker.order_count, broker.audit_events) == before


def test_durable_recovery_streams_and_continues_on_the_same_real_store(
    tmp_path: Path,
) -> None:
    """Recovery and the next contiguous append must share one exact EventStore capability."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'same-store.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=audit,
        durable=True,
        session_id="same-store",
    )
    order = broker.submit(_intent(), _snapshot(_bar(0, open_price="100")))
    head = broker.session_head

    recovered = PaperBroker.rehydrate_durable(audit, expected_head=head)
    recovered.cancel(order.order_id, at=AT + timedelta(seconds=1))
    rows = tuple(store.stream("same-store"))

    assert recovered.durability_mode == "durable"
    assert [row.sequence for row in rows] == list(range(1, len(rows) + 1))
    assert rows[-1].event_id == f"same-store:event:{len(rows):020d}"
    assert recovered.session_head.event_count == len(rows)


def test_generic_record_replay_cannot_continue_durably_into_another_store(
    tmp_path: Path,
) -> None:
    """Caller-supplied store-A rows cannot authorize future writes into store B."""
    store_a = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'store-a.db'}")
    )
    audit_a = AuditLog(store_a, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=audit_a,
        durable=True,
        session_id="store-provenance",
    )
    order = broker.submit(_intent(), _snapshot(_bar(0, open_price="100")))
    rows_a = tuple(store_a.stream("store-provenance"))
    store_b = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'store-b.db'}")
    )
    audit_b = AuditLog(store_b, FrozenClock(AT))

    replayed = PaperBroker.rehydrate(
        rows_a,
        audit_log=audit_b,
        expected_head=broker.session_head,
    )
    replayed.cancel(order.order_id, at=AT + timedelta(seconds=1))

    assert replayed.durability_mode == "current_session"
    assert tuple(store_b.stream("store-provenance")) == ()


def test_durable_recovery_rollback_keeps_store_and_head_contiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed same-store append after recovery must roll back every in-memory field."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'recovery-rollback.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=audit,
        durable=True,
        session_id="recovery-rollback",
    )
    order = broker.submit(_intent(), _snapshot(_bar(0, open_price="100")))
    recovered = PaperBroker.rehydrate_durable(audit, expected_head=broker.session_head)
    before = (
        recovered.list_orders(),
        recovered.fills,
        recovered.cash,
        recovered.audit_events,
        recovered.session_head,
        tuple(store.stream("recovery-rollback")),
    )

    def fail_append(batch: object) -> None:
        del batch
        raise RuntimeError("durable store unavailable")

    monkeypatch.setattr(store, "append_many", fail_append)
    with pytest.raises(RuntimeError, match="unavailable"):
        recovered.cancel(order.order_id, at=AT + timedelta(seconds=1))

    assert (
        recovered.list_orders(),
        recovered.fills,
        recovered.cash,
        recovered.audit_events,
        recovered.session_head,
        tuple(store.stream("recovery-rollback")),
    ) == before


def test_recovery_requires_exact_validated_session_head_type(tmp_path: Path) -> None:
    """Duck-typed mutable head objects cannot establish the external trust boundary."""
    rows, _, audit, broker = _durable_stream(tmp_path, session_id="exact-head")
    head = broker.session_head
    mutable_head = SimpleNamespace(
        session_id=head.session_id,
        event_count=head.event_count,
        operation_count=head.operation_count,
        head_digest=head.head_digest,
    )

    with pytest.raises(ValueError, match="trusted session head"):
        PaperBroker.rehydrate(
            rows,
            audit_log=audit,
            expected_head=mutable_head,  # type: ignore[arg-type]
        )


def test_natural_full_prefix_100_to_101_is_accepted_without_hidden_slicing() -> None:
    """A normal provider prefix remains valid while it is within the public input bound."""
    bars = tuple(_bar(minute, open_price="100") for minute in range(100))
    broker = PaperBroker(starting_cash=Decimal("1000"))
    broker.submit(
        _intent(created_at=bars[-1].at, expires_at=bars[-1].at + timedelta(minutes=10)),
        _snapshot(*bars),
    )
    next_bar = _bar(100, open_price="101")

    fills = broker.on_snapshot(_snapshot(*bars, next_bar), _instrument())

    assert len(fills) == 1
    assert broker.portfolio_snapshot().observed_at == next_bar.at


def test_explicit_rolling_window_continues_after_1024_and_rejects_bad_overlap() -> None:
    """The public rolling contract binds prior count/digest and the retained overlap."""
    first = _bar(0, open_price="100")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    broker.submit(
        _intent(created_at=first.at, expires_at=first.at + timedelta(minutes=2000)),
        _snapshot(first),
    )
    prior_digest = paper_module._bars_digest((first,))
    overlap: tuple[Bar, ...] = (first,)
    window: object = None
    for minute in range(1, 1025):
        next_bar = _bar(minute, open_price="101")
        window = paper_module.RollingMarketWindow(
            instrument_id="AAPL@alpaca",
            observed_at=next_bar.at,
            source_at=next_bar.at,
            prior_bar_count=minute,
            prior_bars_digest=prior_digest,
            overlap=overlap,
            new_bar=next_bar,
            provider="fixture",
            max_age_seconds=60,
        )
        broker.on_rolling_snapshot(window, _instrument())
        prior_digest = paper_module._next_bar_digest(prior_digest, next_bar)
        overlap = (overlap + (next_bar,))[-64:]
    assert isinstance(window, paper_module.RollingMarketWindow)
    before = (broker.list_orders(), broker.fills, broker.cash, broker.audit_events)
    malformed = replace(
        window,
        observed_at=_bar(1025, open_price="102").at,
        source_at=_bar(1025, open_price="102").at,
        prior_bar_count=1025,
        prior_bars_digest=prior_digest,
        new_bar=_bar(1025, open_price="102"),
        overlap=overlap[-63:],
    )
    with pytest.raises(ValueError, match="rolling"):
        broker.on_rolling_snapshot(malformed, _instrument())

    assert (broker.list_orders(), broker.fills, broker.cash, broker.audit_events) == before


def test_submission_admission_keeps_worst_case_market_grammar_consumable(
    tmp_path: Path,
) -> None:
    """The 171st triggerable order is rejected before it can brick the next cohort."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'liveness.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("100000"),
        audit_log=audit,
        durable=True,
        session_id="liveness",
    )
    initial = _snapshot(_bar(0, open_price="100"))
    for index in range(170):
        broker.submit(
            _intent(
                intent_id=f"stop-{index}",
                quantity=Decimal("0.1"),
                order_type=OrderType.STOP,
                trigger_price=Decimal("110"),
                take_profit=Decimal("150"),
            ),
            initial,
        )
    before = (broker.order_count, broker.audit_events, broker.session_head)

    with pytest.raises(ValueError, match="activity"):
        broker.submit(
            _intent(
                intent_id="stop-170",
                quantity=Decimal("0.1"),
                order_type=OrderType.STOP,
                trigger_price=Decimal("110"),
                take_profit=Decimal("150"),
            ),
            initial,
        )
    assert (broker.order_count, broker.audit_events, broker.session_head) == before

    crossing = _snapshot(
        *initial.bars,
        _bar(1, open_price="110", high="115", low="109", volume="1000"),
    )
    fills = broker.on_snapshot(crossing, _instrument())
    restored = PaperBroker.rehydrate_durable(audit, expected_head=broker.session_head)

    assert len(fills) == 170
    assert restored.list_orders() == broker.list_orders()
    assert restored.fills == broker.fills


def test_admission_budget_includes_open_position_instruments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Position-only market rows count against a multi-instrument cohort's exact grammar."""
    monkeypatch.setattr(paper_module, "_MAX_GROUP_ACTIVITIES", 7)
    broker = PaperBroker(starting_cash=Decimal("10000"))
    b0 = _snapshot(_bar(0, open_price="100"), instrument_id="BBB@alpaca")
    broker.submit(
        _intent(intent_id="bbb-position", instrument_id="BBB@alpaca"),
        b0,
    )
    b1 = _snapshot(
        *b0.bars,
        _bar(1, open_price="100"),
        instrument_id="BBB@alpaca",
    )
    broker.on_snapshot(b1, _instrument(instrument_id="BBB@alpaca"))
    a1 = _snapshot(_bar(1, open_price="100"), instrument_id="AAA@alpaca")
    for index in range(1):
        broker.submit(
            _intent(
                intent_id=f"aaa-stop-{index}",
                instrument_id="AAA@alpaca",
                quantity=Decimal("0.1"),
                order_type=OrderType.STOP,
                trigger_price=Decimal("110"),
                take_profit=Decimal("150"),
                created_at=AT + timedelta(minutes=1),
                expires_at=AT + timedelta(minutes=10),
            ),
            a1,
        )
    before = (broker.order_count, broker.audit_events)
    with pytest.raises(ValueError, match="activity"):
        broker.submit(
            _intent(
                intent_id="aaa-stop-1",
                instrument_id="AAA@alpaca",
                quantity=Decimal("0.1"),
                order_type=OrderType.STOP,
                trigger_price=Decimal("110"),
                take_profit=Decimal("150"),
                created_at=AT + timedelta(minutes=1),
                expires_at=AT + timedelta(minutes=10),
            ),
            a1,
        )
    assert (broker.order_count, broker.audit_events) == before

    a2 = _snapshot(
        *a1.bars,
        _bar(2, open_price="110", high="115", low="109", volume="100"),
        instrument_id="AAA@alpaca",
    )
    b2 = _snapshot(
        *b1.bars,
        _bar(2, open_price="101"),
        instrument_id="BBB@alpaca",
    )
    fills = broker.on_snapshots(
        (
            (a2, _instrument(instrument_id="AAA@alpaca")),
            (b2, _instrument(instrument_id="BBB@alpaca")),
        )
    )
    assert len(fills) == 1


def test_staging_does_not_rescan_or_copy_growing_fill_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Many partial fills process only their deltas, not triangular history prefixes."""
    broker = PaperBroker(starting_cash=Decimal("10000"))
    order = broker.submit(
        _intent(quantity=Decimal("30")),
        _snapshot(_bar(0, open_price="100")),
    )
    broker.on_snapshot(
        _snapshot(_bar(0, open_price="100"), _bar(1, open_price="100", volume="0")),
        _instrument(quantity_step=Decimal("1")),
    )
    clone_history_references = 0
    export_fill_references = 0
    original_export = PortfolioLedger.export_state

    original_deepcopy = copy_module.deepcopy

    def count_deepcopy(
        value: object,
        memo: dict[int, Any] | None = None,
    ) -> object:
        nonlocal clone_history_references
        if isinstance(value, PortfolioLedger):
            clone_history_references += len(original_export(value).fill_ids)
        return original_deepcopy(value, memo)

    def count_export(ledger: PortfolioLedger) -> PortfolioLedgerState:
        nonlocal export_fill_references
        exported = original_export(ledger)
        export_fill_references += len(exported.fill_ids)
        return exported

    monkeypatch.setattr(copy_module, "deepcopy", count_deepcopy)
    monkeypatch.setattr(PortfolioLedger, "export_state", count_export)
    current = order
    for index in range(30):
        unknown = broker.mark_unknown(
            current.order_id,
            at=AT + timedelta(minutes=1, seconds=index * 3 + 1),
        )
        fill = Fill(
            fill_id=f"complexity-fill-{index}",
            order_id=order.order_id,
            instrument_id=order.instrument_id,
            side=Side.BUY,
            quantity=Decimal("1"),
            price=Decimal("100"),
            fee=Decimal("0"),
            filled_at=AT + timedelta(minutes=1, seconds=index * 3 + 1, microseconds=1),
        )
        quantity = Decimal(index + 1)
        current = unknown.model_copy(
            update={
                "status": (
                    OrderStatus.FILLED if index == 29 else OrderStatus.PARTIALLY_FILLED
                ),
                "filled_quantity": quantity,
                "average_fill_price": Decimal("100"),
                "updated_at": AT + timedelta(minutes=1, seconds=index * 3 + 2),
            }
        )
        broker.reconcile_unknown_fills(
            current,
            (fill,),
            instrument=_instrument(quantity_step=Decimal("1")),
        )

    assert clone_history_references <= 30
    assert export_fill_references <= 30


def test_runtime_and_replay_route_every_non_market_operation_through_same_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit, idempotency, resolution, and authoritative truth each have one reducer path."""
    names = (
        "_execute_submit_kernel",
        "_execute_idempotency_kernel",
        "_execute_resolution_kernel",
        "_execute_reconciliation_kernel",
    )
    calls = dict.fromkeys(names, 0)
    for name in names:
        original = getattr(paper_module, name)

        def wrapper(
            *args: object,
            _name: str = name,
            _original: object = original,
            **kwargs: object,
        ) -> object:
            calls[_name] += 1
            return _original(*args, **kwargs)  # type: ignore[operator]

        monkeypatch.setattr(paper_module, name, wrapper)

    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'shared-nonmarket.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=audit,
        durable=True,
        session_id="shared-nonmarket",
    )
    snapshot = _snapshot(_bar(0, open_price="100"))
    intent = _intent()
    order = broker.submit(intent, snapshot)
    broker.submit(intent, snapshot)
    # Persist exact instrument metadata without making the order eligible for a market fill.
    broker.on_snapshot(
        _snapshot(
            _bar(0, open_price="100"),
            _bar(0, open_price="100").model_copy(
                update={"at": AT + timedelta(microseconds=1), "volume": Decimal("0")}
            ),
            observed_at=AT + timedelta(microseconds=1),
            source_at=AT + timedelta(microseconds=1),
        ),
        _instrument(),
    )
    unknown = broker.mark_unknown(order.order_id, at=AT + timedelta(seconds=1))
    fill = Fill(
        fill_id="shared-authoritative-fill",
        order_id=order.order_id,
        instrument_id=order.instrument_id,
        side=Side.BUY,
        quantity=Decimal("1"),
        price=Decimal("100"),
        fee=Decimal("0"),
        filled_at=AT + timedelta(seconds=1, microseconds=1),
    )
    truth = unknown.model_copy(
        update={
            "status": OrderStatus.FILLED,
            "filled_quantity": Decimal("1"),
            "average_fill_price": Decimal("100"),
            "updated_at": AT + timedelta(seconds=2),
        }
    )
    broker.reconcile_unknown_fills(truth, (fill,), instrument=_instrument())
    PaperBroker.rehydrate(
        tuple(store.stream("shared-nonmarket")),
        audit_log=audit,
        expected_head=broker.session_head,
    )

    assert calls["_execute_submit_kernel"] == 2
    assert calls["_execute_idempotency_kernel"] == 2
    assert calls["_execute_resolution_kernel"] == 2
    assert calls["_execute_reconciliation_kernel"] == 2


def test_submit_replay_rejects_split_operation_timestamp_with_recomputed_head(
    tmp_path: Path,
) -> None:
    """All submit evidence must share the canonical snapshot availability timestamp."""
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'split-submit.db'}")
    )
    audit = AuditLog(store, FrozenClock(AT))
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=audit,
        durable=True,
        session_id="split-submit",
    )
    broker.submit(_intent(), _snapshot(_bar(0, open_price="100")))
    rows = list(store.stream("split-submit"))
    late = AT + timedelta(seconds=1)
    for index in range(1, 6):
        rows[index] = replace(rows[index], occurred_at=late)
    checkpoint = _mutable_payload(rows[-1].payload)
    assert isinstance(checkpoint, dict)
    market = _mutable_payload(rows[0].payload)
    assert isinstance(market, dict)
    snapshot = market["snapshot"]
    assert isinstance(snapshot, dict)
    bars = snapshot["bars"]
    assert isinstance(bars, list)
    last_bar = bars[-1]
    assert isinstance(last_bar, dict)
    activities = [
        {
            "kind": row.kind,
            "occurred_at": row.occurred_at.astimezone(UTC).isoformat(),
            "payload": _mutable_payload(row.payload),
        }
        for row in rows[:-1]
    ]
    state_digest = _test_canonical_digest(
        {
            "previous_state_digest": checkpoint["previous_state_digest"],
            "activities": activities,
            "event_sequence": len(rows),
            "operation_sequence": 1,
            "fill_sequence": 0,
            "order_count": 1,
            "instrument_count": 0,
            "snapshot_count": 1,
            "market_cursors": [
                {
                    "instrument_id": "AAPL@alpaca",
                    "total_count": market["bar_count"],
                    "digest": market["bars_digest"],
                    "latest_at": last_bar["at"],
                }
            ],
            "latest_at": late.isoformat(),
            "last_event_key": None,
            "ledger": checkpoint["ledger"],
        }
    )
    checkpoint["state_digest"] = state_digest
    rows[-1] = replace(rows[-1], payload=checkpoint, occurred_at=late)
    forged_head = SessionHead("split-submit", len(rows), 1, state_digest)

    with pytest.raises(ValueError, match="invalid durable paper stream"):
        PaperBroker.rehydrate(
            tuple(rows),
            audit_log=audit,
            expected_head=forged_head,
        )
