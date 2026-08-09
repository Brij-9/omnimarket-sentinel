"""Deterministic paper-broker tests over real FillModel behavior."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from market_sentinel.backtest.engine import CostModel, FillModel
from market_sentinel.domain.clock import FrozenClock
from market_sentinel.domain.enums import AssetClass, OrderStatus, OrderType, Side
from market_sentinel.domain.models import Bar, Instrument, MarketSnapshot, OrderIntent
from market_sentinel.execution.base import BrokerAdapter, BrokerCapabilities
from market_sentinel.execution.paper import DuplicateIntentConflict, PaperBroker
from market_sentinel.execution.state_machine import InvalidOrderTransition
from market_sentinel.operations.audit import AuditEvent, AuditLog
from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventStore

AT = datetime(2026, 8, 9, 10, tzinfo=UTC)


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
) -> MarketSnapshot:
    latest = bars[-1].at
    return MarketSnapshot(
        instrument_id=instrument_id,
        observed_at=latest if observed_at is None else observed_at,
        source_at=latest if source_at is None else source_at,
        bars=tuple(bars),
        provider="fixture",
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
        time_in_force="day",
        product="cash",
        session="regular",
        snapshot_hash="snapshot-hash",
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
    values = {
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
    assert fill_events[-1].payload["cumulative_filled_notional"] == "18.0"
    assert fill_events[-1].payload["remaining_notional"] == "1.0"
    assert fill_events[-1].payload["cumulative_fees"] == "0.0"


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


def test_global_event_regression_rejects_without_mutation() -> None:
    """A past cross-instrument event must not value prior cash with future prices."""
    aapl = _instrument(instrument_id="AAPL@alpaca")
    msft = _instrument(instrument_id="MSFT@alpaca")
    aapl0 = _snapshot(_bar(0, open_price="100"), instrument_id="AAPL@alpaca")
    msft0 = _snapshot(_bar(0, open_price="100"), instrument_id="MSFT@alpaca")
    broker = PaperBroker(starting_cash=Decimal("1000"))
    aapl_order = broker.submit(_intent(intent_id="aapl", instrument_id="AAPL@alpaca"), aapl0)
    broker.submit(_intent(intent_id="msft", instrument_id="MSFT@alpaca"), msft0)
    later = _snapshot(
        *msft0.bars,
        _bar(2, open_price="100"),
        instrument_id="MSFT@alpaca",
    )
    broker.on_snapshot(later, msft)
    before = (broker.list_orders(), broker.fills, broker.cash, broker.audit_events)
    earlier = _snapshot(
        *aapl0.bars,
        _bar(1, open_price="100"),
        instrument_id="AAPL@alpaca",
    )

    with pytest.raises(ValueError, match="global chronology"):
        broker.on_snapshot(earlier, aapl)

    assert (broker.list_orders(), broker.fills, broker.cash, broker.audit_events) == before
    assert broker.get_order(aapl_order.order_id).status is OrderStatus.ACKNOWLEDGED


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
    """Durable persistence failure must precede every in-memory event commit."""
    bar0 = _bar(0, open_price="100")
    recorder = _RecordManyFailure(fail_on=2)
    broker = PaperBroker(
        starting_cash=Decimal("1000"),
        audit_log=recorder,
        durable=True,
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
        (1, "paper.order.submitted", AT),
        (2, "paper.order.transition", AT),
        (3, "paper.order.transition", AT),
        (4, "paper.order.transition", AT),
        (5, "paper.order.transition", AT),
        (6, "paper.state.committed", AT),
        (7, "paper.order.transition", AT + timedelta(minutes=1)),
        (8, "paper.order.fill", AT + timedelta(minutes=1)),
        (9, "paper.state.committed", AT + timedelta(minutes=1)),
    ]
    assert all(row.occurred_at != clock.now() for row in rows)
    assert len({row.event_id for row in rows}) == len(rows)
    restored = PaperBroker.rehydrate(
        rows,
        audit_log=audit,
        starting_cash=Decimal("1000"),
        session_id="durable-session",
    )

    assert restored.get_order(order.order_id) == broker.get_order(order.order_id)
    assert restored.fills == broker.fills
    assert restored.cash == broker.cash
    assert restored.submit(value, _snapshot(bar0)) == restored.get_order(order.order_id)
    assert restored.on_snapshot(event, _instrument()) == ()


def test_two_durable_brokers_never_collide_event_or_fill_ids(tmp_path: Path) -> None:
    """Per-instance counters alone collide when sessions share one durable store."""
    store = EventStore(create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'shared.db'}"))
    audit = AuditLog(store, FrozenClock(AT))
    bar0 = _bar(0, open_price="100")
    fill_ids = []
    event_ids = []
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


def test_rehydrate_uses_append_sequence_when_event_times_arrive_out_of_order(
    tmp_path: Path,
) -> None:
    """A late broker update with an older source time must still survive replay."""
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
        _intent(intent_id="aapl", instrument_id="AAPL@alpaca"),
        aapl0,
    )
    broker.submit(_intent(intent_id="msft", instrument_id="MSFT@alpaca"), msft0)
    broker.on_snapshot(
        _snapshot(
            *msft0.bars,
            _bar(2, open_price="100"),
            instrument_id="MSFT@alpaca",
        ),
        _instrument(instrument_id="MSFT@alpaca"),
    )
    cancelled = broker.cancel(aapl_order.order_id, at=AT + timedelta(minutes=1))

    rows = tuple(store.stream("late-update-session"))
    restored = PaperBroker.rehydrate(
        rows,
        audit_log=audit,
        starting_cash=Decimal("1000"),
        session_id="late-update-session",
    )

    assert restored.get_order(aapl_order.order_id) == cancelled


def test_runtime_durability_semantics_are_explicit_and_validated() -> None:
    """A default in-memory broker must not be mislabeled crash-durable."""
    assert PaperBroker().durability_mode == "current_session"
    with pytest.raises(ValueError, match="durable"):
        PaperBroker(durable=True, session_id="runtime-session")
    with pytest.raises(ValueError, match="session_id"):
        PaperBroker(durable=True, audit_log=_RecordManyFailure(fail_on=99))
    with pytest.raises(ValueError, match="record_many"):
        PaperBroker(
            durable=True,
            audit_log=object(),  # type: ignore[arg-type]
            session_id="runtime-session",
        )
