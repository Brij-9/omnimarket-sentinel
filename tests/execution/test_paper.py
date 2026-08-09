"""Deterministic paper-broker tests over real FillModel behavior."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from market_sentinel.backtest.engine import CostModel, FillModel
from market_sentinel.domain.enums import AssetClass, OrderStatus, OrderType, Side
from market_sentinel.domain.models import Bar, Instrument, MarketSnapshot, OrderIntent
from market_sentinel.execution.base import BrokerAdapter
from market_sentinel.execution.paper import DuplicateIntentConflict, PaperBroker
from market_sentinel.execution.state_machine import InvalidOrderTransition

AT = datetime(2026, 8, 9, 10, tzinfo=UTC)


class _FailingAudit:
    """Exercise broker atomicity when durable audit persistence is unavailable."""

    def __init__(self, failing_kind: str, *, occurrence: int = 1) -> None:
        self.failing_kind = failing_kind
        self.occurrence = occurrence
        self._seen = 0

    def record(
        self,
        event_id: str,
        kind: str,
        aggregate_id: str,
        payload: dict[str, object],
    ) -> None:
        del event_id, aggregate_id, payload
        if kind == self.failing_kind:
            self._seen += 1
        if kind == self.failing_kind and self._seen == self.occurrence:
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
        [str(inspect.signature(PaperBroker)), *vars(broker), repr(broker)]
    ).lower()
    assert "api_key" not in public_surface
    assert "secret" not in public_surface
    assert "credential" not in public_surface


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
        _intent(order_type=OrderType.STOP, stop_loss=Decimal("105"), take_profit=Decimal("120")),
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
            stop_loss=Decimal("105"),
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
    for terminal in (cancelled, rejected, unknown, expired):
        with pytest.raises(InvalidOrderTransition):
            broker.cancel(terminal.order_id, at=AT + timedelta(seconds=2))
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
        _intent(order_type=OrderType.LIMIT, limit_price=None),
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
        audit_log=_FailingAudit("paper.audit.batch", occurrence=2),
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
        audit_log=_FailingAudit("paper.audit.batch"),
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
        "fill_id": "paper-fill-1",
        "quantity": "1",
        "price": "100",
        "fee": "0",
        "cumulative_filled_quantity": "1",
        "remaining_quantity": "0",
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
    assert broker.get_order(order.order_id).status is OrderStatus.FILLED
