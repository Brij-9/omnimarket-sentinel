"""Deterministic, credential-free current-session paper execution."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from threading import RLock
from types import MappingProxyType
from typing import cast

from market_sentinel.backtest.engine import FillModel
from market_sentinel.domain.enums import AssetClass, OrderStatus, OrderType, Side
from market_sentinel.domain.models import (
    Bar,
    BrokerOrder,
    Fill,
    Instrument,
    MarketSnapshot,
    OrderIntent,
    Position,
)
from market_sentinel.execution.base import AuditRecorder, BrokerCapabilities
from market_sentinel.execution.state_machine import (
    OrderStateMachine,
    OrderTransitionEvent,
)
from market_sentinel.portfolio.ledger import PortfolioLedger


class DuplicateIntentConflict(ValueError):
    """Raised when one stable client ID names two different canonical intents."""


@dataclass(frozen=True, slots=True)
class PaperAuditEvent:
    """Safe immutable current-session evidence for paper order activity."""

    event_id: str
    kind: str
    client_intent_id: str
    broker_order_id: str
    occurred_at: datetime
    prior_status: OrderStatus | None = None
    new_status: OrderStatus | None = None
    payload: Mapping[str, object] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class _PaperOrderRecord:
    intent: OrderIntent
    order: BrokerOrder
    remaining_notional: Decimal | None
    stop_triggered: bool = False


@dataclass(frozen=True, slots=True)
class _AuditSpec:
    kind: str
    client_intent_id: str
    broker_order_id: str
    occurred_at: datetime
    prior_status: OrderStatus | None
    new_status: OrderStatus | None
    payload: Mapping[str, object]


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")
_REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_TERMINAL = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
        OrderStatus.EXPIRED,
        OrderStatus.UNKNOWN,
    }
)
_PAPER_CAPABILITIES = BrokerCapabilities(
    broker="paper",
    supported_asset_classes=frozenset(
        {AssetClass.EQUITY, AssetClass.CRYPTO_SPOT, AssetClass.FOREX, AssetClass.COMMODITY}
    ),
    supported_order_types=frozenset(OrderType),
    supports_fractional_quantity=True,
    supports_notional_orders=True,
    supports_partial_fills=True,
    supports_shorting=False,
    supports_leverage=False,
    supports_derivatives=False,
    supports_cancel=True,
    is_paper=True,
)


class PaperBroker:
    """Keep strict durable in-memory state and fill only unseen market events.

    ``OrderIntent`` has no dedicated stop-entry field. For ``STOP`` and
    ``STOP_LIMIT`` only, ``stop_loss`` is therefore the required entry trigger.
    For ``MARKET`` and ``LIMIT`` it remains protective metadata and is never used
    as an entry trigger.
    """

    broker_name = "paper"

    def __init__(
        self,
        *,
        fill_model: FillModel | None = None,
        starting_cash: Decimal = Decimal("10"),
        currency: str = "USD",
        max_volume_participation: Decimal = Decimal("1"),
        audit_log: AuditRecorder | None = None,
    ) -> None:
        _positive_decimal(starting_cash, "starting_cash")
        participation = _positive_decimal(
            max_volume_participation, "max_volume_participation"
        )
        if participation > Decimal("1"):
            raise ValueError("max_volume_participation must be at most 1")
        if not isinstance(currency, str) or not currency:
            raise ValueError("currency must be nonempty")
        self._currency = currency
        self._fill_model = FillModel() if fill_model is None else fill_model
        if not isinstance(self._fill_model, FillModel):
            raise ValueError("fill_model must be a FillModel")
        self._max_volume_participation = participation
        self._audit_log = audit_log
        self._orders: dict[str, _PaperOrderRecord] = {}
        self._client_orders: dict[str, str] = {}
        self._snapshots: dict[str, MarketSnapshot] = {}
        self._fills: list[Fill] = []
        self._audit_events: list[PaperAuditEvent] = []
        self._event_sequence = 0
        self._fill_sequence = 0
        self._ledger = PortfolioLedger(starting_cash=starting_cash, currency=currency)
        self._market_prices: dict[str, Decimal] = {}
        self._latest_at: datetime | None = None
        self._lock = RLock()

    def capabilities(self) -> BrokerCapabilities:
        return _PAPER_CAPABILITIES

    @property
    def order_count(self) -> int:
        with self._lock:
            return len(self._orders)

    @property
    def fills(self) -> tuple[Fill, ...]:
        with self._lock:
            return tuple(self._fills)

    @property
    def audit_events(self) -> tuple[PaperAuditEvent, ...]:
        with self._lock:
            return tuple(self._audit_events)

    @property
    def cash(self) -> Decimal:
        with self._lock:
            return self._ledger.cash

    def submit(self, intent: OrderIntent, snapshot: MarketSnapshot) -> BrokerOrder:
        """Acknowledge one canonical intent without filling its current snapshot."""
        with self._lock:
            _validate_intent(intent)
            _validate_snapshot(snapshot, expected_instrument_id=intent.instrument_id)
            submitted_at = snapshot.observed_at.astimezone(UTC)
            if intent.created_at.astimezone(UTC) > submitted_at:
                raise ValueError("intent creation must not be after submission snapshot")
            if intent.expires_at.astimezone(UTC) <= submitted_at:
                raise ValueError("intent must remain unexpired at submission")

            duplicate_order_id = self._client_orders.get(intent.intent_id)
            if duplicate_order_id is not None:
                existing = self._orders[duplicate_order_id]
                audit_at = max(existing.order.updated_at, submitted_at)
                if existing.intent != intent:
                    self._audit(
                        kind="paper.order.duplicate_conflict",
                        client_intent_id=intent.intent_id,
                        broker_order_id=duplicate_order_id,
                        at=audit_at,
                        prior_status=existing.order.status,
                        new_status=existing.order.status,
                        payload={"outcome": "CONFLICT"},
                    )
                    raise DuplicateIntentConflict(
                        "client intent ID already belongs to a different canonical intent"
                    )
                self._audit(
                    kind="paper.order.duplicate",
                    client_intent_id=intent.intent_id,
                    broker_order_id=duplicate_order_id,
                    at=audit_at,
                    prior_status=existing.order.status,
                    new_status=existing.order.status,
                    payload={"outcome": "IDEMPOTENT_REPLAY"},
                )
                return existing.order

            previous_snapshot = self._snapshots.get(intent.instrument_id)
            if previous_snapshot is not None and previous_snapshot != snapshot:
                raise ValueError("submit snapshot revision requires on_snapshot processing first")

            order_id = _paper_order_id(intent.intent_id)
            order = BrokerOrder(
                order_id=order_id,
                client_order_id=intent.intent_id,
                broker=self.broker_name,
                instrument_id=intent.instrument_id,
                status=OrderStatus.PROPOSED,
                requested_quantity=intent.quantity,
                filled_quantity=Decimal("0"),
                average_fill_price=None,
                submitted_at=submitted_at,
                updated_at=submitted_at,
            )
            lifecycle_specs = [
                _AuditSpec(
                    kind="paper.order.submitted",
                    client_intent_id=intent.intent_id,
                    broker_order_id=order_id,
                    occurred_at=submitted_at,
                    prior_status=None,
                    new_status=OrderStatus.PROPOSED,
                    payload=MappingProxyType(
                        {
                            "instrument_id": intent.instrument_id,
                            "side": intent.side.value,
                            "order_type": intent.order_type.value,
                        }
                    ),
                )
            ]
            for target in (
                OrderStatus.RISK_APPROVED,
                OrderStatus.CONFIRMED,
                OrderStatus.SUBMITTING,
                OrderStatus.ACKNOWLEDGED,
            ):
                staged: list[OrderTransitionEvent] = []
                order = OrderStateMachine.transition(
                    order,
                    target,
                    at=submitted_at,
                    emit=staged.append,
                )
                [transition] = staged
                lifecycle_specs.append(_transition_audit_spec(transition))
            self._audit_many(tuple(lifecycle_specs))
            record = _PaperOrderRecord(
                intent=intent,
                order=order,
                remaining_notional=intent.notional,
            )
            self._orders[order_id] = record
            self._client_orders[intent.intent_id] = order_id
            self._snapshots[intent.instrument_id] = snapshot
            self._latest_at = submitted_at if self._latest_at is None else max(
                self._latest_at, submitted_at
            )
            return order

    def on_snapshot(
        self,
        snapshot: MarketSnapshot,
        instrument: Instrument,
    ) -> tuple[Fill, ...]:
        """Process each strictly unseen bar once in timestamp and order-ID order."""
        with self._lock:
            instrument_id = _instrument_id(instrument)
            if snapshot.instrument_id != instrument_id:
                raise ValueError("snapshot and instrument identity must match")
            if instrument.quote_currency != self._currency:
                raise ValueError("instrument quote currency must match the paper cash ledger")
            previous = self._snapshots.get(instrument_id)
            if previous is not None and snapshot == previous:
                self._audit(
                    kind="paper.snapshot.duplicate",
                    client_intent_id=instrument_id,
                    broker_order_id=instrument_id,
                    at=snapshot.observed_at,
                    prior_status=None,
                    new_status=None,
                    payload={"outcome": "IGNORED"},
                )
                return ()
            if previous is not None and (
                len(snapshot.bars) <= len(previous.bars)
                or snapshot.bars[: len(previous.bars)] != previous.bars
            ):
                raise ValueError("snapshot revision or backward event is not allowed")
            _validate_snapshot(snapshot, expected_instrument_id=instrument_id)
            if instrument.asset_class not in _PAPER_CAPABILITIES.supported_asset_classes:
                raise ValueError("instrument asset class is unsupported by paper execution")
            new_bars = snapshot.bars if previous is None else snapshot.bars[len(previous.bars) :]
            if previous is not None and len(new_bars) != 1:
                raise ValueError("paper snapshots must append exactly one unseen market event")
            produced: list[Fill] = []
            for bar in new_bars:
                self._market_prices[instrument_id] = bar.close
                for order_id in sorted(self._orders):
                    record = self._orders[order_id]
                    if record.order.instrument_id != instrument_id:
                        continue
                    updated, fill = self._process_event(record, bar, instrument)
                    self._orders[order_id] = updated
                    if fill is not None:
                        produced.append(fill)
                if self._latest_at is None or bar.at > self._latest_at:
                    self._latest_at = bar.at
                if self._market_prices:
                    self._ledger.mark(self._market_prices, bar.at)
            self._snapshots[instrument_id] = snapshot
            return tuple(produced)

    def get_order(self, order_id: str) -> BrokerOrder:
        with self._lock:
            try:
                return self._orders[order_id].order
            except KeyError as error:
                raise KeyError("paper order not found") from error

    def get_order_by_client_id(self, client_intent_id: str) -> BrokerOrder:
        with self._lock:
            try:
                return self._orders[self._client_orders[client_intent_id]].order
            except KeyError as error:
                raise KeyError("paper client intent not found") from error

    def positions(self) -> tuple[Position, ...]:
        with self._lock:
            if self._latest_at is None:
                return ()
            return self._ledger.snapshot(self._latest_at).positions

    def cancel(self, order_id: str, *, at: datetime) -> BrokerOrder:
        return self._resolve(order_id, OrderStatus.CANCELLED, at=at)

    def reject(self, order_id: str, *, at: datetime, reason_code: str) -> BrokerOrder:
        if not isinstance(reason_code, str) or _REASON_CODE.fullmatch(reason_code) is None:
            raise ValueError("reason_code must be a safe stable code")
        with self._lock:
            record = self._orders.get(order_id)
            if record is None:
                raise KeyError("paper order not found")
            staged: list[OrderTransitionEvent] = []
            updated = OrderStateMachine.transition(
                record.order,
                OrderStatus.REJECTED,
                at=at,
                emit=staged.append,
            )
            [transition] = staged
            self._audit_many(
                (
                    _AuditSpec(
                        kind="paper.order.rejected",
                        client_intent_id=record.order.client_order_id,
                        broker_order_id=record.order.order_id,
                        occurred_at=_aware_utc(at, "audit timestamp"),
                        prior_status=record.order.status,
                        new_status=OrderStatus.REJECTED,
                        payload=MappingProxyType({"reason_code": reason_code}),
                    ),
                    _transition_audit_spec(transition),
                )
            )
            self._orders[order_id] = replace(record, order=updated)
            return updated

    def expire(self, order_id: str, *, at: datetime) -> BrokerOrder:
        return self._resolve(order_id, OrderStatus.EXPIRED, at=at)

    def mark_unknown(self, order_id: str, *, at: datetime) -> BrokerOrder:
        return self._resolve(order_id, OrderStatus.UNKNOWN, at=at)

    def _resolve(
        self,
        order_id: str,
        status: OrderStatus,
        *,
        at: datetime,
    ) -> BrokerOrder:
        with self._lock:
            record = self._orders.get(order_id)
            if record is None:
                raise KeyError("paper order not found")
            updated = OrderStateMachine.transition(
                record.order,
                status,
                at=at,
                emit=self._emit_transition,
            )
            self._orders[order_id] = replace(record, order=updated)
            return updated

    def _process_event(
        self,
        record: _PaperOrderRecord,
        bar: Bar,
        instrument: Instrument,
    ) -> tuple[_PaperOrderRecord, Fill | None]:
        order = record.order
        if order.status in _TERMINAL or bar.at.astimezone(UTC) <= order.submitted_at:
            return record, None
        if bar.at.astimezone(UTC) >= record.intent.expires_at.astimezone(UTC):
            expired = OrderStateMachine.transition(
                order,
                OrderStatus.EXPIRED,
                at=bar.at,
                emit=self._emit_transition,
            )
            return replace(record, order=expired), None
        if not self._fill_model.can_fill(
            submitted_at=order.submitted_at,
            event_at=bar.at,
        ):
            return record, None

        reference, triggered_only = _reference_price(record, bar)
        if triggered_only:
            triggered = replace(record, stop_triggered=True)
            self._audit(
                kind="paper.order.stop_triggered",
                client_intent_id=order.client_order_id,
                broker_order_id=order.order_id,
                at=bar.at,
                prior_status=order.status,
                new_status=order.status,
                payload={"trigger_price": _decimal_text(cast(Decimal, record.intent.stop_loss))},
            )
            return triggered, None
        if reference is None:
            return record, None

        quantity = _executable_quantity(
            record=record,
            reference_price=reference,
            event_volume=bar.volume,
            instrument=instrument,
            max_volume_participation=self._max_volume_participation,
        )
        if quantity is None:
            return record, None
        fill_id = f"paper-fill-{self._fill_sequence + 1}"
        candidate = self._fill_model.fill(
            fill_id=fill_id,
            order_id=order.order_id,
            instrument=instrument,
            side=record.intent.side,
            quantity=quantity,
            reference_price=reference,
            submitted_at=order.submitted_at,
            filled_at=bar.at,
        )
        if record.remaining_notional is not None:
            maximum_quantity = _floor_to_step(
                record.remaining_notional / candidate.price,
                instrument.quantity_step,
            )
            if maximum_quantity <= Decimal("0"):
                return record, None
            if maximum_quantity < candidate.quantity:
                candidate = self._fill_model.fill(
                    fill_id=fill_id,
                    order_id=order.order_id,
                    instrument=instrument,
                    side=record.intent.side,
                    quantity=maximum_quantity,
                    reference_price=reference,
                    submitted_at=order.submitted_at,
                    filled_at=bar.at,
                )
        if not _respects_order_prices(record.intent, candidate.price):
            return record, None
        notional = candidate.quantity * candidate.price
        requested_value = (
            record.intent.notional
            if record.intent.notional is not None
            else cast(Decimal, record.intent.quantity) * reference
        )
        if requested_value < instrument.minimum_notional:
            return record, None
        if record.remaining_notional is not None and notional > record.remaining_notional:
            return record, None
        if record.intent.side is Side.BUY:
            if notional + candidate.fee > self._ledger.cash:
                return record, None
        else:
            held = next(
                (
                    position.quantity
                    for position in self.positions()
                    if position.instrument_id == order.instrument_id
                ),
                Decimal("0"),
            )
            if candidate.quantity > held:
                return record, None

        old_filled = order.filled_quantity
        new_filled = old_filled + candidate.quantity
        if order.requested_quantity is not None and new_filled > order.requested_quantity:
            raise RuntimeError("paper fill would exceed requested quantity")
        remaining_notional = record.remaining_notional
        notional_complete = False
        if remaining_notional is not None:
            remaining_notional -= notional
            minimum_next = instrument.quantity_step * candidate.price
            if remaining_notional < minimum_next:
                remaining_notional = Decimal("0")
                notional_complete = True
        quantity_complete = (
            order.requested_quantity is not None and new_filled == order.requested_quantity
        )
        target = (
            OrderStatus.FILLED
            if quantity_complete or notional_complete
            else OrderStatus.PARTIALLY_FILLED
        )
        staged_transitions: list[OrderTransitionEvent] = []
        transitioned = OrderStateMachine.transition(
            order,
            target,
            at=bar.at,
            emit=staged_transitions.append,
        )
        weighted_price = (
            candidate.price
            if old_filled == Decimal("0")
            else (
                cast(Decimal, order.average_fill_price) * old_filled
                + candidate.price * candidate.quantity
            )
            / new_filled
        )
        completed_order = transitioned.model_copy(
            update={
                "filled_quantity": new_filled,
                "average_fill_price": weighted_price,
                "updated_at": bar.at.astimezone(UTC),
            }
        )
        remaining_quantity = (
            order.requested_quantity - new_filled
            if order.requested_quantity is not None
            else Decimal("0")
        )
        [transition] = staged_transitions
        self._audit_many(
            (
                _transition_audit_spec(transition),
                _AuditSpec(
                    kind="paper.order.fill",
                    client_intent_id=order.client_order_id,
                    broker_order_id=order.order_id,
                    occurred_at=bar.at.astimezone(UTC),
                    prior_status=order.status,
                    new_status=target,
                    payload=MappingProxyType(
                        {
                            "fill_id": candidate.fill_id,
                            "quantity": _decimal_text(candidate.quantity),
                            "price": _decimal_text(candidate.price),
                            "fee": _decimal_text(candidate.fee),
                            "cumulative_filled_quantity": _decimal_text(new_filled),
                            "remaining_quantity": _decimal_text(remaining_quantity),
                        }
                    ),
                ),
            )
        )
        self._ledger.apply_fill(candidate)
        self._fill_sequence += 1
        self._fills.append(candidate)
        return (
            replace(
                record,
                order=completed_order,
                remaining_notional=remaining_notional,
            ),
            candidate,
        )

    def _emit_transition(self, event: OrderTransitionEvent) -> None:
        self._audit_many((_transition_audit_spec(event),))

    def _audit(
        self,
        *,
        kind: str,
        client_intent_id: str,
        broker_order_id: str,
        at: datetime,
        prior_status: OrderStatus | None,
        new_status: OrderStatus | None,
        payload: Mapping[str, object],
    ) -> None:
        self._audit_many(
            (
                _AuditSpec(
                    kind=kind,
                    client_intent_id=client_intent_id,
                    broker_order_id=broker_order_id,
                    occurred_at=_aware_utc(at, "audit timestamp"),
                    prior_status=prior_status,
                    new_status=new_status,
                    payload=MappingProxyType(dict(payload)),
                ),
            )
        )

    def _audit_many(self, specs: tuple[_AuditSpec, ...]) -> None:
        prepared: list[tuple[PaperAuditEvent, dict[str, object]]] = []
        for offset, spec in enumerate(specs, start=1):
            event_id = f"paper-event-{self._event_sequence + offset}"
            safe_payload = {
                "client_intent_id": spec.client_intent_id,
                "broker_order_id": spec.broker_order_id,
                "occurred_at": spec.occurred_at.isoformat(),
                **dict(spec.payload),
            }
            if spec.prior_status is not None:
                safe_payload["prior_status"] = spec.prior_status.value
            if spec.new_status is not None:
                safe_payload["new_status"] = spec.new_status.value
            prepared.append(
                (
                    PaperAuditEvent(
                        event_id=event_id,
                        kind=spec.kind,
                        client_intent_id=spec.client_intent_id,
                        broker_order_id=spec.broker_order_id,
                        occurred_at=spec.occurred_at,
                        prior_status=spec.prior_status,
                        new_status=spec.new_status,
                        payload=MappingProxyType(dict(spec.payload)),
                    ),
                    safe_payload,
                )
            )
        if self._audit_log is not None:
            if len(prepared) == 1:
                event, safe_payload = prepared[0]
                self._audit_log.record(
                    event.event_id,
                    event.kind,
                    event.broker_order_id,
                    safe_payload,
                )
            else:
                first_event = prepared[0][0]
                last_event = prepared[-1][0]
                self._audit_log.record(
                    f"paper-event-batch-{first_event.event_id}-{last_event.event_id}",
                    "paper.audit.batch",
                    last_event.broker_order_id,
                    {
                        "events": [
                            {"event_id": event.event_id, "kind": event.kind, **payload}
                            for event, payload in prepared
                        ]
                    },
                )
        self._event_sequence += len(prepared)
        self._audit_events.extend(event for event, _ in prepared)


def _transition_audit_spec(event: OrderTransitionEvent) -> _AuditSpec:
    return _AuditSpec(
        kind="paper.order.transition",
        client_intent_id=event.client_intent_id,
        broker_order_id=event.broker_order_id,
        occurred_at=event.occurred_at,
        prior_status=event.prior_status,
        new_status=event.new_status,
        payload=MappingProxyType(
            {
                "prior_status": event.prior_status.value,
                "new_status": event.new_status.value,
            }
        ),
    )


def _validate_intent(intent: object) -> None:
    if not isinstance(intent, OrderIntent):
        raise ValueError("intent must be an OrderIntent")
    for identifier, name in (
        (intent.intent_id, "intent_id"),
        (intent.instrument_id, "instrument_id"),
    ):
        if not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None:
            raise ValueError(f"{name} must be a safe stable identifier")
    if (intent.quantity is None) == (intent.notional is None):
        raise ValueError("exactly one order size must be populated")
    if not isinstance(intent.side, Side) or not isinstance(intent.order_type, OrderType):
        raise ValueError("side and order_type must be canonical enum values")
    if intent.quantity is not None:
        _positive_decimal(intent.quantity, "quantity")
    if intent.notional is not None:
        _positive_decimal(intent.notional, "notional")
    for price, name in (
        (intent.limit_price, "limit_price"),
        (intent.stop_loss, "stop_loss"),
        (intent.take_profit, "take_profit"),
    ):
        if price is not None:
            _positive_decimal(price, name)
    if intent.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and intent.limit_price is None:
        raise ValueError("limit and stop-limit orders require limit_price")
    if intent.order_type in {OrderType.STOP, OrderType.STOP_LIMIT} and intent.stop_loss is None:
        raise ValueError("stop and stop-limit orders require stop_loss as trigger")
    created_at = _aware_utc(intent.created_at, "intent created_at")
    expires_at = _aware_utc(intent.expires_at, "intent expires_at")
    if expires_at <= created_at:
        raise ValueError("intent expires_at must follow created_at")


def _validate_snapshot(snapshot: object, *, expected_instrument_id: str) -> None:
    if not isinstance(snapshot, MarketSnapshot):
        raise ValueError("snapshot must be a MarketSnapshot")
    if snapshot.instrument_id != expected_instrument_id:
        raise ValueError("snapshot instrument identity must match the order")
    observed_at = _aware_utc(snapshot.observed_at, "snapshot observed_at")
    source_at = _aware_utc(snapshot.source_at, "snapshot source_at")
    if source_at > observed_at:
        raise ValueError("snapshot source timestamp cannot be in the future")
    if (observed_at - source_at).total_seconds() > snapshot.max_age_seconds:
        raise ValueError("snapshot is stale at observation time")
    if not isinstance(snapshot.bars, tuple) or not snapshot.bars:
        raise ValueError("snapshot bars must be a nonempty tuple")
    previous_at: datetime | None = None
    for bar in snapshot.bars:
        if not isinstance(bar, Bar):
            raise ValueError("snapshot bars must contain Bar records")
        at = _aware_utc(bar.at, "bar timestamp")
        if previous_at is not None and at <= previous_at:
            raise ValueError("snapshot bars must be strictly chronological")
        if at > source_at:
            raise ValueError("bar timestamp cannot be after snapshot source timestamp")
        for price in (bar.open, bar.high, bar.low, bar.close):
            _positive_decimal(price, "bar price")
        if not bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high:
            raise ValueError("snapshot bar OHLC values are inconsistent")
        _nonnegative_decimal(bar.volume, "bar volume")
        previous_at = at
    if snapshot.bars[-1].at.astimezone(UTC) != source_at:
        raise ValueError("snapshot source timestamp must equal its latest bar")


def _reference_price(
    record: _PaperOrderRecord,
    bar: Bar,
) -> tuple[Decimal | None, bool]:
    intent = record.intent
    if intent.order_type is OrderType.MARKET:
        return bar.open, False
    if intent.order_type is OrderType.LIMIT:
        return _limit_reference(intent, bar), False
    trigger = cast(Decimal, intent.stop_loss)
    crossed = bar.high >= trigger if intent.side is Side.BUY else bar.low <= trigger
    if intent.order_type is OrderType.STOP:
        if not crossed:
            return None, False
        return (
            max(bar.open, trigger) if intent.side is Side.BUY else min(bar.open, trigger),
            False,
        )
    if not record.stop_triggered:
        return (None, True) if crossed else (None, False)
    return _limit_reference(intent, bar), False


def _limit_reference(intent: OrderIntent, bar: Bar) -> Decimal | None:
    limit = cast(Decimal, intent.limit_price)
    if intent.side is Side.BUY:
        return min(bar.open, limit) if bar.low <= limit else None
    return max(bar.open, limit) if bar.high >= limit else None


def _respects_order_prices(intent: OrderIntent, fill_price: Decimal) -> bool:
    if intent.limit_price is not None:
        if intent.side is Side.BUY and fill_price > intent.limit_price:
            return False
        if intent.side is Side.SELL and fill_price < intent.limit_price:
            return False
    if intent.stop_loss is not None and intent.take_profit is not None:
        if intent.order_type in {OrderType.STOP, OrderType.STOP_LIMIT}:
            if intent.side is Side.BUY:
                return intent.stop_loss <= fill_price < intent.take_profit
            return intent.stop_loss >= fill_price > intent.take_profit
        if intent.side is Side.BUY:
            return intent.stop_loss < fill_price < intent.take_profit
        return intent.stop_loss > fill_price > intent.take_profit
    return True


def _executable_quantity(
    *,
    record: _PaperOrderRecord,
    reference_price: Decimal,
    event_volume: Decimal,
    instrument: Instrument,
    max_volume_participation: Decimal,
) -> Decimal | None:
    step = _positive_decimal(instrument.quantity_step, "instrument quantity step")
    _positive_decimal(instrument.price_tick, "instrument price tick")
    _positive_decimal(instrument.minimum_notional, "instrument minimum notional")
    liquidity = _floor_to_step(event_volume * max_volume_participation, step)
    if liquidity <= Decimal("0"):
        return None
    if record.order.requested_quantity is not None:
        requested = _positive_decimal(record.order.requested_quantity, "requested quantity")
        if _floor_to_step(requested, step) != requested:
            raise ValueError("requested quantity must align with instrument quantity step")
        remaining = requested - record.order.filled_quantity
    else:
        remaining_notional = cast(Decimal, record.remaining_notional)
        remaining = _floor_to_step(remaining_notional / reference_price, step)
    quantity = min(remaining, liquidity)
    return quantity if quantity > Decimal("0") else None


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    try:
        return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step
    except (InvalidOperation, OverflowError, ZeroDivisionError) as error:
        raise ValueError("quantity step arithmetic must remain finite") from error


def _instrument_id(instrument: object) -> str:
    if not isinstance(instrument, Instrument):
        raise ValueError("instrument must be an Instrument")
    return f"{instrument.symbol}@{instrument.venue}"


def _paper_order_id(intent_id: str) -> str:
    digest = hashlib.sha256(intent_id.encode("utf-8")).hexdigest()[:24]
    return f"paper-{digest}"


def _positive_decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= Decimal("0"):
        raise ValueError(f"{name} must be a finite positive Decimal")
    return value


def _nonnegative_decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value < Decimal("0"):
        raise ValueError(f"{name} must be a finite nonnegative Decimal")
    return value


def _aware_utc(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")
