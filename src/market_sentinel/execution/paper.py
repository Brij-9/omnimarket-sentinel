"""Deterministic, transactional, credential-free paper execution."""

from __future__ import annotations

import copy
import hashlib
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
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
from market_sentinel.execution.state_machine import OrderStateMachine, OrderTransitionEvent
from market_sentinel.operations.audit import AuditEvent
from market_sentinel.portfolio.ledger import PortfolioLedger
from market_sentinel.storage.events import EventRecord


class DuplicateIntentConflict(ValueError):
    """Raised when one stable client ID names two different canonical intents."""


@dataclass(frozen=True, slots=True)
class PaperAuditEvent:
    """Safe immutable evidence for one paper execution activity."""

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
    submitted_source_at: datetime
    remaining_notional: Decimal | None
    cumulative_filled_notional: Decimal = Decimal("0")
    cumulative_fees: Decimal = Decimal("0")
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


@dataclass(slots=True)
class _PaperState:
    orders: dict[str, _PaperOrderRecord]
    client_orders: dict[str, str]
    snapshots: dict[str, MarketSnapshot]
    fills: list[Fill]
    event_sequence: int
    fill_sequence: int
    ledger: PortfolioLedger
    market_prices: dict[str, Decimal]
    latest_at: datetime | None
    last_event_key: tuple[datetime, datetime, str] | None

    def clone(self) -> _PaperState:
        return _PaperState(
            orders=dict(self.orders),
            client_orders=dict(self.client_orders),
            snapshots=dict(self.snapshots),
            fills=list(self.fills),
            event_sequence=self.event_sequence,
            fill_sequence=self.fill_sequence,
            ledger=copy.deepcopy(self.ledger),
            market_prices=dict(self.market_prices),
            latest_at=self.latest_at,
            last_event_key=self.last_event_key,
        )


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")
_REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_CLOSED = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
        OrderStatus.EXPIRED,
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
    """Maintain an atomic current-session ledger, optionally backed by durable events.

    The default mode is explicitly process-local ``current_session`` state. Crash-durable
    operation requires ``durable=True``, a stable caller-supplied ``session_id``, and an
    ``AuditRecorder`` whose ``record_many`` is transactional. Each committed operation
    persists its first-class activity rows plus a replayable state event in one batch.
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
        durable: bool = False,
        session_id: str | None = None,
    ) -> None:
        initial_cash = _positive_decimal(starting_cash, "starting_cash")
        participation = _positive_decimal(
            max_volume_participation, "max_volume_participation"
        )
        if participation > Decimal("1"):
            raise ValueError("max_volume_participation must be at most 1")
        if not isinstance(currency, str) or not currency or currency != currency.strip():
            raise ValueError("currency must be nonempty and trimmed")
        if type(durable) is not bool:
            raise ValueError("durable must be an exact bool")
        if durable and audit_log is None:
            raise ValueError("durable paper execution requires an audit recorder")
        if audit_log is not None and not callable(getattr(audit_log, "record_many", None)):
            raise ValueError("audit recorder must expose transactional record_many")
        if durable and session_id is None:
            raise ValueError("durable paper execution requires a stable session_id")
        effective_session = uuid.uuid4().hex if session_id is None else session_id
        if (
            not isinstance(effective_session, str)
            or _IDENTIFIER.fullmatch(effective_session) is None
        ):
            raise ValueError("session_id must be a safe stable identifier")
        model = FillModel() if fill_model is None else fill_model
        if not isinstance(model, FillModel):
            raise ValueError("fill_model must be a FillModel")

        self._fill_model = model
        self._starting_cash = initial_cash
        self._currency = currency
        self._max_volume_participation = participation
        self._audit_log = audit_log
        self._durable = durable
        self._session_id = effective_session
        self._audit_events: list[PaperAuditEvent] = []
        self._state = _PaperState(
            orders={},
            client_orders={},
            snapshots={},
            fills=[],
            event_sequence=0,
            fill_sequence=0,
            ledger=PortfolioLedger(starting_cash=initial_cash, currency=currency),
            market_prices={},
            latest_at=None,
            last_event_key=None,
        )
        self._lock = RLock()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def durability_mode(self) -> str:
        return "durable" if self._durable else "current_session"

    def capabilities(self) -> BrokerCapabilities:
        return _PAPER_CAPABILITIES

    @property
    def order_count(self) -> int:
        with self._lock:
            return len(self._state.orders)

    @property
    def fills(self) -> tuple[Fill, ...]:
        with self._lock:
            return tuple(self._state.fills)

    @property
    def audit_events(self) -> tuple[PaperAuditEvent, ...]:
        with self._lock:
            return tuple(self._audit_events)

    @property
    def cash(self) -> Decimal:
        with self._lock:
            return self._state.ledger.cash

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
            staged = self._state.clone()
            specs: list[_AuditSpec] = []

            duplicate_order_id = staged.client_orders.get(intent.intent_id)
            if duplicate_order_id is not None:
                existing = staged.orders[duplicate_order_id]
                at = max(existing.order.updated_at, submitted_at)
                kind = (
                    "paper.order.duplicate"
                    if existing.intent == intent
                    else "paper.order.duplicate_conflict"
                )
                specs.append(
                    _activity_spec(
                        kind=kind,
                        order=existing.order,
                        at=at,
                        payload={
                            "outcome": (
                                "IDEMPOTENT_REPLAY"
                                if existing.intent == intent
                                else "CONFLICT"
                            )
                        },
                    )
                )
                self._commit(staged, specs, at=at)
                if existing.intent != intent:
                    raise DuplicateIntentConflict(
                        "client intent ID already belongs to a different canonical intent"
                    )
                return existing.order

            previous_snapshot = staged.snapshots.get(intent.instrument_id)
            if previous_snapshot is not None and previous_snapshot != snapshot:
                raise ValueError("submit snapshot revision requires on_snapshot processing first")

            order_id = _paper_order_id(self._session_id, intent.intent_id)
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
            specs.append(
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
                            "intent": _intent_payload(intent),
                        }
                    ),
                )
            )
            for target in (
                OrderStatus.RISK_APPROVED,
                OrderStatus.CONFIRMED,
                OrderStatus.SUBMITTING,
                OrderStatus.ACKNOWLEDGED,
            ):
                order, transition = _stage_transition(order, target, submitted_at)
                specs.append(_transition_spec(transition))
            staged.orders[order_id] = _PaperOrderRecord(
                intent=intent,
                order=order,
                submitted_source_at=snapshot.source_at.astimezone(UTC),
                remaining_notional=intent.notional,
            )
            staged.client_orders[intent.intent_id] = order_id
            staged.snapshots[intent.instrument_id] = snapshot
            staged.latest_at = (
                submitted_at
                if staged.latest_at is None
                else max(staged.latest_at, submitted_at)
            )
            self._commit(staged, specs, at=submitted_at)
            return order

    def on_snapshot(
        self,
        snapshot: MarketSnapshot,
        instrument: Instrument,
    ) -> tuple[Fill, ...]:
        return self.on_snapshots(((snapshot, instrument),))

    def on_snapshots(
        self,
        events: tuple[tuple[MarketSnapshot, Instrument], ...],
    ) -> tuple[Fill, ...]:
        """Atomically process a globally sorted batch of current market events."""
        with self._lock:
            if not isinstance(events, tuple) or not events:
                raise ValueError("snapshot batch must be a nonempty tuple")
            validated: list[
                tuple[tuple[datetime, datetime, str], MarketSnapshot, Instrument, bool]
            ] = []
            seen_instruments: set[str] = set()
            for item in events:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError("snapshot batch entries must contain snapshot and instrument")
                snapshot, instrument = item
                instrument_id = _instrument_id(instrument)
                if instrument_id in seen_instruments:
                    raise ValueError("snapshot batch must contain an instrument at most once")
                seen_instruments.add(instrument_id)
                if instrument.quote_currency != self._currency:
                    raise ValueError("instrument quote currency must match the paper cash ledger")
                if snapshot.instrument_id != instrument_id:
                    raise ValueError("snapshot and instrument identity must match")
                previous = self._state.snapshots.get(instrument_id)
                duplicate = previous is not None and snapshot == previous
                if not duplicate and previous is not None:
                    if len(snapshot.bars) > len(previous.bars) + 1:
                        raise ValueError("snapshot must contain exactly one unseen event")
                    if (
                        len(snapshot.bars) != len(previous.bars) + 1
                        or snapshot.bars[: len(previous.bars)] != previous.bars
                    ):
                        raise ValueError("snapshot revision or backward event is not allowed")
                _validate_snapshot(snapshot, expected_instrument_id=instrument_id)
                if instrument.asset_class not in _PAPER_CAPABILITIES.supported_asset_classes:
                    raise ValueError("instrument asset class is unsupported by paper execution")
                key = _event_key(snapshot)
                validated.append((key, snapshot, instrument, duplicate))

            validated.sort(key=lambda item: item[0])
            new_keys = [key for key, _, _, duplicate in validated if not duplicate]
            if new_keys:
                if (
                    self._state.last_event_key is not None
                    and new_keys[0] <= self._state.last_event_key
                ):
                    raise ValueError("global chronology regression is not allowed")
                if any(
                    current <= previous
                    for previous, current in zip(new_keys, new_keys[1:], strict=False)
                ):
                    raise ValueError("global chronology keys must be strictly increasing")

            staged = self._state.clone()
            specs: list[_AuditSpec] = []
            produced: list[Fill] = []
            commit_at = max(snapshot.observed_at for _, snapshot, _, _ in validated)
            for key, snapshot, instrument, duplicate in validated:
                if duplicate:
                    specs.append(
                        _AuditSpec(
                            kind="paper.snapshot.duplicate",
                            client_intent_id=snapshot.instrument_id,
                            broker_order_id=snapshot.instrument_id,
                            occurred_at=snapshot.observed_at,
                            prior_status=None,
                            new_status=None,
                            payload=MappingProxyType({"outcome": "IGNORED"}),
                        )
                    )
                    continue
                [bar] = snapshot.bars[len(staged.snapshots[snapshot.instrument_id].bars) :]
                liquidity = self._fill_model.liquidity_budget(
                    event_volume=bar.volume,
                    max_participation=self._max_volume_participation,
                    quantity_step=instrument.quantity_step,
                )
                staged.market_prices[snapshot.instrument_id] = bar.close
                for order_id in sorted(staged.orders):
                    record = staged.orders[order_id]
                    if record.order.instrument_id != snapshot.instrument_id:
                        continue
                    record, fill, liquidity, order_specs = self._process_order(
                        state=staged,
                        record=record,
                        snapshot=snapshot,
                        bar=bar,
                        instrument=instrument,
                        available_liquidity=liquidity,
                    )
                    staged.orders[order_id] = record
                    specs.extend(order_specs)
                    if fill is not None:
                        produced.append(fill)
                staged.snapshots[snapshot.instrument_id] = snapshot
                staged.latest_at = snapshot.observed_at.astimezone(UTC)
                staged.last_event_key = key

            if new_keys and staged.market_prices and staged.latest_at is not None:
                staged.ledger.mark(staged.market_prices, staged.latest_at)
            self._commit(staged, specs, at=commit_at)
            return tuple(produced)

    def get_order(self, order_id: str) -> BrokerOrder:
        with self._lock:
            try:
                return self._state.orders[order_id].order
            except KeyError as error:
                raise KeyError("paper order not found") from error

    def get_order_by_client_id(self, client_intent_id: str) -> BrokerOrder:
        with self._lock:
            try:
                order_id = self._state.client_orders[client_intent_id]
                return self._state.orders[order_id].order
            except KeyError as error:
                raise KeyError("paper client intent not found") from error

    def list_orders(self) -> tuple[BrokerOrder, ...]:
        with self._lock:
            return tuple(self._state.orders[key].order for key in sorted(self._state.orders))

    def open_orders(self) -> tuple[BrokerOrder, ...]:
        return tuple(order for order in self.list_orders() if order.status not in _CLOSED)

    def positions(self) -> tuple[Position, ...]:
        with self._lock:
            if self._state.latest_at is None:
                return ()
            return self._state.ledger.snapshot(self._state.latest_at).positions

    def cancel(self, order_id: str, *, at: datetime) -> BrokerOrder:
        return self._resolve(order_id, OrderStatus.CANCELLED, at=at)

    def reject(self, order_id: str, *, at: datetime, reason_code: str) -> BrokerOrder:
        if not isinstance(reason_code, str) or _REASON_CODE.fullmatch(reason_code) is None:
            raise ValueError("reason_code must be a safe stable code")
        return self._resolve(
            order_id,
            OrderStatus.REJECTED,
            at=at,
            activity=("paper.order.rejected", {"reason_code": reason_code}),
        )

    def expire(self, order_id: str, *, at: datetime) -> BrokerOrder:
        return self._resolve(order_id, OrderStatus.EXPIRED, at=at)

    def mark_unknown(self, order_id: str, *, at: datetime) -> BrokerOrder:
        return self._resolve(order_id, OrderStatus.UNKNOWN, at=at)

    def reconcile_unknown(
        self,
        order_id: str,
        broker_status: OrderStatus,
        *,
        at: datetime,
    ) -> BrokerOrder:
        with self._lock:
            record = self._state.orders.get(order_id)
            if record is None:
                raise KeyError("paper order not found")
            if record.order.status is not OrderStatus.UNKNOWN:
                raise ValueError("only UNKNOWN orders can be reconciled")
        return self._resolve(order_id, broker_status, at=at)

    def _resolve(
        self,
        order_id: str,
        status: OrderStatus,
        *,
        at: datetime,
        activity: tuple[str, Mapping[str, object]] | None = None,
    ) -> BrokerOrder:
        with self._lock:
            staged = self._state.clone()
            record = staged.orders.get(order_id)
            if record is None:
                raise KeyError("paper order not found")
            updated, transition = _stage_transition(record.order, status, at)
            specs: list[_AuditSpec] = []
            if activity is not None:
                specs.append(
                    _activity_spec(
                        kind=activity[0],
                        order=record.order,
                        at=at,
                        payload=activity[1],
                        new_status=status,
                    )
                )
            specs.append(_transition_spec(transition))
            staged.orders[order_id] = replace(record, order=updated)
            staged.latest_at = max(
                _aware_utc(at, "resolution timestamp"),
                staged.latest_at or _aware_utc(at, "resolution timestamp"),
            )
            self._commit(staged, specs, at=at)
            return updated

    def _process_order(
        self,
        *,
        state: _PaperState,
        record: _PaperOrderRecord,
        snapshot: MarketSnapshot,
        bar: Bar,
        instrument: Instrument,
        available_liquidity: Decimal,
    ) -> tuple[_PaperOrderRecord, Fill | None, Decimal, tuple[_AuditSpec, ...]]:
        order = record.order
        observed_at = snapshot.observed_at.astimezone(UTC)
        source_at = bar.at.astimezone(UTC)
        if order.status in _CLOSED or order.status is OrderStatus.UNKNOWN:
            return record, None, available_liquidity, ()
        if source_at <= record.submitted_source_at:
            return record, None, available_liquidity, ()
        if observed_at >= record.intent.expires_at.astimezone(UTC):
            expired, transition = _stage_transition(order, OrderStatus.EXPIRED, observed_at)
            return (
                replace(record, order=expired),
                None,
                available_liquidity,
                (_transition_spec(transition),),
            )
        if not self._fill_model.can_fill(
            submitted_at=order.submitted_at,
            event_at=observed_at,
        ):
            return record, None, available_liquidity, ()

        reference, triggered, may_fill = _reference_price(record, bar)
        specs: list[_AuditSpec] = []
        if triggered:
            record = replace(record, stop_triggered=True)
            specs.append(
                _activity_spec(
                    kind="paper.order.stop_triggered",
                    order=order,
                    at=observed_at,
                    payload={
                        "trigger_price": _decimal_text(
                            cast(Decimal, record.intent.trigger_price)
                        )
                    },
                )
            )
        if not may_fill or reference is None:
            return record, None, available_liquidity, tuple(specs)

        remaining_quantity = _remaining_quantity(record, reference, instrument)
        quantity = self._fill_model.allocate_quantity(
            remaining_quantity=remaining_quantity,
            available_liquidity=available_liquidity,
            quantity_step=instrument.quantity_step,
        )
        if quantity <= Decimal("0"):
            return record, None, available_liquidity, tuple(specs)
        fill_id = f"{self._session_id}:fill:{state.fill_sequence + 1:020d}"
        candidate = self._fill_model.fill(
            fill_id=fill_id,
            order_id=order.order_id,
            instrument=instrument,
            side=record.intent.side,
            quantity=quantity,
            reference_price=reference,
            submitted_at=order.submitted_at,
            filled_at=observed_at,
        )
        if record.remaining_notional is not None:
            affordable_quantity = self._fill_model.allocate_quantity(
                remaining_quantity=record.remaining_notional / candidate.price,
                available_liquidity=available_liquidity,
                quantity_step=instrument.quantity_step,
            )
            if affordable_quantity <= Decimal("0"):
                return record, None, available_liquidity, tuple(specs)
            if affordable_quantity < candidate.quantity:
                candidate = self._fill_model.fill(
                    fill_id=fill_id,
                    order_id=order.order_id,
                    instrument=instrument,
                    side=record.intent.side,
                    quantity=affordable_quantity,
                    reference_price=reference,
                    submitted_at=order.submitted_at,
                    filled_at=observed_at,
                )
        if not _respects_order_prices(record.intent, candidate.price):
            return record, None, available_liquidity, tuple(specs)
        notional = candidate.quantity * candidate.price
        requested_value = (
            record.intent.notional
            if record.intent.notional is not None
            else cast(Decimal, record.intent.quantity) * reference
        )
        if requested_value < instrument.minimum_notional:
            return record, None, available_liquidity, tuple(specs)
        if record.remaining_notional is not None and notional > record.remaining_notional:
            return record, None, available_liquidity, tuple(specs)
        if record.intent.side is Side.BUY:
            if notional + candidate.fee > state.ledger.cash:
                return record, None, available_liquidity, tuple(specs)
        else:
            held = next(
                (
                    position.quantity
                    for position in _state_positions(state)
                    if position.instrument_id == order.instrument_id
                ),
                Decimal("0"),
            )
            if candidate.quantity > held:
                return record, None, available_liquidity, tuple(specs)

        old_filled = order.filled_quantity
        new_filled = old_filled + candidate.quantity
        if order.requested_quantity is not None and new_filled > order.requested_quantity:
            raise RuntimeError("paper fill would exceed requested quantity")
        remaining_notional = record.remaining_notional
        if remaining_notional is not None:
            remaining_notional -= notional
            if remaining_notional < Decimal("0"):
                raise RuntimeError("paper notional fill would create a negative residual")
        quantity_complete = (
            order.requested_quantity is not None and new_filled == order.requested_quantity
        )
        notional_complete = remaining_notional == Decimal("0")
        target = (
            OrderStatus.FILLED
            if quantity_complete or notional_complete
            else OrderStatus.PARTIALLY_FILLED
        )
        transitioned, transition = _stage_transition(order, target, observed_at)
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
                "updated_at": observed_at,
            }
        )
        cumulative_notional = record.cumulative_filled_notional + notional
        cumulative_fees = record.cumulative_fees + candidate.fee
        completed_record = replace(
            record,
            order=completed_order,
            remaining_notional=remaining_notional,
            cumulative_filled_notional=cumulative_notional,
            cumulative_fees=cumulative_fees,
        )
        state.ledger.apply_fill(candidate)
        state.fill_sequence += 1
        state.fills.append(candidate)
        specs.extend(
            (
                _transition_spec(transition),
                _activity_spec(
                    kind="paper.order.fill",
                    order=order,
                    at=observed_at,
                    payload={
                        "fill_id": candidate.fill_id,
                        "quantity": _decimal_text(candidate.quantity),
                        "price": _decimal_text(candidate.price),
                        "fee": _decimal_text(candidate.fee),
                        "cumulative_filled_quantity": _decimal_text(new_filled),
                        "remaining_quantity": _decimal_text(
                            order.requested_quantity - new_filled
                            if order.requested_quantity is not None
                            else Decimal("0")
                        ),
                        "requested_notional": (
                            _decimal_text(record.intent.notional)
                            if record.intent.notional is not None
                            else None
                        ),
                        "cumulative_filled_notional": _decimal_text(cumulative_notional),
                        "remaining_notional": (
                            _decimal_text(remaining_notional)
                            if remaining_notional is not None
                            else None
                        ),
                        "cumulative_fees": _decimal_text(cumulative_fees),
                    },
                    new_status=target,
                ),
            )
        )
        return (
            completed_record,
            candidate,
            available_liquidity - candidate.quantity,
            tuple(specs),
        )

    def _commit(
        self,
        staged: _PaperState,
        specs: list[_AuditSpec],
        *,
        at: datetime,
    ) -> None:
        occurred_at = _aware_utc(at, "commit timestamp")
        final_sequence = staged.event_sequence + len(specs) + 1
        staged.event_sequence = final_sequence
        specs.append(
            _AuditSpec(
                kind="paper.state.committed",
                client_intent_id=self._session_id,
                broker_order_id=self._session_id,
                occurred_at=occurred_at,
                prior_status=None,
                new_status=None,
                payload=MappingProxyType(_state_payload(staged)),
            )
        )
        first_sequence = self._state.event_sequence + 1
        prepared: list[PaperAuditEvent] = []
        durable_events: list[AuditEvent] = []
        for offset, spec in enumerate(specs):
            event_id = f"{self._session_id}:event:{first_sequence + offset:020d}"
            paper_event = PaperAuditEvent(
                event_id=event_id,
                kind=spec.kind,
                client_intent_id=spec.client_intent_id,
                broker_order_id=spec.broker_order_id,
                occurred_at=spec.occurred_at,
                prior_status=spec.prior_status,
                new_status=spec.new_status,
                payload=MappingProxyType(dict(spec.payload)),
            )
            if spec.kind != "paper.state.committed":
                prepared.append(paper_event)
            durable_events.append(
                AuditEvent(
                    event_id=event_id,
                    kind=spec.kind,
                    aggregate_id=self._session_id,
                    payload=_audit_payload(paper_event),
                    occurred_at=spec.occurred_at,
                )
            )
        if self._audit_log is not None:
            self._audit_log.record_many(tuple(durable_events))
        self._state = staged
        self._audit_events.extend(prepared)

    @classmethod
    def rehydrate(
        cls,
        records: Iterable[EventRecord],
        *,
        audit_log: AuditRecorder,
        starting_cash: Decimal,
        session_id: str,
        currency: str = "USD",
        fill_model: FillModel | None = None,
        max_volume_participation: Decimal = Decimal("1"),
    ) -> PaperBroker:
        """Restore the last atomically committed state for one durable session."""
        rows = tuple(records)
        if not rows or any(row.aggregate_id != session_id for row in rows):
            raise ValueError("rehydration requires one nonempty matching session stream")
        state_rows = [row for row in rows if row.kind == "paper.state.committed"]
        if not state_rows:
            raise ValueError("rehydration stream lacks a committed state")
        latest_state = max(state_rows, key=lambda row: row.sequence)
        broker = cls(
            fill_model=fill_model,
            starting_cash=starting_cash,
            currency=currency,
            max_volume_participation=max_volume_participation,
            audit_log=audit_log,
            durable=True,
            session_id=session_id,
        )
        broker._state = _state_from_payload(
            latest_state.payload,
            starting_cash=starting_cash,
            currency=currency,
        )
        broker._audit_events = [
            _paper_event_from_record(row)
            for row in sorted(rows, key=lambda item: item.sequence)
            if row.kind != "paper.state.committed"
        ]
        return broker


def _stage_transition(
    order: BrokerOrder,
    status: OrderStatus,
    at: datetime,
) -> tuple[BrokerOrder, OrderTransitionEvent]:
    events: list[OrderTransitionEvent] = []
    updated = OrderStateMachine.transition(order, status, at=at, emit=events.append)
    [event] = events
    return updated, event


def _transition_spec(event: OrderTransitionEvent) -> _AuditSpec:
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


def _activity_spec(
    *,
    kind: str,
    order: BrokerOrder,
    at: datetime,
    payload: Mapping[str, object],
    new_status: OrderStatus | None = None,
) -> _AuditSpec:
    return _AuditSpec(
        kind=kind,
        client_intent_id=order.client_order_id,
        broker_order_id=order.order_id,
        occurred_at=_aware_utc(at, "audit timestamp"),
        prior_status=order.status,
        new_status=order.status if new_status is None else new_status,
        payload=MappingProxyType(dict(payload)),
    )


def _audit_payload(event: PaperAuditEvent) -> Mapping[str, object]:
    payload: dict[str, object] = {
        "client_intent_id": event.client_intent_id,
        "broker_order_id": event.broker_order_id,
        **dict(event.payload),
    }
    if event.prior_status is not None:
        payload["prior_status"] = event.prior_status.value
    if event.new_status is not None:
        payload["new_status"] = event.new_status.value
    return payload


def _validate_intent(intent: object) -> None:
    if not isinstance(intent, OrderIntent):
        raise ValueError("intent must be an OrderIntent")
    for identifier, name in (
        (intent.intent_id, "intent_id"),
        (intent.instrument_id, "instrument_id"),
    ):
        if not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None:
            raise ValueError(f"{name} must be a safe stable identifier")
    if not isinstance(intent.side, Side) or not isinstance(intent.order_type, OrderType):
        raise ValueError("side and order_type must be canonical enum values")
    if (intent.quantity is None) == (intent.notional is None):
        raise ValueError("exactly one order size must be populated")
    if intent.quantity is not None:
        _positive_decimal(intent.quantity, "quantity")
    if intent.notional is not None:
        _positive_decimal(intent.notional, "notional")
    for price, name in (
        (intent.limit_price, "limit_price"),
        (intent.trigger_price, "trigger_price"),
        (intent.stop_loss, "stop_loss"),
        (intent.take_profit, "take_profit"),
    ):
        if price is not None:
            _positive_decimal(price, name)
    expected = {
        OrderType.MARKET: (False, False),
        OrderType.LIMIT: (True, False),
        OrderType.STOP: (False, True),
        OrderType.STOP_LIMIT: (True, True),
    }[intent.order_type]
    if (intent.limit_price is not None, intent.trigger_price is not None) != expected:
        raise ValueError("order type requires its exact limit/trigger fields")
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
) -> tuple[Decimal | None, bool, bool]:
    intent = record.intent
    if intent.order_type is OrderType.MARKET:
        return bar.open, False, True
    if intent.order_type is OrderType.LIMIT:
        return _limit_reference(intent, bar), False, True
    trigger = cast(Decimal, intent.trigger_price)
    if record.stop_triggered:
        if intent.order_type is OrderType.STOP:
            return bar.open, False, True
        return _limit_reference(intent, bar), False, True
    crossed = bar.high >= trigger if intent.side is Side.BUY else bar.low <= trigger
    if not crossed:
        return None, False, False
    if intent.order_type is OrderType.STOP:
        reference = max(bar.open, trigger) if intent.side is Side.BUY else min(bar.open, trigger)
        return reference, True, True
    return None, True, False


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
        if intent.side is Side.BUY:
            return intent.stop_loss < fill_price < intent.take_profit
        return intent.stop_loss > fill_price > intent.take_profit
    return True


def _remaining_quantity(
    record: _PaperOrderRecord,
    reference_price: Decimal,
    instrument: Instrument,
) -> Decimal:
    _positive_decimal(instrument.quantity_step, "instrument quantity step")
    _positive_decimal(instrument.price_tick, "instrument price tick")
    _positive_decimal(instrument.minimum_notional, "instrument minimum notional")
    if record.order.requested_quantity is not None:
        requested = _positive_decimal(record.order.requested_quantity, "requested quantity")
        allocated = FillModel().allocate_quantity(
            remaining_quantity=requested,
            available_liquidity=requested,
            quantity_step=instrument.quantity_step,
        )
        if allocated != requested:
            raise ValueError("requested quantity must align with instrument quantity step")
        return requested - record.order.filled_quantity
    return cast(Decimal, record.remaining_notional) / reference_price


def _state_positions(state: _PaperState) -> tuple[Position, ...]:
    if state.latest_at is None:
        return ()
    return state.ledger.snapshot(state.latest_at).positions


def _event_key(snapshot: MarketSnapshot) -> tuple[datetime, datetime, str]:
    return (
        snapshot.observed_at.astimezone(UTC),
        snapshot.source_at.astimezone(UTC),
        snapshot.instrument_id,
    )


def _instrument_id(instrument: object) -> str:
    if not isinstance(instrument, Instrument):
        raise ValueError("instrument must be an Instrument")
    if not isinstance(instrument.asset_class, AssetClass):
        raise ValueError("instrument asset class must be canonical")
    return f"{instrument.symbol}@{instrument.venue}"


def _paper_order_id(session_id: str, intent_id: str) -> str:
    session_digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
    intent_digest = hashlib.sha256(intent_id.encode("utf-8")).hexdigest()[:24]
    return f"paper-{session_digest}-{intent_digest}"


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


def _datetime_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _intent_payload(intent: OrderIntent) -> Mapping[str, object]:
    return {
        "intent_id": intent.intent_id,
        "instrument_id": intent.instrument_id,
        "side": intent.side.value,
        "quantity": _optional_decimal(intent.quantity),
        "notional": _optional_decimal(intent.notional),
        "order_type": intent.order_type.value,
        "limit_price": _optional_decimal(intent.limit_price),
        "trigger_price": _optional_decimal(intent.trigger_price),
        "stop_loss": _optional_decimal(intent.stop_loss),
        "take_profit": _optional_decimal(intent.take_profit),
        "time_in_force": intent.time_in_force,
        "product": intent.product,
        "session": intent.session,
        "snapshot_hash": intent.snapshot_hash,
        "created_at": _datetime_text(intent.created_at),
        "expires_at": _datetime_text(intent.expires_at),
    }


def _intent_from_payload(payload: Mapping[str, object]) -> OrderIntent:
    return OrderIntent.model_validate(dict(payload))


def _order_payload(order: BrokerOrder) -> Mapping[str, object]:
    return {
        "order_id": order.order_id,
        "client_order_id": order.client_order_id,
        "broker": order.broker,
        "instrument_id": order.instrument_id,
        "status": order.status.value,
        "requested_quantity": _optional_decimal(order.requested_quantity),
        "filled_quantity": _decimal_text(order.filled_quantity),
        "average_fill_price": _optional_decimal(order.average_fill_price),
        "submitted_at": _datetime_text(order.submitted_at),
        "updated_at": _datetime_text(order.updated_at),
    }


def _fill_payload(fill: Fill) -> Mapping[str, object]:
    return {
        "fill_id": fill.fill_id,
        "order_id": fill.order_id,
        "instrument_id": fill.instrument_id,
        "side": fill.side.value,
        "quantity": _decimal_text(fill.quantity),
        "price": _decimal_text(fill.price),
        "fee": _decimal_text(fill.fee),
        "filled_at": _datetime_text(fill.filled_at),
    }


def _bar_payload(bar: Bar) -> Mapping[str, object]:
    return {
        "at": _datetime_text(bar.at),
        "open": _decimal_text(bar.open),
        "high": _decimal_text(bar.high),
        "low": _decimal_text(bar.low),
        "close": _decimal_text(bar.close),
        "volume": _decimal_text(bar.volume),
    }


def _snapshot_payload(snapshot: MarketSnapshot) -> Mapping[str, object]:
    return {
        "instrument_id": snapshot.instrument_id,
        "observed_at": _datetime_text(snapshot.observed_at),
        "source_at": _datetime_text(snapshot.source_at),
        "bars": [_bar_payload(bar) for bar in snapshot.bars],
        "provider": snapshot.provider,
        "max_age_seconds": snapshot.max_age_seconds,
    }


def _state_payload(state: _PaperState) -> dict[str, object]:
    return {
        "schema_version": 1,
        "event_sequence": state.event_sequence,
        "fill_sequence": state.fill_sequence,
        "latest_at": _datetime_text(state.latest_at) if state.latest_at is not None else None,
        "last_event_key": (
            [
                _datetime_text(state.last_event_key[0]),
                _datetime_text(state.last_event_key[1]),
                state.last_event_key[2],
            ]
            if state.last_event_key is not None
            else None
        ),
        "orders": [
            {
                "intent": _intent_payload(record.intent),
                "order": _order_payload(record.order),
                "submitted_source_at": _datetime_text(record.submitted_source_at),
                "remaining_notional": _optional_decimal(record.remaining_notional),
                "cumulative_filled_notional": _decimal_text(
                    record.cumulative_filled_notional
                ),
                "cumulative_fees": _decimal_text(record.cumulative_fees),
                "stop_triggered": record.stop_triggered,
            }
            for _, record in sorted(state.orders.items())
        ],
        "snapshots": [
            _snapshot_payload(snapshot)
            for _, snapshot in sorted(state.snapshots.items())
        ],
        "fills": [_fill_payload(fill) for fill in state.fills],
        "market_prices": {
            key: _decimal_text(value) for key, value in sorted(state.market_prices.items())
        },
    }


def _state_from_payload(
    payload: Mapping[str, object],
    *,
    starting_cash: Decimal,
    currency: str,
) -> _PaperState:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported paper state schema")
    orders: dict[str, _PaperOrderRecord] = {}
    client_orders: dict[str, str] = {}
    raw_orders = cast(Iterable[Mapping[str, object]], payload["orders"])
    for raw in raw_orders:
        intent = _intent_from_payload(cast(Mapping[str, object], raw["intent"]))
        order = BrokerOrder.model_validate(dict(cast(Mapping[str, object], raw["order"])))
        record = _PaperOrderRecord(
            intent=intent,
            order=order,
            submitted_source_at=datetime.fromisoformat(cast(str, raw["submitted_source_at"])),
            remaining_notional=_decimal_or_none(raw.get("remaining_notional")),
            cumulative_filled_notional=Decimal(
                cast(str, raw["cumulative_filled_notional"])
            ),
            cumulative_fees=Decimal(cast(str, raw["cumulative_fees"])),
            stop_triggered=cast(bool, raw["stop_triggered"]),
        )
        orders[order.order_id] = record
        client_orders[intent.intent_id] = order.order_id
    snapshots: dict[str, MarketSnapshot] = {}
    for raw_snapshot in cast(Iterable[Mapping[str, object]], payload["snapshots"]):
        snapshot = MarketSnapshot.model_validate(dict(raw_snapshot))
        snapshots[snapshot.instrument_id] = snapshot
    fills = [
        Fill.model_validate(dict(raw_fill))
        for raw_fill in cast(Iterable[Mapping[str, object]], payload["fills"])
    ]
    ledger = PortfolioLedger(starting_cash=starting_cash, currency=currency)
    for fill in fills:
        ledger.apply_fill(fill)
    prices = {
        key: Decimal(cast(str, value))
        for key, value in cast(Mapping[str, object], payload["market_prices"]).items()
    }
    latest_raw = payload.get("latest_at")
    latest_at = datetime.fromisoformat(cast(str, latest_raw)) if latest_raw is not None else None
    if prices and latest_at is not None:
        ledger.mark(prices, latest_at)
    key_raw = payload.get("last_event_key")
    last_event_key = None
    if key_raw is not None:
        key_values = tuple(cast(Iterable[object], key_raw))
        last_event_key = (
            datetime.fromisoformat(cast(str, key_values[0])),
            datetime.fromisoformat(cast(str, key_values[1])),
            cast(str, key_values[2]),
        )
    return _PaperState(
        orders=orders,
        client_orders=client_orders,
        snapshots=snapshots,
        fills=fills,
        event_sequence=cast(int, payload["event_sequence"]),
        fill_sequence=cast(int, payload["fill_sequence"]),
        ledger=ledger,
        market_prices=prices,
        latest_at=latest_at,
        last_event_key=last_event_key,
    )


def _paper_event_from_record(row: EventRecord) -> PaperAuditEvent:
    prior = row.payload.get("prior_status")
    new = row.payload.get("new_status")
    return PaperAuditEvent(
        event_id=row.event_id,
        kind=row.kind,
        client_intent_id=cast(str, row.payload.get("client_intent_id", row.aggregate_id)),
        broker_order_id=cast(str, row.payload.get("broker_order_id", row.aggregate_id)),
        occurred_at=row.occurred_at,
        prior_status=OrderStatus(cast(str, prior)) if prior is not None else None,
        new_status=OrderStatus(cast(str, new)) if new is not None else None,
        payload=row.payload,
    )


def _optional_decimal(value: Decimal | None) -> str | None:
    return _decimal_text(value) if value is not None else None


def _decimal_or_none(value: object) -> Decimal | None:
    return None if value is None else Decimal(cast(str, value))
