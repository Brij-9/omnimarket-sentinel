"""Deterministic, transactional, credential-free paper execution."""

from __future__ import annotations

import copy
import hashlib
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from threading import RLock
from types import MappingProxyType
from typing import cast

from market_sentinel.backtest.engine import CostModel, FillModel
from market_sentinel.domain.enums import AssetClass, OrderStatus, OrderType, Side
from market_sentinel.domain.models import (
    Bar,
    BrokerOrder,
    Fill,
    Instrument,
    MarketSnapshot,
    OrderIntent,
    PortfolioSnapshot,
    Position,
)
from market_sentinel.execution.base import AuditRecorder, BrokerCapabilities
from market_sentinel.execution.state_machine import OrderStateMachine, OrderTransitionEvent
from market_sentinel.operations.audit import AuditEvent
from market_sentinel.portfolio.ledger import (
    PortfolioLedger,
    PortfolioLedgerPositionState,
    PortfolioLedgerState,
)
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


@dataclass(frozen=True, slots=True)
class _ReplayConfiguration:
    session_id: str
    starting_cash: Decimal
    currency: str
    costs: CostModel
    max_volume_participation: Decimal


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
    instruments: dict[str, Instrument]
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
            instruments=dict(self.instruments),
            latest_at=self.latest_at,
            last_event_key=self.last_event_key,
        )


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")
_VENUE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_CURRENCY = re.compile(r"^[A-Z][A-Z0-9]{2,11}$")
_TIMEZONE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_+./-]{0,127}$")
_SNAPSHOT_HASH = re.compile(r"^[0-9a-f]{64}$")
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
        if not isinstance(currency, str) or _CURRENCY.fullmatch(currency) is None:
            raise ValueError("currency must be a safe uppercase identifier")
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
            instruments={},
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

    def portfolio_snapshot(self) -> PortfolioSnapshot:
        """Return the exact current marked portfolio at the global watermark."""
        with self._lock:
            if self._state.latest_at is None:
                raise ValueError("paper portfolio has no availability watermark")
            return self._state.ledger.snapshot(self._state.latest_at)

    def submit(self, intent: OrderIntent, snapshot: MarketSnapshot) -> BrokerOrder:
        """Acknowledge one canonical intent without filling its current snapshot."""
        with self._lock:
            if not isinstance(intent, OrderIntent):
                raise ValueError("intent must be an OrderIntent")
            _validate_snapshot(snapshot, expected_instrument_id=intent.instrument_id)
            _validate_intent(intent, snapshot=snapshot)
            submitted_at = snapshot.observed_at.astimezone(UTC)
            _validate_watermark(self._state, submitted_at)
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
        with self._lock:
            _validate_watermark(self._state, snapshot.observed_at.astimezone(UTC))
            if len(_active_instrument_ids(self._state)) > 1:
                raise ValueError(
                    "multiple active instruments require one atomic on_snapshots batch"
                )
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
            active_instruments = _active_instrument_ids(self._state)
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
                _validate_watermark(
                    self._state,
                    snapshot.observed_at.astimezone(UTC),
                )
                if instrument.asset_class not in _PAPER_CAPABILITIES.supported_asset_classes:
                    raise ValueError("instrument asset class is unsupported by paper execution")
                key = _event_key(snapshot)
                validated.append((key, snapshot, instrument, duplicate))

            if len(active_instruments) > 1 and seen_instruments != active_instruments:
                raise ValueError(
                    "snapshot batch must include all active instruments"
                )

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
                previous_instrument = staged.instruments.get(snapshot.instrument_id)
                if previous_instrument is not None and previous_instrument != instrument:
                    raise ValueError("instrument metadata changed within the paper session")
                staged.instruments[snapshot.instrument_id] = instrument
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
                staged.latest_at = max(
                    snapshot.observed_at.astimezone(UTC),
                    staged.latest_at or snapshot.observed_at.astimezone(UTC),
                )
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

    def reconcile_unknown_fills(
        self,
        authoritative_order: BrokerOrder,
        new_fills: tuple[Fill, ...],
        *,
        instrument: Instrument,
    ) -> BrokerOrder:
        """Atomically apply authoritative fill evidence for one UNKNOWN order."""
        with self._lock:
            if not isinstance(authoritative_order, BrokerOrder):
                raise ValueError("authoritative fill order is invalid")
            staged = self._state.clone()
            record = staged.orders.get(authoritative_order.order_id)
            if record is None or record.order.status is not OrderStatus.UNKNOWN:
                raise ValueError("authoritative fill requires one UNKNOWN paper order")
            current = record.order
            at = _aware_utc(authoritative_order.updated_at, "authoritative update")
            _validate_watermark(self._state, at)
            instrument_id = _instrument_id(instrument)
            if instrument.quote_currency != self._currency:
                raise ValueError("authoritative fill instrument currency is invalid")
            if (
                instrument_id != current.instrument_id
                or authoritative_order.client_order_id != current.client_order_id
                or authoritative_order.broker != self.broker_name
                or authoritative_order.instrument_id != current.instrument_id
                or authoritative_order.submitted_at != current.submitted_at
                or authoritative_order.requested_quantity != current.requested_quantity
                or authoritative_order.status
                not in {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED}
                or authoritative_order.filled_quantity <= current.filled_quantity
                or authoritative_order.average_fill_price is None
                or not isinstance(new_fills, tuple)
                or not new_fills
            ):
                raise ValueError("authoritative fill order identity or status is invalid")
            prior_instrument = staged.instruments.get(instrument_id)
            if prior_instrument is not None and prior_instrument != instrument:
                raise ValueError("authoritative fill instrument metadata changed")

            existing_fill_ids = {fill.fill_id for fill in staged.fills}
            delta = authoritative_order.filled_quantity - current.filled_quantity
            prior_at = max(
                (
                    fill.filled_at
                    for fill in staged.fills
                    if fill.order_id == current.order_id
                ),
                default=current.submitted_at,
            )
            prior_key = (prior_at, "")
            fill_notional = Decimal("0")
            fill_fees = Decimal("0")
            fill_quantity = Decimal("0")
            seen_fill_ids: set[str] = set()
            for fill in new_fills:
                if not isinstance(fill, Fill):
                    raise ValueError("authoritative fill evidence is invalid")
                filled_at = _aware_utc(fill.filled_at, "authoritative fill timestamp")
                key = (filled_at, fill.fill_id)
                if (
                    not isinstance(fill.fill_id, str)
                    or _IDENTIFIER.fullmatch(fill.fill_id) is None
                    or fill.fill_id in existing_fill_ids
                    or fill.fill_id in seen_fill_ids
                    or fill.order_id != current.order_id
                    or fill.instrument_id != current.instrument_id
                    or fill.side is not record.intent.side
                    or filled_at < current.submitted_at
                    or filled_at > at
                    or key <= prior_key
                ):
                    raise ValueError("authoritative fill identity or chronology is invalid")
                _positive_decimal(fill.quantity, "authoritative fill quantity")
                _positive_decimal(fill.price, "authoritative fill price")
                _nonnegative_decimal(fill.fee, "authoritative fill fee")
                if not _respects_order_prices(record.intent, fill.price):
                    raise ValueError("authoritative fill price violates order protection")
                if record.intent.side is Side.BUY:
                    if fill.quantity * fill.price + fill.fee > staged.ledger.cash:
                        raise ValueError("authoritative fill exceeds paper cash")
                else:
                    held = next(
                        (
                            position.quantity
                            for position in _state_positions(staged)
                            if position.instrument_id == current.instrument_id
                        ),
                        Decimal("0"),
                    )
                    if fill.quantity > held:
                        raise ValueError("authoritative fill exceeds paper position")
                staged.ledger.apply_fill(fill)
                seen_fill_ids.add(fill.fill_id)
                fill_quantity += fill.quantity
                fill_notional += fill.quantity * fill.price
                fill_fees += fill.fee
                prior_key = key
            if fill_quantity != delta:
                raise ValueError("authoritative fill quantity does not match broker delta")
            total_notional = record.cumulative_filled_notional + fill_notional
            expected_average = total_notional / authoritative_order.filled_quantity
            if authoritative_order.average_fill_price != expected_average:
                raise ValueError("authoritative fill weighted average is inconsistent")
            if (
                current.requested_quantity is not None
                and authoritative_order.filled_quantity > current.requested_quantity
            ):
                raise ValueError("authoritative fill would overfill paper order")
            remaining_notional = record.remaining_notional
            if remaining_notional is not None:
                remaining_notional -= fill_notional
                if remaining_notional < Decimal("0"):
                    raise ValueError("authoritative fill exceeds requested notional")
            complete = (
                authoritative_order.filled_quantity == current.requested_quantity
                if current.requested_quantity is not None
                else remaining_notional == Decimal("0")
            )
            if complete is not (authoritative_order.status is OrderStatus.FILLED):
                raise ValueError("authoritative fill status is inconsistent with remaining size")

            transition = OrderTransitionEvent(
                prior_status=OrderStatus.UNKNOWN,
                new_status=authoritative_order.status,
                client_intent_id=current.client_order_id,
                broker_order_id=current.order_id,
                occurred_at=at,
            )
            specs: list[_AuditSpec] = [_transition_spec(transition)]
            cumulative_quantity = current.filled_quantity
            cumulative_notional = record.cumulative_filled_notional
            cumulative_fees = record.cumulative_fees
            for fill in new_fills:
                cumulative_quantity += fill.quantity
                cumulative_notional += fill.quantity * fill.price
                cumulative_fees += fill.fee
                specs.append(
                    _activity_spec(
                        kind="paper.order.fill",
                        order=current,
                        at=at,
                        new_status=authoritative_order.status,
                        payload={
                            "fill_id": fill.fill_id,
                            "quantity": _decimal_text(fill.quantity),
                            "price": _decimal_text(fill.price),
                            "fee": _decimal_text(fill.fee),
                            "instrument": _instrument_payload(instrument),
                            "source": "AUTHORITATIVE_RECONCILIATION",
                            "cumulative_filled_quantity": _decimal_text(
                                cumulative_quantity
                            ),
                            "cumulative_filled_notional": _decimal_text(
                                cumulative_notional
                            ),
                            "cumulative_fees": _decimal_text(cumulative_fees),
                        },
                    )
                )
            staged.orders[current.order_id] = replace(
                record,
                order=authoritative_order,
                remaining_notional=remaining_notional,
                cumulative_filled_notional=total_notional,
                cumulative_fees=record.cumulative_fees + fill_fees,
            )
            staged.fills.extend(new_fills)
            staged.fill_sequence += len(new_fills)
            staged.market_prices[instrument_id] = new_fills[-1].price
            staged.instruments[instrument_id] = instrument
            staged.latest_at = at
            staged.ledger.mark(staged.market_prices, at)
            self._commit(staged, specs, at=at)
            return authoritative_order

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
            resolved_at = _aware_utc(at, "resolution timestamp")
            _validate_watermark(self._state, resolved_at)
            updated, transition = _stage_transition(record.order, status, resolved_at)
            specs: list[_AuditSpec] = []
            if activity is not None:
                specs.append(
                    _activity_spec(
                        kind=activity[0],
                        order=record.order,
                        at=resolved_at,
                        payload=activity[1],
                        new_status=status,
                    )
                )
            specs.append(_transition_spec(transition))
            staged.orders[order_id] = replace(record, order=updated)
            staged.latest_at = resolved_at
            self._commit(staged, specs, at=resolved_at)
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
                        "instrument": _instrument_payload(instrument),
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
                payload=MappingProxyType(
                    _state_payload(
                        staged,
                        session_id=self._session_id,
                        starting_cash=self._starting_cash,
                        currency=self._currency,
                        costs=self._fill_model.costs,
                        max_volume_participation=self._max_volume_participation,
                    )
                ),
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
        starting_cash: Decimal | None = None,
        session_id: str | None = None,
        currency: str | None = None,
        fill_model: FillModel | None = None,
        max_volume_participation: Decimal | None = None,
    ) -> PaperBroker:
        """Restore the last atomically committed state for one durable session."""
        try:
            rows = tuple(records)
            if not rows:
                raise ValueError
            derived_session = rows[0].aggregate_id
            if (
                not isinstance(derived_session, str)
                or _IDENTIFIER.fullmatch(derived_session) is None
                or (session_id is not None and session_id != derived_session)
            ):
                raise ValueError
            prior_database_sequence = 0
            prior_occurred_at: datetime | None = None
            committed: list[tuple[_ReplayConfiguration, _PaperState]] = []
            fill_instruments: dict[str, Instrument] = {}
            pending_activity: list[EventRecord] = []
            for local_sequence, row in enumerate(rows, start=1):
                if (
                    not isinstance(row, EventRecord)
                    or type(row.sequence) is not int
                    or row.sequence <= prior_database_sequence
                    or row.aggregate_id != derived_session
                    or row.event_id
                    != f"{derived_session}:event:{local_sequence:020d}"
                    or not isinstance(row.kind, str)
                    or _IDENTIFIER.fullmatch(row.kind) is None
                ):
                    raise ValueError
                occurred_at = _aware_utc(row.occurred_at, "replay event timestamp")
                if prior_occurred_at is not None and occurred_at < prior_occurred_at:
                    raise ValueError
                if not isinstance(row.payload, Mapping):
                    raise ValueError
                prior_database_sequence = row.sequence
                prior_occurred_at = occurred_at
                if row.kind == "paper.state.committed":
                    state_payload = dict(row.payload)
                    if (
                        state_payload.pop("client_intent_id", None) != derived_session
                        or state_payload.pop("broker_order_id", None) != derived_session
                    ):
                        raise ValueError
                    configuration = _configuration_from_payload(state_payload)
                    state = _state_from_payload(
                        state_payload,
                        configuration=configuration,
                    )
                    if (
                        configuration.session_id != derived_session
                        or state.event_sequence != local_sequence
                        or any(
                            state.instruments.get(instrument_id) != instrument
                            for instrument_id, instrument in fill_instruments.items()
                        )
                    ):
                        raise ValueError
                    _validate_replay_group(
                        tuple(pending_activity),
                        state=state,
                        previous_state=committed[-1][1] if committed else None,
                        session_id=derived_session,
                    )
                    pending_activity.clear()
                    committed.append((configuration, state))
                else:
                    client_id = row.payload.get("client_intent_id")
                    broker_id = row.payload.get("broker_order_id")
                    if (
                        not isinstance(client_id, str)
                        or _IDENTIFIER.fullmatch(client_id) is None
                        or not isinstance(broker_id, str)
                        or _IDENTIFIER.fullmatch(broker_id) is None
                    ):
                        raise ValueError
                    if row.kind == "paper.order.fill":
                        instrument = _instrument_from_payload(
                            _strict_mapping(row.payload.get("instrument"))
                        )
                        instrument_id = _instrument_id(instrument)
                        previous_instrument = fill_instruments.get(instrument_id)
                        if (
                            previous_instrument is not None
                            and previous_instrument != instrument
                        ):
                            raise ValueError
                        fill_instruments[instrument_id] = instrument
                    pending_activity.append(row)
            if not committed or rows[-1].kind != "paper.state.committed":
                raise ValueError
            configuration, latest_state = committed[-1]
            if any(item != configuration for item, _ in committed):
                raise ValueError
            if latest_state.event_sequence != len(rows):
                raise ValueError
            if starting_cash is not None and starting_cash != configuration.starting_cash:
                raise ValueError
            if currency is not None and currency != configuration.currency:
                raise ValueError
            if fill_model is not None and fill_model.costs != configuration.costs:
                raise ValueError
            if (
                max_volume_participation is not None
                and max_volume_participation != configuration.max_volume_participation
            ):
                raise ValueError
            broker = cls(
                fill_model=FillModel(costs=configuration.costs),
                starting_cash=configuration.starting_cash,
                currency=configuration.currency,
                max_volume_participation=configuration.max_volume_participation,
                audit_log=audit_log,
                durable=True,
                session_id=configuration.session_id,
            )
            broker._state = latest_state
            broker._audit_events = [
                _paper_event_from_record(row)
                for row in rows
                if row.kind != "paper.state.committed"
            ]
            return broker
        except (AttributeError, KeyError, TypeError, ValueError, ArithmeticError) as error:
            raise ValueError("invalid durable paper stream") from error


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


def _validate_intent(intent: object, *, snapshot: MarketSnapshot) -> None:
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
    if intent.stop_loss is None or intent.take_profit is None:
        raise ValueError("paper entries require stop_loss and take_profit")
    if intent.time_in_force != "day":
        raise ValueError("paper time_in_force must be day")
    if intent.product != "cash":
        raise ValueError("paper product must be cash")
    if intent.session != "regular":
        raise ValueError("paper session must be regular")
    if (
        not isinstance(intent.snapshot_hash, str)
        or _SNAPSHOT_HASH.fullmatch(intent.snapshot_hash) is None
    ):
        raise ValueError("snapshot_hash must be a lowercase sha256 digest")
    _validate_protective_prices(intent, snapshot)
    created_at = _aware_utc(intent.created_at, "intent created_at")
    expires_at = _aware_utc(intent.expires_at, "intent expires_at")
    if expires_at <= created_at:
        raise ValueError("intent expires_at must follow created_at")


def _validate_snapshot(snapshot: object, *, expected_instrument_id: str) -> None:
    if not isinstance(snapshot, MarketSnapshot):
        raise ValueError("snapshot must be a MarketSnapshot")
    if snapshot.instrument_id != expected_instrument_id:
        raise ValueError("snapshot instrument identity must match the order")
    if (
        not isinstance(snapshot.provider, str)
        or _IDENTIFIER.fullmatch(snapshot.provider) is None
    ):
        raise ValueError("snapshot provider must be a safe identifier")
    if type(snapshot.max_age_seconds) is not int or snapshot.max_age_seconds < 0:
        raise ValueError("snapshot max_age_seconds must be a nonnegative integer")
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


def _validate_protective_prices(
    intent: OrderIntent,
    snapshot: MarketSnapshot,
) -> None:
    assert intent.stop_loss is not None and intent.take_profit is not None
    references: tuple[Decimal, ...]
    if intent.order_type is OrderType.MARKET:
        references = (snapshot.bars[-1].close,)
    elif intent.order_type is OrderType.LIMIT:
        references = (cast(Decimal, intent.limit_price),)
    elif intent.order_type is OrderType.STOP:
        references = (cast(Decimal, intent.trigger_price),)
    else:
        references = (
            cast(Decimal, intent.trigger_price),
            cast(Decimal, intent.limit_price),
        )
    if intent.side is Side.BUY and not (
        intent.stop_loss < min(references)
        and max(references) < intent.take_profit
    ):
        raise ValueError("buy protection must bracket every executable entry reference")
    if intent.side is Side.SELL and not (
        intent.stop_loss > max(references)
        and min(references) > intent.take_profit
    ):
        raise ValueError("sell protection must bracket every executable entry reference")


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


def _active_instrument_ids(state: _PaperState) -> set[str]:
    return {
        record.order.instrument_id
        for record in state.orders.values()
        if record.order.status not in _CLOSED
    }


def _validate_watermark(state: _PaperState, at: datetime) -> None:
    if state.latest_at is not None and at < state.latest_at:
        raise ValueError("global operation chronology regression is not allowed")


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
    if (
        not isinstance(instrument.symbol, str)
        or _VENUE_IDENTIFIER.fullmatch(instrument.symbol) is None
        or not isinstance(instrument.venue, str)
        or _VENUE_IDENTIFIER.fullmatch(instrument.venue) is None
        or not isinstance(instrument.quote_currency, str)
        or _CURRENCY.fullmatch(instrument.quote_currency) is None
        or not isinstance(instrument.timezone, str)
        or _TIMEZONE.fullmatch(instrument.timezone) is None
        or (
            instrument.session_calendar is not None
            and (
                not isinstance(instrument.session_calendar, str)
                or _IDENTIFIER.fullmatch(instrument.session_calendar) is None
            )
        )
    ):
        raise ValueError("instrument metadata must contain safe bounded identifiers")
    _positive_decimal(instrument.price_tick, "instrument price tick")
    _positive_decimal(instrument.quantity_step, "instrument quantity step")
    _positive_decimal(instrument.minimum_notional, "instrument minimum notional")
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
    _require_exact_keys(
        payload,
        {
            "intent_id",
            "instrument_id",
            "side",
            "quantity",
            "notional",
            "order_type",
            "limit_price",
            "trigger_price",
            "stop_loss",
            "take_profit",
            "time_in_force",
            "product",
            "session",
            "snapshot_hash",
            "created_at",
            "expires_at",
        },
    )
    return OrderIntent(
        intent_id=_strict_identifier(payload["intent_id"]),
        instrument_id=_strict_identifier(payload["instrument_id"]),
        side=Side(_strict_string(payload["side"])),
        quantity=_parse_optional_decimal(payload["quantity"], positive=True),
        notional=_parse_optional_decimal(payload["notional"], positive=True),
        order_type=OrderType(_strict_string(payload["order_type"])),
        limit_price=_parse_optional_decimal(payload["limit_price"], positive=True),
        trigger_price=_parse_optional_decimal(payload["trigger_price"], positive=True),
        stop_loss=_parse_optional_decimal(payload["stop_loss"], positive=True),
        take_profit=_parse_optional_decimal(payload["take_profit"], positive=True),
        time_in_force=_strict_string(payload["time_in_force"]),
        product=_strict_string(payload["product"]),
        session=_strict_string(payload["session"]),
        snapshot_hash=_strict_string(payload["snapshot_hash"]),
        created_at=_parse_datetime(payload["created_at"]),
        expires_at=_parse_datetime(payload["expires_at"]),
    )


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


def _order_from_payload(payload: Mapping[str, object]) -> BrokerOrder:
    _require_exact_keys(
        payload,
        {
            "order_id",
            "client_order_id",
            "broker",
            "instrument_id",
            "status",
            "requested_quantity",
            "filled_quantity",
            "average_fill_price",
            "submitted_at",
            "updated_at",
        },
    )
    filled_quantity = _parse_decimal(payload["filled_quantity"], nonnegative=True)
    return BrokerOrder(
        order_id=_strict_identifier(payload["order_id"]),
        client_order_id=_strict_identifier(payload["client_order_id"]),
        broker=_strict_string(payload["broker"]),
        instrument_id=_strict_identifier(payload["instrument_id"]),
        status=OrderStatus(_strict_string(payload["status"])),
        requested_quantity=_parse_optional_decimal(
            payload["requested_quantity"],
            positive=True,
        ),
        filled_quantity=filled_quantity,
        average_fill_price=_parse_optional_decimal(
            payload["average_fill_price"],
            positive=True,
        ),
        submitted_at=_parse_datetime(payload["submitted_at"]),
        updated_at=_parse_datetime(payload["updated_at"]),
    )


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


def _fill_from_payload(payload: Mapping[str, object]) -> Fill:
    _require_exact_keys(
        payload,
        {
            "fill_id",
            "order_id",
            "instrument_id",
            "side",
            "quantity",
            "price",
            "fee",
            "filled_at",
        },
    )
    return Fill(
        fill_id=_strict_identifier(payload["fill_id"]),
        order_id=_strict_identifier(payload["order_id"]),
        instrument_id=_strict_identifier(payload["instrument_id"]),
        side=Side(_strict_string(payload["side"])),
        quantity=_parse_decimal(payload["quantity"], positive=True),
        price=_parse_decimal(payload["price"], positive=True),
        fee=_parse_decimal(payload["fee"], nonnegative=True),
        filled_at=_parse_datetime(payload["filled_at"]),
    )


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


def _bar_from_payload(payload: Mapping[str, object]) -> Bar:
    _require_exact_keys(payload, {"at", "open", "high", "low", "close", "volume"})
    return Bar(
        at=_parse_datetime(payload["at"]),
        open=_parse_decimal(payload["open"], positive=True),
        high=_parse_decimal(payload["high"], positive=True),
        low=_parse_decimal(payload["low"], positive=True),
        close=_parse_decimal(payload["close"], positive=True),
        volume=_parse_decimal(payload["volume"], nonnegative=True),
    )


def _snapshot_from_payload(payload: Mapping[str, object]) -> MarketSnapshot:
    _require_exact_keys(
        payload,
        {
            "instrument_id",
            "observed_at",
            "source_at",
            "bars",
            "provider",
            "max_age_seconds",
        },
    )
    bars = tuple(
        _bar_from_payload(_strict_mapping(item))
        for item in _strict_sequence(payload["bars"])
    )
    max_age = _strict_nonnegative_int(payload["max_age_seconds"])
    snapshot = MarketSnapshot(
        instrument_id=_strict_identifier(payload["instrument_id"]),
        observed_at=_parse_datetime(payload["observed_at"]),
        source_at=_parse_datetime(payload["source_at"]),
        bars=bars,
        provider=_strict_identifier(payload["provider"]),
        max_age_seconds=max_age,
    )
    _validate_snapshot(snapshot, expected_instrument_id=snapshot.instrument_id)
    return snapshot


def _instrument_payload(instrument: Instrument) -> Mapping[str, object]:
    return {
        "symbol": instrument.symbol,
        "venue": instrument.venue,
        "asset_class": instrument.asset_class.value,
        "quote_currency": instrument.quote_currency,
        "timezone": instrument.timezone,
        "price_tick": _decimal_text(instrument.price_tick),
        "quantity_step": _decimal_text(instrument.quantity_step),
        "minimum_notional": _decimal_text(instrument.minimum_notional),
        "session_calendar": instrument.session_calendar,
    }


def _instrument_from_payload(payload: Mapping[str, object]) -> Instrument:
    _require_exact_keys(
        payload,
        {
            "symbol",
            "venue",
            "asset_class",
            "quote_currency",
            "timezone",
            "price_tick",
            "quantity_step",
            "minimum_notional",
            "session_calendar",
        },
    )
    session_calendar_value = payload["session_calendar"]
    if session_calendar_value is not None:
        session_calendar = _strict_identifier(session_calendar_value)
    else:
        session_calendar = None
    instrument = Instrument(
        symbol=_strict_string(payload["symbol"]),
        venue=_strict_string(payload["venue"]),
        asset_class=AssetClass(_strict_string(payload["asset_class"])),
        quote_currency=_strict_string(payload["quote_currency"]),
        timezone=_strict_string(payload["timezone"]),
        price_tick=_parse_decimal(payload["price_tick"], positive=True),
        quantity_step=_parse_decimal(payload["quantity_step"], positive=True),
        minimum_notional=_parse_decimal(payload["minimum_notional"], positive=True),
        session_calendar=session_calendar,
    )
    _instrument_id(instrument)
    return instrument


def _cost_payload(costs: CostModel) -> Mapping[str, object]:
    latency_microseconds = (
        costs.latency.days * 86_400_000_000
        + costs.latency.seconds * 1_000_000
        + costs.latency.microseconds
    )
    return {
        "fee_bps": _decimal_text(costs.fee_bps),
        "spread_bps": _decimal_text(costs.spread_bps),
        "slippage_bps": _decimal_text(costs.slippage_bps),
        "latency_microseconds": latency_microseconds,
    }


def _capabilities_payload() -> Mapping[str, object]:
    return {
        "schema_version": 1,
        "broker": _PAPER_CAPABILITIES.broker,
        "supported_asset_classes": sorted(
            item.value for item in _PAPER_CAPABILITIES.supported_asset_classes
        ),
        "supported_order_types": sorted(
            item.value for item in _PAPER_CAPABILITIES.supported_order_types
        ),
        "supports_fractional_quantity": _PAPER_CAPABILITIES.supports_fractional_quantity,
        "supports_notional_orders": _PAPER_CAPABILITIES.supports_notional_orders,
        "supports_partial_fills": _PAPER_CAPABILITIES.supports_partial_fills,
        "supports_shorting": _PAPER_CAPABILITIES.supports_shorting,
        "supports_leverage": _PAPER_CAPABILITIES.supports_leverage,
        "supports_derivatives": _PAPER_CAPABILITIES.supports_derivatives,
        "supports_cancel": _PAPER_CAPABILITIES.supports_cancel,
        "is_paper": _PAPER_CAPABILITIES.is_paper,
    }


def _ledger_state_payload(ledger: PortfolioLedger) -> Mapping[str, object]:
    state = ledger.export_state()
    return {
        "schema_version": 1,
        "starting_cash": _decimal_text(state.starting_cash),
        "currency": state.currency,
        "cash": _decimal_text(state.cash),
        "positions": [
            {
                "instrument_id": position.instrument_id,
                "quantity": _decimal_text(position.quantity),
                "average_price": _decimal_text(position.average_price),
            }
            for position in state.positions
        ],
        "market_prices": [
            {"instrument_id": instrument_id, "price": _decimal_text(price)}
            for instrument_id, price in state.market_prices
        ],
        "fill_ids": list(state.fill_ids),
        "gross_realized_pnl": _decimal_text(state.gross_realized_pnl),
        "fees": _decimal_text(state.fees),
        "equity": _decimal_text(state.equity),
        "peak_equity": _decimal_text(state.peak_equity),
        "drawdown": _decimal_text(state.drawdown),
    }


def _configuration_from_payload(payload: Mapping[str, object]) -> _ReplayConfiguration:
    if payload.get("schema_version") != 2:
        raise ValueError
    session_id = _strict_identifier(payload["session_id"])
    configuration = _strict_mapping(payload["configuration"])
    _require_exact_keys(
        configuration,
        {
            "starting_cash",
            "currency",
            "cost_model",
            "max_volume_participation",
        },
    )
    cost_payload = _strict_mapping(configuration["cost_model"])
    _require_exact_keys(
        cost_payload,
        {"fee_bps", "spread_bps", "slippage_bps", "latency_microseconds"},
    )
    latency = timedelta(
        microseconds=_strict_nonnegative_int(cost_payload["latency_microseconds"])
    )
    costs = CostModel(
        fee_bps=_parse_decimal(cost_payload["fee_bps"], nonnegative=True),
        spread_bps=_parse_decimal(cost_payload["spread_bps"], nonnegative=True),
        slippage_bps=_parse_decimal(cost_payload["slippage_bps"], nonnegative=True),
        latency=latency,
    )
    starting_cash = _parse_decimal(configuration["starting_cash"], positive=True)
    currency = _strict_string(configuration["currency"])
    participation = _parse_decimal(
        configuration["max_volume_participation"],
        positive=True,
    )
    if _CURRENCY.fullmatch(currency) is None or participation > Decimal("1"):
        raise ValueError
    _validate_capabilities_payload(_strict_mapping(payload["capabilities"]))
    return _ReplayConfiguration(
        session_id=session_id,
        starting_cash=starting_cash,
        currency=currency,
        costs=costs,
        max_volume_participation=participation,
    )


def _validate_capabilities_payload(payload: Mapping[str, object]) -> None:
    expected = _capabilities_payload()
    _require_exact_keys(payload, set(expected))
    if _strict_nonnegative_int(payload["schema_version"]) != 1:
        raise ValueError
    for name in (
        "supports_fractional_quantity",
        "supports_notional_orders",
        "supports_partial_fills",
        "supports_shorting",
        "supports_leverage",
        "supports_derivatives",
        "supports_cancel",
        "is_paper",
    ):
        if type(payload[name]) is not bool or payload[name] is not expected[name]:
            raise ValueError
    if (
        _strict_string(payload["broker"]) != expected["broker"]
        or tuple(_strict_sequence(payload["supported_asset_classes"]))
        != tuple(cast(list[str], expected["supported_asset_classes"]))
        or tuple(_strict_sequence(payload["supported_order_types"]))
        != tuple(cast(list[str], expected["supported_order_types"]))
    ):
        raise ValueError


def _ledger_state_from_payload(payload: Mapping[str, object]) -> PortfolioLedgerState:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "starting_cash",
            "currency",
            "cash",
            "positions",
            "market_prices",
            "fill_ids",
            "gross_realized_pnl",
            "fees",
            "equity",
            "peak_equity",
            "drawdown",
        },
    )
    if _strict_nonnegative_int(payload["schema_version"]) != 1:
        raise ValueError
    positions = tuple(
        _ledger_position_from_payload(_strict_mapping(item))
        for item in _strict_sequence(payload["positions"])
    )
    prices: list[tuple[str, Decimal]] = []
    for item in _strict_sequence(payload["market_prices"]):
        raw = _strict_mapping(item)
        _require_exact_keys(raw, {"instrument_id", "price"})
        prices.append(
            (
                _strict_identifier(raw["instrument_id"]),
                _parse_decimal(raw["price"], positive=True),
            )
        )
    fill_ids = tuple(_strict_identifier(item) for item in _strict_sequence(payload["fill_ids"]))
    return PortfolioLedgerState(
        starting_cash=_parse_decimal(payload["starting_cash"], positive=True),
        currency=_strict_string(payload["currency"]),
        cash=_parse_decimal(payload["cash"]),
        positions=positions,
        market_prices=tuple(prices),
        fill_ids=fill_ids,
        gross_realized_pnl=_parse_decimal(payload["gross_realized_pnl"]),
        fees=_parse_decimal(payload["fees"], nonnegative=True),
        equity=_parse_decimal(payload["equity"]),
        peak_equity=_parse_decimal(payload["peak_equity"]),
        drawdown=_parse_decimal(payload["drawdown"], nonnegative=True),
    )


def _ledger_position_from_payload(
    payload: Mapping[str, object],
) -> PortfolioLedgerPositionState:
    _require_exact_keys(payload, {"instrument_id", "quantity", "average_price"})
    return PortfolioLedgerPositionState(
        instrument_id=_strict_identifier(payload["instrument_id"]),
        quantity=_parse_decimal(payload["quantity"], positive=True),
        average_price=_parse_decimal(payload["average_price"], positive=True),
    )


def _state_payload(
    state: _PaperState,
    *,
    session_id: str,
    starting_cash: Decimal,
    currency: str,
    costs: CostModel,
    max_volume_participation: Decimal,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "session_id": session_id,
        "configuration": {
            "starting_cash": _decimal_text(starting_cash),
            "currency": currency,
            "cost_model": _cost_payload(costs),
            "max_volume_participation": _decimal_text(max_volume_participation),
        },
        "capabilities": _capabilities_payload(),
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
        "instruments": [
            _instrument_payload(instrument)
            for _, instrument in sorted(state.instruments.items())
        ],
        "ledger": _ledger_state_payload(state.ledger),
    }


def _state_from_payload(
    payload: Mapping[str, object],
    *,
    configuration: _ReplayConfiguration,
) -> _PaperState:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "session_id",
            "configuration",
            "capabilities",
            "event_sequence",
            "fill_sequence",
            "latest_at",
            "last_event_key",
            "orders",
            "snapshots",
            "fills",
            "market_prices",
            "instruments",
            "ledger",
        },
    )
    if _configuration_from_payload(payload) != configuration:
        raise ValueError
    event_sequence = _strict_nonnegative_int(payload["event_sequence"])
    fill_sequence = _strict_nonnegative_int(payload["fill_sequence"])

    snapshots: dict[str, MarketSnapshot] = {}
    for raw_snapshot in _strict_sequence(payload["snapshots"]):
        snapshot = _snapshot_from_payload(_strict_mapping(raw_snapshot))
        if snapshot.instrument_id in snapshots:
            raise ValueError
        snapshots[snapshot.instrument_id] = snapshot

    orders: dict[str, _PaperOrderRecord] = {}
    client_orders: dict[str, str] = {}
    for raw_value in _strict_sequence(payload["orders"]):
        raw = _strict_mapping(raw_value)
        _require_exact_keys(
            raw,
            {
                "intent",
                "order",
                "submitted_source_at",
                "remaining_notional",
                "cumulative_filled_notional",
                "cumulative_fees",
                "stop_triggered",
            },
        )
        intent = _intent_from_payload(_strict_mapping(raw["intent"]))
        order_snapshot = snapshots.get(intent.instrument_id)
        if order_snapshot is None:
            raise ValueError
        order = _order_from_payload(_strict_mapping(raw["order"]))
        submitted_source_at = _parse_datetime(raw["submitted_source_at"])
        submission_bars = tuple(
            bar for bar in order_snapshot.bars if bar.at <= submitted_source_at
        )
        if not submission_bars or submission_bars[-1].at != submitted_source_at:
            raise ValueError
        _validate_intent(
            intent,
            snapshot=order_snapshot.model_copy(update={"bars": submission_bars}),
        )
        if (
            order.order_id in orders
            or intent.intent_id in client_orders
            or order.order_id != _paper_order_id(configuration.session_id, intent.intent_id)
            or order.client_order_id != intent.intent_id
            or order.broker != _PAPER_CAPABILITIES.broker
            or order.instrument_id != intent.instrument_id
            or order.requested_quantity != intent.quantity
            or order.updated_at < order.submitted_at
        ):
            raise ValueError
        remaining_notional = _parse_optional_decimal(
            raw["remaining_notional"],
            nonnegative=True,
        )
        cumulative_notional = _parse_decimal(
            raw["cumulative_filled_notional"],
            nonnegative=True,
        )
        cumulative_fees = _parse_decimal(raw["cumulative_fees"], nonnegative=True)
        stop_triggered = raw["stop_triggered"]
        if type(stop_triggered) is not bool:
            raise ValueError
        if (intent.notional is None) is not (remaining_notional is None):
            raise ValueError
        record = _PaperOrderRecord(
            intent=intent,
            order=order,
            submitted_source_at=submitted_source_at,
            remaining_notional=remaining_notional,
            cumulative_filled_notional=cumulative_notional,
            cumulative_fees=cumulative_fees,
            stop_triggered=stop_triggered,
        )
        orders[order.order_id] = record
        client_orders[intent.intent_id] = order.order_id

    fills: list[Fill] = []
    fill_ids: set[str] = set()
    for raw_fill in _strict_sequence(payload["fills"]):
        fill = _fill_from_payload(_strict_mapping(raw_fill))
        if fill.fill_id in fill_ids:
            raise ValueError
        fills.append(fill)
        fill_ids.add(fill.fill_id)
    if fill_sequence != len(fills):
        raise ValueError

    prices = {
        _strict_identifier(key): _parse_decimal(value, positive=True)
        for key, value in _strict_mapping(payload["market_prices"]).items()
    }
    instruments: dict[str, Instrument] = {}
    for raw_instrument in _strict_sequence(payload["instruments"]):
        instrument = _instrument_from_payload(_strict_mapping(raw_instrument))
        instrument_id = _instrument_id(instrument)
        if instrument_id in instruments:
            raise ValueError
        instruments[instrument_id] = instrument

    ledger_state = _ledger_state_from_payload(_strict_mapping(payload["ledger"]))
    if (
        ledger_state.starting_cash != configuration.starting_cash
        or ledger_state.currency != configuration.currency
        or dict(ledger_state.market_prices) != prices
        or set(ledger_state.fill_ids) != fill_ids
    ):
        raise ValueError
    ledger = PortfolioLedger.from_state(ledger_state)

    latest_raw = payload["latest_at"]
    latest_at = _parse_datetime(latest_raw) if latest_raw is not None else None
    key_raw = payload["last_event_key"]
    last_event_key = None
    if key_raw is not None:
        key_values = _strict_sequence(key_raw)
        if len(key_values) != 3:
            raise ValueError
        last_event_key = (
            _parse_datetime(key_values[0]),
            _parse_datetime(key_values[1]),
            _strict_identifier(key_values[2]),
        )

    _validate_replayed_relationships(
        orders=orders,
        fills=fills,
        snapshots=snapshots,
        instruments=instruments,
        ledger_state=ledger_state,
        latest_at=latest_at,
    )
    return _PaperState(
        orders=orders,
        client_orders=client_orders,
        snapshots=snapshots,
        fills=fills,
        event_sequence=event_sequence,
        fill_sequence=fill_sequence,
        ledger=ledger,
        market_prices=prices,
        instruments=instruments,
        latest_at=latest_at,
        last_event_key=last_event_key,
    )


def _validate_replayed_relationships(
    *,
    orders: Mapping[str, _PaperOrderRecord],
    fills: list[Fill],
    snapshots: Mapping[str, MarketSnapshot],
    instruments: Mapping[str, Instrument],
    ledger_state: PortfolioLedgerState,
    latest_at: datetime | None,
) -> None:
    if latest_at is None:
        raise ValueError
    fills_by_order: dict[str, list[Fill]] = {order_id: [] for order_id in orders}
    for fill in fills:
        record = orders.get(fill.order_id)
        if (
            record is None
            or fill.instrument_id != record.order.instrument_id
            or fill.side is not record.intent.side
            or fill.filled_at < record.order.submitted_at
            or fill.filled_at > latest_at
            or fill.instrument_id not in instruments
        ):
            raise ValueError
        fills_by_order[fill.order_id].append(fill)
    for order_id, record in orders.items():
        order_fills = fills_by_order[order_id]
        filled_quantity = sum((fill.quantity for fill in order_fills), Decimal("0"))
        filled_notional = sum(
            (fill.quantity * fill.price for fill in order_fills),
            Decimal("0"),
        )
        fees = sum((fill.fee for fill in order_fills), Decimal("0"))
        average = (
            filled_notional / filled_quantity
            if filled_quantity > Decimal("0")
            else None
        )
        if (
            record.order.filled_quantity != filled_quantity
            or record.order.average_fill_price != average
            or record.cumulative_filled_notional != filled_notional
            or record.cumulative_fees != fees
            or record.order.updated_at > latest_at
            or record.submitted_source_at
            > snapshots[record.order.instrument_id].source_at
        ):
            raise ValueError
        if record.intent.quantity is not None:
            remaining = record.intent.quantity - filled_quantity
            if remaining < Decimal("0") or record.remaining_notional is not None:
                raise ValueError
        else:
            assert record.intent.notional is not None
            remaining = record.intent.notional - filled_notional
            if remaining < Decimal("0") or record.remaining_notional != remaining:
                raise ValueError
        is_complete = remaining == Decimal("0")
        if (
            (record.order.status is OrderStatus.FILLED) is not is_complete
            or (
                record.order.status is OrderStatus.PARTIALLY_FILLED
                and filled_quantity == Decimal("0")
            )
        ):
            raise ValueError

    reconstructed = PortfolioLedger(
        starting_cash=ledger_state.starting_cash,
        currency=ledger_state.currency,
    )
    for fill in fills:
        reconstructed.apply_fill(fill)
    reconstructed_state = reconstructed.export_state()
    if (
        reconstructed_state.cash != ledger_state.cash
        or reconstructed_state.positions != ledger_state.positions
        or reconstructed_state.fill_ids != ledger_state.fill_ids
        or reconstructed_state.gross_realized_pnl != ledger_state.gross_realized_pnl
        or reconstructed_state.fees != ledger_state.fees
    ):
        raise ValueError


def _require_exact_keys(payload: Mapping[str, object], expected: set[str]) -> None:
    if set(payload) != expected:
        raise ValueError


def _strict_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError
    return cast(Mapping[str, object], value)


def _strict_sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError
    return tuple(value)


def _strict_string(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError
    return value


def _strict_identifier(value: object) -> str:
    result = _strict_string(value)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ValueError
    return result


def _strict_nonnegative_int(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError
    return value


def _parse_decimal(
    value: object,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Decimal:
    if (
        positive and nonnegative
        or not isinstance(value, str)
        or not value
        or len(value) > 128
        or re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value) is None
    ):
        raise ValueError
    result = Decimal(value)
    if (
        not result.is_finite()
        or (positive and result <= Decimal("0"))
        or (nonnegative and result < Decimal("0"))
    ):
        raise ValueError
    return result


def _parse_optional_decimal(
    value: object,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Decimal | None:
    if value is None:
        return None
    return _parse_decimal(value, positive=positive, nonnegative=nonnegative)


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise ValueError
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError
    normalized = result.astimezone(UTC)
    if _datetime_text(normalized) != value:
        raise ValueError
    return normalized


def _validate_replay_group(
    records: tuple[EventRecord, ...],
    *,
    state: _PaperState,
    previous_state: _PaperState | None,
    session_id: str,
) -> None:
    if not records and previous_state is None:
        raise ValueError
    statuses = (
        {}
        if previous_state is None
        else {
            order_id: record.order.status
            for order_id, record in previous_state.orders.items()
        }
    )
    if previous_state is not None:
        if (
            not set(previous_state.orders).issubset(state.orders)
            or state.fills[: len(previous_state.fills)] != previous_state.fills
            or any(
                state.instruments.get(instrument_id) != instrument
                for instrument_id, instrument in previous_state.instruments.items()
            )
        ):
            raise ValueError
        for order_id, previous in previous_state.orders.items():
            current = state.orders[order_id]
            if (
                current.intent != previous.intent
                or current.order.order_id != previous.order.order_id
                or current.order.client_order_id != previous.order.client_order_id
                or current.order.instrument_id != previous.order.instrument_id
                or current.order.submitted_at != previous.order.submitted_at
                or current.order.requested_quantity != previous.order.requested_quantity
                or current.order.filled_quantity < previous.order.filled_quantity
            ):
                raise ValueError

    authoritative_edges = {
        (
            _strict_identifier(record.payload.get("broker_order_id")),
            OrderStatus(_strict_string(record.payload.get("prior_status"))),
            OrderStatus(_strict_string(record.payload.get("new_status"))),
        )
        for record in records
        if record.kind == "paper.order.fill"
        and record.payload.get("source") == "AUTHORITATIVE_RECONCILIATION"
    }
    last_transition: dict[str, tuple[OrderStatus, OrderStatus]] = {}
    fills_by_id = {fill.fill_id: fill for fill in state.fills}
    for row in records:
        payload = _strict_mapping(row.payload)
        client_id = _strict_identifier(payload.get("client_intent_id"))
        broker_id = _strict_identifier(payload.get("broker_order_id"))
        if row.kind == "paper.snapshot.duplicate":
            _require_exact_keys(
                payload,
                {"client_intent_id", "broker_order_id", "outcome"},
            )
            if (
                client_id != broker_id
                or client_id not in state.snapshots
                or payload["outcome"] != "IGNORED"
            ):
                raise ValueError
            continue

        current_record = state.orders.get(broker_id)
        if (
            current_record is None
            or current_record.order.client_order_id != client_id
            or broker_id != _paper_order_id(session_id, client_id)
        ):
            raise ValueError

        if row.kind == "paper.order.submitted":
            _require_exact_keys(
                payload,
                {
                    "client_intent_id",
                    "broker_order_id",
                    "instrument_id",
                    "side",
                    "order_type",
                    "intent",
                    "new_status",
                },
            )
            intent = _intent_from_payload(_strict_mapping(payload["intent"]))
            if (
                broker_id in statuses
                or intent != current_record.intent
                or payload["instrument_id"] != intent.instrument_id
                or payload["side"] != intent.side.value
                or payload["order_type"] != intent.order_type.value
                or payload["new_status"] != OrderStatus.PROPOSED.value
            ):
                raise ValueError
            statuses[broker_id] = OrderStatus.PROPOSED
            continue

        prior = OrderStatus(_strict_string(payload.get("prior_status")))
        new = OrderStatus(_strict_string(payload.get("new_status")))
        if row.kind == "paper.order.transition":
            _require_exact_keys(
                payload,
                {"client_intent_id", "broker_order_id", "prior_status", "new_status"},
            )
            if statuses.get(broker_id) is not prior:
                raise ValueError
            if (broker_id, prior, new) not in authoritative_edges:
                OrderStateMachine.transition(
                    prior,
                    new,
                    at=row.occurred_at,
                    emit=lambda _event: None,
                    client_intent_id=client_id,
                    broker_order_id=broker_id,
                )
            elif prior is not OrderStatus.UNKNOWN or new not in {
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.FILLED,
            }:
                raise ValueError
            statuses[broker_id] = new
            last_transition[broker_id] = (prior, new)
            continue

        if row.kind in {"paper.order.duplicate", "paper.order.duplicate_conflict"}:
            _require_exact_keys(
                payload,
                {
                    "client_intent_id",
                    "broker_order_id",
                    "outcome",
                    "prior_status",
                    "new_status",
                },
            )
            expected = (
                "IDEMPOTENT_REPLAY"
                if row.kind == "paper.order.duplicate"
                else "CONFLICT"
            )
            if (
                prior is not new
                or statuses.get(broker_id) is not prior
                or payload["outcome"] != expected
            ):
                raise ValueError
            continue

        if row.kind == "paper.order.rejected":
            _require_exact_keys(
                payload,
                {
                    "client_intent_id",
                    "broker_order_id",
                    "reason_code",
                    "prior_status",
                    "new_status",
                },
            )
            reason = payload["reason_code"]
            if (
                not isinstance(reason, str)
                or _REASON_CODE.fullmatch(reason) is None
                or statuses.get(broker_id) is not prior
                or new is not OrderStatus.REJECTED
            ):
                raise ValueError
            continue

        if row.kind == "paper.order.stop_triggered":
            _require_exact_keys(
                payload,
                {
                    "client_intent_id",
                    "broker_order_id",
                    "trigger_price",
                    "prior_status",
                    "new_status",
                },
            )
            if (
                prior is not new
                or statuses.get(broker_id) is not prior
                or _parse_decimal(payload["trigger_price"], positive=True)
                != current_record.intent.trigger_price
            ):
                raise ValueError
            continue

        if row.kind != "paper.order.fill":
            raise ValueError
        authoritative = "source" in payload
        common_fill_keys = {
            "client_intent_id",
            "broker_order_id",
            "fill_id",
            "quantity",
            "price",
            "fee",
            "instrument",
            "cumulative_filled_quantity",
            "cumulative_filled_notional",
            "cumulative_fees",
            "prior_status",
            "new_status",
        }
        _require_exact_keys(
            payload,
            common_fill_keys
            | (
                {"source"}
                if authoritative
                else {"remaining_quantity", "requested_notional", "remaining_notional"}
            ),
        )
        fill_id = _strict_identifier(payload["fill_id"])
        fill = fills_by_id.get(fill_id)
        instrument = _instrument_from_payload(_strict_mapping(payload["instrument"]))
        if (
            fill is None
            or fill.order_id != broker_id
            or fill.quantity != _parse_decimal(payload["quantity"], positive=True)
            or fill.price != _parse_decimal(payload["price"], positive=True)
            or fill.fee != _parse_decimal(payload["fee"], nonnegative=True)
            or state.instruments.get(fill.instrument_id) != instrument
            or statuses.get(broker_id) is not new
            or (
                authoritative
                and (
                    payload["source"] != "AUTHORITATIVE_RECONCILIATION"
                    or prior is not OrderStatus.UNKNOWN
                )
            )
            or (
                not authoritative
                and last_transition.get(broker_id) != (prior, new)
            )
        ):
            raise ValueError
        _parse_decimal(payload["cumulative_filled_quantity"], positive=True)
        _parse_decimal(payload["cumulative_filled_notional"], positive=True)
        _parse_decimal(payload["cumulative_fees"], nonnegative=True)
        if not authoritative:
            _parse_decimal(payload["remaining_quantity"], nonnegative=True)
            _parse_optional_decimal(payload["requested_notional"], positive=True)
            _parse_optional_decimal(payload["remaining_notional"], nonnegative=True)

    if set(statuses) != set(state.orders) or any(
        statuses[order_id] is not record.order.status
        for order_id, record in state.orders.items()
    ):
        raise ValueError


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
