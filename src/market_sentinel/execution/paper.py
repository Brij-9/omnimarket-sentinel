"""Deterministic, transactional, credential-free paper execution."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import islice
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
from market_sentinel.operations.audit import AuditEvent, AuditLog
from market_sentinel.portfolio.ledger import PortfolioLedger, PortfolioLedgerCompactState
from market_sentinel.storage.events import EventRecord, EventStore, validate_event_payload


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
class SessionHead:
    """Externally retained trust anchor for one complete paper session stream."""

    session_id: str
    event_count: int
    operation_count: int
    head_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.session_id, str)
            or _IDENTIFIER.fullmatch(self.session_id) is None
            or type(self.event_count) is not int
            or self.event_count < 0
            or type(self.operation_count) is not int
            or self.operation_count < 0
            or not isinstance(self.head_digest, str)
            or _SNAPSHOT_HASH.fullmatch(self.head_digest) is None
        ):
            raise ValueError("session head must contain canonical bounded values")


@dataclass(frozen=True, slots=True)
class RollingMarketWindow:
    """One explicit post-prefix market delta bound to the retained cursor."""

    instrument_id: str
    observed_at: datetime
    source_at: datetime
    prior_bar_count: int
    prior_bars_digest: str
    overlap: tuple[Bar, ...]
    new_bar: Bar
    provider: str
    max_age_seconds: int


@dataclass(frozen=True, slots=True)
class _PaperOrderRecord:
    intent: OrderIntent
    order: BrokerOrder
    submitted_source_at: datetime
    submission_reference_price: Decimal
    remaining_notional: Decimal | None
    cumulative_filled_notional: Decimal = Decimal("0")
    cumulative_fees: Decimal = Decimal("0")
    stop_triggered: bool = False
    fill_count: int = 0
    last_fill_at: datetime | None = None
    last_fill_id: str | None = None


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


@dataclass(frozen=True, slots=True)
class _MarketInput:
    snapshot: MarketSnapshot
    instrument: Instrument | None
    duplicate: bool


@dataclass(frozen=True, slots=True)
class _MarketCursor:
    total_count: int
    digest: str
    window: tuple[Bar, ...]


@dataclass(slots=True)
class _PaperState:
    orders: dict[str, _PaperOrderRecord]
    client_orders: dict[str, str]
    snapshots: dict[str, MarketSnapshot]
    cursors: dict[str, _MarketCursor]
    fills: list[Fill]
    event_sequence: int
    operation_sequence: int
    fill_sequence: int
    state_digest: str
    fill_ids_digest: str
    ledger: PortfolioLedger
    market_prices: dict[str, Decimal]
    instruments: dict[str, Instrument]
    latest_at: datetime | None
    last_event_key: tuple[datetime, datetime, str] | None


@dataclass(frozen=True, slots=True)
class _Missing:
    pass


_MISSING = _Missing()


@dataclass(slots=True)
class _StateJournal:
    """Key-scoped undo data for one staged in-place operation."""

    state: _PaperState
    order_values: dict[str, _PaperOrderRecord | _Missing]
    client_values: dict[str, str | _Missing]
    snapshot_values: dict[str, MarketSnapshot | _Missing]
    cursor_values: dict[str, _MarketCursor | _Missing]
    instrument_values: dict[str, Instrument | _Missing]
    market_price_values: dict[str, Decimal | _Missing]
    fills_length: int
    ledger_state: PortfolioLedgerCompactState
    event_sequence: int
    operation_sequence: int
    fill_sequence: int
    state_digest: str
    fill_ids_digest: str
    latest_at: datetime | None
    last_event_key: tuple[datetime, datetime, str] | None

    @classmethod
    def capture(
        cls,
        state: _PaperState,
        *,
        order_ids: Iterable[str] = (),
        client_ids: Iterable[str] = (),
        instrument_ids: Iterable[str] = (),
    ) -> _StateJournal:
        order_keys = tuple(order_ids)
        client_keys = tuple(client_ids)
        instrument_keys = tuple(instrument_ids)
        return cls(
            state=state,
            order_values=_capture_mapping_values(state.orders, order_keys),
            client_values=_capture_mapping_values(state.client_orders, client_keys),
            snapshot_values=_capture_mapping_values(state.snapshots, instrument_keys),
            cursor_values=_capture_mapping_values(state.cursors, instrument_keys),
            instrument_values=_capture_mapping_values(state.instruments, instrument_keys),
            market_price_values=_capture_mapping_values(
                state.market_prices,
                instrument_keys,
            ),
            fills_length=len(state.fills),
            ledger_state=state.ledger.compact_state(),
            event_sequence=state.event_sequence,
            operation_sequence=state.operation_sequence,
            fill_sequence=state.fill_sequence,
            state_digest=state.state_digest,
            fill_ids_digest=state.fill_ids_digest,
            latest_at=state.latest_at,
            last_event_key=state.last_event_key,
        )

    def rollback(self) -> None:
        added_fills = tuple(self.state.fills[self.fills_length :])
        self.state.ledger.restore_compact_state(
            self.ledger_state,
            added_fill_ids=tuple(fill.fill_id for fill in added_fills),
        )
        del self.state.fills[self.fills_length :]
        _restore_mapping_values(self.state.orders, self.order_values)
        _restore_mapping_values(self.state.client_orders, self.client_values)
        _restore_mapping_values(self.state.snapshots, self.snapshot_values)
        _restore_mapping_values(self.state.cursors, self.cursor_values)
        _restore_mapping_values(self.state.instruments, self.instrument_values)
        _restore_mapping_values(self.state.market_prices, self.market_price_values)
        self.state.event_sequence = self.event_sequence
        self.state.operation_sequence = self.operation_sequence
        self.state.fill_sequence = self.fill_sequence
        self.state.state_digest = self.state_digest
        self.state.fill_ids_digest = self.fill_ids_digest
        self.state.latest_at = self.latest_at
        self.state.last_event_key = self.last_event_key


def _capture_mapping_values[MappingValue](
    mapping: Mapping[str, MappingValue],
    keys: Iterable[str],
) -> dict[str, MappingValue | _Missing]:
    return {key: mapping.get(key, _MISSING) for key in keys}


def _restore_mapping_values[MappingValue](
    mapping: dict[str, MappingValue],
    values: Mapping[str, MappingValue | _Missing],
) -> None:
    for key, value in values.items():
        if isinstance(value, _Missing):
            mapping.pop(key, None)
        else:
            mapping[key] = value


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")
_VENUE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_CURRENCY = re.compile(r"^[A-Z][A-Z0-9]{2,11}$")
_TIMEZONE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_+./-]{0,127}$")
_SNAPSHOT_HASH = re.compile(r"^[0-9a-f]{64}$")
_REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_PROVIDERS = frozenset({"fixture", "backtest", "yfinance", "alpaca", "groww", "ccxt", "paper"})
_MAX_DECIMAL_DIGITS = 34
_MAX_DECIMAL_ADJUSTED_EXPONENT = 64
_MAX_DECIMAL_TEXT = 96
_MAX_INITIAL_BARS = 1024
_MARKET_CURSOR_WINDOW = 64
_MAX_SESSION_EVENTS = 1_000_000
_MAX_SESSION_ORDERS = 4_096
_MAX_SESSION_FILLS = 100_000
_MAX_SESSION_INSTRUMENTS = 128
_MAX_GROUP_ACTIVITIES = 512
_MAX_EVENT_PAYLOAD_BYTES = 65_536
_MAX_FILLS_PER_ORDER = 10_000
_EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()
_EMPTY_BAR_DIGEST = hashlib.sha256(b"paper-market-bars-v1").hexdigest()
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
    operation requires ``durable=True``, a stable caller-supplied ``session_id``, and the
    application's exact EventStore-backed ``AuditLog``. Durable continuation must use
    :meth:`rehydrate_durable` with that same log and an externally protected session head.
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
        if durable and (
            type(audit_log) is not AuditLog
            or type(audit_log.event_store) is not EventStore
        ):
            raise ValueError("durable paper execution requires the application AuditLog")
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
        replay_configuration = _ReplayConfiguration(
            session_id=effective_session,
            starting_cash=initial_cash,
            currency=currency,
            costs=model.costs,
            max_volume_participation=participation,
        )
        self._state = _PaperState(
            orders={},
            client_orders={},
            snapshots={},
            cursors={},
            fills=[],
            event_sequence=0,
            operation_sequence=0,
            fill_sequence=0,
            state_digest=_initial_state_digest(replay_configuration),
            fill_ids_digest=_EMPTY_DIGEST,
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

    @property
    def session_head(self) -> SessionHead:
        """Return the head a caller must protect separately from the event store."""
        with self._lock:
            return SessionHead(
                session_id=self._session_id,
                event_count=self._state.event_sequence,
                operation_count=self._state.operation_sequence,
                head_digest=self._state.state_digest,
            )

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
            _validate_intent_shape(intent)
            if intent.intent_id in self._state.client_orders:
                order_id = self._state.client_orders[intent.intent_id]
                journal = _StateJournal.capture(
                    self._state,
                    order_ids=(order_id,),
                    client_ids=(intent.intent_id,),
                )
                try:
                    staged, specs, existing, conflict, at = _execute_idempotency_kernel(
                        self._state,
                        intent,
                    )
                    self._commit(staged, specs, at=at)
                except BaseException:
                    journal.rollback()
                    raise
                if conflict:
                    raise DuplicateIntentConflict(
                        "client intent ID already belongs to a different canonical intent"
                    )
                return existing.order
            order_id = _paper_order_id(self._session_id, intent.intent_id)
            journal = _StateJournal.capture(
                self._state,
                order_ids=(order_id,),
                client_ids=(intent.intent_id,),
                instrument_ids=(intent.instrument_id,),
            )
            try:
                staged, specs, order, submitted_at = _execute_submit_kernel(
                    self._state,
                    intent,
                    snapshot,
                    session_id=self._session_id,
                )
                self._commit(staged, specs, at=submitted_at)
            except BaseException:
                journal.rollback()
                raise
            return order

    def on_snapshot(
        self,
        snapshot: MarketSnapshot,
        instrument: Instrument,
    ) -> tuple[Fill, ...]:
        with self._lock:
            _validate_watermark(self._state, snapshot.observed_at.astimezone(UTC))
            if len(_relevant_instrument_ids(self._state)) != 1:
                raise ValueError(
                    "market cohort requires one atomic on_snapshots batch"
                )
            return self.on_snapshots(((snapshot, instrument),))

    def on_snapshots(
        self,
        events: tuple[tuple[MarketSnapshot, Instrument], ...],
    ) -> tuple[Fill, ...]:
        """Process natural full prefixes up to the 1,024-bar public input bound."""
        with self._lock:
            if not isinstance(events, tuple) or not events:
                raise ValueError("snapshot batch must be a nonempty tuple")
            normalized: list[tuple[MarketSnapshot, Instrument]] = []
            for item in events:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError("snapshot batch entries must contain snapshot and instrument")
                snapshot, instrument = item
                instrument_id = _instrument_id(instrument)
                if not isinstance(snapshot, MarketSnapshot):
                    raise ValueError("snapshot must be a MarketSnapshot")
                if snapshot.instrument_id != instrument_id:
                    raise ValueError("snapshot and instrument identity must match")
                normalized.append(
                    (
                        _normalize_full_prefix(self._state, snapshot, instrument_id),
                        instrument,
                    )
                )
            return self._process_snapshots(tuple(normalized))

    def on_rolling_snapshot(
        self,
        window: RollingMarketWindow,
        instrument: Instrument,
    ) -> tuple[Fill, ...]:
        """Process one explicit cursor-bound 64-bar overlap plus one unseen bar."""
        with self._lock:
            if len(_relevant_instrument_ids(self._state)) != 1:
                raise ValueError(
                    "market cohort requires one atomic on_rolling_snapshots batch"
                )
            return self.on_rolling_snapshots(((window, instrument),))

    def on_rolling_snapshots(
        self,
        events: tuple[tuple[RollingMarketWindow, Instrument], ...],
    ) -> tuple[Fill, ...]:
        """Atomically process an exact same-time cohort of explicit rolling deltas."""
        with self._lock:
            if not isinstance(events, tuple) or not events:
                raise ValueError("rolling snapshot batch must be a nonempty tuple")
            normalized: list[tuple[MarketSnapshot, Instrument]] = []
            for item in events:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError("rolling entries must contain window and instrument")
                window, instrument = item
                instrument_id = _instrument_id(instrument)
                if not isinstance(window, RollingMarketWindow):
                    raise ValueError("rolling input must be a RollingMarketWindow")
                if window.instrument_id != instrument_id:
                    raise ValueError("rolling input and instrument identity must match")
                normalized.append(
                    (
                        _snapshot_from_rolling_window(self._state, window, instrument),
                        instrument,
                    )
                )
            return self._process_snapshots(tuple(normalized))

    def _process_snapshots(
        self,
        events: tuple[tuple[MarketSnapshot, Instrument], ...],
    ) -> tuple[Fill, ...]:
        """Process prevalidated bounded snapshots through the shared market kernel."""
        with self._lock:
            validated: list[
                tuple[tuple[datetime, datetime, str], MarketSnapshot, Instrument, bool]
            ] = []
            seen_instruments: set[str] = set()
            relevant_instruments = _relevant_instrument_ids(self._state)
            for item in events:
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
                _validate_snapshot(
                    snapshot,
                    expected_instrument_id=instrument_id,
                    max_bars=_MARKET_CURSOR_WINDOW + 1,
                )
                _validate_watermark(
                    self._state,
                    snapshot.observed_at.astimezone(UTC),
                )
                if instrument.asset_class not in _PAPER_CAPABILITIES.supported_asset_classes:
                    raise ValueError("instrument asset class is unsupported by paper execution")
                key = _event_key(snapshot)
                validated.append((key, snapshot, instrument, duplicate))

            if seen_instruments != relevant_instruments:
                raise ValueError("snapshot batch must include the exact relevant cohort")
            observed_instants = {
                snapshot.observed_at.astimezone(UTC) for _, snapshot, _, _ in validated
            }
            if len(observed_instants) != 1:
                raise ValueError("snapshot cohort must share one observed_at")

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

            commit_at = max(snapshot.observed_at for _, snapshot, _, _ in validated)
            order_ids = tuple(
                order_id
                for order_id, record in self._state.orders.items()
                if record.order.instrument_id in seen_instruments
            )
            journal = _StateJournal.capture(
                self._state,
                order_ids=order_ids,
                instrument_ids=seen_instruments,
            )
            try:
                staged, specs, produced = _execute_market_kernel(
                    self._state,
                    tuple(
                        _MarketInput(snapshot, instrument, duplicate)
                        for _, snapshot, instrument, duplicate in validated
                    ),
                    fill_model=self._fill_model,
                    max_volume_participation=self._max_volume_participation,
                    session_id=self._session_id,
                )
                self._commit(staged, specs, at=commit_at)
            except BaseException:
                journal.rollback()
                raise
            return produced

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
            journal = _StateJournal.capture(
                self._state,
                order_ids=(authoritative_order.order_id,),
                instrument_ids=(_instrument_id(instrument),),
            )
            try:
                staged, specs, at = _execute_reconciliation_kernel(
                    self._state,
                    authoritative_order,
                    new_fills,
                    instrument,
                    currency=self._currency,
                )
                self._commit(staged, specs, at=at)
            except BaseException:
                journal.rollback()
                raise
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
            journal = _StateJournal.capture(self._state, order_ids=(order_id,))
            try:
                staged, specs, updated, resolved_at = _execute_resolution_kernel(
                    self._state,
                    order_id,
                    status,
                    at=at,
                    activity=activity,
                )
                self._commit(staged, specs, at=resolved_at)
            except BaseException:
                journal.rollback()
                raise
            return updated

    @staticmethod
    def _process_order(
        *,
        state: _PaperState,
        record: _PaperOrderRecord,
        snapshot: MarketSnapshot,
        bar: Bar,
        instrument: Instrument,
        available_liquidity: Decimal,
        fill_model: FillModel,
        session_id: str,
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
        if not fill_model.can_fill(
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
        quantity = fill_model.allocate_quantity(
            remaining_quantity=remaining_quantity,
            available_liquidity=available_liquidity,
            quantity_step=instrument.quantity_step,
        )
        if quantity <= Decimal("0"):
            return record, None, available_liquidity, tuple(specs)
        fill_id = f"{session_id}:fill:{state.fill_sequence + 1:020d}"
        candidate = fill_model.fill(
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
            affordable_quantity = fill_model.allocate_quantity(
                remaining_quantity=record.remaining_notional / candidate.price,
                available_liquidity=available_liquidity,
                quantity_step=instrument.quantity_step,
            )
            if affordable_quantity <= Decimal("0"):
                return record, None, available_liquidity, tuple(specs)
            if affordable_quantity < candidate.quantity:
                candidate = fill_model.fill(
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
            fill_count=record.fill_count + 1,
            last_fill_at=candidate.filled_at,
            last_fill_id=candidate.fill_id,
        )
        state.ledger.apply_fill(candidate)
        state.fill_sequence += 1
        state.fills.append(candidate)
        state.fill_ids_digest = _next_fill_ids_digest(
            state.fill_ids_digest,
            candidate.fill_id,
        )
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
                        "filled_at": _datetime_text(candidate.filled_at),
                        "observed_at": _datetime_text(observed_at),
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
        if not specs:
            raise ValueError("paper operation requires first-class activity evidence")
        previous_event_sequence = self._state.event_sequence
        previous_operation_sequence = self._state.operation_sequence
        previous_state_digest = self._state.state_digest
        activity_count = len(specs)
        first_sequence = previous_event_sequence + 1
        final_sequence = previous_event_sequence + activity_count + 1
        staged.event_sequence = final_sequence
        staged.operation_sequence = previous_operation_sequence + 1
        _validate_state_limits(staged, activity_count=activity_count)
        staged.state_digest = _next_state_digest(
            previous_state_digest,
            _spec_activity_facts(specs),
            staged,
        )
        specs.append(
            _AuditSpec(
                kind="paper.state.committed",
                client_intent_id=self._session_id,
                broker_order_id=self._session_id,
                occurred_at=occurred_at,
                prior_status=None,
                new_status=None,
                payload=MappingProxyType(
                    _checkpoint_payload(
                        staged,
                        session_id=self._session_id,
                        starting_cash=self._starting_cash,
                        currency=self._currency,
                        costs=self._fill_model.costs,
                        max_volume_participation=self._max_volume_participation,
                        first_activity_sequence=first_sequence,
                        activity_count=activity_count,
                        operation_kind=_operation_kind(specs),
                        previous_state_digest=previous_state_digest,
                    )
                ),
            )
        )
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
                payload=_freeze_mapping(spec.payload),
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
        for event in durable_events:
            if _payload_size(event.payload) > _MAX_EVENT_PAYLOAD_BYTES:
                raise ValueError("paper durable event payload exceeds bounded size")
        if self._audit_log is not None:
            self._audit_log.record_many(tuple(durable_events))
        self._state = staged
        self._audit_events.extend(prepared)

    @classmethod
    def rehydrate(
        cls,
        records: Iterable[EventRecord],
        *,
        audit_log: AuditRecorder | None = None,
        expected_head: SessionHead | None = None,
        starting_cash: Decimal | None = None,
        session_id: str | None = None,
        currency: str | None = None,
        fill_model: FillModel | None = None,
        max_volume_participation: Decimal | None = None,
    ) -> PaperBroker:
        """Replay caller-supplied records into non-durable current-session state."""
        del audit_log
        return cls._rehydrate_records(
            records,
            expected_head=expected_head,
            starting_cash=starting_cash,
            session_id=session_id,
            currency=currency,
            fill_model=fill_model,
            max_volume_participation=max_volume_participation,
            continuation_audit=None,
        )

    @classmethod
    def rehydrate_durable(
        cls,
        audit_log: AuditLog,
        *,
        expected_head: SessionHead,
        starting_cash: Decimal | None = None,
        session_id: str | None = None,
        currency: str | None = None,
        fill_model: FillModel | None = None,
        max_volume_participation: Decimal | None = None,
    ) -> PaperBroker:
        """Recover and continue from the exact EventStore retained by ``audit_log``."""
        if type(audit_log) is not AuditLog or type(audit_log.event_store) is not EventStore:
            raise ValueError("durable recovery requires an EventStore-backed AuditLog")
        if type(expected_head) is not SessionHead:
            raise ValueError("trusted session head must be an exact SessionHead")
        return cls._rehydrate_records(
            audit_log.event_store.stream(expected_head.session_id),
            expected_head=expected_head,
            starting_cash=starting_cash,
            session_id=session_id,
            currency=currency,
            fill_model=fill_model,
            max_volume_participation=max_volume_participation,
            continuation_audit=audit_log,
        )

    @classmethod
    def _rehydrate_records(
        cls,
        records: Iterable[EventRecord],
        *,
        expected_head: SessionHead | None,
        starting_cash: Decimal | None,
        session_id: str | None,
        currency: str | None,
        fill_model: FillModel | None,
        max_volume_participation: Decimal | None,
        continuation_audit: AuditLog | None,
    ) -> PaperBroker:
        if type(expected_head) is not SessionHead:
            raise ValueError("trusted session head is required for durable recovery")
        try:
            rows = tuple(islice(iter(records), _MAX_SESSION_EVENTS + 1))
            if not rows or len(rows) > _MAX_SESSION_EVENTS:
                raise ValueError
            derived_session = rows[0].aggregate_id
            if (
                not isinstance(derived_session, str)
                or _IDENTIFIER.fullmatch(derived_session) is None
                or expected_head.session_id != derived_session
                or (session_id is not None and session_id != derived_session)
            ):
                raise ValueError
            prior_database_sequence = 0
            prior_occurred_at: datetime | None = None
            configuration: _ReplayConfiguration | None = None
            state: _PaperState | None = None
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
                    or _payload_size(row.payload) > _MAX_EVENT_PAYLOAD_BYTES
                ):
                    raise ValueError
                occurred_at = _aware_utc(row.occurred_at, "replay event timestamp")
                validate_event_payload(row.payload)
                if prior_occurred_at is not None and occurred_at < prior_occurred_at:
                    raise ValueError
                if not isinstance(row.payload, Mapping):
                    raise ValueError
                prior_database_sequence = row.sequence
                prior_occurred_at = occurred_at
                if row.kind == "paper.state.committed":
                    checkpoint = dict(row.payload)
                    if (
                        checkpoint.pop("client_intent_id", None) != derived_session
                        or checkpoint.pop("broker_order_id", None) != derived_session
                        or not pending_activity
                    ):
                        raise ValueError
                    configuration = _configuration_from_checkpoint(
                        checkpoint,
                        session_id=derived_session,
                        existing=configuration,
                    )
                    if state is None:
                        state = _blank_state(configuration)
                    _validate_checkpoint_envelope(
                        checkpoint,
                        state=state,
                        records=tuple(pending_activity),
                        local_sequence=local_sequence,
                        session_id=derived_session,
                    )
                    candidate = _reduce_activity_group(
                        state,
                        tuple(pending_activity),
                        configuration=configuration,
                    )
                    candidate.event_sequence = local_sequence
                    candidate.operation_sequence = state.operation_sequence + 1
                    _validate_state_limits(
                        candidate,
                        activity_count=len(pending_activity),
                    )
                    expected_state_digest = _next_state_digest(
                        state.state_digest,
                        _record_activity_facts(tuple(pending_activity)),
                        candidate,
                    )
                    if (
                        checkpoint["state_digest"] != expected_state_digest
                        or _json_ready(checkpoint["ledger"])
                        != _json_ready(_ledger_checkpoint_payload(candidate))
                    ):
                        raise ValueError
                    candidate.state_digest = expected_state_digest
                    state = candidate
                    pending_activity.clear()
                else:
                    pending_activity.append(row)
                    if len(pending_activity) > _MAX_GROUP_ACTIVITIES:
                        raise ValueError
            if (
                configuration is None
                or state is None
                or pending_activity
                or rows[-1].kind != "paper.state.committed"
                or state.event_sequence != len(rows)
                or expected_head.event_count != state.event_sequence
                or expected_head.operation_count != state.operation_sequence
                or expected_head.head_digest != state.state_digest
            ):
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
                audit_log=continuation_audit,
                durable=continuation_audit is not None,
                session_id=configuration.session_id,
            )
            broker._state = state
            broker._audit_events = [
                _paper_event_from_record(row)
                for row in rows
                if row.kind != "paper.state.committed"
            ]
            return broker
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            ArithmeticError,
            RecursionError,
        ) as error:
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


def _execute_submit_kernel(
    previous: _PaperState,
    intent: OrderIntent,
    snapshot: MarketSnapshot,
    *,
    session_id: str,
) -> tuple[_PaperState, list[_AuditSpec], BrokerOrder, datetime]:
    """Deterministically validate and project one new canonical submission."""
    _validate_intent_shape(intent)
    if intent.intent_id in previous.client_orders:
        raise ValueError("new submission kernel requires an unused client intent ID")
    _validate_snapshot(snapshot, expected_instrument_id=intent.instrument_id)
    _validate_intent(intent, snapshot=snapshot)
    submitted_at = snapshot.observed_at.astimezone(UTC)
    _validate_watermark(previous, submitted_at)
    if intent.created_at.astimezone(UTC) > submitted_at:
        raise ValueError("intent creation must not be after submission snapshot")
    if intent.expires_at.astimezone(UTC) <= submitted_at:
        raise ValueError("intent must remain unexpired at submission")
    _validate_admission_liveness(previous, intent.instrument_id)
    staged = previous
    previous_snapshot = staged.snapshots.get(intent.instrument_id)
    if previous_snapshot is not None and previous_snapshot != snapshot:
        raise ValueError("submit snapshot revision requires on_snapshot processing first")
    if previous_snapshot is None:
        cursor = _cursor_from_bars(snapshot.bars)
        market_kind = "paper.market.submission"
        market_payload: Mapping[str, object] = MappingProxyType(
            {
                "snapshot": _snapshot_payload(snapshot),
                "bar_count": cursor.total_count,
                "bars_digest": cursor.digest,
            }
        )
    else:
        cursor = staged.cursors[intent.instrument_id]
        market_kind = "paper.market.submission_reference"
        market_payload = MappingProxyType(
            {"bar_count": cursor.total_count, "bars_digest": cursor.digest}
        )
    specs: list[_AuditSpec] = [
        _AuditSpec(
            kind=market_kind,
            client_intent_id=intent.instrument_id,
            broker_order_id=intent.instrument_id,
            occurred_at=submitted_at,
            prior_status=None,
            new_status=None,
            payload=market_payload,
        )
    ]
    order_id = _paper_order_id(session_id, intent.intent_id)
    order = BrokerOrder(
        order_id=order_id,
        client_order_id=intent.intent_id,
        broker=_PAPER_CAPABILITIES.broker,
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
        submission_reference_price=snapshot.bars[-1].close,
        remaining_notional=intent.notional,
    )
    staged.client_orders[intent.intent_id] = order_id
    staged.snapshots[intent.instrument_id] = _bounded_snapshot(snapshot)
    staged.cursors[intent.instrument_id] = cursor
    staged.latest_at = max(staged.latest_at or submitted_at, submitted_at)
    return staged, specs, order, submitted_at


def _execute_idempotency_kernel(
    previous: _PaperState,
    intent: OrderIntent,
) -> tuple[_PaperState, list[_AuditSpec], _PaperOrderRecord, bool, datetime]:
    """Deterministically project one canonical duplicate or conflict outcome."""
    _validate_intent_shape(intent)
    order_id = previous.client_orders.get(intent.intent_id)
    if order_id is None:
        raise ValueError("idempotency kernel requires an existing client intent ID")
    staged = previous
    existing = staged.orders[order_id]
    at = staged.latest_at or existing.order.updated_at
    conflict = existing.intent != intent
    specs = [
        _activity_spec(
            kind=("paper.order.duplicate_conflict" if conflict else "paper.order.duplicate"),
            order=existing.order,
            at=at,
            payload={
                "outcome": "CONFLICT" if conflict else "IDEMPOTENT_REPLAY",
                "intent": _intent_payload(intent),
            },
        )
    ]
    return staged, specs, existing, conflict, at


def _execute_resolution_kernel(
    previous: _PaperState,
    order_id: str,
    status: OrderStatus,
    *,
    at: datetime,
    activity: tuple[str, Mapping[str, object]] | None = None,
) -> tuple[_PaperState, list[_AuditSpec], BrokerOrder, datetime]:
    """Deterministically project one non-fill broker status resolution."""
    staged = previous
    record = staged.orders.get(order_id)
    if record is None:
        raise KeyError("paper order not found")
    resolved_at = _aware_utc(at, "resolution timestamp")
    _validate_watermark(previous, resolved_at)
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
    return staged, specs, updated, resolved_at


def _execute_market_kernel(
    previous: _PaperState,
    inputs: tuple[_MarketInput, ...],
    *,
    fill_model: FillModel,
    max_volume_participation: Decimal,
    session_id: str,
) -> tuple[_PaperState, list[_AuditSpec], tuple[Fill, ...]]:
    """Purely project one canonical cohort into state and exact activity grammar."""
    staged = previous
    specs: list[_AuditSpec] = []
    produced: list[Fill] = []
    for market_input in inputs:
        snapshot = market_input.snapshot
        instrument = market_input.instrument
        key = _event_key(snapshot)
        if market_input.duplicate:
            cursor = staged.cursors.get(snapshot.instrument_id)
            if cursor is None:
                raise ValueError("duplicate market input has no cursor")
            specs.append(
                _AuditSpec(
                    kind="paper.snapshot.duplicate",
                    client_intent_id=snapshot.instrument_id,
                    broker_order_id=snapshot.instrument_id,
                    occurred_at=snapshot.observed_at,
                    prior_status=None,
                    new_status=None,
                    payload=MappingProxyType(
                        {
                            "outcome": "IGNORED",
                            "bar_count": cursor.total_count,
                            "bars_digest": cursor.digest,
                        }
                    ),
                )
            )
            continue
        if instrument is None:
            raise ValueError("new market input requires exact instrument metadata")
        prior_snapshot = staged.snapshots.get(snapshot.instrument_id)
        prior_cursor = staged.cursors.get(snapshot.instrument_id)
        if prior_snapshot is None or prior_cursor is None:
            raise ValueError("market input has no submitted snapshot")
        [bar] = snapshot.bars[len(prior_snapshot.bars) :]
        cursor = _MarketCursor(
            total_count=prior_cursor.total_count + 1,
            digest=_next_bar_digest(prior_cursor.digest, bar),
            window=(prior_cursor.window + (bar,))[-_MARKET_CURSOR_WINDOW:],
        )
        liquidity = fill_model.liquidity_budget(
            event_volume=bar.volume,
            max_participation=max_volume_participation,
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
            record, fill, liquidity, order_specs = PaperBroker._process_order(
                state=staged,
                record=record,
                snapshot=snapshot,
                bar=bar,
                instrument=instrument,
                available_liquidity=liquidity,
                fill_model=fill_model,
                session_id=session_id,
            )
            staged.orders[order_id] = record
            specs.extend(order_specs)
            if fill is not None:
                produced.append(fill)
        staged.snapshots[snapshot.instrument_id] = _bounded_snapshot(snapshot)
        staged.cursors[snapshot.instrument_id] = cursor
        observed_at = snapshot.observed_at.astimezone(UTC)
        staged.latest_at = max(observed_at, staged.latest_at or observed_at)
        staged.last_event_key = key
        specs.append(
            _AuditSpec(
                kind="paper.market.observed",
                client_intent_id=snapshot.instrument_id,
                broker_order_id=snapshot.instrument_id,
                occurred_at=snapshot.observed_at,
                prior_status=None,
                new_status=None,
                payload=MappingProxyType(
                    {
                        "instrument": _instrument_payload(instrument),
                        "instrument_id": snapshot.instrument_id,
                        "observed_at": _datetime_text(snapshot.observed_at),
                        "source_at": _datetime_text(snapshot.source_at),
                        "provider": snapshot.provider,
                        "max_age_seconds": snapshot.max_age_seconds,
                        "bar_count": cursor.total_count,
                        "bars_digest": cursor.digest,
                        "bar": _bar_payload(bar),
                    }
                ),
            )
        )
    if (
        any(not market_input.duplicate for market_input in inputs)
        and staged.market_prices
        and staged.latest_at is not None
    ):
        staged.ledger.mark(staged.market_prices, staged.latest_at)
    return staged, specs, tuple(produced)


def _execute_reconciliation_kernel(
    previous: _PaperState,
    authoritative_order: BrokerOrder,
    new_fills: tuple[Fill, ...],
    instrument: Instrument,
    *,
    currency: str,
) -> tuple[_PaperState, list[_AuditSpec], datetime]:
    """Purely validate and project one authoritative UNKNOWN-fill operation."""
    if not isinstance(authoritative_order, BrokerOrder):
        raise ValueError("authoritative fill order is invalid")
    if (
        not isinstance(new_fills, tuple)
        or not new_fills
        or len(new_fills) + 1 > _MAX_GROUP_ACTIVITIES
    ):
        raise ValueError("authoritative fill evidence exceeds activity capacity")
    staged = previous
    record = staged.orders.get(authoritative_order.order_id)
    if record is None or record.order.status is not OrderStatus.UNKNOWN:
        raise ValueError("authoritative fill requires one UNKNOWN paper order")
    current = record.order
    at = _aware_utc(authoritative_order.updated_at, "authoritative update")
    _validate_watermark(previous, at)
    instrument_id = _instrument_id(instrument)
    if instrument.quote_currency != currency:
        raise ValueError("authoritative fill instrument currency is invalid")
    if (
        instrument_id != current.instrument_id
        or authoritative_order.client_order_id != current.client_order_id
        or authoritative_order.broker != _PAPER_CAPABILITIES.broker
        or authoritative_order.instrument_id != current.instrument_id
        or authoritative_order.submitted_at != current.submitted_at
        or authoritative_order.requested_quantity != current.requested_quantity
        or authoritative_order.status
        not in {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED}
        or authoritative_order.filled_quantity <= current.filled_quantity
        or authoritative_order.average_fill_price is None
    ):
        raise ValueError("authoritative fill order identity or status is invalid")
    prior_instrument = staged.instruments.get(instrument_id)
    if prior_instrument is None or prior_instrument != instrument:
        raise ValueError("authoritative fill requires exact persisted instrument metadata")
    if instrument.asset_class not in _PAPER_CAPABILITIES.supported_asset_classes:
        raise ValueError("authoritative fill instrument capability is unsupported")
    if _original_requested_notional(record) < instrument.minimum_notional:
        raise ValueError("authoritative order violates venue minimum notional")

    delta = authoritative_order.filled_quantity - current.filled_quantity
    prior_key = (
        record.last_fill_at or current.submitted_at,
        record.last_fill_id or "",
    )
    fill_notional = Decimal("0")
    fill_fees = Decimal("0")
    fill_quantity = Decimal("0")
    seen_fill_ids: set[str] = set()
    available_cash = staged.ledger.cash
    available_position = next(
        (
            position.quantity
            for position in _state_positions(staged)
            if position.instrument_id == current.instrument_id
        ),
        Decimal("0"),
    )
    for fill in new_fills:
        if not isinstance(fill, Fill):
            raise ValueError("authoritative fill evidence is invalid")
        filled_at = _aware_utc(fill.filled_at, "authoritative fill timestamp")
        key = (filled_at, fill.fill_id)
        if (
            not isinstance(fill.fill_id, str)
            or _IDENTIFIER.fullmatch(fill.fill_id) is None
            or staged.ledger.has_fill_id(fill.fill_id)
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
        if (
            fill.quantity % instrument.quantity_step != Decimal("0")
            or fill.price % instrument.price_tick != Decimal("0")
        ):
            raise ValueError("authoritative fill violates venue precision")
        if not _respects_order_prices(record.intent, fill.price):
            raise ValueError("authoritative fill price violates order protection")
        if record.intent.side is Side.BUY:
            consideration = fill.quantity * fill.price + fill.fee
            if consideration > available_cash:
                raise ValueError("authoritative fill exceeds paper cash")
            available_cash -= consideration
        else:
            if fill.quantity > available_position:
                raise ValueError("authoritative fill exceeds paper position")
            available_position -= fill.quantity
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
                    "filled_at": _datetime_text(fill.filled_at),
                    "observed_at": _datetime_text(at),
                    "instrument": _instrument_payload(instrument),
                    "source": "AUTHORITATIVE_RECONCILIATION",
                    "cumulative_filled_quantity": _decimal_text(cumulative_quantity),
                    "cumulative_filled_notional": _decimal_text(cumulative_notional),
                    "cumulative_fees": _decimal_text(cumulative_fees),
                },
            )
        )
    # Pair each staged ledger mutation with the journal-visible fill suffix.  Any
    # later mark, persistence, or commit failure can then remove exactly these IDs
    # without visiting or copying historical fill identity state.
    for fill in new_fills:
        staged.ledger.apply_fill(fill)
        staged.fills.append(fill)
    staged.orders[current.order_id] = replace(
        record,
        order=authoritative_order,
        remaining_notional=remaining_notional,
        cumulative_filled_notional=total_notional,
        cumulative_fees=record.cumulative_fees + fill_fees,
        fill_count=record.fill_count + len(new_fills),
        last_fill_at=new_fills[-1].filled_at,
        last_fill_id=new_fills[-1].fill_id,
    )
    for fill in new_fills:
        staged.fill_ids_digest = _next_fill_ids_digest(
            staged.fill_ids_digest,
            fill.fill_id,
        )
    staged.fill_sequence += len(new_fills)
    staged.market_prices[instrument_id] = new_fills[-1].price
    staged.instruments[instrument_id] = instrument
    staged.latest_at = at
    staged.ledger.mark(staged.market_prices, at)
    return staged, specs, at


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


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {key: _freeze_value(item) for key, item in value.items()}
    )


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(cast(Mapping[str, object], value))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _validate_intent(intent: object, *, snapshot: MarketSnapshot) -> None:
    _validate_intent_shape(intent)
    assert isinstance(intent, OrderIntent)
    _validate_protective_prices(intent, snapshot)


def _validate_intent_shape(intent: object) -> None:
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
    if intent.side is Side.BUY and intent.stop_loss >= intent.take_profit:
        raise ValueError("buy protection must keep stop_loss below take_profit")
    if intent.side is Side.SELL and intent.stop_loss <= intent.take_profit:
        raise ValueError("sell protection must keep stop_loss above take_profit")
    known_references = tuple(
        price for price in (intent.trigger_price, intent.limit_price) if price is not None
    )
    if known_references:
        if intent.side is Side.BUY and not (
            intent.stop_loss < min(known_references)
            and max(known_references) < intent.take_profit
        ):
            raise ValueError("buy protection must bracket every declared entry price")
        if intent.side is Side.SELL and not (
            intent.stop_loss > max(known_references)
            and min(known_references) > intent.take_profit
        ):
            raise ValueError("sell protection must bracket every declared entry price")
    if intent.order_type is OrderType.STOP_LIMIT:
        assert intent.limit_price is not None and intent.trigger_price is not None
        if intent.side is Side.BUY and intent.limit_price < intent.trigger_price:
            raise ValueError("buy stop-limit protection requires limit at or above trigger")
        if intent.side is Side.SELL and intent.limit_price > intent.trigger_price:
            raise ValueError("sell stop-limit protection requires limit at or below trigger")
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
    created_at = _aware_utc(intent.created_at, "intent created_at")
    expires_at = _aware_utc(intent.expires_at, "intent expires_at")
    if expires_at <= created_at:
        raise ValueError("intent expires_at must follow created_at")


def _validate_snapshot(
    snapshot: object,
    *,
    expected_instrument_id: str,
    max_bars: int = _MAX_INITIAL_BARS,
) -> None:
    if not isinstance(snapshot, MarketSnapshot):
        raise ValueError("snapshot must be a MarketSnapshot")
    if snapshot.instrument_id != expected_instrument_id:
        raise ValueError("snapshot instrument identity must match the order")
    if (
        not isinstance(snapshot.provider, str)
        or snapshot.provider not in _PROVIDERS
    ):
        raise ValueError("snapshot provider must be canonical")
    if type(snapshot.max_age_seconds) is not int or snapshot.max_age_seconds < 0:
        raise ValueError("snapshot max_age_seconds must be a nonnegative integer")
    observed_at = _aware_utc(snapshot.observed_at, "snapshot observed_at")
    source_at = _aware_utc(snapshot.source_at, "snapshot source_at")
    if source_at > observed_at:
        raise ValueError("snapshot source timestamp cannot be in the future")
    if (observed_at - source_at).total_seconds() > snapshot.max_age_seconds:
        raise ValueError("snapshot is stale at observation time")
    if (
        not isinstance(snapshot.bars, tuple)
        or not snapshot.bars
        or len(snapshot.bars) > max_bars
    ):
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


def _normalize_full_prefix(
    state: _PaperState,
    snapshot: MarketSnapshot,
    instrument_id: str,
) -> MarketSnapshot:
    """Normalize a natural bounded provider prefix against the retained cursor window."""
    previous = state.snapshots.get(instrument_id)
    cursor = state.cursors.get(instrument_id)
    if previous is None or cursor is None:
        raise ValueError("snapshot has no submitted paper cursor")
    # A provider rewriting an already accepted prefix is a history revision even
    # when the rewritten candle is internally malformed.  Classify it before the
    # generic bar validator so callers receive the stable cursor-contract error.
    if len(snapshot.bars) == cursor.total_count and snapshot != previous:
        raise ValueError("snapshot revision or backward event is not allowed")
    _validate_snapshot(
        snapshot,
        expected_instrument_id=instrument_id,
        max_bars=_MAX_INITIAL_BARS,
    )
    if snapshot == previous:
        return snapshot
    same_metadata = (
        snapshot.observed_at == previous.observed_at
        and snapshot.source_at == previous.source_at
        and snapshot.provider == previous.provider
        and snapshot.max_age_seconds == previous.max_age_seconds
    )
    if len(snapshot.bars) == cursor.total_count:
        if same_metadata and snapshot.bars[-len(cursor.window) :] == cursor.window:
            return previous
        raise ValueError("snapshot revision or backward event is not allowed")
    if cursor.total_count >= _MAX_INITIAL_BARS:
        raise ValueError("snapshot requires the explicit rolling window API")
    if len(snapshot.bars) != cursor.total_count + 1:
        raise ValueError("snapshot must contain exactly one unseen event")
    overlap = snapshot.bars[-(len(cursor.window) + 1) : -1]
    if overlap != cursor.window:
        raise ValueError("snapshot trailing overlap does not match the retained cursor")
    return snapshot.model_copy(update={"bars": cursor.window + (snapshot.bars[-1],)})


def _snapshot_from_rolling_window(
    state: _PaperState,
    window: RollingMarketWindow,
    instrument: Instrument,
) -> MarketSnapshot:
    """Validate one explicit rolling input and return its bounded kernel snapshot."""
    if type(window) is not RollingMarketWindow:
        raise ValueError("rolling input must be a RollingMarketWindow")
    instrument_id = _instrument_id(instrument)
    cursor = state.cursors.get(instrument_id)
    if (
        cursor is None
        or window.instrument_id != instrument_id
        or type(window.prior_bar_count) is not int
        or window.prior_bar_count != cursor.total_count
        or not isinstance(window.prior_bars_digest, str)
        or window.prior_bars_digest != cursor.digest
        or _SNAPSHOT_HASH.fullmatch(window.prior_bars_digest) is None
        or not isinstance(window.overlap, tuple)
        or len(window.overlap) != len(cursor.window)
        or window.overlap != cursor.window
        or not isinstance(window.new_bar, Bar)
    ):
        raise ValueError("rolling input does not match the retained market cursor")
    snapshot = MarketSnapshot(
        instrument_id=instrument_id,
        observed_at=window.observed_at,
        source_at=window.source_at,
        bars=window.overlap + (window.new_bar,),
        provider=window.provider,
        max_age_seconds=window.max_age_seconds,
    )
    _validate_snapshot(
        snapshot,
        expected_instrument_id=instrument_id,
        max_bars=_MARKET_CURSOR_WINDOW + 1,
    )
    return snapshot


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


def _original_requested_notional(
    record: _PaperOrderRecord,
) -> Decimal:
    if record.intent.notional is not None:
        return record.intent.notional
    quantity = cast(Decimal, record.intent.quantity)
    declared = tuple(
        price
        for price in (record.intent.limit_price, record.intent.trigger_price)
        if price is not None
    )
    if declared:
        return quantity * min(declared)
    return quantity * record.submission_reference_price


def _state_positions(state: _PaperState) -> tuple[Position, ...]:
    if state.latest_at is None:
        return ()
    return state.ledger.snapshot(state.latest_at).positions


def _relevant_instrument_ids(state: _PaperState) -> set[str]:
    active = {
        record.order.instrument_id
        for record in state.orders.values()
        if record.order.status not in _CLOSED
    }
    positions = {position.instrument_id for position in _state_positions(state)}
    return active | positions


def _validate_admission_liveness(state: _PaperState, instrument_id: str) -> None:
    """Ensure the worst next cohort grammar fits before admitting active exposure."""
    active_records = [
        record for record in state.orders.values() if record.order.status not in _CLOSED
    ]
    active_instruments = {
        record.order.instrument_id for record in active_records
    } | {instrument_id}
    position_instruments = {position.instrument_id for position in _state_positions(state)}
    market_rows = len(active_instruments | position_instruments)
    worst_order_rows = 3 * (len(active_records) + 1)
    if market_rows + worst_order_rows > _MAX_GROUP_ACTIVITIES:
        raise ValueError("submission would exceed the next market activity capacity")


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
    if not _bounded_decimal(value) or cast(Decimal, value) <= Decimal("0"):
        raise ValueError(f"{name} must be a finite positive Decimal")
    return cast(Decimal, value)


def _nonnegative_decimal(value: object, name: str) -> Decimal:
    if not _bounded_decimal(value) or cast(Decimal, value) < Decimal("0"):
        raise ValueError(f"{name} must be a finite nonnegative Decimal")
    return cast(Decimal, value)


def _bounded_decimal(value: object) -> bool:
    return _canonical_decimal_text(value) is not None


def _canonical_decimal_text(value: object) -> str | None:
    if not isinstance(value, Decimal) or not value.is_finite():
        return None
    sign, raw_digits, raw_exponent = value.as_tuple()
    if not isinstance(raw_exponent, int) or not raw_digits:
        return None
    last = len(raw_digits)
    while last > 1 and raw_digits[last - 1] == 0:
        last -= 1
    digits = raw_digits[:last]
    if not any(digits):
        return "0"
    exponent = raw_exponent + len(raw_digits) - last
    if len(digits) > _MAX_DECIMAL_DIGITS:
        return None
    adjusted = len(digits) + exponent - 1
    if abs(adjusted) > _MAX_DECIMAL_ADJUSTED_EXPONENT:
        return None
    point = len(digits) + exponent
    body_length = (
        len(digits) + exponent
        if exponent >= 0
        else len(digits) + 1
        if point > 0
        else 2 - point + len(digits)
    )
    if body_length + sign > _MAX_DECIMAL_TEXT:
        return None
    coefficient = "".join(str(digit) for digit in digits)
    if exponent >= 0:
        body = coefficient + "0" * exponent
    elif point > 0:
        body = coefficient[:point] + "." + coefficient[point:]
    else:
        body = "0." + "0" * (-point) + coefficient
    return ("-" if sign else "") + body


def _aware_utc(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _decimal_text(value: Decimal) -> str:
    encoded = _canonical_decimal_text(value)
    if encoded is None:
        raise ValueError("Decimal exceeds canonical paper encoding bounds")
    return encoded


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


def _bars_digest(bars: tuple[Bar, ...]) -> str:
    digest = _EMPTY_BAR_DIGEST
    for bar in bars:
        digest = _next_bar_digest(digest, bar)
    return digest


def _next_bar_digest(previous_digest: str, bar: Bar) -> str:
    encoded = json.dumps(
        _bar_payload(bar),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(bytes.fromhex(previous_digest) + b"\0" + encoded).hexdigest()


def _cursor_from_bars(bars: tuple[Bar, ...]) -> _MarketCursor:
    return _MarketCursor(
        total_count=len(bars),
        digest=_bars_digest(bars),
        window=bars[-_MARKET_CURSOR_WINDOW:],
    )


def _bounded_snapshot(snapshot: MarketSnapshot) -> MarketSnapshot:
    return snapshot.model_copy(
        update={"bars": snapshot.bars[-_MARKET_CURSOR_WINDOW:]}
    )


def _operation_kind(specs: list[_AuditSpec]) -> str:
    kinds = {spec.kind for spec in specs}
    if "paper.order.submitted" in kinds:
        return "SUBMIT"
    if "paper.market.observed" in kinds or "paper.snapshot.duplicate" in kinds:
        return "MARKET"
    if "paper.order.duplicate" in kinds or "paper.order.duplicate_conflict" in kinds:
        return "IDEMPOTENCY"
    if any(
        spec.kind == "paper.order.fill"
        and spec.payload.get("source") == "AUTHORITATIVE_RECONCILIATION"
        for spec in specs
    ):
        return "RECONCILIATION"
    return "RESOLUTION"


def _state_digest(state: _PaperState) -> str:
    return state.state_digest


def _initial_state_digest(configuration: _ReplayConfiguration) -> str:
    return _canonical_digest(
        {
            "schema_version": 3,
            "session_id": configuration.session_id,
            "starting_cash": _decimal_text(configuration.starting_cash),
            "currency": configuration.currency,
            "cost_model": _cost_payload(configuration.costs),
            "max_volume_participation": _decimal_text(
                configuration.max_volume_participation
            ),
            "capabilities": _capabilities_payload(),
        }
    )


def _next_fill_ids_digest(previous_digest: str, fill_id: str) -> str:
    return hashlib.sha256(
        bytes.fromhex(previous_digest) + b"\0" + fill_id.encode("utf-8")
    ).hexdigest()


def _ledger_checkpoint_payload(state: _PaperState) -> Mapping[str, object]:
    ledger_state = state.ledger.compact_state()
    positions_digest = hashlib.sha256(
        json.dumps(
            [
                {
                    "instrument_id": item.instrument_id,
                    "quantity": _decimal_text(item.quantity),
                    "average_price": _decimal_text(item.average_price),
                }
                for item in ledger_state.positions
            ],
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "cash": _decimal_text(ledger_state.cash),
        "market_prices": [
            {"instrument_id": key, "price": _decimal_text(value)}
            for key, value in ledger_state.market_prices
        ],
        "gross_realized_pnl": _decimal_text(ledger_state.gross_realized_pnl),
        "fees": _decimal_text(ledger_state.fees),
        "equity": _decimal_text(ledger_state.equity),
        "peak_equity": _decimal_text(ledger_state.peak_equity),
        "drawdown": _decimal_text(ledger_state.drawdown),
        "positions_digest": positions_digest,
        "fill_ids_digest": state.fill_ids_digest,
    }


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        _json_ready(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _spec_activity_facts(specs: list[_AuditSpec]) -> list[Mapping[str, object]]:
    facts: list[Mapping[str, object]] = []
    for spec in specs:
        payload: dict[str, object] = {
            "client_intent_id": spec.client_intent_id,
            "broker_order_id": spec.broker_order_id,
            **dict(spec.payload),
        }
        if spec.prior_status is not None:
            payload["prior_status"] = spec.prior_status.value
        if spec.new_status is not None:
            payload["new_status"] = spec.new_status.value
        facts.append(
            {
                "kind": spec.kind,
                "occurred_at": _datetime_text(spec.occurred_at),
                "payload": payload,
            }
        )
    return facts


def _record_activity_facts(records: tuple[EventRecord, ...]) -> list[Mapping[str, object]]:
    return [
        {
            "kind": record.kind,
            "occurred_at": _datetime_text(record.occurred_at),
            "payload": record.payload,
        }
        for record in records
    ]


def _next_state_digest(
    previous_digest: str,
    activity_facts: list[Mapping[str, object]],
    state: _PaperState,
) -> str:
    return _canonical_digest(
        {
            "previous_state_digest": previous_digest,
            "activities": activity_facts,
            "event_sequence": state.event_sequence,
            "operation_sequence": state.operation_sequence,
            "fill_sequence": state.fill_sequence,
            "order_count": len(state.orders),
            "instrument_count": len(state.instruments),
            "snapshot_count": len(state.snapshots),
            "market_cursors": [
                {
                    "instrument_id": instrument_id,
                    "total_count": cursor.total_count,
                    "digest": cursor.digest,
                    "latest_at": _datetime_text(cursor.window[-1].at),
                }
                for instrument_id, cursor in sorted(state.cursors.items())
            ],
            "latest_at": (
                _datetime_text(state.latest_at) if state.latest_at is not None else None
            ),
            "last_event_key": (
                [
                    _datetime_text(state.last_event_key[0]),
                    _datetime_text(state.last_event_key[1]),
                    state.last_event_key[2],
                ]
                if state.last_event_key is not None
                else None
            ),
            "ledger": _ledger_checkpoint_payload(state),
        }
    )


def _checkpoint_payload(
    state: _PaperState,
    *,
    session_id: str,
    starting_cash: Decimal,
    currency: str,
    costs: CostModel,
    max_volume_participation: Decimal,
    first_activity_sequence: int,
    activity_count: int,
    operation_kind: str,
    previous_state_digest: str,
) -> dict[str, object]:
    configuration: Mapping[str, object] | None = None
    capabilities: Mapping[str, object] | None = None
    if state.operation_sequence == 1:
        configuration = {
            "starting_cash": _decimal_text(starting_cash),
            "currency": currency,
            "cost_model": _cost_payload(costs),
            "max_volume_participation": _decimal_text(max_volume_participation),
        }
        capabilities = _capabilities_payload()
    return {
        "schema_version": 3,
        "session_id": session_id,
        "event_sequence": state.event_sequence,
        "operation_sequence": state.operation_sequence,
        "operation_id": f"{session_id}:operation:{state.operation_sequence:020d}",
        "operation_kind": operation_kind,
        "first_activity_sequence": first_activity_sequence,
        "activity_count": activity_count,
        "previous_state_digest": previous_state_digest,
        "state_digest": _state_digest(state),
        "configuration": configuration,
        "capabilities": capabilities,
        "ledger": _ledger_checkpoint_payload(state),
    }


def _validate_state_limits(state: _PaperState, *, activity_count: int) -> None:
    instrument_ids = (
        set(state.snapshots)
        | set(state.cursors)
        | set(state.instruments)
        | set(state.market_prices)
        | {record.order.instrument_id for record in state.orders.values()}
        | {position.instrument_id for position in _state_positions(state)}
    )
    if (
        state.event_sequence > _MAX_SESSION_EVENTS
        or len(state.orders) > _MAX_SESSION_ORDERS
        or len(state.fills) > _MAX_SESSION_FILLS
        or len(instrument_ids) > _MAX_SESSION_INSTRUMENTS
        or activity_count > _MAX_GROUP_ACTIVITIES
        or any(record.fill_count > _MAX_FILLS_PER_ORDER for record in state.orders.values())
        or any(
            len(snapshot.bars) > _MARKET_CURSOR_WINDOW
            for snapshot in state.snapshots.values()
        )
        or set(state.snapshots) != set(state.cursors)
        or any(
            cursor.total_count < len(cursor.window)
            or len(cursor.window) > _MARKET_CURSOR_WINDOW
            or not isinstance(cursor.digest, str)
            or _SNAPSHOT_HASH.fullmatch(cursor.digest) is None
            for cursor in state.cursors.values()
        )
    ):
        raise ValueError("paper session exceeds a bounded durable limit")


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _payload_size(payload: Mapping[str, object]) -> int:
    validate_event_payload(payload)
    return len(
        json.dumps(
            _json_ready(payload),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _configuration_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    session_id: str,
    existing: _ReplayConfiguration | None,
) -> _ReplayConfiguration:
    _require_exact_keys(
        checkpoint,
        {
            "schema_version",
            "session_id",
            "event_sequence",
            "operation_sequence",
            "operation_id",
            "operation_kind",
            "first_activity_sequence",
            "activity_count",
            "previous_state_digest",
            "state_digest",
            "configuration",
            "capabilities",
            "ledger",
        },
    )
    if checkpoint["schema_version"] != 3 or checkpoint["session_id"] != session_id:
        raise ValueError
    for digest_name in ("previous_state_digest", "state_digest"):
        digest = checkpoint[digest_name]
        if not isinstance(digest, str) or _SNAPSHOT_HASH.fullmatch(digest) is None:
            raise ValueError
    if existing is not None:
        if checkpoint["configuration"] is not None or checkpoint["capabilities"] is not None:
            raise ValueError
        return existing
    configuration = _strict_mapping(checkpoint["configuration"])
    _require_exact_keys(
        configuration,
        {"starting_cash", "currency", "cost_model", "max_volume_participation"},
    )
    costs_payload = _strict_mapping(configuration["cost_model"])
    _require_exact_keys(
        costs_payload,
        {"fee_bps", "spread_bps", "slippage_bps", "latency_microseconds"},
    )
    costs = CostModel(
        fee_bps=_parse_decimal(costs_payload["fee_bps"], nonnegative=True),
        spread_bps=_parse_decimal(costs_payload["spread_bps"], nonnegative=True),
        slippage_bps=_parse_decimal(costs_payload["slippage_bps"], nonnegative=True),
        latency=timedelta(
            microseconds=_strict_nonnegative_int(costs_payload["latency_microseconds"])
        ),
    )
    starting_cash = _parse_decimal(configuration["starting_cash"], positive=True)
    currency = _strict_string(configuration["currency"])
    participation = _parse_decimal(
        configuration["max_volume_participation"],
        positive=True,
    )
    if (
        _CURRENCY.fullmatch(currency) is None
        or participation > Decimal("1")
        or checkpoint["capabilities"] is None
    ):
        raise ValueError
    _validate_capabilities_payload(_strict_mapping(checkpoint["capabilities"]))
    return _ReplayConfiguration(
        session_id=session_id,
        starting_cash=starting_cash,
        currency=currency,
        costs=costs,
        max_volume_participation=participation,
    )


def _blank_state(configuration: _ReplayConfiguration) -> _PaperState:
    return _PaperState(
        orders={},
        client_orders={},
        snapshots={},
        cursors={},
        fills=[],
        event_sequence=0,
        operation_sequence=0,
        fill_sequence=0,
        state_digest=_initial_state_digest(configuration),
        fill_ids_digest=_EMPTY_DIGEST,
        ledger=PortfolioLedger(
            starting_cash=configuration.starting_cash,
            currency=configuration.currency,
        ),
        market_prices={},
        instruments={},
        latest_at=None,
        last_event_key=None,
    )


def _replay_operation_kind(records: tuple[EventRecord, ...]) -> str:
    kinds = {record.kind for record in records}
    if "paper.order.submitted" in kinds:
        return "SUBMIT"
    if "paper.market.observed" in kinds or "paper.snapshot.duplicate" in kinds:
        return "MARKET"
    if "paper.order.duplicate" in kinds or "paper.order.duplicate_conflict" in kinds:
        return "IDEMPOTENCY"
    if any(
        record.kind == "paper.order.fill"
        and record.payload.get("source") == "AUTHORITATIVE_RECONCILIATION"
        for record in records
    ):
        return "RECONCILIATION"
    return "RESOLUTION"


def _validate_checkpoint_envelope(
    checkpoint: Mapping[str, object],
    *,
    state: _PaperState,
    records: tuple[EventRecord, ...],
    local_sequence: int,
    session_id: str,
) -> None:
    operation_sequence = _strict_nonnegative_int(checkpoint["operation_sequence"])
    first_activity = _strict_nonnegative_int(checkpoint["first_activity_sequence"])
    activity_count = _strict_nonnegative_int(checkpoint["activity_count"])
    event_sequence = _strict_nonnegative_int(checkpoint["event_sequence"])
    operation_kind = _strict_string(checkpoint["operation_kind"])
    if (
        operation_sequence != state.operation_sequence + 1
        or checkpoint["operation_id"]
        != f"{session_id}:operation:{operation_sequence:020d}"
        or first_activity != state.event_sequence + 1
        or activity_count != len(records)
        or event_sequence != local_sequence
        or checkpoint["previous_state_digest"] != _state_digest(state)
        or operation_kind != _replay_operation_kind(records)
        or records[0].event_id
        != f"{session_id}:event:{first_activity:020d}"
    ):
        raise ValueError


def _reduce_activity_group(
    previous: _PaperState,
    records: tuple[EventRecord, ...],
    *,
    configuration: _ReplayConfiguration,
) -> _PaperState:
    operation_kind = _replay_operation_kind(records)
    _validate_operation_grammar(records, operation_kind)
    if operation_kind == "MARKET":
        return _replay_market_group(
            previous,
            records,
            configuration=configuration,
        )
    if operation_kind == "RECONCILIATION":
        return _replay_reconciliation_group(
            previous,
            records,
            configuration=configuration,
        )
    if operation_kind == "SUBMIT":
        return _replay_submit_group(previous, records, configuration=configuration)
    if operation_kind == "IDEMPOTENCY":
        return _replay_idempotency_group(previous, records)
    if operation_kind == "RESOLUTION":
        return _replay_resolution_group(previous, records)
    raise ValueError


def _replay_submit_group(
    previous: _PaperState,
    records: tuple[EventRecord, ...],
    *,
    configuration: _ReplayConfiguration,
) -> _PaperState:
    market_payload = _strict_mapping(records[0].payload)
    submitted_payload = _strict_mapping(records[1].payload)
    _require_exact_keys(
        submitted_payload,
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
    intent = _intent_from_payload(_strict_mapping(submitted_payload["intent"]))
    if records[0].kind == "paper.market.submission":
        _require_exact_keys(
            market_payload,
            {
                "client_intent_id",
                "broker_order_id",
                "snapshot",
                "bar_count",
                "bars_digest",
            },
        )
        snapshot = _snapshot_from_payload(_strict_mapping(market_payload["snapshot"]))
    else:
        _require_exact_keys(
            market_payload,
            {
                "client_intent_id",
                "broker_order_id",
                "bar_count",
                "bars_digest",
            },
        )
        existing_snapshot = previous.snapshots.get(intent.instrument_id)
        if existing_snapshot is None:
            raise ValueError
        snapshot = existing_snapshot
    candidate, expected_specs, _, _ = _execute_submit_kernel(
        previous,
        intent,
        snapshot,
        session_id=configuration.session_id,
    )
    _require_specs_equal(expected_specs, records)
    return candidate


def _replay_idempotency_group(
    previous: _PaperState,
    records: tuple[EventRecord, ...],
) -> _PaperState:
    payload = _strict_mapping(records[0].payload)
    _require_exact_keys(
        payload,
        {
            "client_intent_id",
            "broker_order_id",
            "outcome",
            "intent",
            "prior_status",
            "new_status",
        },
    )
    intent = _intent_from_payload(_strict_mapping(payload["intent"]))
    candidate, expected_specs, _, _, _ = _execute_idempotency_kernel(previous, intent)
    _require_specs_equal(expected_specs, records)
    return candidate


def _replay_resolution_group(
    previous: _PaperState,
    records: tuple[EventRecord, ...],
) -> _PaperState:
    transition_payload = _strict_mapping(records[-1].payload)
    _require_exact_keys(
        transition_payload,
        {"client_intent_id", "broker_order_id", "prior_status", "new_status"},
    )
    order_id = _strict_identifier(transition_payload["broker_order_id"])
    status = OrderStatus(_strict_string(transition_payload["new_status"]))
    at = _aware_utc(records[-1].occurred_at, "replay resolution timestamp")
    activity: tuple[str, Mapping[str, object]] | None = None
    if len(records) == 2:
        rejected_payload = _strict_mapping(records[0].payload)
        _require_exact_keys(
            rejected_payload,
            {
                "client_intent_id",
                "broker_order_id",
                "reason_code",
                "prior_status",
                "new_status",
            },
        )
        reason_code = rejected_payload["reason_code"]
        if not isinstance(reason_code, str) or _REASON_CODE.fullmatch(reason_code) is None:
            raise ValueError
        activity = ("paper.order.rejected", {"reason_code": reason_code})
    candidate, expected_specs, _, _ = _execute_resolution_kernel(
        previous,
        order_id,
        status,
        at=at,
        activity=activity,
    )
    _require_specs_equal(expected_specs, records)
    return candidate


def _require_specs_equal(
    expected_specs: list[_AuditSpec],
    records: tuple[EventRecord, ...],
) -> None:
    expected_bytes = json.dumps(
        _json_ready(_spec_activity_facts(expected_specs)),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    recorded_bytes = json.dumps(
        _json_ready(_record_activity_facts(records)),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if expected_bytes != recorded_bytes:
        raise ValueError


def _validate_operation_grammar(
    records: tuple[EventRecord, ...],
    operation_kind: str,
) -> None:
    kinds = tuple(record.kind for record in records)
    if operation_kind == "SUBMIT":
        expected_tail = (
            "paper.order.submitted",
            "paper.order.transition",
            "paper.order.transition",
            "paper.order.transition",
            "paper.order.transition",
        )
        if (
            len(kinds) != 6
            or kinds[0]
            not in {"paper.market.submission", "paper.market.submission_reference"}
            or kinds[1:] != expected_tail
        ):
            raise ValueError
        return
    if operation_kind == "MARKET":
        if any(
            kind
            not in {
                "paper.order.stop_triggered",
                "paper.order.transition",
                "paper.order.fill",
                "paper.market.observed",
                "paper.snapshot.duplicate",
            }
            for kind in kinds
        ):
            raise ValueError
        return
    if operation_kind == "IDEMPOTENCY":
        if len(kinds) != 1 or kinds[0] not in {
            "paper.order.duplicate",
            "paper.order.duplicate_conflict",
        }:
            raise ValueError
        return
    if operation_kind == "RECONCILIATION":
        if (
            len(kinds) < 2
            or kinds[0] != "paper.order.transition"
            or any(kind != "paper.order.fill" for kind in kinds[1:])
        ):
            raise ValueError
        return
    if operation_kind != "RESOLUTION" or kinds not in {
        ("paper.order.transition",),
        ("paper.order.rejected", "paper.order.transition"),
    }:
        raise ValueError


def _replay_reconciliation_group(
    previous: _PaperState,
    records: tuple[EventRecord, ...],
    *,
    configuration: _ReplayConfiguration,
) -> _PaperState:
    if len(records) < 2 or records[0].kind != "paper.order.transition":
        raise ValueError
    transition_payload = _strict_mapping(records[0].payload)
    _require_exact_keys(
        transition_payload,
        {"client_intent_id", "broker_order_id", "prior_status", "new_status"},
    )
    order_id = _strict_identifier(transition_payload["broker_order_id"])
    client_id = _strict_identifier(transition_payload["client_intent_id"])
    record = previous.orders.get(order_id)
    status = OrderStatus(_strict_string(transition_payload["new_status"]))
    observed_at = _aware_utc(records[0].occurred_at, "replay reconciliation timestamp")
    if (
        record is None
        or record.order.client_order_id != client_id
        or record.order.status is not OrderStatus.UNKNOWN
        or transition_payload["prior_status"] != OrderStatus.UNKNOWN.value
        or status not in {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED}
    ):
        raise ValueError

    fills: list[Fill] = []
    instrument: Instrument | None = None
    for row in records[1:]:
        if row.kind != "paper.order.fill":
            raise ValueError
        payload = _strict_mapping(row.payload)
        _require_exact_keys(
            payload,
            {
                "client_intent_id",
                "broker_order_id",
                "fill_id",
                "quantity",
                "price",
                "fee",
                "filled_at",
                "observed_at",
                "instrument",
                "source",
                "cumulative_filled_quantity",
                "cumulative_filled_notional",
                "cumulative_fees",
                "prior_status",
                "new_status",
            },
        )
        row_instrument = _instrument_from_payload(
            _strict_mapping(payload["instrument"])
        )
        if instrument is None:
            instrument = row_instrument
        if (
            row_instrument != instrument
            or payload["client_intent_id"] != client_id
            or payload["broker_order_id"] != order_id
            or payload["source"] != "AUTHORITATIVE_RECONCILIATION"
            or payload["prior_status"] != OrderStatus.UNKNOWN.value
            or payload["new_status"] != status.value
            or _parse_datetime(payload["observed_at"]) != observed_at
            or row.occurred_at != observed_at
        ):
            raise ValueError
        fills.append(
            Fill(
                fill_id=_strict_identifier(payload["fill_id"]),
                order_id=order_id,
                instrument_id=record.order.instrument_id,
                side=record.intent.side,
                quantity=_parse_decimal(payload["quantity"], positive=True),
                price=_parse_decimal(payload["price"], positive=True),
                fee=_parse_decimal(payload["fee"], nonnegative=True),
                filled_at=_parse_datetime(payload["filled_at"]),
            )
        )
    if instrument is None:
        raise ValueError
    cumulative_quantity = record.order.filled_quantity + sum(
        (fill.quantity for fill in fills),
        Decimal("0"),
    )
    cumulative_notional = record.cumulative_filled_notional + sum(
        (fill.quantity * fill.price for fill in fills),
        Decimal("0"),
    )
    authoritative_order = record.order.model_copy(
        update={
            "status": status,
            "filled_quantity": cumulative_quantity,
            "average_fill_price": cumulative_notional / cumulative_quantity,
            "updated_at": observed_at,
        }
    )
    candidate, expected_specs, _ = _execute_reconciliation_kernel(
        previous,
        authoritative_order,
        tuple(fills),
        instrument,
        currency=configuration.currency,
    )
    expected_bytes = json.dumps(
        _json_ready(_spec_activity_facts(expected_specs)),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    recorded_bytes = json.dumps(
        _json_ready(_record_activity_facts(records)),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if expected_bytes != recorded_bytes:
        raise ValueError
    return candidate


def _replay_market_group(
    previous: _PaperState,
    records: tuple[EventRecord, ...],
    *,
    configuration: _ReplayConfiguration,
) -> _PaperState:
    inputs: list[_MarketInput] = []
    seen_instruments: set[str] = set()
    for row in records:
        if row.kind == "paper.market.observed":
            payload = _strict_mapping(row.payload)
            _require_exact_keys(
                payload,
                {
                    "client_intent_id",
                    "broker_order_id",
                    "instrument",
                    "instrument_id",
                    "observed_at",
                    "source_at",
                    "provider",
                    "max_age_seconds",
                    "bar_count",
                    "bars_digest",
                    "bar",
                },
            )
            instrument = _instrument_from_payload(
                _strict_mapping(payload["instrument"])
            )
            instrument_id = _instrument_id(instrument)
            prior_snapshot = previous.snapshots.get(instrument_id)
            prior_cursor = previous.cursors.get(instrument_id)
            if (
                prior_snapshot is None
                or prior_cursor is None
                or instrument_id in seen_instruments
            ):
                raise ValueError
            bar = _bar_from_payload(_strict_mapping(payload["bar"]))
            snapshot = MarketSnapshot(
                instrument_id=instrument_id,
                observed_at=_parse_datetime(payload["observed_at"]),
                source_at=_parse_datetime(payload["source_at"]),
                bars=prior_snapshot.bars + (bar,),
                provider=_strict_string(payload["provider"]),
                max_age_seconds=_strict_nonnegative_int(payload["max_age_seconds"]),
            )
            _validate_snapshot(
                snapshot,
                expected_instrument_id=instrument_id,
                max_bars=_MARKET_CURSOR_WINDOW + 1,
            )
            next_count = prior_cursor.total_count + 1
            next_digest = _next_bar_digest(prior_cursor.digest, bar)
            prior_instrument = previous.instruments.get(instrument_id)
            if (
                payload["client_intent_id"] != instrument_id
                or payload["broker_order_id"] != instrument_id
                or payload["instrument_id"] != instrument_id
                or row.occurred_at != snapshot.observed_at
                or _strict_nonnegative_int(payload["bar_count"]) != next_count
                or payload["bars_digest"] != next_digest
                or (prior_instrument is not None and prior_instrument != instrument)
                or instrument.quote_currency != configuration.currency
            ):
                raise ValueError
            inputs.append(_MarketInput(snapshot, instrument, False))
            seen_instruments.add(instrument_id)
        elif row.kind == "paper.snapshot.duplicate":
            payload = _strict_mapping(row.payload)
            _require_exact_keys(
                payload,
                {
                    "client_intent_id",
                    "broker_order_id",
                    "outcome",
                    "bar_count",
                    "bars_digest",
                },
            )
            instrument_id = _strict_identifier(payload["client_intent_id"])
            duplicate_snapshot = previous.snapshots.get(instrument_id)
            cursor = previous.cursors.get(instrument_id)
            duplicate_instrument = previous.instruments.get(instrument_id)
            if (
                duplicate_snapshot is None
                or cursor is None
                or instrument_id in seen_instruments
                or payload["broker_order_id"] != instrument_id
                or payload["outcome"] != "IGNORED"
                or row.occurred_at != duplicate_snapshot.observed_at
                or _strict_nonnegative_int(payload["bar_count"]) != cursor.total_count
                or payload["bars_digest"] != cursor.digest
            ):
                raise ValueError
            inputs.append(_MarketInput(duplicate_snapshot, duplicate_instrument, True))
            seen_instruments.add(instrument_id)

    if not inputs or seen_instruments != _relevant_instrument_ids(previous):
        raise ValueError
    observed_instants = {
        market_input.snapshot.observed_at.astimezone(UTC) for market_input in inputs
    }
    if len(observed_instants) != 1:
        raise ValueError
    inputs.sort(key=lambda item: _event_key(item.snapshot))
    new_keys = [
        _event_key(market_input.snapshot)
        for market_input in inputs
        if not market_input.duplicate
    ]
    if new_keys:
        if previous.last_event_key is not None and new_keys[0] <= previous.last_event_key:
            raise ValueError
        if any(
            current <= prior
            for prior, current in zip(new_keys, new_keys[1:], strict=False)
        ):
            raise ValueError
    candidate, expected_specs, _ = _execute_market_kernel(
        previous,
        tuple(inputs),
        fill_model=FillModel(costs=configuration.costs),
        max_volume_participation=configuration.max_volume_participation,
        session_id=configuration.session_id,
    )
    expected_bytes = json.dumps(
        _json_ready(_spec_activity_facts(expected_specs)),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    recorded_bytes = json.dumps(
        _json_ready(_record_activity_facts(records)),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if expected_bytes != recorded_bytes:
        raise ValueError
    return candidate


def _validate_nonmutating_order_activity(
    row: EventRecord,
    record: _PaperOrderRecord,
) -> None:
    payload = _strict_mapping(row.payload)
    prior = OrderStatus(_strict_string(payload["prior_status"]))
    new = OrderStatus(_strict_string(payload["new_status"]))
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
        if prior is not new or prior is not record.order.status or payload["outcome"] != expected:
            raise ValueError
        return
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
        prior is not record.order.status
        or new is not OrderStatus.REJECTED
        or not isinstance(reason, str)
        or _REASON_CODE.fullmatch(reason) is None
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
        or len(value) > _MAX_DECIMAL_TEXT
        or re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value) is None
    ):
        raise ValueError
    result = Decimal(value)
    if (
        not _bounded_decimal(result)
        or _canonical_decimal_text(result) != value
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
        payload=_freeze_mapping(row.payload),
    )


def _optional_decimal(value: Decimal | None) -> str | None:
    return _decimal_text(value) if value is not None else None
