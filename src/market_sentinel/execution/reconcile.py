"""Exact broker/ledger reconciliation and a durable generation-aware kill switch."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Final
from uuid import uuid4

from market_sentinel.domain.clock import Clock
from market_sentinel.domain.enums import OrderStatus, Side
from market_sentinel.execution.canonical import CanonicalEncodingError, canonical_decimal
from market_sentinel.execution.safety import (
    SafetyEvent,
    SafetyIntegrityError,
    SafetyRepository,
)
from market_sentinel.portfolio.ledger import PortfolioLedger, PortfolioLedgerPositionState
from market_sentinel.storage.events import EventHeadConflict, EventRecord

KILL_SWITCH_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_HEALTHY_RECONCILIATION"
RECONCILIATION_MAX_AGE = timedelta(seconds=60)
RECONCILIATION_AGGREGATE: Final = "live-reconciliation"
KILL_SWITCH_AGGREGATE: Final = "live-kill-switch"
LIVE_INTERLOCK_AGGREGATE: Final = "live-submission-interlock"
_BROKER_HASH_DOMAIN = b"omnimarket-sentinel:broker-reconciliation:v1\x00"
_LEDGER_HASH_DOMAIN = b"omnimarket-sentinel:ledger-reconciliation:v1\x00"
_REASON_ORDER = (
    "PROVIDER_UNAVAILABLE",
    "PROVIDER_DATA_INVALID",
    "PROVIDER_DATA_FUTURE",
    "PROVIDER_DATA_STALE",
    "CURRENCY_MISMATCH",
    "CASH_MISMATCH",
    "POSITION_UNKNOWN",
    "POSITION_MISSING",
    "POSITION_SIDE_MISMATCH",
    "POSITION_QUANTITY_MISMATCH",
    "ORDER_UNKNOWN",
    "ORDER_MISSING",
    "ORDER_INSTRUMENT_MISMATCH",
    "ORDER_SIDE_MISMATCH",
    "ORDER_QUANTITY_MISMATCH",
    "ORDER_FILL_MISMATCH",
    "ORDER_STATUS_MISMATCH",
)


class KillSwitchError(RuntimeError):
    """Safe rejection of an unauthorized or unsafe kill-switch clear."""


@dataclass(frozen=True, slots=True)
class BrokerPositionRecord:
    """Strict side-aware position facts read from a broker account."""

    instrument_id: str
    side: Side
    quantity: Decimal

    def __post_init__(self) -> None:
        _nonempty_text(self.instrument_id)
        _exact_enum(self.side, Side)
        _positive_decimal(self.quantity)


@dataclass(frozen=True, slots=True)
class BrokerOpenOrderRecord:
    """Every open-order field needed to detect unknown or changed broker state."""

    client_intent_id: str
    broker_order_id: str
    instrument_id: str
    side: Side
    quantity: Decimal
    filled_quantity: Decimal
    status: OrderStatus

    def __post_init__(self) -> None:
        _nonempty_text(self.client_intent_id)
        _nonempty_text(self.broker_order_id)
        _nonempty_text(self.instrument_id)
        _exact_enum(self.side, Side)
        quantity = _positive_decimal(self.quantity)
        filled = _nonnegative_decimal(self.filled_quantity)
        if filled > quantity:
            raise ValueError("filled quantity exceeds requested quantity")
        _exact_enum(self.status, OrderStatus)


@dataclass(frozen=True, slots=True)
class BrokerReconciliationSnapshot:
    """Immutable provider source facts; health is always derived by Reconciler."""

    broker: str
    currency: str
    cash: Decimal
    positions: tuple[BrokerPositionRecord, ...]
    open_orders: tuple[BrokerOpenOrderRecord, ...]
    observed_at: datetime

    def __post_init__(self) -> None:
        _nonempty_text(self.broker)
        _nonempty_text(self.currency)
        _finite_decimal(self.cash)
        if type(self.positions) is not tuple or not all(
            type(item) is BrokerPositionRecord for item in self.positions
        ):
            raise ValueError("positions must be strict broker position records")
        if type(self.open_orders) is not tuple or not all(
            type(item) is BrokerOpenOrderRecord for item in self.open_orders
        ):
            raise ValueError("open orders must be strict broker order records")
        _aware_utc(self.observed_at)


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Derived report whose identity and sequence are verified against the event store."""

    report_id: str
    broker: str
    healthy: bool
    reason_codes: tuple[str, ...]
    broker_hash: str
    ledger_hash: str
    checked_at: datetime
    sequence: int


@dataclass(frozen=True, slots=True)
class SafetyFence:
    """Exact durable heads that must remain unchanged through a live claim."""

    reconciliation_head: str
    kill_switch_head: str | None
    interlock_head: str | None


class Reconciler:
    """Compare exact states and persist every report plus fail-closed switch changes."""

    def __init__(self, *, safety_repository: SafetyRepository, clock: Clock) -> None:
        if type(safety_repository) is not SafetyRepository:
            raise ValueError("reconciliation requires an authenticated safety repository")
        self._safety = safety_repository
        self._clock = clock

    @property
    def safety_repository(self) -> SafetyRepository:
        """Expose the exact authenticated durability boundary used for safety state."""
        return self._safety

    def compare(
        self,
        source: object,
        ledger: PortfolioLedger,
        expected_open_orders: tuple[BrokerOpenOrderRecord, ...],
    ) -> ReconciliationReport:
        """Derive, hash, and durably record exact state without trusting a health input."""
        if type(ledger) is not PortfolioLedger:
            raise ValueError("reconciliation requires an exact PortfolioLedger")
        instant = _aware_utc(self._clock.now())
        reasons: set[str] = set()
        if not _valid_expected_orders(expected_open_orders):
            reasons.add("PROVIDER_DATA_INVALID")
            expected_open_orders = ()
        if type(source) is not BrokerReconciliationSnapshot or not _valid_source(source):
            return self._record_report(
                broker="unavailable",
                broker_hash=_domain_hash({"state": "invalid"}),
                ledger_hash=_ledger_hash(ledger),
                reasons={"PROVIDER_DATA_INVALID"},
                checked_at=instant,
            )
        observed_at = _aware_utc(source.observed_at)
        if observed_at > instant:
            reasons.add("PROVIDER_DATA_FUTURE")
        elif instant - observed_at > RECONCILIATION_MAX_AGE:
            reasons.add("PROVIDER_DATA_STALE")
        state = ledger.export_state()
        if source.currency != state.currency:
            reasons.add("CURRENCY_MISMATCH")
        if source.cash != state.cash:
            reasons.add("CASH_MISMATCH")
        _compare_positions(source.positions, state.positions, reasons)
        _compare_orders(source.open_orders, expected_open_orders, reasons)
        return self._record_report(
            broker=source.broker,
            broker_hash=_broker_hash(source),
            ledger_hash=_ledger_hash(ledger),
            reasons=reasons,
            checked_at=instant,
        )

    def read_and_compare(
        self,
        source: Callable[[], BrokerReconciliationSnapshot],
        ledger: PortfolioLedger,
        expected_open_orders: tuple[BrokerOpenOrderRecord, ...],
    ) -> ReconciliationReport:
        """Convert any provider read/parsing uncertainty into a durable unhealthy report."""
        try:
            snapshot = source()
        except BaseException:
            snapshot = None
            unavailable = True
        else:
            unavailable = False
        if unavailable:
            instant = _aware_utc(self._clock.now())
            return self._record_report(
                broker="unavailable",
                broker_hash=_domain_hash({"state": "unavailable"}),
                ledger_hash=_ledger_hash(ledger),
                reasons={"PROVIDER_UNAVAILABLE"},
                checked_at=instant,
            )
        return self.compare(snapshot, ledger, expected_open_orders)

    def is_current_healthy(
        self,
        report: object,
        *,
        broker: str,
        ledger: PortfolioLedger,
    ) -> bool:
        """Verify an unexpired report is the exact persisted reconciliation head."""
        if (
            type(report) is not ReconciliationReport
            or type(broker) is not str
            or type(ledger) is not PortfolioLedger
            or not report.healthy
            or report.reason_codes
            or report.broker != broker
            or report.ledger_hash != _ledger_hash(ledger)
        ):
            return False
        try:
            checked_at = _aware_utc(report.checked_at)
            now = _aware_utc(self._clock.now())
        except (TypeError, ValueError):
            return False
        if checked_at > now or now - checked_at > RECONCILIATION_MAX_AGE:
            return False
        try:
            rows = self._safety.stream_verified(RECONCILIATION_AGGREGATE)
        except SafetyIntegrityError:
            return False
        if not rows:
            return False
        head = rows[-1]
        payload = head.payload
        return (
            head.event_id == report.report_id
            and head.sequence == report.sequence
            and head.kind == "reconciliation.healthy"
            and payload.get("healthy") is True
            and payload.get("broker") == report.broker
            and payload.get("broker_hash") == report.broker_hash
            and payload.get("ledger_hash") == report.ledger_hash
            and head.occurred_at == checked_at
        )

    def kill_switch_active(self) -> bool:
        """Replay generations: a clear covers only activations it explicitly observed."""
        try:
            kill_rows = self._safety.stream_verified(KILL_SWITCH_AGGREGATE)
            interlock_rows = self._safety.stream_verified(LIVE_INTERLOCK_AGGREGATE)
        except SafetyIntegrityError:
            return True
        return _kill_rows_active(kill_rows) or _interlock_rows_active(interlock_rows)

    def safety_fence(
        self,
        report: object,
        *,
        broker: str,
        ledger: PortfolioLedger,
    ) -> SafetyFence:
        """Capture exact healthy heads for a subsequent conditional claim transaction."""
        if (
            type(report) is not ReconciliationReport
            or type(broker) is not str
            or type(ledger) is not PortfolioLedger
            or not report.healthy
            or report.reason_codes
            or report.broker != broker
            or report.ledger_hash != _ledger_hash(ledger)
        ):
            raise KillSwitchError("reconciliation report is not current")
        now = _aware_utc(self._clock.now())
        try:
            reconciliation_rows = self._safety.stream_verified(RECONCILIATION_AGGREGATE)
            kill_rows = self._safety.stream_verified(KILL_SWITCH_AGGREGATE)
            interlock_rows = self._safety.stream_verified(LIVE_INTERLOCK_AGGREGATE)
        except SafetyIntegrityError:
            safety_state = None
        else:
            safety_state = (reconciliation_rows, kill_rows, interlock_rows)
        if safety_state is None:
            raise KillSwitchError("safety state failed authentication")
        reconciliation_rows, kill_rows, interlock_rows = safety_state
        if not reconciliation_rows:
            raise KillSwitchError("reconciliation report is not current")
        head = reconciliation_rows[-1]
        if (
            head.event_id != report.report_id
            or head.sequence != report.sequence
            or head.kind != "reconciliation.healthy"
            or head.occurred_at != report.checked_at
            or head.occurred_at > now
            or now - head.occurred_at > RECONCILIATION_MAX_AGE
        ):
            raise KillSwitchError("reconciliation report is not current")
        if _kill_rows_active(kill_rows) or _interlock_rows_active(interlock_rows):
            raise KillSwitchError("kill switch is active")
        return SafetyFence(
            reconciliation_head=head.event_id,
            kill_switch_head=kill_rows[-1].event_id if kill_rows else None,
            interlock_head=interlock_rows[-1].event_id if interlock_rows else None,
        )

    def clear_kill_switch(self, acknowledgement: str) -> None:
        """Clear only through a newly healthy generation with exact local acknowledgement."""
        if type(acknowledgement) is not str or acknowledgement != KILL_SWITCH_ACKNOWLEDGEMENT:
            raise KillSwitchError("kill-switch acknowledgement is not exact")
        try:
            interlock_rows = self._safety.stream_verified(LIVE_INTERLOCK_AGGREGATE)
            kill_rows = self._safety.stream_verified(KILL_SWITCH_AGGREGATE)
            reconciliation_rows = self._safety.stream_verified(RECONCILIATION_AGGREGATE)
        except SafetyIntegrityError:
            safety_state = None
        else:
            safety_state = (interlock_rows, kill_rows, reconciliation_rows)
        if safety_state is None:
            raise KillSwitchError("safety state failed authentication")
        interlock_rows, kill_rows, reconciliation_rows = safety_state
        if _interlock_rows_active(interlock_rows):
            raise KillSwitchError("a live submission remains unresolved")
        if not _kill_rows_active(kill_rows):
            return
        activations = [row for row in kill_rows if row.kind == "kill_switch.activated"]
        if not activations:
            raise KillSwitchError("kill-switch state is malformed")
        latest_activation = activations[-1]
        if not reconciliation_rows:
            raise KillSwitchError("a new healthy reconciliation is required")
        head = reconciliation_rows[-1]
        now = _aware_utc(self._clock.now())
        if (
            head.kind != "reconciliation.healthy"
            or head.sequence <= latest_activation.sequence
            or head.occurred_at > now
            or now - head.occurred_at > RECONCILIATION_MAX_AGE
        ):
            raise KillSwitchError("a new healthy reconciliation is required")
        clear = SafetyEvent(
            f"kill-clear-{uuid4().hex}",
            "kill_switch.cleared",
            KILL_SWITCH_AGGREGATE,
            {
                "activation_event_id": latest_activation.event_id,
                "activation_sequence": latest_activation.sequence,
            },
            now,
        )
        try:
            self._safety.record_many_if_heads(
                (clear,),
                {
                    KILL_SWITCH_AGGREGATE: kill_rows[-1].event_id if kill_rows else None,
                    RECONCILIATION_AGGREGATE: head.event_id,
                    LIVE_INTERLOCK_AGGREGATE: (
                        interlock_rows[-1].event_id if interlock_rows else None
                    ),
                },
            )
        except EventHeadConflict:
            conflicted = True
            persistence_failed = False
        except BaseException:
            conflicted = False
            persistence_failed = True
        else:
            conflicted = False
            persistence_failed = False
        if persistence_failed:
            raise KillSwitchError("kill-switch persistence failed")
        if conflicted:
            raise KillSwitchError("safety state changed")

    def record_submission_unknown(self, intent_id: str, submission_id: str) -> None:
        """Atomically persist UNKNOWN, resolve its interlock, and activate the kill switch."""
        instant = _aware_utc(self._clock.now())
        nonce = uuid4().hex
        self._safety.record_many(
            (
                SafetyEvent(
                    f"live-unknown-{nonce}",
                    "live.unknown",
                    _nonempty_text(intent_id),
                    {"reason_code": "SUBMISSION_UNKNOWN"},
                    instant,
                ),
                SafetyEvent(
                    f"interlock-unknown-{nonce}",
                    "live.interlock_resolved",
                    LIVE_INTERLOCK_AGGREGATE,
                    {
                        "resolution": "unknown",
                        "submission_id": _nonempty_text(submission_id),
                    },
                    instant,
                ),
                SafetyEvent(
                    f"kill-unknown-{nonce}",
                    "kill_switch.activated",
                    KILL_SWITCH_AGGREGATE,
                    {"reason_codes": ["SUBMISSION_UNKNOWN"]},
                    instant,
                ),
            )
        )

    def _record_report(
        self,
        *,
        broker: str,
        broker_hash: str,
        ledger_hash: str,
        reasons: set[str],
        checked_at: datetime,
    ) -> ReconciliationReport:
        ordered = tuple(code for code in _REASON_ORDER if code in reasons)
        healthy = not ordered
        report_id = f"reconciliation-{uuid4().hex}"
        report_event = SafetyEvent(
            report_id,
            "reconciliation.healthy" if healthy else "reconciliation.unhealthy",
            RECONCILIATION_AGGREGATE,
            {
                "broker": broker,
                "broker_hash": broker_hash,
                "healthy": healthy,
                "ledger_hash": ledger_hash,
                "reason_codes": list(ordered),
            },
            checked_at,
        )
        try:
            if healthy:
                self._safety.record_many((report_event,))
            else:
                self._safety.record_many(
                    (
                        report_event,
                        SafetyEvent(
                            f"kill-{uuid4().hex}",
                            "kill_switch.activated",
                            KILL_SWITCH_AGGREGATE,
                            {"reason_codes": list(ordered)},
                            checked_at,
                        ),
                    )
                )
            matching = [
                row
                for row in self._safety.stream_verified(RECONCILIATION_AGGREGATE)
                if row.event_id == report_id
            ]
        except BaseException:
            matching = []
        if len(matching) != 1:
            raise RuntimeError("reconciliation persistence could not be verified")
        return ReconciliationReport(
            report_id=report_id,
            broker=broker,
            healthy=healthy,
            reason_codes=ordered,
            broker_hash=broker_hash,
            ledger_hash=ledger_hash,
            checked_at=checked_at,
            sequence=matching[0].sequence,
        )


def _kill_rows_active(rows: tuple[EventRecord, ...]) -> bool:
    latest_activation: EventRecord | None = None
    active = False
    for row in rows:
        if row.kind == "kill_switch.activated":
            reasons = row.payload.get("reason_codes")
            if (
                type(reasons) is not tuple
                or not reasons
                or not all(type(reason) is str and reason for reason in reasons)
            ):
                return True
            latest_activation = row
            active = True
        elif row.kind == "kill_switch.cleared":
            activation_event_id = row.payload.get("activation_event_id")
            activation_sequence = row.payload.get("activation_sequence")
            if (
                latest_activation is None
                or type(activation_event_id) is not str
                or not activation_event_id
                or type(activation_sequence) is not int
                or activation_sequence <= 0
                or latest_activation.event_id != activation_event_id
                or latest_activation.sequence != activation_sequence
                or activation_sequence >= row.sequence
            ):
                return True
            active = False
        else:
            return True
    return active


def _interlock_rows_active(rows: tuple[EventRecord, ...]) -> bool:
    started: set[str] = set()
    resolved: set[str] = set()
    for row in rows:
        submission_id = row.payload.get("submission_id")
        if type(submission_id) is not str or not submission_id:
            return True
        if row.kind == "live.interlock_started":
            if submission_id in started:
                return True
            started.add(submission_id)
        elif row.kind == "live.interlock_resolved":
            if submission_id not in started or submission_id in resolved:
                return True
            resolved.add(submission_id)
        else:
            return True
    return bool(started - resolved)


def _compare_positions(
    actual: tuple[BrokerPositionRecord, ...],
    expected: tuple[PortfolioLedgerPositionState, ...],
    reasons: set[str],
) -> None:
    actual_by_id = {item.instrument_id: item for item in actual}
    expected_by_id = {item.instrument_id: item for item in expected}
    if len(actual_by_id) != len(actual):
        reasons.add("PROVIDER_DATA_INVALID")
    for instrument_id in sorted(actual_by_id.keys() - expected_by_id.keys()):
        del instrument_id
        reasons.add("POSITION_UNKNOWN")
    for instrument_id in sorted(expected_by_id.keys() - actual_by_id.keys()):
        del instrument_id
        reasons.add("POSITION_MISSING")
    for instrument_id in sorted(actual_by_id.keys() & expected_by_id.keys()):
        broker_position = actual_by_id[instrument_id]
        ledger_position = expected_by_id[instrument_id]
        if broker_position.side is not Side.BUY:
            reasons.add("POSITION_SIDE_MISMATCH")
        if broker_position.quantity != ledger_position.quantity:
            reasons.add("POSITION_QUANTITY_MISMATCH")


def _compare_orders(
    actual: tuple[BrokerOpenOrderRecord, ...],
    expected: tuple[BrokerOpenOrderRecord, ...],
    reasons: set[str],
) -> None:
    actual_by_id = {item.client_intent_id: item for item in actual}
    expected_by_id = {item.client_intent_id: item for item in expected}
    if len(actual_by_id) != len(actual):
        reasons.add("PROVIDER_DATA_INVALID")
    if actual_by_id.keys() - expected_by_id.keys():
        reasons.add("ORDER_UNKNOWN")
    if expected_by_id.keys() - actual_by_id.keys():
        reasons.add("ORDER_MISSING")
    for client_id in sorted(actual_by_id.keys() & expected_by_id.keys()):
        left = actual_by_id[client_id]
        right = expected_by_id[client_id]
        if (
            left.broker_order_id != right.broker_order_id
            or left.instrument_id != right.instrument_id
        ):
            reasons.add("ORDER_INSTRUMENT_MISMATCH")
        if left.side is not right.side:
            reasons.add("ORDER_SIDE_MISMATCH")
        if left.quantity != right.quantity:
            reasons.add("ORDER_QUANTITY_MISMATCH")
        if left.filled_quantity != right.filled_quantity:
            reasons.add("ORDER_FILL_MISMATCH")
        if left.status is not right.status:
            reasons.add("ORDER_STATUS_MISMATCH")


def _valid_source(source: BrokerReconciliationSnapshot) -> bool:
    try:
        source.__post_init__()
        return len({item.instrument_id for item in source.positions}) == len(
            source.positions
        ) and len({item.client_intent_id for item in source.open_orders}) == len(source.open_orders)
    except (TypeError, ValueError):
        return False


def _valid_expected_orders(value: object) -> bool:
    if type(value) is not tuple or not all(type(item) is BrokerOpenOrderRecord for item in value):
        return False
    try:
        for item in value:
            item.__post_init__()
    except (TypeError, ValueError):
        return False
    return len({item.client_intent_id for item in value}) == len(value)


def _broker_hash(source: BrokerReconciliationSnapshot) -> str:
    payload = {
        "broker": source.broker,
        "cash": _decimal_text(source.cash),
        "currency": source.currency,
        "observed_at": _time_text(source.observed_at),
        "open_orders": [
            [
                item.client_intent_id,
                item.broker_order_id,
                item.instrument_id,
                item.side.value,
                _decimal_text(item.quantity),
                _decimal_text(item.filled_quantity),
                item.status.value,
            ]
            for item in sorted(source.open_orders, key=lambda entry: entry.client_intent_id)
        ],
        "positions": [
            [item.instrument_id, item.side.value, _decimal_text(item.quantity)]
            for item in sorted(source.positions, key=lambda entry: entry.instrument_id)
        ],
    }
    return _domain_hash(payload)


def _ledger_hash(ledger: PortfolioLedger) -> str:
    state = ledger.compact_state()
    payload = {
        "cash": _decimal_text(state.cash),
        "currency": state.currency,
        "drawdown": _decimal_text(state.drawdown),
        "equity": _decimal_text(state.equity),
        "fees": _decimal_text(state.fees),
        "fill_count": state.fill_count,
        "gross_realized_pnl": _decimal_text(state.gross_realized_pnl),
        "market_prices": [
            [instrument_id, _decimal_text(price)] for instrument_id, price in state.market_prices
        ],
        "peak_equity": _decimal_text(state.peak_equity),
        "positions": [
            [
                item.instrument_id,
                _decimal_text(item.quantity),
                _decimal_text(item.average_price),
            ]
            for item in state.positions
        ],
        "starting_cash": _decimal_text(state.starting_cash),
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(_LEDGER_HASH_DOMAIN + encoded.encode("utf-8")).hexdigest()


def _domain_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(_BROKER_HASH_DOMAIN + encoded.encode("utf-8")).hexdigest()


def _nonempty_text(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("text field must be nonempty and trimmed")
    return value


def _exact_enum(value: object, enum_type: type[Side] | type[OrderStatus]) -> object:
    if type(value) is not enum_type:
        raise ValueError("enum field has an unknown value")
    return value


def _finite_decimal(value: object) -> Decimal:
    if type(value) is not Decimal or not value.is_finite():
        raise ValueError("numeric field must be a finite Decimal")
    return value


def _positive_decimal(value: object) -> Decimal:
    result = _finite_decimal(value)
    if result <= 0:
        raise ValueError("numeric field must be positive")
    return result


def _nonnegative_decimal(value: object) -> Decimal:
    result = _finite_decimal(value)
    if result < 0:
        raise ValueError("numeric field must be nonnegative")
    return result


def _decimal_text(value: Decimal) -> str:
    try:
        result = canonical_decimal(value)
    except CanonicalEncodingError:
        result = None
    if result is None:
        raise ValueError("numeric field is malformed")
    return result


def _aware_utc(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _time_text(value: datetime) -> str:
    return _aware_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")
