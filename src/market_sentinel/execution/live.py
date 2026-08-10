"""Fail-closed live submission after durable exact local confirmation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Protocol
from uuid import uuid4

from sqlalchemy.exc import IntegrityError

from market_sentinel.brokers.preflight import PreflightReport
from market_sentinel.domain.clock import Clock
from market_sentinel.domain.enums import OrderStatus
from market_sentinel.domain.models import BrokerOrder, MarketSnapshot, OrderIntent, RiskDecision
from market_sentinel.execution.approval import ApprovalError, ApprovalService, OrderConfirmation
from market_sentinel.execution.reconcile import (
    KILL_SWITCH_AGGREGATE,
    LIVE_INTERLOCK_AGGREGATE,
    RECONCILIATION_AGGREGATE,
    KillSwitchError,
    Reconciler,
    ReconciliationReport,
    SafetyFence,
)
from market_sentinel.operations.audit import AuditEvent, AuditLog
from market_sentinel.portfolio.ledger import PortfolioLedger
from market_sentinel.storage.events import EventHeadConflict


class LiveOrderError(RuntimeError):
    """One stable, secret-free live rejection reason."""


class LiveBroker(Protocol):
    """Only the injected provider operations used by the live coordinator."""

    @property
    def broker_name(self) -> str: ...

    def preflight(self) -> PreflightReport: ...

    def submit(self, intent: OrderIntent, snapshot: MarketSnapshot) -> BrokerOrder: ...

    def get_order_by_client_id(self, client_intent_id: str) -> BrokerOrder: ...


class LiveOrderService:
    """Apply every gate in a fixed order and call an injected broker at most once."""

    def __init__(
        self,
        *,
        broker: LiveBroker,
        approval_service: ApprovalService,
        reconciler: Reconciler,
        audit_log: AuditLog,
        clock: Clock,
        ledger: PortfolioLedger,
    ) -> None:
        if type(approval_service) is not ApprovalService:
            raise ValueError("live execution requires the exact approval service")
        if type(reconciler) is not Reconciler:
            raise ValueError("live execution requires the exact reconciler")
        if type(audit_log) is not AuditLog:
            raise ValueError("live execution durability requires an exact AuditLog")
        if audit_log.event_store is not reconciler.audit_log.event_store:
            raise ValueError("live execution and reconciliation must share one EventStore")
        if type(ledger) is not PortfolioLedger:
            raise ValueError("live execution requires an exact PortfolioLedger")
        broker_name = getattr(broker, "broker_name", None)
        if type(broker_name) is not str or not broker_name:
            raise ValueError("live broker identity is malformed")
        self._broker = broker
        self._approval = approval_service
        self._reconciler = reconciler
        self._audit = audit_log
        self._clock = clock
        self._ledger = ledger
        self._broker_name = broker_name

    def submit_confirmed(
        self,
        *,
        intent: OrderIntent,
        risk_decision: RiskDecision,
        snapshot: MarketSnapshot,
        confirmation: OrderConfirmation,
        preflight: PreflightReport,
        reconciliation: ReconciliationReport,
    ) -> BrokerOrder:
        """Submit once after gates, atomically claimed confirmation, and start audit."""
        self._require_preflight(preflight)
        instant = _aware_utc(self._clock.now())
        self._require_risk(intent, risk_decision, instant)
        try:
            current = self._reconciler.is_current_healthy(
                reconciliation,
                broker=self._broker_name,
                ledger=self._ledger,
            )
        except BaseException:
            raise LiveOrderError("RECONCILIATION_NOT_CURRENT") from None
        if not current:
            raise LiveOrderError("RECONCILIATION_NOT_CURRENT")
        try:
            kill_switch_active = self._reconciler.kill_switch_active()
        except BaseException:
            raise LiveOrderError("KILL_SWITCH_ACTIVE") from None
        if kill_switch_active:
            raise LiveOrderError("KILL_SWITCH_ACTIVE")
        try:
            self._approval.verify(
                intent,
                risk_decision,
                confirmation,
                broker=self._broker_name,
            )
        except (ApprovalError, TypeError, ValueError):
            raise LiveOrderError("CONFIRMATION_INVALID") from None
        self._require_snapshot(intent, snapshot, instant)
        try:
            fence = self._reconciler.safety_fence(
                reconciliation,
                broker=self._broker_name,
                ledger=self._ledger,
            )
        except KillSwitchError:
            raise LiveOrderError("SAFETY_STATE_CHANGED") from None
        except BaseException:
            raise LiveOrderError("AUDIT_PERSISTENCE_FAILED") from None
        claim_instant = _aware_utc(self._clock.now())
        self._require_risk(intent, risk_decision, claim_instant)
        try:
            self._approval.verify(
                intent,
                risk_decision,
                confirmation,
                broker=self._broker_name,
            )
        except (ApprovalError, TypeError, ValueError):
            raise LiveOrderError("CONFIRMATION_INVALID") from None
        self._require_snapshot(intent, snapshot, claim_instant)
        self._claim_and_start(intent, confirmation, claim_instant, fence)

        submitted: BrokerOrder | None = None
        try:
            candidate = self._broker.submit(intent, snapshot)
            response_instant = _aware_utc(self._clock.now())
            if _valid_acknowledgement(
                candidate,
                intent,
                self._broker_name,
                response_instant,
            ):
                submitted = candidate
        except BaseException:
            submitted = None
        if submitted is None:
            submitted = self._query_ambiguous(intent)
        if submitted is None:
            self._persist_unknown(intent.intent_id, confirmation.confirmation_id)
            raise LiveOrderError("SUBMISSION_UNKNOWN") from None
        try:
            acknowledged_at = _aware_utc(self._clock.now())
            nonce = uuid4().hex
            self._audit.record_many(
                (
                    AuditEvent(
                        f"live-ack-{nonce}",
                        "live.acknowledged",
                        intent.intent_id,
                        {
                            "broker": self._broker_name,
                            "broker_order_id": submitted.order_id,
                            "client_intent_id": intent.intent_id,
                            "status": submitted.status.value,
                        },
                        acknowledged_at,
                    ),
                    AuditEvent(
                        f"interlock-ack-{nonce}",
                        "live.interlock_resolved",
                        LIVE_INTERLOCK_AGGREGATE,
                        {
                            "resolution": "acknowledged",
                            "submission_id": confirmation.confirmation_id,
                        },
                        acknowledged_at,
                    ),
                )
            )
        except BaseException:
            self._persist_unknown(intent.intent_id, confirmation.confirmation_id)
            raise LiveOrderError("SUBMISSION_UNKNOWN") from None
        return submitted

    def _require_preflight(self, report: object) -> None:
        try:
            fresh = self._broker.preflight()
        except BaseException:
            raise LiveOrderError("PREFLIGHT_NOT_READY") from None
        if type(report) is not PreflightReport or type(fresh) is not PreflightReport:
            raise LiveOrderError("PREFLIGHT_NOT_READY")
        if not _valid_preflight(fresh, self._broker_name) or report != fresh:
            raise LiveOrderError("PREFLIGHT_NOT_READY")

    def _require_risk(
        self,
        intent: object,
        decision: object,
        now: datetime,
    ) -> None:
        if type(intent) is not OrderIntent or type(decision) is not RiskDecision:
            raise LiveOrderError("RISK_NOT_APPROVED")
        try:
            decided_at = _aware_utc(decision.decided_at)
            expires_at = _aware_utc(decision.expires_at)
            intent_created = _aware_utc(intent.created_at)
            intent_expires = _aware_utc(intent.expires_at)
        except (TypeError, ValueError):
            raise LiveOrderError("RISK_NOT_APPROVED") from None
        if (
            decided_at > now
            or expires_at <= now
            or expires_at <= decided_at
            or (expires_at - decided_at).total_seconds() > 60
            or intent_created > now
            or intent_expires <= now
        ):
            raise LiveOrderError("RISK_STALE")
        if (
            type(decision.approved) is not bool
            or not decision.approved
            or type(decision.reason_codes) is not tuple
            or decision.reason_codes
            or decision.portfolio_hash != self._ledger.position_hash()
            or intent.snapshot_hash != decision.portfolio_hash
            or not _positive_decimal(decision.approved_quantity)
            or not _positive_decimal(decision.approved_notional)
        ):
            raise LiveOrderError("RISK_NOT_APPROVED")
        if intent.quantity is not None and decision.approved_quantity != intent.quantity:
            raise LiveOrderError("RISK_NOT_APPROVED")
        if intent.notional is not None and decision.approved_notional != intent.notional:
            raise LiveOrderError("RISK_NOT_APPROVED")

    def _require_snapshot(
        self,
        intent: OrderIntent,
        snapshot: object,
        now: datetime,
    ) -> None:
        if type(snapshot) is not MarketSnapshot or snapshot.instrument_id != intent.instrument_id:
            raise LiveOrderError("SNAPSHOT_INVALID")
        try:
            observed_at = _aware_utc(snapshot.observed_at)
            source_at = _aware_utc(snapshot.source_at)
        except (TypeError, ValueError):
            raise LiveOrderError("SNAPSHOT_INVALID") from None
        if (
            type(snapshot.max_age_seconds) is not int
            or snapshot.max_age_seconds < 0
            or source_at > observed_at
            or observed_at > now
            or source_at > now
            or now - source_at > timedelta(seconds=snapshot.max_age_seconds)
        ):
            raise LiveOrderError("SNAPSHOT_INVALID")

    def _claim_and_start(
        self,
        intent: OrderIntent,
        confirmation: OrderConfirmation,
        instant: datetime,
        fence: SafetyFence,
    ) -> None:
        confirmation_aggregate = f"live-confirmation:{confirmation.confirmation_id}"
        claim_event_id = f"live-confirmation-{confirmation.confirmation_id}"
        nonce = uuid4().hex
        try:
            self._audit.record_many_if_heads(
                (
                    AuditEvent(
                        f"live-claim-audit-{nonce}",
                        "live.confirmation_claimed",
                        intent.intent_id,
                        {
                            "broker": self._broker_name,
                            "confirmation_fingerprint": confirmation.fingerprint,
                            "expires_at": _time_text(confirmation.expires_at),
                        },
                        instant,
                    ),
                    AuditEvent(
                        claim_event_id,
                        "live.confirmation_consumed",
                        confirmation_aggregate,
                        {"intent_id": intent.intent_id},
                        instant,
                    ),
                    AuditEvent(
                        f"live-start-{nonce}",
                        "live.submission_started",
                        intent.intent_id,
                        {
                            "broker": self._broker_name,
                            "client_intent_id": intent.intent_id,
                        },
                        instant,
                    ),
                    AuditEvent(
                        f"interlock-start-{nonce}",
                        "live.interlock_started",
                        LIVE_INTERLOCK_AGGREGATE,
                        {
                            "intent_id": intent.intent_id,
                            "submission_id": confirmation.confirmation_id,
                        },
                        instant,
                    ),
                ),
                {
                    RECONCILIATION_AGGREGATE: fence.reconciliation_head,
                    KILL_SWITCH_AGGREGATE: fence.kill_switch_head,
                    LIVE_INTERLOCK_AGGREGATE: fence.interlock_head,
                    confirmation_aggregate: None,
                },
            )
        except EventHeadConflict:
            if tuple(self._audit.event_store.stream(confirmation_aggregate)):
                raise LiveOrderError("CONFIRMATION_USED") from None
            raise LiveOrderError("SAFETY_STATE_CHANGED") from None
        except IntegrityError:
            if tuple(self._audit.event_store.stream(confirmation_aggregate)):
                raise LiveOrderError("CONFIRMATION_USED") from None
            raise LiveOrderError("AUDIT_PERSISTENCE_FAILED") from None
        except BaseException:
            raise LiveOrderError("AUDIT_PERSISTENCE_FAILED") from None

    def _query_ambiguous(self, intent: OrderIntent) -> BrokerOrder | None:
        try:
            candidate = self._broker.get_order_by_client_id(intent.intent_id)
        except BaseException:
            return None
        if not _valid_acknowledgement(
            candidate,
            intent,
            self._broker_name,
            _aware_utc(self._clock.now()),
        ):
            return None
        return candidate

    def _persist_unknown(self, intent_id: str, submission_id: str) -> None:
        try:
            self._reconciler.record_submission_unknown(intent_id, submission_id)
        except BaseException:
            raise LiveOrderError("UNKNOWN_PERSISTENCE_FAILED") from None


def _valid_acknowledgement(
    order: object,
    intent: OrderIntent,
    broker: str,
    now: datetime,
) -> bool:
    try:
        if type(order) is not BrokerOrder:
            return False
        submitted_at = _aware_utc(order.submitted_at)
        updated_at = _aware_utc(order.updated_at)
        filled = order.filled_quantity
        requested = order.requested_quantity
        average = order.average_fill_price
        status_consistent = (
            (order.status is OrderStatus.ACKNOWLEDGED and filled == 0 and average is None)
            or (
                order.status is OrderStatus.PARTIALLY_FILLED
                and type(filled) is Decimal
                and filled > 0
                and average is not None
                and (requested is None or filled < requested)
            )
            or (
                order.status is OrderStatus.FILLED
                and type(filled) is Decimal
                and filled > 0
                and average is not None
                and (requested is None or filled == requested)
            )
        )
        return (
            type(order.order_id) is str
            and bool(order.order_id)
            and order.client_order_id == intent.intent_id
            and order.broker == broker
            and order.instrument_id == intent.instrument_id
            and type(order.status) is OrderStatus
            and order.status
            in {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED}
            and status_consistent
            and requested == intent.quantity
            and type(filled) is Decimal
            and filled.is_finite()
            and filled >= 0
            and (requested is None or filled <= requested)
            and (average is None or _positive_decimal(average))
            and (filled == 0 or average is not None)
            and intent.created_at <= submitted_at <= updated_at <= now
        )
    except (AttributeError, TypeError, ValueError, ArithmeticError):
        return False


def _valid_preflight(report: PreflightReport, broker: str) -> bool:
    try:
        if report.broker != broker or type(report.gates) is not tuple or not report.gates:
            return False
        names: set[str] = set()
        for gate in report.gates:
            if (
                type(gate.name) is not str
                or not gate.name
                or gate.name in names
                or type(gate.passed) is not bool
                or type(gate.reason_code) is not str
                or not gate.reason_code
            ):
                return False
            names.add(gate.name)
        return report.ready
    except (AttributeError, TypeError, ValueError):
        return False


def _positive_decimal(value: object) -> bool:
    return type(value) is Decimal and value.is_finite() and value > 0


def _aware_utc(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _time_text(value: datetime) -> str:
    return _aware_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")
