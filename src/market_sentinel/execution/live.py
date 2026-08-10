"""Fail-closed live submission after durable exact local confirmation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Protocol

from market_sentinel.brokers.preflight import PreflightReport, required_gate_names
from market_sentinel.domain.clock import Clock
from market_sentinel.domain.enums import OrderStatus
from market_sentinel.domain.models import (
    BrokerOrder,
    GateResult,
    MarketSnapshot,
    OrderIntent,
    RiskDecision,
)
from market_sentinel.execution.approval import ApprovalError, ApprovalService, OrderConfirmation
from market_sentinel.execution.base import BrokerCapabilities
from market_sentinel.execution.reconcile import (
    KillSwitchError,
    Reconciler,
    ReconciliationReport,
    SafetyFence,
)
from market_sentinel.execution.safety import (
    LiveSafetyCapability,
    SafetyAlreadyUsedError,
    SafetyIntegrityError,
    SafetyStateChangedError,
)
from market_sentinel.portfolio.ledger import PortfolioLedger
from market_sentinel.security import redact_secret_text
from market_sentinel.storage.events import EventHeadConflict

_AUDIT_REDACTOR = redact_secret_text


class LiveOrderError(RuntimeError):
    """One stable, secret-free live rejection reason."""


class LiveBroker(Protocol):
    """Only the injected provider operations used by the live coordinator."""

    @property
    def broker_name(self) -> str: ...

    def preflight(self) -> PreflightReport: ...

    def capabilities(self) -> BrokerCapabilities: ...

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
        safety_capability: LiveSafetyCapability,
        clock: Clock,
        ledger: PortfolioLedger,
    ) -> None:
        if type(approval_service) is not ApprovalService:
            raise ValueError("live execution requires the exact approval service")
        if type(reconciler) is not Reconciler:
            raise ValueError("live execution requires the exact reconciler")
        if type(safety_capability) is not LiveSafetyCapability:
            raise ValueError("live execution requires its exact safety capability")
        try:
            live_store_identity = safety_capability.store_identity
        except SafetyIntegrityError:
            raise ValueError(
                "live execution requires a factory-registered safety capability"
            ) from None
        if (
            live_store_identity is not reconciler.safety_store_identity
            or live_store_identity is not approval_service.safety_store_identity
        ):
            raise ValueError("live safety services must share one EventStore")
        if type(ledger) is not PortfolioLedger:
            raise ValueError("live execution requires an exact PortfolioLedger")
        broker_name = getattr(broker, "broker_name", None)
        if type(broker_name) is not str or not broker_name:
            raise ValueError("live broker identity is malformed")
        self._broker = broker
        self._approval = approval_service
        self._reconciler = reconciler
        self._safety = safety_capability
        self._clock = clock
        self._ledger = ledger
        self._broker_name = broker_name
        self._audit_redactor = _AUDIT_REDACTOR

    @property
    def audit_boundary_intact(self) -> bool:
        """Report whether this exact service retains its construction-time redactor."""
        return (
            self._audit_redactor is _AUDIT_REDACTOR
            and self._audit_redactor is redact_secret_text
        )

    @property
    def safety_store_identity(self) -> object:
        """Return the opaque shared safety-store identity used by live execution."""
        return self._approval.safety_store_identity

    @property
    def broker_name(self) -> str:
        """Return the validated broker identity bound at construction."""
        return self._broker_name

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
        audit_redactor = self._audit_redactor
        if not self.audit_boundary_intact:
            raise LiveOrderError("AUDIT_PERSISTENCE_FAILED")
        self._require_preflight(preflight)
        self._require_capabilities(intent)
        instant = _aware_utc(self._clock.now())
        self._require_risk(intent, risk_decision, instant)
        try:
            current = self._reconciler.is_current_healthy(
                reconciliation,
                broker=self._broker_name,
                ledger=self._ledger,
            )
        except BaseException:
            current = False
        if not current:
            raise LiveOrderError("RECONCILIATION_NOT_CURRENT")
        try:
            kill_switch_active = self._reconciler.kill_switch_active()
        except BaseException:
            kill_switch_active = True
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
            confirmation_valid = False
        else:
            confirmation_valid = True
        if not confirmation_valid:
            raise LiveOrderError("CONFIRMATION_INVALID")
        self._require_snapshot(intent, snapshot, instant)
        try:
            fence = self._reconciler.safety_fence(
                reconciliation,
                broker=self._broker_name,
                ledger=self._ledger,
            )
        except KillSwitchError:
            fence = None
            fence_reason = "SAFETY_STATE_CHANGED"
        except BaseException:
            fence = None
            fence_reason = "AUDIT_PERSISTENCE_FAILED"
        else:
            fence_reason = ""
        if fence is None:
            raise LiveOrderError(fence_reason)
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
            confirmation_valid = False
        else:
            confirmation_valid = True
        if not confirmation_valid:
            raise LiveOrderError("CONFIRMATION_INVALID")
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
            self._safety.record_acknowledgement(
                intent_id=intent.intent_id,
                broker=self._broker_name,
                broker_order_id=audit_redactor(submitted.order_id),
                status=submitted.status.value,
                submission_id=confirmation.confirmation_id,
                occurred_at=acknowledged_at,
            )
        except BaseException:
            acknowledgement_persisted = False
        else:
            acknowledgement_persisted = True
        if not acknowledgement_persisted:
            self._persist_unknown(intent.intent_id, confirmation.confirmation_id)
            raise LiveOrderError("SUBMISSION_UNKNOWN") from None
        return submitted

    def _require_preflight(self, report: object) -> None:
        try:
            fresh = self._broker.preflight()
        except BaseException:
            fresh = None
        if type(report) is not PreflightReport or type(fresh) is not PreflightReport:
            raise LiveOrderError("PREFLIGHT_NOT_READY")
        if not _valid_preflight(fresh, self._broker_name) or report != fresh:
            raise LiveOrderError("PREFLIGHT_NOT_READY")

    def _require_capabilities(self, intent: OrderIntent) -> None:
        try:
            capabilities = self._broker.capabilities()
        except BaseException:
            capabilities = None
        if (
            type(capabilities) is not BrokerCapabilities
            or capabilities.broker != self._broker_name
            or (intent.notional is not None and not capabilities.supports_notional_orders)
        ):
            raise LiveOrderError("ORDER_UNSUPPORTED")

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
            timestamps = None
        else:
            timestamps = (decided_at, expires_at, intent_created, intent_expires)
        if timestamps is None:
            raise LiveOrderError("RISK_NOT_APPROVED")
        decided_at, expires_at, intent_created, intent_expires = timestamps
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
            timestamps = None
        else:
            timestamps = (observed_at, source_at)
        if timestamps is None:
            raise LiveOrderError("SNAPSHOT_INVALID")
        observed_at, source_at = timestamps
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
        try:
            self._safety.claim_and_start(
                intent_id=intent.intent_id,
                broker=self._broker_name,
                confirmation_id=confirmation.confirmation_id,
                fingerprint=confirmation.fingerprint,
                expires_at=confirmation.expires_at,
                reconciliation_head=fence.reconciliation_head,
                kill_switch_head=fence.kill_switch_head,
                interlock_head=fence.interlock_head,
                occurred_at=instant,
            )
        except SafetyAlreadyUsedError:
            raise LiveOrderError("CONFIRMATION_USED") from None
        except SafetyStateChangedError:
            raise LiveOrderError("SAFETY_STATE_CHANGED") from None
        except SafetyIntegrityError:
            raise LiveOrderError("CONFIRMATION_INVALID") from None
        except EventHeadConflict:
            failure = "conflict"
        except BaseException:
            failure = "persistence"
        else:
            failure = ""
        if not failure:
            return
        if failure == "conflict":
            raise LiveOrderError("SAFETY_STATE_CHANGED")
        raise LiveOrderError("AUDIT_PERSISTENCE_FAILED")

    def _query_ambiguous(self, intent: OrderIntent) -> BrokerOrder | None:
        try:
            candidate = self._broker.get_order_by_client_id(intent.intent_id)
        except BaseException:
            candidate = None
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
            self._safety.record_unknown(
                intent_id=intent_id,
                submission_id=submission_id,
                occurred_at=_aware_utc(self._clock.now()),
            )
        except BaseException:
            persisted = False
        else:
            persisted = True
        if not persisted:
            raise LiveOrderError("UNKNOWN_PERSISTENCE_FAILED")


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
        requested_notional = order.requested_notional
        average = order.average_fill_price
        if (
            type(order.order_id) is not str
            or not order.order_id
            or type(order.client_order_id) is not str
            or order.client_order_id != intent.intent_id
            or type(order.broker) is not str
            or order.broker != broker
            or type(order.instrument_id) is not str
            or order.instrument_id != intent.instrument_id
            or type(order.status) is not OrderStatus
            or type(filled) is not Decimal
            or not filled.is_finite()
            or filled < 0
            or (average is not None and not _positive_decimal(average))
        ):
            return False
        if intent.quantity is not None:
            if (
                type(requested) is not Decimal
                or not requested.is_finite()
                or requested <= 0
                or requested != intent.quantity
                or requested_notional is not None
                or filled > requested
            ):
                return False
        elif (
            intent.notional is None
            or requested is not None
            or type(requested_notional) is not Decimal
            or not requested_notional.is_finite()
            or requested_notional <= 0
            or requested_notional != intent.notional
        ):
            return False
        status_consistent = (
            (
                order.status is OrderStatus.ACKNOWLEDGED
                and filled == Decimal("0")
                and average is None
            )
            or (
                order.status is OrderStatus.PARTIALLY_FILLED
                and filled > 0
                and average is not None
                and type(requested) is Decimal
                and filled < requested
            )
            or (
                order.status is OrderStatus.FILLED
                and filled > 0
                and average is not None
                and type(requested) is Decimal
                and filled == requested
            )
        )
        return (
            order.status
            in {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED}
            and status_consistent
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
                type(gate) is not GateResult
                or type(gate.name) is not str
                or not gate.name
                or gate.name in names
                or type(gate.passed) is not bool
                or type(gate.reason_code) is not str
                or not gate.reason_code
                or (gate.passed and gate.reason_code != "OK")
            ):
                return False
            names.add(gate.name)
        return names == required_gate_names(broker) and report.ready
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
