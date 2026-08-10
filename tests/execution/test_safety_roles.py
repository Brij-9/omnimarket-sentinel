"""Adversarial proof that authenticated writers cannot cross safety roles."""

from datetime import timedelta

import pytest

from market_sentinel.domain.clock import FrozenClock
from market_sentinel.execution.approval import ApprovalService
from market_sentinel.execution.reconcile import Reconciler
from market_sentinel.execution.safety import (
    ApprovalSafetyCapability,
    LiveSafetyCapability,
    ReconciliationSafetyCapability,
    create_safety_capabilities,
)
from market_sentinel.operations.audit import AuditLog
from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventStore
from tests.factories import DEFAULT_INSTANT

KEY = b"task-14-role-scope-test-key-material!"


def _roles() -> tuple[
    FrozenClock,
    ApprovalSafetyCapability,
    ReconciliationSafetyCapability,
    LiveSafetyCapability,
]:
    clock = FrozenClock(DEFAULT_INSTANT)
    store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
    approval, reconciliation, live = create_safety_capabilities(
        audit_log=AuditLog(store, clock),
        key=KEY,
        nonce_source=lambda: b"s" * 32,
    )
    return clock, approval, reconciliation, live


def test_services_expose_no_authenticated_writer_or_capability() -> None:
    """Service holders cannot retrieve any role handle, root, signer, or generic writer."""
    clock, approval_safety, reconciliation_safety, _live = _roles()
    approval = ApprovalService(clock=clock, safety_capability=approval_safety)
    reconciler = Reconciler(safety_capability=reconciliation_safety, clock=clock)
    for service in (approval, reconciler):
        for forbidden in ("safety_repository", "safety_capability", "record_many", "sign"):
            with pytest.raises(AttributeError):
                object.__getattribute__(service, forbidden)


def test_role_capabilities_cannot_cross_write_domains() -> None:
    """Each injected object lacks every other role's transition methods."""
    _clock, approval, reconciliation, live = _roles()
    for capability, forbidden in (
        (approval, "persist_report"),
        (approval, "record_unknown"),
        (reconciliation, "issue_confirmation"),
        (reconciliation, "record_unknown"),
        (live, "issue_confirmation"),
        (live, "persist_report"),
        (live, "clear_kill_switch"),
    ):
        with pytest.raises(AttributeError):
            object.__getattribute__(capability, forbidden)


def test_services_reject_a_capability_for_another_role() -> None:
    """Exact constructor checks prevent accidental or adversarial cross-role injection."""
    clock, approval, reconciliation, _live = _roles()
    with pytest.raises(ValueError, match="exact safety capability"):
        ApprovalService(clock=clock, safety_capability=reconciliation)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact safety capability"):
        Reconciler(safety_capability=approval, clock=clock)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="factory-registered"):
        ApprovalService(clock=clock, safety_capability=ApprovalSafetyCapability())
    with pytest.raises(ValueError, match="factory-registered"):
        Reconciler(safety_capability=ReconciliationSafetyCapability(), clock=clock)


def test_narrow_capabilities_recheck_transition_prerequisites() -> None:
    """Possession of a role object does not bypass phrase, schema, or prior-state checks."""
    _clock, approval, reconciliation, live = _roles()
    with pytest.raises(ValueError, match="phrase"):
        approval.issue_confirmation(
            phrase="",
            broker="alpaca",
            created_at=DEFAULT_INSTANT,
            expires_at=DEFAULT_INSTANT + timedelta(minutes=5),
            fingerprint="a" * 64,
            risk_decision_hash="b" * 64,
        )
    with pytest.raises(ValueError, match="reasons"):
        reconciliation.persist_report(
            broker="alpaca",
            broker_hash="a" * 64,
            ledger_hash="b" * 64,
            reason_codes=("CALLER_CHOSEN",),
            checked_at=DEFAULT_INSTANT,
        )
    with pytest.raises(ValueError, match="transition"):
        live.record_unknown(
            intent_id="intent",
            submission_id="submission",
            occurred_at=DEFAULT_INSTANT,
        )
