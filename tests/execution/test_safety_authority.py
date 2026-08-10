"""Adversarial tests for non-exported signing authority and authoritative live claims."""

from __future__ import annotations

from datetime import timedelta
from inspect import signature
from itertools import count

import pytest

import market_sentinel.execution as execution_api
import market_sentinel.execution.safety as safety_module
from market_sentinel.domain.clock import FrozenClock
from market_sentinel.execution.safety import (
    ApprovalSafetyCapability,
    LiveSafetyCapability,
    ReconciliationSafetyCapability,
    SafetyIntegrityError,
    create_safety_capabilities,
)
from market_sentinel.operations.audit import AuditLog
from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventStore
from tests.factories import DEFAULT_INSTANT

KEY = b"task-14-authority-test-key-material!!"


def _authority() -> tuple[
    FrozenClock,
    AuditLog,
    ApprovalSafetyCapability,
    ReconciliationSafetyCapability,
    LiveSafetyCapability,
]:
    nonces = count(1)
    clock = FrozenClock(DEFAULT_INSTANT)
    audit = AuditLog(EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:")), clock)
    approval, reconciliation, live = create_safety_capabilities(
        audit_log=audit,
        key=KEY,
        nonce_source=lambda: next(nonces).to_bytes(32, "big"),
    )
    return clock, audit, approval, reconciliation, live


def _issue(approval: ApprovalSafetyCapability, *, suffix: str = "") -> tuple[str, str]:
    confirmation_id, _nonce, _mac = approval.issue_confirmation(
        phrase="I_CONFIRM_REAL_MONEY_ORDER",
        broker="alpaca",
        created_at=DEFAULT_INSTANT,
        expires_at=DEFAULT_INSTANT + timedelta(minutes=5),
        fingerprint=("a" if not suffix else "b") * 64,
        risk_decision_hash="c" * 64,
    )
    return confirmation_id, ("a" if not suffix else "b") * 64


def _claim(
    live: LiveSafetyCapability,
    *,
    confirmation_id: str,
    fingerprint: str,
    reconciliation_head: str,
    kill_head: str | None,
    interlock_head: str | None,
) -> None:
    live.claim_and_start(
        intent_id=f"intent-{confirmation_id[:8]}",
        broker="alpaca",
        confirmation_id=confirmation_id,
        fingerprint=fingerprint,
        expires_at=DEFAULT_INSTANT + timedelta(minutes=5),
        reconciliation_head=reconciliation_head,
        kill_switch_head=kill_head,
        interlock_head=interlock_head,
        occurred_at=DEFAULT_INSTANT,
    )


def test_public_api_and_capability_graph_expose_no_generic_signing_path() -> None:
    """A role holder must not reach a root, key, signer, or generic event writer."""
    _clock, _audit, approval, reconciliation, live = _authority()
    assert not hasattr(execution_api, "SafetyAuthenticator")
    assert not hasattr(safety_module, "SafetyAuthenticator")
    assert not hasattr(safety_module, "SafetyRepository")
    for capability in (
        approval,
        reconciliation,
        live,
    ):
        assert not hasattr(capability, "_repository")
        assert not hasattr(capability, "sign")
        assert not hasattr(capability, "record_many")
        assert not hasattr(capability, "stream_verified")
        for slot in capability.__slots__:
            value = object.__getattribute__(capability, slot)
            assert type(value).__name__ not in {
                "_SafetyAuthority",
                "_SafetyMac",
                "_ApprovalRole",
                "_ReconciliationRole",
                "_LiveRole",
                "AuditLog",
                "EventStore",
            }
            if callable(value) and slot != "__weakref__":
                assert not {"kind", "aggregate_id", "payload", "batch"} & set(
                    signature(value).parameters
                )
                with pytest.raises(TypeError):
                    value(kind="reconciliation.healthy", aggregate_id="live-reconciliation")
                for cell in value.__closure__ or ():
                    assert type(cell.cell_contents).__name__ not in {
                        "_SafetyAuthority",
                        "_SafetyMac",
                        "_ApprovalRole",
                        "_ReconciliationRole",
                        "_LiveRole",
                        "AuditLog",
                        "EventStore",
                    }
                    assert type(cell.cell_contents) is not bytes


def test_capability_class_swap_cannot_confuse_roles() -> None:
    """Distinct immutable layouts prevent turning one issued handle into another role."""
    _clock, _audit, approval, reconciliation, _live = _authority()
    with pytest.raises((AttributeError, TypeError)):
        approval.__class__ = type(reconciliation)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="cannot be constructed"):
        ApprovalSafetyCapability(
            token=object(),
            issue=reconciliation.persist_report,  # type: ignore[arg-type]
            read=reconciliation.reconciliation_events,  # type: ignore[arg-type]
            store_identity=reconciliation.store_identity,
        )


def test_live_claim_rejects_unhealthy_reconciliation_without_started_row() -> None:
    """Passing the exact unhealthy head cannot turn its label into live authority."""
    _clock, audit, approval, reconciliation, live = _authority()
    confirmation_id, fingerprint = _issue(approval)
    reconciliation.persist_report(
        broker="alpaca",
        broker_hash="d" * 64,
        ledger_hash="e" * 64,
        reason_codes=("CASH_MISMATCH",),
        checked_at=DEFAULT_INSTANT,
    )
    reconciliation_head = tuple(audit.event_store.stream("live-reconciliation"))[-1].event_id
    kill_head = tuple(audit.event_store.stream("live-kill-switch"))[-1].event_id

    with pytest.raises(SafetyIntegrityError):
        _claim(
            live,
            confirmation_id=confirmation_id,
            fingerprint=fingerprint,
            reconciliation_head=reconciliation_head,
            kill_head=kill_head,
            interlock_head=None,
        )
    assert tuple(audit.event_store.stream(f"intent-{confirmation_id[:8]}")) == ()


def test_live_claim_rejects_stale_healthy_reconciliation_without_started_row() -> None:
    """A signed healthy head older than the fixed window is not live authority."""
    _clock, audit, approval, reconciliation, live = _authority()
    confirmation_id, fingerprint = _issue(approval)
    report = reconciliation.persist_report(
        broker="alpaca",
        broker_hash="d" * 64,
        ledger_hash="e" * 64,
        reason_codes=(),
        checked_at=DEFAULT_INSTANT - timedelta(seconds=61),
    )

    with pytest.raises(SafetyIntegrityError):
        _claim(
            live,
            confirmation_id=confirmation_id,
            fingerprint=fingerprint,
            reconciliation_head=report.event_id,
            kill_head=None,
            interlock_head=None,
        )
    assert tuple(audit.event_store.stream(f"intent-{confirmation_id[:8]}")) == ()


def test_live_claim_rejects_public_forged_healthy_and_clear_rows() -> None:
    """Unsigned matching heads can deny service but can never authorize a claim."""
    _clock, audit, approval, _reconciliation, live = _authority()
    confirmation_id, fingerprint = _issue(approval)
    audit.record(
        "forged-healthy",
        "reconciliation.healthy",
        "live-reconciliation",
        {
            "broker": "alpaca",
            "broker_hash": "d" * 64,
            "healthy": True,
            "ledger_hash": "e" * 64,
            "reason_codes": [],
        },
    )
    audit.record(
        "forged-clear",
        "kill_switch.cleared",
        "live-kill-switch",
        {"activation_event_id": "missing", "activation_sequence": 1},
    )

    with pytest.raises(SafetyIntegrityError):
        _claim(
            live,
            confirmation_id=confirmation_id,
            fingerprint=fingerprint,
            reconciliation_head="forged-healthy",
            kill_head="forged-clear",
            interlock_head=None,
        )
    assert tuple(audit.event_store.stream(f"intent-{confirmation_id[:8]}")) == ()


def test_live_claim_rejects_active_kill_and_unresolved_interlock() -> None:
    """Exact caller-supplied active heads cannot bypass semantic safety replay."""
    _clock, audit, approval, reconciliation, live = _authority()
    first_id, first_fingerprint = _issue(approval)
    report = reconciliation.persist_report(
        broker="alpaca",
        broker_hash="d" * 64,
        ledger_hash="e" * 64,
        reason_codes=(),
        checked_at=DEFAULT_INSTANT,
    )
    reconciliation.persist_report(
        broker="alpaca",
        broker_hash="f" * 64,
        ledger_hash="e" * 64,
        reason_codes=("CASH_MISMATCH",),
        checked_at=DEFAULT_INSTANT,
    )
    active_report = tuple(audit.event_store.stream("live-reconciliation"))[-1]
    active_kill = tuple(audit.event_store.stream("live-kill-switch"))[-1]
    with pytest.raises(SafetyIntegrityError):
        _claim(
            live,
            confirmation_id=first_id,
            fingerprint=first_fingerprint,
            reconciliation_head=active_report.event_id,
            kill_head=active_kill.event_id,
            interlock_head=None,
        )

    # A separate authority isolates the unresolved-interlock case behind a healthy report.
    _clock2, audit2, approval2, reconciliation2, live2 = _authority()
    first_id, first_fingerprint = _issue(approval2)
    report = reconciliation2.persist_report(
        broker="alpaca",
        broker_hash="d" * 64,
        ledger_hash="e" * 64,
        reason_codes=(),
        checked_at=DEFAULT_INSTANT,
    )
    _claim(
        live2,
        confirmation_id=first_id,
        fingerprint=first_fingerprint,
        reconciliation_head=report.event_id,
        kill_head=None,
        interlock_head=None,
    )
    second_id, second_fingerprint = _issue(approval2, suffix="2")
    interlock_head = tuple(audit2.event_store.stream("live-submission-interlock"))[-1].event_id
    with pytest.raises(SafetyIntegrityError):
        _claim(
            live2,
            confirmation_id=second_id,
            fingerprint=second_fingerprint,
            reconciliation_head=report.event_id,
            kill_head=None,
            interlock_head=interlock_head,
        )
    assert tuple(audit2.event_store.stream(f"intent-{second_id[:8]}")) == ()
