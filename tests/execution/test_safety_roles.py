"""Adversarial proof that authenticated writers cannot cross safety roles."""

import pytest

from market_sentinel.domain.clock import FrozenClock
from market_sentinel.execution.approval import ApprovalService
from market_sentinel.execution.reconcile import Reconciler
from market_sentinel.execution.safety import SafetyAuthenticator, SafetyRepository
from market_sentinel.operations.audit import AuditLog
from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventStore
from tests.factories import DEFAULT_INSTANT

KEY = b"task-14-role-scope-test-key-material!"


def _root() -> tuple[FrozenClock, SafetyRepository]:
    clock = FrozenClock(DEFAULT_INSTANT)
    store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
    return clock, SafetyRepository(
        audit_log=AuditLog(store, clock),
        authenticator=SafetyAuthenticator(key=KEY, nonce_source=lambda: b"s" * 32),
    )


def test_generic_signer_cannot_forge_confirmation_issuance_without_phrase() -> None:
    """A repository holder cannot manufacture phrase authority for an exact intent."""
    clock, repository = _root()
    approval = ApprovalService(clock=clock, safety_capability=repository.approval_capability())
    with pytest.raises(AttributeError):
        object.__getattribute__(repository, "record_many")
    with pytest.raises(AttributeError):
        object.__getattribute__(approval, "safety_repository")


def test_generic_signer_cannot_forge_healthy_report_without_compare() -> None:
    """Matching public hashes alone cannot manufacture a healthy reconciliation authority."""
    clock, repository = _root()
    reconciler = Reconciler(safety_capability=repository.reconciliation_capability(), clock=clock)
    with pytest.raises(AttributeError):
        object.__getattribute__(repository, "record_many")
    with pytest.raises(AttributeError):
        object.__getattribute__(reconciler, "safety_repository")


def test_generic_signer_cannot_clear_kill_switch_without_ack_or_healthy_transition() -> None:
    """A signed marker cannot bypass exact acknowledgement and fresh-health prerequisites."""
    _clock, repository = _root()
    with pytest.raises(AttributeError):
        object.__getattribute__(repository, "record_many")
    with pytest.raises(ValueError, match="acknowledgement"):
        repository.reconciliation_capability().clear_kill_switch(
            acknowledgement="",
            now=DEFAULT_INSTANT,
        )


def test_role_capabilities_cannot_cross_write_domains() -> None:
    """Each injected object lacks every other role's transition methods."""
    _clock, repository = _root()
    approval = repository.approval_capability()
    reconciliation = repository.reconciliation_capability()
    live = repository.live_capability()

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
    clock, repository = _root()
    with pytest.raises(ValueError, match="exact safety capability"):
        ApprovalService(
            clock=clock,
            safety_capability=repository.reconciliation_capability(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="exact safety capability"):
        Reconciler(
            safety_capability=repository.approval_capability(),  # type: ignore[arg-type]
            clock=clock,
        )


def test_narrow_capabilities_recheck_transition_prerequisites() -> None:
    """Possession of a role object does not bypass phrase, schema, or prior-state checks."""
    _clock, repository = _root()
    with pytest.raises(ValueError, match="phrase"):
        repository.approval_capability().issue_confirmation(
            phrase="",
            broker="alpaca",
            created_at=DEFAULT_INSTANT,
            expires_at=DEFAULT_INSTANT,
            fingerprint="a" * 64,
            risk_decision_hash="b" * 64,
        )
    with pytest.raises(ValueError, match="reasons"):
        repository.reconciliation_capability().persist_report(
            broker="alpaca",
            broker_hash="a" * 64,
            ledger_hash="b" * 64,
            reason_codes=("CALLER_CHOSEN",),
            checked_at=DEFAULT_INSTANT,
        )
    with pytest.raises(ValueError, match="transition"):
        repository.live_capability().record_unknown(
            intent_id="intent",
            submission_id="submission",
            occurred_at=DEFAULT_INSTANT,
        )
