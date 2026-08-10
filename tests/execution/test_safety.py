from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from decimal import Context, Decimal, localcontext
from pathlib import Path

import pytest

import market_sentinel.execution as execution_api
from market_sentinel.domain.clock import FrozenClock
from market_sentinel.execution.canonical import CanonicalEncodingError, canonical_decimal
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

KEY = b"task-14-test-safety-key-material!!"
OTHER_KEY = b"task-14-other-safety-key-material!"


def _roles(
    path: Path | None = None,
    *,
    key: bytes = KEY,
    nonce_source: Callable[[], bytes] = lambda: b"n" * 32,
) -> tuple[
    AuditLog,
    ApprovalSafetyCapability,
    ReconciliationSafetyCapability,
    LiveSafetyCapability,
]:
    url = "sqlite+pysqlite:///:memory:" if path is None else f"sqlite+pysqlite:///{path}"
    clock = FrozenClock(DEFAULT_INSTANT)
    audit = AuditLog(EventStore(create_engine_and_schema(url)), clock)
    approval, reconciliation, live = create_safety_capabilities(
        audit_log=audit, key=key, nonce_source=nonce_source
    )
    return audit, approval, reconciliation, live


@pytest.mark.parametrize("precision", [10, 28, 60])
def test_decimal_canonicalization_is_independent_of_ambient_context(precision: int) -> None:
    """Changing Decimal context must not change exact safety bytes or collapse values."""
    with localcontext(Context(prec=precision)):
        assert canonical_decimal(Decimal("1.2300")) == "1.23"
        assert canonical_decimal(Decimal("-0.000")) == "0"
        assert canonical_decimal(Decimal("12345678901234567890123456789.1")) == (
            "12345678901234567890123456789.1"
        )
        assert canonical_decimal(Decimal("12345678901234567890123456789.2")) == (
            "12345678901234567890123456789.2"
        )


@pytest.mark.parametrize(
    "bad",
    [True, "1", Decimal("NaN"), Decimal("Infinity"), Decimal((0, (1,), 100_000))],
)
def test_decimal_canonicalization_rejects_wrong_or_resource_hostile_values(bad: object) -> None:
    """Unknown types, nonfinite values, and huge exponents must fail before expansion."""
    with pytest.raises(CanonicalEncodingError):
        canonical_decimal(bad)


def test_unsigned_public_safety_row_cannot_authorize_replay() -> None:
    """A matching public AuditLog row is not authenticated safety authority."""
    audit, _approval, reconciliation, _live = _roles()
    audit.record(
        "forged",
        "reconciliation.healthy",
        "live-reconciliation",
        {"broker": "alpaca", "healthy": True, "broker_hash": "a" * 64, "ledger_hash": "b" * 64},
    )

    with pytest.raises(SafetyIntegrityError):
        reconciliation.reconciliation_events()


def test_signed_rows_survive_restart_with_same_key_and_wrong_key_fails(tmp_path: Path) -> None:
    """Only possession of the same local key can replay persisted safety authority."""
    path = tmp_path / "safety.db"
    _audit, _approval, reconciliation, _live = _roles(path)
    row = reconciliation.persist_report(
        broker="alpaca",
        broker_hash="a" * 64,
        ledger_hash="b" * 64,
        reason_codes=("CASH_MISMATCH",),
        checked_at=DEFAULT_INSTANT,
    )

    _audit2, _approval2, restarted, _live2 = _roles(path)
    assert restarted.reconciliation_events()[0].event_id == row.event_id
    _audit3, _approval3, wrong, _live3 = _roles(path, key=OTHER_KEY)
    with pytest.raises(SafetyIntegrityError):
        wrong.reconciliation_events()


def test_key_and_nonce_validation_has_no_public_signer_fallback() -> None:
    """The one-shot authority factory rejects malformed local key and nonce material."""
    assert not hasattr(execution_api, "SafetyAuthenticator")
    with pytest.raises(ValueError):
        _roles(key=b"short")
    _audit, approval, _reconciliation, _live = _roles(nonce_source=lambda: b"short")
    with pytest.raises(ValueError):
        approval.issue_confirmation(
            phrase="I_CONFIRM_REAL_MONEY_ORDER",
            broker="alpaca",
            created_at=DEFAULT_INSTANT,
            expires_at=DEFAULT_INSTANT + timedelta(minutes=5),
            fingerprint="a" * 64,
            risk_decision_hash="b" * 64,
        )


def test_nonce_source_exception_is_sanitized_without_context() -> None:
    """An injected entropy failure cannot attach implementation or secret text."""

    def failed() -> bytes:
        raise RuntimeError("api_key=secret-token-123")

    _audit, approval, _reconciliation, _live = _roles(nonce_source=failed)
    with pytest.raises(ValueError) as captured:
        approval.issue_confirmation(
            phrase="I_CONFIRM_REAL_MONEY_ORDER",
            broker="alpaca",
            created_at=DEFAULT_INSTANT,
            expires_at=DEFAULT_INSTANT + timedelta(minutes=5),
            fingerprint="a" * 64,
            risk_decision_hash="b" * 64,
        )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "secret-token-123" not in repr(captured.value)


def test_safety_event_rejects_naive_time() -> None:
    """Safety MACs never depend on local timezone interpretation."""
    _audit, _approval, reconciliation, _live = _roles()
    with pytest.raises(ValueError):
        reconciliation.persist_report(
            broker="alpaca",
            broker_hash="a" * 64,
            ledger_hash="b" * 64,
            reason_codes=(),
            checked_at=(DEFAULT_INSTANT + timedelta(seconds=1)).replace(tzinfo=None),
        )


def test_authenticated_safety_configuration_exports_only_the_narrow_factory() -> None:
    """Task 15 can inject a local key without receiving generic signing authority."""
    assert execution_api.create_safety_capabilities is create_safety_capabilities
    assert not hasattr(execution_api, "SafetyRepository")
    assert not hasattr(execution_api, "SafetyAuthenticator")
