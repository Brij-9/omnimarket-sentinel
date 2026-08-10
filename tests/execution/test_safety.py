from __future__ import annotations

from datetime import timedelta
from decimal import Context, Decimal, localcontext
from pathlib import Path

import pytest

from market_sentinel.domain.clock import FrozenClock
from market_sentinel.execution import SafetyAuthenticator as PublicSafetyAuthenticator
from market_sentinel.execution.canonical import CanonicalEncodingError, canonical_decimal
from market_sentinel.execution.safety import (
    SafetyAuthenticator,
    SafetyEvent,
    SafetyIntegrityError,
    SafetyRepository,
)
from market_sentinel.operations.audit import AuditLog
from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventStore
from tests.factories import DEFAULT_INSTANT

KEY = b"task-14-test-safety-key-material!!"
OTHER_KEY = b"task-14-other-safety-key-material!"


def _repository(path: Path | None = None, *, key: bytes = KEY) -> SafetyRepository:
    url = "sqlite+pysqlite:///:memory:" if path is None else f"sqlite+pysqlite:///{path}"
    clock = FrozenClock(DEFAULT_INSTANT)
    audit = AuditLog(EventStore(create_engine_and_schema(url)), clock)
    authenticator = SafetyAuthenticator(key=key, nonce_source=lambda: b"n" * 32)
    return SafetyRepository(audit_log=audit, authenticator=authenticator)


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
    repository = _repository()
    repository.audit_log.record(
        "forged",
        "reconciliation.healthy",
        "live-reconciliation",
        {"broker": "alpaca", "healthy": True, "broker_hash": "a" * 64, "ledger_hash": "b" * 64},
    )

    with pytest.raises(SafetyIntegrityError):
        repository.stream_verified("live-reconciliation")


def test_signed_rows_survive_restart_with_same_key_and_wrong_key_fails(tmp_path: Path) -> None:
    """Only possession of the same local key can replay persisted safety authority."""
    path = tmp_path / "safety.db"
    repository = _repository(path)
    repository.record_many(
        (
            SafetyEvent(
                event_id="signed",
                kind="kill_switch.activated",
                aggregate_id="live-kill-switch",
                payload={"reason_codes": ["TEST"]},
                occurred_at=DEFAULT_INSTANT,
            ),
        )
    )

    assert _repository(path).stream_verified("live-kill-switch")[0].event_id == "signed"
    with pytest.raises(SafetyIntegrityError):
        _repository(path, key=OTHER_KEY).stream_verified("live-kill-switch")


def test_authenticator_repr_and_errors_never_disclose_key_or_nonce() -> None:
    """Local authentication material must stay out of representations and errors."""
    authenticator = SafetyAuthenticator(key=KEY, nonce_source=lambda: b"z" * 32)
    assert KEY.decode() not in repr(authenticator)
    assert (b"z" * 32).decode() not in repr(authenticator)
    assert not hasattr(authenticator, "key")
    assert not hasattr(authenticator, "nonce_source")
    assert len(authenticator.new_nonce()) == 64


def test_safety_authenticator_rejects_short_or_missing_key_and_nonce() -> None:
    """There is no deterministic or empty production fallback for key/nonce material."""
    with pytest.raises(ValueError):
        SafetyAuthenticator(key=b"short", nonce_source=lambda: b"n" * 32)
    with pytest.raises(ValueError):
        SafetyAuthenticator(key=KEY, nonce_source=lambda: b"short").new_nonce()
    with pytest.raises(ValueError):
        SafetyAuthenticator(
            key=KEY,
            nonce_source=lambda: "not-bytes",  # type: ignore[arg-type,return-value]
        ).new_nonce()


def test_nonce_source_exception_is_sanitized_without_context() -> None:
    """An injected entropy failure cannot attach implementation or secret text."""

    def failed() -> bytes:
        raise RuntimeError("api_key=secret-token-123")

    authenticator = SafetyAuthenticator(key=KEY, nonce_source=failed)
    with pytest.raises(ValueError) as captured:
        authenticator.new_nonce()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "secret-token-123" not in repr(captured.value)


def test_safety_event_rejects_naive_time() -> None:
    """Safety MACs never depend on local timezone interpretation."""
    repository = _repository()
    event = SafetyEvent(
        "naive",
        "kill_switch.activated",
        "live-kill-switch",
        {},
        (DEFAULT_INSTANT + timedelta(seconds=1)).replace(tzinfo=None),
    )
    with pytest.raises(ValueError):
        repository.record_many((event,))


def test_authenticated_safety_configuration_has_an_explicit_public_api() -> None:
    """Task 15 can wire a local key without importing a private implementation detail."""
    assert PublicSafetyAuthenticator is SafetyAuthenticator


def test_safety_mac_is_stable_across_contexts_and_distinguishes_long_decimals() -> None:
    """Canonical decimal payloads cannot collide before authenticated safety persistence."""
    authenticator = SafetyAuthenticator(key=KEY, nonce_source=lambda: b"n" * 32)

    def mac(value: Decimal, precision: int) -> str:
        with localcontext(Context(prec=precision)):
            return authenticator.sign(
                event_id="decimal-event",
                kind="reconciliation.healthy",
                aggregate_id="live-reconciliation",
                occurred_at=DEFAULT_INSTANT,
                payload={"exact_decimal": canonical_decimal(value)},
            )

    first = Decimal("12345678901234567890123456789.1")
    second = Decimal("12345678901234567890123456789.2")
    assert len({mac(first, precision) for precision in (10, 28, 60)}) == 1
    assert mac(first, 28) != mac(second, 28)
