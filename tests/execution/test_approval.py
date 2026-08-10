from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Context, Decimal, localcontext
from pathlib import Path
from unittest.mock import patch

import pytest

from market_sentinel.domain.clock import FrozenClock
from market_sentinel.domain.enums import OrderType, Side
from market_sentinel.domain.models import OrderIntent, RiskDecision
from market_sentinel.execution.approval import (
    CONFIRMATION_PHRASE,
    ApprovalError,
    ApprovalService,
    risk_decision_hash,
)
from market_sentinel.execution.safety import create_safety_capabilities
from market_sentinel.operations.audit import AuditLog
from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventStore
from tests.factories import DEFAULT_INSTANT, intent, risk_decision

KEY = b"task-14-approval-test-key-material!!"


def _authenticated_service(
    clock: FrozenClock,
    path: Path | None = None,
    *,
    key: bytes = KEY,
) -> tuple[ApprovalService, EventStore]:
    url = "sqlite+pysqlite:///:memory:" if path is None else f"sqlite+pysqlite:///{path}"
    store = EventStore(create_engine_and_schema(url))
    approval, _reconciliation, _live = create_safety_capabilities(
        audit_log=AuditLog(store, clock),
        key=key,
        nonce_source=lambda: b"a" * 32,
    )
    return ApprovalService(clock=clock, safety_capability=approval), store


def _approved(*, quantity: str = "0.1", snapshot_hash: str = "a" * 64) -> RiskDecision:
    return risk_decision(
        approved=True,
        reason_codes=(),
        approved_quantity=quantity,
        approved_notional="10",
        portfolio_hash=snapshot_hash,
        expires_at=DEFAULT_INSTANT + timedelta(minutes=1),
    )


def _intent() -> OrderIntent:
    return intent(
        quantity="0.1",
        notional=None,
        limit_price="100",
        stop_loss="98",
        take_profit="104",
    )


def test_confirmation_expires_and_is_bound_to_every_order_parameter() -> None:
    """Changing size or waiting through expiry must invalidate the exact capability."""
    clock = FrozenClock(DEFAULT_INSTANT)
    service = _authenticated_service(clock)[0]
    original = _intent()
    risk = _approved()
    confirmation = service.create(original, risk, phrase=CONFIRMATION_PHRASE, broker="alpaca")

    changed = original.model_copy(update={"quantity": Decimal("0.2")})
    with pytest.raises(ApprovalError, match="fingerprint"):
        service.verify(changed, risk, confirmation, broker="alpaca")

    clock.advance(timedelta(minutes=5))
    with pytest.raises(ApprovalError, match="expired"):
        service.verify(original, risk, confirmation, broker="alpaca")


@pytest.mark.parametrize(
    "change",
    [
        {"intent_id": "other"},
        {"instrument_id": "MSFT@alpaca"},
        {"side": Side.SELL, "stop_loss": Decimal("104"), "take_profit": Decimal("98")},
        {"quantity": Decimal("0.2")},
        {"order_type": OrderType.MARKET, "limit_price": None},
        {"limit_price": Decimal("101")},
        {"stop_loss": Decimal("97")},
        {"take_profit": Decimal("105")},
        {"time_in_force": "gtc"},
        {"product": "margin"},
        {"session": "extended"},
        {"snapshot_hash": "b" * 64},
        {"created_at": DEFAULT_INSTANT - timedelta(seconds=1)},
        {"expires_at": DEFAULT_INSTANT + timedelta(seconds=30)},
    ],
)
def test_confirmation_fingerprint_rejects_each_changed_intent_field(
    change: dict[str, object],
) -> None:
    """Dropping any intent field from the fingerprint would let that mutation pass."""
    service = _authenticated_service(FrozenClock(DEFAULT_INSTANT))[0]
    original = _intent()
    risk = _approved()
    confirmation = service.create(original, risk, phrase=CONFIRMATION_PHRASE, broker="alpaca")

    with pytest.raises(ApprovalError, match="fingerprint"):
        service.verify(original.model_copy(update=change), risk, confirmation, broker="alpaca")


def test_confirmation_binds_broker_and_complete_risk_decision() -> None:
    """A broker change or a freshly forged risk decision must require new confirmation."""
    service = _authenticated_service(FrozenClock(DEFAULT_INSTANT))[0]
    original = _intent()
    risk = _approved()
    confirmation = service.create(original, risk, phrase=CONFIRMATION_PHRASE, broker="alpaca")

    with pytest.raises(ApprovalError, match="fingerprint"):
        service.verify(original, risk, confirmation, broker="groww")
    with pytest.raises(ApprovalError, match="fingerprint"):
        service.verify(
            original,
            risk.model_copy(update={"expires_at": risk.expires_at - timedelta(seconds=1)}),
            confirmation,
            broker="alpaca",
        )


def test_confirmation_requires_exact_phrase_without_retaining_it() -> None:
    """A near-match must fail and the capability representation must not disclose the phrase."""
    service = _authenticated_service(FrozenClock(DEFAULT_INSTANT))[0]
    with pytest.raises(ApprovalError, match="phrase"):
        service.create(_intent(), _approved(), phrase=f"{CONFIRMATION_PHRASE} ", broker="alpaca")

    confirmation = service.create(
        _intent(), _approved(), phrase=CONFIRMATION_PHRASE, broker="alpaca"
    )
    assert CONFIRMATION_PHRASE not in repr(confirmation)


def test_confirmation_uses_canonical_decimal_and_utc_encodings() -> None:
    """Equivalent Decimal scales and timezone offsets must produce the same exact meaning."""
    service = _authenticated_service(FrozenClock(DEFAULT_INSTANT))[0]
    original = _intent()
    risk = _approved()
    confirmation = service.create(original, risk, phrase=CONFIRMATION_PHRASE, broker="alpaca")
    equivalent = original.model_copy(
        update={"quantity": Decimal("0.100"), "limit_price": Decimal("100.00")}
    )
    equivalent_risk = risk.model_copy(
        update={"approved_quantity": Decimal("0.100"), "approved_notional": Decimal("10.00")}
    )
    service.verify(equivalent, equivalent_risk, confirmation, broker="alpaca")


@pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity"), "0.1"])
def test_confirmation_rejects_malformed_or_nonfinite_numeric_records(bad: object) -> None:
    """Defensive verification must reject values bypassing Pydantic validation."""
    service = _authenticated_service(FrozenClock(DEFAULT_INSTANT))[0]
    original = _intent()
    risk = _approved()
    confirmation = service.create(original, risk, phrase=CONFIRMATION_PHRASE, broker="alpaca")
    malformed = original.model_copy(update={"quantity": bad})

    with pytest.raises(ApprovalError, match="malformed"):
        service.verify(malformed, risk, confirmation, broker="alpaca")


def test_confirmation_rejects_future_and_naive_capability_times() -> None:
    """Clock rollback and context-dependent naive timestamps must fail closed."""
    service = _authenticated_service(FrozenClock(DEFAULT_INSTANT))[0]
    original = _intent()
    risk = _approved()
    confirmation = service.create(original, risk, phrase=CONFIRMATION_PHRASE, broker="alpaca")

    future = replace(confirmation, created_at=DEFAULT_INSTANT + timedelta(seconds=1))
    with pytest.raises(ApprovalError, match="future"):
        service.verify(original, risk, future, broker="alpaca")
    naive = replace(confirmation, expires_at=datetime(2026, 8, 9, 10, 5))
    with pytest.raises(ApprovalError, match="malformed"):
        service.verify(original, risk, naive, broker="alpaca")


def test_confirmation_rejects_defensively_forged_negative_intent_values() -> None:
    """A fingerprint must never normalize a domain-invalid numeric intent into a capability."""
    service = _authenticated_service(FrozenClock(DEFAULT_INSTANT))[0]
    original = _intent()
    risk = _approved()
    confirmation = service.create(original, risk, phrase=CONFIRMATION_PHRASE, broker="alpaca")
    forged = original.model_copy(update={"quantity": Decimal("-0.1")})

    with pytest.raises(ApprovalError, match="malformed"):
        service.verify(forged, risk, confirmation, broker="alpaca")


def test_confirmation_rejects_a_risk_decision_with_an_overlong_freshness_window() -> None:
    """A forged far-future expiry must not extend RiskEngine's one-minute approval window."""
    service = _authenticated_service(FrozenClock(DEFAULT_INSTANT))[0]
    risk = _approved().model_copy(update={"expires_at": DEFAULT_INSTANT + timedelta(minutes=10)})

    with pytest.raises(ApprovalError, match="freshness"):
        service.create(_intent(), risk, phrase=CONFIRMATION_PHRASE, broker="alpaca")


@pytest.mark.parametrize(
    "forged",
    [
        _intent().model_copy(update={"order_type": OrderType.MARKET}),
        _intent().model_copy(update={"trigger_price": Decimal("99")}),
        _intent().model_copy(update={"stop_loss": Decimal("105")}),
        _intent().model_copy(update={"created_at": DEFAULT_INSTANT + timedelta(seconds=1)}),
        _intent().model_copy(update={"expires_at": DEFAULT_INSTANT}),
    ],
)
def test_create_rejects_defensively_forged_order_invariants(forged: OrderIntent) -> None:
    """Confirmation creation must re-enforce the full OrderIntent domain boundary."""
    service = _authenticated_service(FrozenClock(DEFAULT_INSTANT))[0]
    with pytest.raises(ApprovalError, match="intent"):
        service.create(forged, _approved(), phrase=CONFIRMATION_PHRASE, broker="alpaca")


def test_confirmation_is_issued_durably_before_return_and_replays_with_same_key(
    tmp_path: Path,
) -> None:
    """A confirmation is authority only after its signed issuance survives restart."""
    path = tmp_path / "confirmation.db"
    clock = FrozenClock(DEFAULT_INSTANT)
    service, store = _authenticated_service(clock, path)
    confirmation = service.create(
        _intent(), _approved(), phrase=CONFIRMATION_PHRASE, broker="alpaca"
    )
    rows = tuple(store.stream(f"live-confirmation:{confirmation.confirmation_id}"))
    assert [row.kind for row in rows] == ["confirmation.issued"]
    assert rows[0].payload["safety_mac"] == confirmation.mac
    restarted, _ = _authenticated_service(clock, path)
    restarted.verify(_intent(), _approved(), confirmation, broker="alpaca")


def test_publicly_constructed_or_modified_confirmation_is_rejected() -> None:
    """Knowing every public order field and digest cannot forge persisted phrase authority."""
    clock = FrozenClock(DEFAULT_INSTANT)
    service, _store = _authenticated_service(clock)
    issued = service.create(_intent(), _approved(), phrase=CONFIRMATION_PHRASE, broker="alpaca")
    forged = replace(issued, confirmation_id="f" * 64)
    modified = replace(issued, nonce="00" * 32)
    for candidate in (forged, modified):
        with pytest.raises(ApprovalError):
            service.verify(_intent(), _approved(), candidate, broker="alpaca")


def test_wrong_key_restart_cannot_verify_an_issued_confirmation(tmp_path: Path) -> None:
    """A copied SQLite file without the local safety key grants no live authority."""
    path = tmp_path / "wrong-key.db"
    clock = FrozenClock(DEFAULT_INSTANT)
    service, _ = _authenticated_service(clock, path)
    confirmation = service.create(
        _intent(), _approved(), phrase=CONFIRMATION_PHRASE, broker="alpaca"
    )
    wrong, _ = _authenticated_service(clock, path, key=b"task-14-approval-wrong-key-material!")
    with pytest.raises(ApprovalError):
        wrong.verify(_intent(), _approved(), confirmation, broker="alpaca")


def test_confirmation_issuance_persistence_failure_returns_no_capability() -> None:
    """No confirmation object escapes when durable signed issuance fails."""
    clock = FrozenClock(DEFAULT_INSTANT)
    service, store = _authenticated_service(clock)
    store._engine.dispose()  # noqa: SLF001 - deliberate local persistence failure
    with pytest.raises(ApprovalError, match="persistence"):
        service.create(_intent(), _approved(), phrase=CONFIRMATION_PHRASE, broker="alpaca")


def test_intent_and_risk_hashes_do_not_collide_or_change_with_decimal_context() -> None:
    """Long exact values retain distinct fingerprint bytes at every ambient precision."""
    fingerprints: list[str] = []
    risk_hashes: list[str] = []
    for precision in (10, 28, 60):
        with localcontext(Context(prec=precision)):
            value = Decimal("12345678901234567890123456789.1")
            order = _intent().model_copy(update={"quantity": value})
            risk = _approved().model_copy(update={"approved_quantity": value})
            service, _ = _authenticated_service(FrozenClock(DEFAULT_INSTANT))
            fingerprints.append(
                service.create(order, risk, phrase=CONFIRMATION_PHRASE, broker="alpaca").fingerprint
            )
            risk_hashes.append(risk_decision_hash(risk))
    assert len(set(fingerprints)) == 1
    assert len(set(risk_hashes)) == 1

    other_value = Decimal("12345678901234567890123456789.2")
    other_order = _intent().model_copy(update={"quantity": other_value})
    other_risk = _approved().model_copy(update={"approved_quantity": other_value})
    service, _ = _authenticated_service(FrozenClock(DEFAULT_INSTANT))
    other = service.create(other_order, other_risk, phrase=CONFIRMATION_PHRASE, broker="alpaca")
    assert other.fingerprint != fingerprints[0]
    assert risk_decision_hash(other_risk) != risk_hashes[0]


def test_huge_exponent_is_rejected_without_expanding_fingerprint_payload() -> None:
    """Resource-hostile Decimal exponents fail at the bounded canonical boundary."""
    huge = Decimal((0, (1,), 100_000))
    order = _intent().model_copy(update={"quantity": huge})
    risk = _approved().model_copy(update={"approved_quantity": huge})
    service, _ = _authenticated_service(FrozenClock(DEFAULT_INSTANT))
    with pytest.raises(ApprovalError, match="numeric"):
        service.create(order, risk, phrase=CONFIRMATION_PHRASE, broker="alpaca")


def test_issuance_persistence_exception_has_no_secret_cause_or_context() -> None:
    """A failing local store cannot attach secret-bearing implementation exceptions."""
    service, _ = _authenticated_service(FrozenClock(DEFAULT_INSTANT))
    with (
        patch.object(
            AuditLog,
            "record_many_if_heads",
            side_effect=RuntimeError("api_key=secret-token-123"),
        ),
        pytest.raises(ApprovalError) as captured,
    ):
        service.create(_intent(), _approved(), phrase=CONFIRMATION_PHRASE, broker="alpaca")
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "secret-token-123" not in repr(captured.value)
