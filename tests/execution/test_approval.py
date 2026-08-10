from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from market_sentinel.domain.clock import FrozenClock
from market_sentinel.domain.enums import OrderType, Side
from market_sentinel.domain.models import OrderIntent, RiskDecision
from market_sentinel.execution.approval import (
    CONFIRMATION_PHRASE,
    ApprovalError,
    ApprovalService,
)
from tests.factories import DEFAULT_INSTANT, intent, risk_decision


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
    service = ApprovalService(clock=clock)
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
    service = ApprovalService(clock=FrozenClock(DEFAULT_INSTANT))
    original = _intent()
    risk = _approved()
    confirmation = service.create(original, risk, phrase=CONFIRMATION_PHRASE, broker="alpaca")

    with pytest.raises(ApprovalError, match="fingerprint"):
        service.verify(original.model_copy(update=change), risk, confirmation, broker="alpaca")


def test_confirmation_binds_broker_and_complete_risk_decision() -> None:
    """A broker change or a freshly forged risk decision must require new confirmation."""
    service = ApprovalService(clock=FrozenClock(DEFAULT_INSTANT))
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
    service = ApprovalService(clock=FrozenClock(DEFAULT_INSTANT))
    with pytest.raises(ApprovalError, match="phrase"):
        service.create(_intent(), _approved(), phrase=f"{CONFIRMATION_PHRASE} ", broker="alpaca")

    confirmation = service.create(
        _intent(), _approved(), phrase=CONFIRMATION_PHRASE, broker="alpaca"
    )
    assert CONFIRMATION_PHRASE not in repr(confirmation)


def test_confirmation_uses_canonical_decimal_and_utc_encodings() -> None:
    """Equivalent Decimal scales and timezone offsets must produce the same exact meaning."""
    service = ApprovalService(clock=FrozenClock(DEFAULT_INSTANT))
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
    service = ApprovalService(clock=FrozenClock(DEFAULT_INSTANT))
    original = _intent()
    risk = _approved()
    confirmation = service.create(original, risk, phrase=CONFIRMATION_PHRASE, broker="alpaca")
    malformed = original.model_copy(update={"quantity": bad})

    with pytest.raises(ApprovalError, match="malformed"):
        service.verify(malformed, risk, confirmation, broker="alpaca")


def test_confirmation_rejects_future_and_naive_capability_times() -> None:
    """Clock rollback and context-dependent naive timestamps must fail closed."""
    service = ApprovalService(clock=FrozenClock(DEFAULT_INSTANT))
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
    service = ApprovalService(clock=FrozenClock(DEFAULT_INSTANT))
    original = _intent()
    risk = _approved()
    confirmation = service.create(original, risk, phrase=CONFIRMATION_PHRASE, broker="alpaca")
    forged = original.model_copy(update={"quantity": Decimal("-0.1")})

    with pytest.raises(ApprovalError, match="malformed"):
        service.verify(forged, risk, confirmation, broker="alpaca")


def test_confirmation_rejects_a_risk_decision_with_an_overlong_freshness_window() -> None:
    """A forged far-future expiry must not extend RiskEngine's one-minute approval window."""
    service = ApprovalService(clock=FrozenClock(DEFAULT_INSTANT))
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
    service = ApprovalService(clock=FrozenClock(DEFAULT_INSTANT))
    with pytest.raises(ApprovalError, match="intent"):
        service.create(forged, _approved(), phrase=CONFIRMATION_PHRASE, broker="alpaca")
