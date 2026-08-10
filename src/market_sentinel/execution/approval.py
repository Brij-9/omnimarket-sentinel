"""Exact, short-lived capabilities for locally confirmed live order intents."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from market_sentinel.domain.clock import Clock
from market_sentinel.domain.enums import OrderType, Side
from market_sentinel.domain.models import OrderIntent, RiskDecision

CONFIRMATION_PHRASE = "I_CONFIRM_REAL_MONEY_ORDER"
DEFAULT_CONFIRMATION_LIFETIME = timedelta(minutes=5)
_FINGERPRINT_DOMAIN = b"omnimarket-sentinel:live-order-confirmation:v1\x00"
_RISK_DOMAIN = b"omnimarket-sentinel:risk-decision:v1\x00"
_BROKER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class ApprovalError(ValueError):
    """A safe reason for rejecting an absent, changed, stale, or malformed capability."""


@dataclass(frozen=True, slots=True)
class OrderConfirmation:
    """Secret-free evidence that one exact live intent was explicitly confirmed."""

    confirmation_id: str
    fingerprint: str
    broker: str
    created_at: datetime
    expires_at: datetime


class ApprovalService:
    """Create and verify exact, expiring confirmation fingerprints."""

    def __init__(self, *, clock: Clock) -> None:
        self._clock = clock

    def create(
        self,
        intent: OrderIntent,
        risk_decision: RiskDecision,
        *,
        phrase: str,
        broker: str | None = None,
        lifetime: timedelta = DEFAULT_CONFIRMATION_LIFETIME,
    ) -> OrderConfirmation:
        """Create a code-only confirmation after exact phrase and risk validation."""
        if type(phrase) is not str or phrase != CONFIRMATION_PHRASE:
            raise ApprovalError("confirmation phrase is not exact")
        instant = _aware_utc(self._clock.now())
        resolved_broker = _resolve_broker(intent, broker)
        if (
            type(lifetime) is not timedelta
            or not timedelta(0) < lifetime <= DEFAULT_CONFIRMATION_LIFETIME
        ):
            raise ApprovalError("confirmation lifetime is malformed")
        _validate_intent(intent, instant)
        _validate_risk(intent, risk_decision, instant)
        expires_at = instant + lifetime
        fingerprint = _fingerprint(intent, risk_decision, resolved_broker, expires_at)
        return OrderConfirmation(
            confirmation_id=fingerprint,
            fingerprint=fingerprint,
            broker=resolved_broker,
            created_at=instant,
            expires_at=expires_at,
        )

    def verify(
        self,
        intent: OrderIntent,
        risk_decision: RiskDecision,
        confirmation: OrderConfirmation,
        *,
        broker: str | None = None,
    ) -> None:
        """Reject every changed, future, expired, malformed, or unapproved capability."""
        if type(confirmation) is not OrderConfirmation:
            raise ApprovalError("confirmation is malformed")
        try:
            instant = _aware_utc(self._clock.now())
            created_at = _aware_utc(confirmation.created_at)
            expires_at = _aware_utc(confirmation.expires_at)
        except (TypeError, ValueError):
            raise ApprovalError("confirmation is malformed") from None
        if created_at > instant:
            raise ApprovalError("confirmation is from the future")
        if expires_at <= instant:
            raise ApprovalError("confirmation expired")
        if expires_at <= created_at or expires_at - created_at > DEFAULT_CONFIRMATION_LIFETIME:
            raise ApprovalError("confirmation is malformed")
        try:
            resolved_broker = _resolve_broker(intent, broker)
            _validate_intent(intent, instant)
            expected = _fingerprint(intent, risk_decision, resolved_broker, expires_at)
        except (TypeError, ValueError, ApprovalError):
            raise ApprovalError("confirmation input is malformed") from None
        if (
            confirmation.broker != resolved_broker
            or not _sha256_text(confirmation.confirmation_id)
            or not _sha256_text(confirmation.fingerprint)
            or confirmation.confirmation_id != confirmation.fingerprint
            or confirmation.fingerprint != expected
        ):
            raise ApprovalError("confirmation fingerprint mismatch")
        _validate_risk(intent, risk_decision, instant)


def risk_decision_hash(decision: RiskDecision) -> str:
    """Hash every risk-decision field with domain-separated canonical encodings."""
    if type(decision) is not RiskDecision:
        raise ApprovalError("risk decision is malformed")
    payload = {
        "approved": _exact_bool(decision.approved),
        "approved_notional": _optional_decimal(decision.approved_notional),
        "approved_quantity": _optional_decimal(decision.approved_quantity),
        "decided_at": _time_text(decision.decided_at),
        "expires_at": _time_text(decision.expires_at),
        "portfolio_hash": _text(decision.portfolio_hash),
        "reason_codes": [_text(item) for item in _exact_tuple(decision.reason_codes)],
    }
    return _domain_hash(_RISK_DOMAIN, payload)


def _fingerprint(
    intent: OrderIntent,
    risk_decision: RiskDecision,
    broker: str,
    confirmation_expires_at: datetime,
) -> str:
    _validate_intent(intent)
    payload = {
        "broker": _broker_text(broker),
        "confirmation_expires_at": _time_text(confirmation_expires_at),
        "intent": {
            "created_at": _time_text(intent.created_at),
            "expires_at": _time_text(intent.expires_at),
            "instrument_id": _text(intent.instrument_id),
            "intent_id": _text(intent.intent_id),
            "limit_price": _optional_decimal(intent.limit_price),
            "notional": _optional_decimal(intent.notional),
            "order_type": _enum_text(intent.order_type, OrderType),
            "product": _text(intent.product),
            "quantity": _optional_decimal(intent.quantity),
            "session": _text(intent.session),
            "side": _enum_text(intent.side, Side),
            "snapshot_hash": _text(intent.snapshot_hash),
            "stop_loss": _optional_decimal(intent.stop_loss),
            "take_profit": _optional_decimal(intent.take_profit),
            "time_in_force": _text(intent.time_in_force),
            "trigger_price": _optional_decimal(intent.trigger_price),
        },
        "risk_decision_hash": risk_decision_hash(risk_decision),
    }
    return _domain_hash(_FINGERPRINT_DOMAIN, payload)


def _validate_risk(intent: OrderIntent, decision: RiskDecision, now: datetime) -> None:
    if type(intent) is not OrderIntent or type(decision) is not RiskDecision:
        raise ApprovalError("risk decision is malformed")
    try:
        decided_at = _aware_utc(decision.decided_at)
        expires_at = _aware_utc(decision.expires_at)
        approved = _exact_bool(decision.approved)
        reasons = _exact_tuple(decision.reason_codes)
        quantity = _positive_optional_decimal(decision.approved_quantity)
        notional = _positive_optional_decimal(decision.approved_notional)
        portfolio_hash = _text(decision.portfolio_hash)
    except (TypeError, ValueError, ApprovalError):
        raise ApprovalError("risk decision is malformed") from None
    if decided_at > now:
        raise ApprovalError("risk decision is from the future")
    if expires_at <= now:
        raise ApprovalError("risk decision expired")
    if expires_at <= decided_at or expires_at - decided_at > timedelta(seconds=60):
        raise ApprovalError("risk decision freshness window is malformed")
    if not approved or reasons:
        raise ApprovalError("risk decision is not approved")
    if quantity is None or notional is None:
        raise ApprovalError("risk decision is not exact")
    if portfolio_hash != intent.snapshot_hash:
        raise ApprovalError("risk snapshot mismatch")
    if intent.quantity is not None and quantity != _decimal(intent.quantity):
        raise ApprovalError("risk quantity mismatch")
    if intent.notional is not None and notional != _decimal(intent.notional):
        raise ApprovalError("risk notional mismatch")


def _validate_intent(intent: object, now: datetime | None = None) -> None:
    if type(intent) is not OrderIntent:
        raise ApprovalError("order intent is malformed")
    try:
        _text(intent.intent_id)
        _text(intent.instrument_id)
        _enum_text(intent.side, Side)
        _enum_text(intent.order_type, OrderType)
        _text(intent.time_in_force)
        _text(intent.product)
        _text(intent.session)
        _text(intent.snapshot_hash)
        created_at = _aware_utc(intent.created_at)
        expires_at = _aware_utc(intent.expires_at)
        quantity = _positive_optional_decimal(intent.quantity)
        notional = _positive_optional_decimal(intent.notional)
        for value in (
            intent.limit_price,
            intent.trigger_price,
            intent.stop_loss,
            intent.take_profit,
        ):
            _positive_optional_decimal(value)
    except (TypeError, ValueError, ApprovalError):
        raise ApprovalError("order intent is malformed") from None
    expected_price_fields = {
        OrderType.MARKET: (False, False),
        OrderType.LIMIT: (True, False),
        OrderType.STOP: (False, True),
        OrderType.STOP_LIMIT: (True, True),
    }
    expected_limit, expected_trigger = expected_price_fields[intent.order_type]
    if (
        (quantity is None) == (notional is None)
        or expires_at <= created_at
        or (intent.limit_price is not None) is not expected_limit
        or (intent.trigger_price is not None) is not expected_trigger
        or (now is not None and (created_at > now or expires_at <= now))
    ):
        raise ApprovalError("order intent is malformed")
    if intent.order_type is OrderType.STOP_LIMIT:
        assert intent.limit_price is not None and intent.trigger_price is not None
        if (intent.side is Side.BUY and intent.limit_price < intent.trigger_price) or (
            intent.side is Side.SELL and intent.limit_price > intent.trigger_price
        ):
            raise ApprovalError("order intent is malformed")
    if (
        intent.stop_loss is not None
        and intent.take_profit is not None
        and (
            (intent.side is Side.BUY and intent.stop_loss >= intent.take_profit)
            or (intent.side is Side.SELL and intent.stop_loss <= intent.take_profit)
        )
    ):
        raise ApprovalError("order intent is malformed")


def _resolve_broker(intent: OrderIntent, broker: str | None) -> str:
    if type(intent) is not OrderIntent:
        raise ApprovalError("order intent is malformed")
    value = broker
    if value is None:
        parts = intent.instrument_id.rsplit("@", 1)
        if len(parts) != 2:
            raise ApprovalError("broker is required")
        value = parts[1]
    return _broker_text(value)


def _broker_text(value: object) -> str:
    if type(value) is not str or _BROKER.fullmatch(value) is None:
        raise ApprovalError("broker is malformed")
    return value


def _text(value: object) -> str:
    if type(value) is not str or not value:
        raise ApprovalError("text field is malformed")
    return value


def _exact_bool(value: object) -> bool:
    if type(value) is not bool:
        raise ApprovalError("boolean field is malformed")
    return value


def _exact_tuple(value: object) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ApprovalError("tuple field is malformed")
    return value


def _enum_text(value: object, enum_type: type[OrderType] | type[Side]) -> str:
    if type(value) is not enum_type:
        raise ApprovalError("enum field is malformed")
    return value.value


def _optional_decimal(value: object) -> str | None:
    return None if value is None else _decimal_text(_decimal(value))


def _positive_optional_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    result = _decimal(value)
    if result <= 0:
        raise ApprovalError("numeric field is malformed")
    return result


def _decimal(value: object) -> Decimal:
    if type(value) is not Decimal or not value.is_finite():
        raise ApprovalError("numeric field is malformed")
    return value


def _decimal_text(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    return format(value.normalize(), "f")


def _aware_utc(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ApprovalError("timestamp is malformed")
    return value.astimezone(UTC)


def _time_text(value: object) -> str:
    return _aware_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _domain_hash(domain: bytes, payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(domain + encoded.encode("utf-8")).hexdigest()


def _sha256_text(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)
