"""Authenticated durable repository for live-safety authority events."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Final
from uuid import uuid4

from market_sentinel.operations.audit import AuditEvent, AuditLog
from market_sentinel.security import redact_mapping
from market_sentinel.storage.events import EventRecord

_MAC_DOMAIN: Final = b"omnimarket-sentinel:safety-event-mac:v1\x00"
_MAC_FIELD: Final = "safety_mac"
_VERSION_FIELD: Final = "safety_version"
_VERSION: Final = 1


_CONFIRMATION_PHRASE: Final = "I_CONFIRM_REAL_MONEY_ORDER"
_KILL_ACK: Final = "I_ACKNOWLEDGE_HEALTHY_RECONCILIATION"
_RECONCILIATION: Final = "live-reconciliation"
_KILL_SWITCH: Final = "live-kill-switch"
_INTERLOCK: Final = "live-submission-interlock"
_RECONCILIATION_REASONS: Final = frozenset(
    {
        "PROVIDER_UNAVAILABLE",
        "PROVIDER_DATA_INVALID",
        "PROVIDER_DATA_FUTURE",
        "PROVIDER_DATA_STALE",
        "CURRENCY_MISMATCH",
        "CASH_MISMATCH",
        "POSITION_UNKNOWN",
        "POSITION_MISSING",
        "POSITION_SIDE_MISMATCH",
        "POSITION_QUANTITY_MISMATCH",
        "ORDER_UNKNOWN",
        "ORDER_MISSING",
        "ORDER_INSTRUMENT_MISMATCH",
        "ORDER_SIDE_MISMATCH",
        "ORDER_QUANTITY_MISMATCH",
        "ORDER_FILL_MISMATCH",
        "ORDER_STATUS_MISMATCH",
    }
)
_ACKNOWLEDGED_STATUSES: Final = frozenset({"acknowledged", "partially_filled", "filled"})


class SafetyIntegrityError(RuntimeError):
    """Persisted live-safety state is unsigned, malformed, or authenticated by another key."""


class SafetyAuthenticator:
    """Injected local HMAC and cryptographic nonce capability; key material is never rendered."""

    __slots__ = ("_nonce_source", "_signer")
    _nonce_source: Callable[[], bytes]
    _signer: Callable[[bytes], str]

    def __init__(self, *, key: bytes, nonce_source: Callable[[], bytes]) -> None:
        if type(key) is not bytes or not 32 <= len(key) <= 64:
            raise ValueError("safety authentication key is malformed")
        if not callable(nonce_source):
            raise ValueError("safety nonce source is malformed")
        secret = bytes(key)
        object.__setattr__(
            self,
            "_signer",
            lambda message: hmac.new(secret, message, hashlib.sha256).hexdigest(),
        )
        object.__setattr__(self, "_nonce_source", nonce_source)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("safety authenticator is immutable")

    def __repr__(self) -> str:
        return "SafetyAuthenticator(configured=True)"

    def new_nonce(self) -> str:
        """Return a bounded opaque nonce from the injected cryptographic source."""
        try:
            value = self._nonce_source()
        except BaseException:
            value = None
        if type(value) is not bytes or not 16 <= len(value) <= 64:
            raise ValueError("safety nonce source returned malformed data")
        return value.hex()

    def sign(
        self,
        *,
        event_id: str,
        kind: str,
        aggregate_id: str,
        occurred_at: datetime,
        payload: Mapping[str, object],
    ) -> str:
        """Authenticate one sequence-independent exact event identity and payload."""
        message = _mac_message(event_id, kind, aggregate_id, occurred_at, payload)
        return self._signer(message)

    def verify(self, event: EventRecord) -> bool:
        """Verify an exact persisted event without exposing authentication material."""
        try:
            payload = dict(event.payload)
            mac = payload.pop(_MAC_FIELD)
            version = payload.pop(_VERSION_FIELD)
            if type(mac) is not str or version != _VERSION:
                return False
            expected = self.sign(
                event_id=event.event_id,
                kind=event.kind,
                aggregate_id=event.aggregate_id,
                occurred_at=event.occurred_at,
                payload=payload,
            )
            return hmac.compare_digest(mac, expected)
        except (KeyError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class _SafetyEvent:
    """One typed unsigned request; SafetyRepository alone adds authentication."""

    event_id: str
    kind: str
    aggregate_id: str
    payload: Mapping[str, object]
    occurred_at: datetime


class SafetyRepository:
    """Read-only authenticated root which grants narrowly typed role capabilities."""

    __slots__ = ("_audit", "_authenticator", "_store_identity")

    def __init__(self, *, audit_log: AuditLog, authenticator: SafetyAuthenticator) -> None:
        if type(audit_log) is not AuditLog or type(authenticator) is not SafetyAuthenticator:
            raise ValueError("safety repository requires exact durable authenticated capabilities")
        self._audit = audit_log
        self._authenticator = authenticator
        self._store_identity = object()

    @property
    def event_store_identity(self) -> object:
        """Opaque identity used to require one shared durable transaction boundary."""
        return self._store_identity

    def approval_capability(self) -> ApprovalSafetyCapability:
        return ApprovalSafetyCapability._from_repository(self)

    def reconciliation_capability(self) -> ReconciliationSafetyCapability:
        return ReconciliationSafetyCapability._from_repository(self)

    def live_capability(self) -> LiveSafetyCapability:
        return LiveSafetyCapability._from_repository(self)

    def _record_many(self, batch: tuple[_SafetyEvent, ...]) -> None:
        self._audit.record_many(self._signed_batch(batch))

    def _record_many_if_heads(
        self,
        batch: tuple[_SafetyEvent, ...],
        expected_heads: Mapping[str, str | None],
    ) -> None:
        """Sign then conditionally persist against exact durable aggregate heads."""
        self._audit.record_many_if_heads(self._signed_batch(batch), expected_heads)

    def stream_verified(self, aggregate_id: str) -> tuple[EventRecord, ...]:
        """Return an aggregate only when every row is valid under the injected key."""
        if type(aggregate_id) is not str or not aggregate_id:
            raise SafetyIntegrityError("safety aggregate identity is malformed")
        try:
            rows = tuple(self._audit.event_store.stream(aggregate_id))
        except BaseException:
            rows = ()
            failed = True
        else:
            failed = False
        if failed or any(not self._authenticator.verify(row) for row in rows):
            raise SafetyIntegrityError("persisted safety state failed authentication")
        return rows

    def _signed_batch(self, batch: tuple[_SafetyEvent, ...]) -> tuple[AuditEvent, ...]:
        if type(batch) is not tuple or not batch:
            raise ValueError("safety batch must be a nonempty tuple")
        signed: list[AuditEvent] = []
        for event in batch:
            if type(event) is not _SafetyEvent:
                raise ValueError("safety batch contains a malformed event")
            if (
                type(event.event_id) is not str
                or not event.event_id
                or type(event.kind) is not str
                or not event.kind
                or type(event.aggregate_id) is not str
                or not event.aggregate_id
            ):
                raise ValueError("safety event identity is malformed")
            occurred_at = _aware_utc(event.occurred_at)
            payload = dict(redact_mapping(event.payload))
            if _MAC_FIELD in payload or _VERSION_FIELD in payload:
                raise ValueError("reserved safety authentication field")
            mac = self._authenticator.sign(
                event_id=event.event_id,
                kind=event.kind,
                aggregate_id=event.aggregate_id,
                occurred_at=occurred_at,
                payload=payload,
            )
            payload[_VERSION_FIELD] = _VERSION
            payload[_MAC_FIELD] = mac
            signed.append(
                AuditEvent(
                    event.event_id,
                    event.kind,
                    event.aggregate_id,
                    MappingProxyType(payload),
                    occurred_at,
                )
            )
        return tuple(signed)


class _Capability:
    __slots__ = ("_repository",)

    def __init__(self, repository: SafetyRepository, token: object) -> None:
        if token is not _CAPABILITY_TOKEN or type(repository) is not SafetyRepository:
            raise ValueError("safety capability cannot be constructed directly")
        self._repository = repository

    @property
    def store_identity(self) -> object:
        return self._repository.event_store_identity


_CAPABILITY_TOKEN = object()


class ApprovalSafetyCapability(_Capability):
    """Only the approval role can persist and read confirmation issuance state."""

    @classmethod
    def _from_repository(cls, repository: SafetyRepository) -> ApprovalSafetyCapability:
        return cls(repository, _CAPABILITY_TOKEN)

    def issue_confirmation(
        self,
        *,
        phrase: str,
        broker: str,
        created_at: datetime,
        expires_at: datetime,
        fingerprint: str,
        risk_decision_hash: str,
    ) -> tuple[str, str, str]:
        if phrase != _CONFIRMATION_PHRASE or type(phrase) is not str:
            raise ValueError("confirmation phrase is not exact")
        _exact_text(broker)
        created = _aware_utc(created_at)
        expires = _aware_utc(expires_at)
        if expires <= created or expires - created > timedelta(minutes=5):
            raise ValueError("confirmation lifetime is malformed")
        _sha256(fingerprint)
        _sha256(risk_decision_hash)
        seed = self._repository._authenticator.new_nonce()
        nonce = self._repository._authenticator.new_nonce()
        confirmation_id = hashlib.sha256(
            b"omnimarket-sentinel:confirmation-issuance-id:v1\x00" + bytes.fromhex(seed)
        ).hexdigest()
        aggregate = f"live-confirmation:{confirmation_id}"
        event_id = f"confirmation-issued-{confirmation_id}"
        self._repository._record_many_if_heads(
            (
                _SafetyEvent(
                    event_id,
                    "confirmation.issued",
                    aggregate,
                    {
                        "broker": broker,
                        "created_at": _time_text(created),
                        "expires_at": _time_text(expires),
                        "fingerprint": fingerprint,
                        "nonce": nonce,
                        "risk_decision_hash": risk_decision_hash,
                    },
                    created,
                ),
            ),
            {aggregate: None},
        )
        rows = self._repository.stream_verified(aggregate)
        if len(rows) != 1 or rows[0].event_id != event_id:
            raise RuntimeError("confirmation persistence failed")
        mac = rows[0].payload.get(_MAC_FIELD)
        if type(mac) is not str:
            raise RuntimeError("confirmation persistence failed")
        return confirmation_id, nonce, mac

    def confirmation_events(self, confirmation_id: str) -> tuple[EventRecord, ...]:
        _sha256(confirmation_id)
        return self._repository.stream_verified(f"live-confirmation:{confirmation_id}")


class ReconciliationSafetyCapability(_Capability):
    """Persist only derived reconciliation reports and prerequisite-checked clears."""

    @classmethod
    def _from_repository(cls, repository: SafetyRepository) -> ReconciliationSafetyCapability:
        return cls(repository, _CAPABILITY_TOKEN)

    def reconciliation_events(self) -> tuple[EventRecord, ...]:
        return self._repository.stream_verified(_RECONCILIATION)

    def kill_switch_events(self) -> tuple[EventRecord, ...]:
        return self._repository.stream_verified(_KILL_SWITCH)

    def interlock_events(self) -> tuple[EventRecord, ...]:
        return self._repository.stream_verified(_INTERLOCK)

    def persist_report(
        self,
        *,
        broker: str,
        broker_hash: str,
        ledger_hash: str,
        reason_codes: tuple[str, ...],
        checked_at: datetime,
    ) -> EventRecord:
        _exact_text(broker)
        _sha256(broker_hash)
        _sha256(ledger_hash)
        if type(reason_codes) is not tuple or any(
            type(x) is not str or not x for x in reason_codes
        ):
            raise ValueError("reconciliation reasons are malformed")
        if (
            len(set(reason_codes)) != len(reason_codes)
            or not set(reason_codes) <= _RECONCILIATION_REASONS
        ):
            raise ValueError("reconciliation reasons are malformed")
        instant = _aware_utc(checked_at)
        healthy = not reason_codes
        report_id = f"reconciliation-{uuid4().hex}"
        report = _SafetyEvent(
            report_id,
            "reconciliation.healthy" if healthy else "reconciliation.unhealthy",
            _RECONCILIATION,
            {
                "broker": broker,
                "broker_hash": broker_hash,
                "healthy": healthy,
                "ledger_hash": ledger_hash,
                "reason_codes": list(reason_codes),
            },
            instant,
        )
        batch: tuple[_SafetyEvent, ...] = (report,)
        if not healthy:
            batch += (
                _SafetyEvent(
                    f"kill-{uuid4().hex}",
                    "kill_switch.activated",
                    _KILL_SWITCH,
                    {"reason_codes": list(reason_codes)},
                    instant,
                ),
            )
        self._repository._record_many(batch)
        matching = [row for row in self.reconciliation_events() if row.event_id == report_id]
        if len(matching) != 1:
            raise RuntimeError("reconciliation persistence could not be verified")
        return matching[0]

    def clear_kill_switch(self, *, acknowledgement: str, now: datetime) -> None:
        if type(acknowledgement) is not str or acknowledgement != _KILL_ACK:
            raise ValueError("kill-switch acknowledgement is not exact")
        instant = _aware_utc(now)
        interlocks = self.interlock_events()
        kills = self.kill_switch_events()
        reconciliations = self.reconciliation_events()
        if _interlock_active(interlocks):
            raise ValueError("a live submission remains unresolved")
        if not _kill_active(kills):
            return
        activations = [row for row in kills if row.kind == "kill_switch.activated"]
        if not activations or not reconciliations:
            raise ValueError("a new healthy reconciliation is required")
        activation = activations[-1]
        healthy = reconciliations[-1]
        if (
            healthy.kind != "reconciliation.healthy"
            or healthy.sequence <= activation.sequence
            or healthy.occurred_at > instant
            or instant - healthy.occurred_at > timedelta(seconds=60)
        ):
            raise ValueError("a new healthy reconciliation is required")
        self._repository._record_many_if_heads(
            (
                _SafetyEvent(
                    f"kill-clear-{uuid4().hex}",
                    "kill_switch.cleared",
                    _KILL_SWITCH,
                    {
                        "activation_event_id": activation.event_id,
                        "activation_sequence": activation.sequence,
                    },
                    instant,
                ),
            ),
            {
                _KILL_SWITCH: kills[-1].event_id,
                _RECONCILIATION: healthy.event_id,
                _INTERLOCK: interlocks[-1].event_id if interlocks else None,
            },
        )


class LiveSafetyCapability(_Capability):
    """Persist only exact live claim/start/acknowledgement/unknown transitions."""

    @classmethod
    def _from_repository(cls, repository: SafetyRepository) -> LiveSafetyCapability:
        return cls(repository, _CAPABILITY_TOKEN)

    def claim_and_start(
        self,
        *,
        intent_id: str,
        broker: str,
        confirmation_id: str,
        fingerprint: str,
        expires_at: datetime,
        reconciliation_head: str,
        kill_switch_head: str | None,
        interlock_head: str | None,
        occurred_at: datetime,
    ) -> None:
        _exact_text(intent_id)
        _exact_text(broker)
        _sha256(confirmation_id)
        _sha256(fingerprint)
        _exact_text(reconciliation_head)
        instant = _aware_utc(occurred_at)
        aggregate = f"live-confirmation:{confirmation_id}"
        rows = self._repository.stream_verified(aggregate)
        if len(rows) > 1 and rows[-1].kind == "live.confirmation_consumed":
            raise SafetyAlreadyUsedError("confirmation already used")
        if len(rows) != 1 or rows[0].kind != "confirmation.issued":
            raise SafetyIntegrityError("confirmation issuance is invalid")
        issued = rows[0]
        if (
            issued.payload.get("broker") != broker
            or issued.payload.get("fingerprint") != fingerprint
            or issued.payload.get("expires_at") != _time_text(expires_at)
            or issued.occurred_at > instant
            or _aware_utc(expires_at) <= instant
        ):
            raise SafetyIntegrityError("confirmation issuance is invalid")
        nonce = uuid4().hex
        self._repository._record_many_if_heads(
            (
                _SafetyEvent(
                    f"live-claim-audit-{nonce}",
                    "live.confirmation_claimed",
                    intent_id,
                    {
                        "broker": broker,
                        "confirmation_fingerprint": fingerprint,
                        "expires_at": _time_text(expires_at),
                    },
                    instant,
                ),
                _SafetyEvent(
                    f"live-confirmation-{confirmation_id}",
                    "live.confirmation_consumed",
                    aggregate,
                    {"intent_id": intent_id},
                    instant,
                ),
                _SafetyEvent(
                    f"live-start-{nonce}",
                    "live.submission_started",
                    intent_id,
                    {"broker": broker, "client_intent_id": intent_id},
                    instant,
                ),
                _SafetyEvent(
                    f"interlock-start-{nonce}",
                    "live.interlock_started",
                    _INTERLOCK,
                    {
                        "broker": broker,
                        "intent_id": intent_id,
                        "submission_id": confirmation_id,
                    },
                    instant,
                ),
            ),
            {
                _RECONCILIATION: reconciliation_head,
                _KILL_SWITCH: kill_switch_head,
                _INTERLOCK: interlock_head,
                aggregate: issued.event_id,
            },
        )

    def record_acknowledgement(
        self,
        *,
        intent_id: str,
        broker: str,
        broker_order_id: str,
        status: str,
        submission_id: str,
        occurred_at: datetime,
    ) -> None:
        for value in (intent_id, broker, broker_order_id, status, submission_id):
            _exact_text(value)
        if status not in _ACKNOWLEDGED_STATUSES:
            raise ValueError("live acknowledgement status is invalid")
        instant = _aware_utc(occurred_at)
        interlocks = self._repository.stream_verified(_INTERLOCK)
        if (
            not _interlock_active(interlocks)
            or interlocks[-1].payload.get("submission_id") != submission_id
            or interlocks[-1].payload.get("intent_id") != intent_id
            or interlocks[-1].payload.get("broker") != broker
        ):
            raise ValueError("live acknowledgement transition is invalid")
        nonce = uuid4().hex
        self._repository._record_many_if_heads(
            (
                _SafetyEvent(
                    f"live-ack-{nonce}",
                    "live.acknowledged",
                    intent_id,
                    {
                        "broker": broker,
                        "broker_order_id": broker_order_id,
                        "client_intent_id": intent_id,
                        "status": status,
                    },
                    instant,
                ),
                _SafetyEvent(
                    f"interlock-ack-{nonce}",
                    "live.interlock_resolved",
                    _INTERLOCK,
                    {"resolution": "acknowledged", "submission_id": submission_id},
                    instant,
                ),
            ),
            {_INTERLOCK: interlocks[-1].event_id},
        )

    def record_unknown(self, *, intent_id: str, submission_id: str, occurred_at: datetime) -> None:
        _exact_text(intent_id)
        _exact_text(submission_id)
        instant = _aware_utc(occurred_at)
        interlocks = self._repository.stream_verified(_INTERLOCK)
        kills = self._repository.stream_verified(_KILL_SWITCH)
        if (
            not _interlock_active(interlocks)
            or interlocks[-1].payload.get("submission_id") != submission_id
            or interlocks[-1].payload.get("intent_id") != intent_id
        ):
            raise ValueError("live unknown transition is invalid")
        nonce = uuid4().hex
        self._repository._record_many_if_heads(
            (
                _SafetyEvent(
                    f"live-unknown-{nonce}",
                    "live.unknown",
                    intent_id,
                    {"reason_code": "SUBMISSION_UNKNOWN"},
                    instant,
                ),
                _SafetyEvent(
                    f"interlock-unknown-{nonce}",
                    "live.interlock_resolved",
                    _INTERLOCK,
                    {"resolution": "unknown", "submission_id": submission_id},
                    instant,
                ),
                _SafetyEvent(
                    f"kill-unknown-{nonce}",
                    "kill_switch.activated",
                    _KILL_SWITCH,
                    {"reason_codes": ["SUBMISSION_UNKNOWN"]},
                    instant,
                ),
            ),
            {
                _INTERLOCK: interlocks[-1].event_id,
                _KILL_SWITCH: kills[-1].event_id if kills else None,
            },
        )


class SafetyAlreadyUsedError(RuntimeError):
    """A confirmation aggregate already contains a consumption event."""


def _exact_text(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("safety text is malformed")
    return value


def _sha256(value: object) -> str:
    text = _exact_text(value)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError("safety digest is malformed")
    return text


def _kill_active(rows: tuple[EventRecord, ...]) -> bool:
    latest: EventRecord | None = None
    active = False
    for row in rows:
        if row.kind == "kill_switch.activated":
            reasons = row.payload.get("reason_codes")
            if type(reasons) is not tuple or not reasons:
                return True
            latest, active = row, True
        elif row.kind == "kill_switch.cleared":
            if (
                latest is None
                or row.payload.get("activation_event_id") != latest.event_id
                or row.payload.get("activation_sequence") != latest.sequence
                or latest.sequence >= row.sequence
            ):
                return True
            active = False
        else:
            return True
    return active


def _interlock_active(rows: tuple[EventRecord, ...]) -> bool:
    active = False
    submission: str | None = None
    for row in rows:
        if row.kind == "live.interlock_started":
            if active or type(row.payload.get("submission_id")) is not str:
                return True
            submission_id = row.payload["submission_id"]
            assert type(submission_id) is str
            submission, active = submission_id, True
        elif row.kind == "live.interlock_resolved":
            if not active or row.payload.get("submission_id") != submission:
                return True
            active = False
        else:
            return True
    return active


def _mac_message(
    event_id: object,
    kind: object,
    aggregate_id: object,
    occurred_at: object,
    payload: object,
) -> bytes:
    if not all(type(item) is str and item for item in (event_id, kind, aggregate_id)):
        raise ValueError("safety event identity is malformed")
    if not isinstance(payload, Mapping):
        raise ValueError("safety payload is malformed")
    encoded = json.dumps(
        {
            "aggregate_id": aggregate_id,
            "event_id": event_id,
            "kind": kind,
            "occurred_at": _time_text(occurred_at),
            "payload": dict(payload),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded) > 65_536:
        raise ValueError("safety event exceeds bounded encoding")
    return _MAC_DOMAIN + encoded.encode("utf-8")


def _aware_utc(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("safety timestamp is malformed")
    return value.astimezone(UTC)


def _time_text(value: object) -> str:
    return _aware_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")
