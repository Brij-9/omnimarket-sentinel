"""Authenticated durable repository for live-safety authority events."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Final, NoReturn, SupportsIndex
from uuid import uuid4
from weakref import WeakKeyDictionary, finalize

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


class SafetyStateChangedError(SafetyIntegrityError):
    """Authenticated claim prerequisites are absent, stale, unhealthy, or changed."""


class _SafetyMac:
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
        return "_SafetyMac(configured=True)"

    def new_nonce(self) -> str:
        """Return a bounded opaque nonce from the injected cryptographic source."""
        try:
            value = self._nonce_source()
        except BaseException:
            value = None
        if type(value) is not bytes or not 16 <= len(value) <= 64:
            raise ValueError("safety nonce source returned malformed data")
        return value.hex()

    def _mac(
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
            expected = self._mac(
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
    """One internal typed request; the unexported authority alone adds authentication."""

    event_id: str
    kind: str
    aggregate_id: str
    payload: Mapping[str, object]
    occurred_at: datetime


class _SafetyAuthority:
    """Internal authenticated persistence authority; never returned by the public factory."""

    __slots__ = ("_audit", "_authenticator", "_store_identity")

    def __init__(self, *, audit_log: AuditLog, authenticator: _SafetyMac) -> None:
        if type(audit_log) is not AuditLog or type(authenticator) is not _SafetyMac:
            raise ValueError("safety repository requires exact durable authenticated capabilities")
        self._audit = audit_log
        self._authenticator = authenticator
        self._store_identity = object()

    @property
    def event_store_identity(self) -> object:
        """Opaque identity used to require one shared durable transaction boundary."""
        return self._store_identity

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
            mac = self._authenticator._mac(
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


class _InternalRole:
    __slots__ = ("_repository",)

    def __init__(self, repository: _SafetyAuthority, token: object) -> None:
        if token is not _CAPABILITY_TOKEN or type(repository) is not _SafetyAuthority:
            raise ValueError("safety capability cannot be constructed directly")
        self._repository = repository

    @property
    def store_identity(self) -> object:
        return self._repository.event_store_identity


_CAPABILITY_TOKEN = object()


class _ApprovalRole(_InternalRole):
    """Only the approval role can persist and read confirmation issuance state."""

    @classmethod
    def _from_repository(cls, repository: _SafetyAuthority) -> _ApprovalRole:
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


class _ReconciliationRole(_InternalRole):
    """Persist only derived reconciliation reports and prerequisite-checked clears."""

    @classmethod
    def _from_repository(cls, repository: _SafetyAuthority) -> _ReconciliationRole:
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
        if not _kill_active(kills, reconciliations):
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


class _LiveRole(_InternalRole):
    """Persist only exact live claim/start/acknowledgement/unknown transitions."""

    @classmethod
    def _from_repository(cls, repository: _SafetyAuthority) -> _LiveRole:
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
        reconciliations = self._repository.stream_verified(_RECONCILIATION)
        kills = self._repository.stream_verified(_KILL_SWITCH)
        interlocks = self._repository.stream_verified(_INTERLOCK)
        actual_reconciliation_head = reconciliations[-1].event_id if reconciliations else None
        actual_kill_head = kills[-1].event_id if kills else None
        actual_interlock_head = interlocks[-1].event_id if interlocks else None
        if (
            actual_reconciliation_head != reconciliation_head
            or actual_kill_head != kill_switch_head
            or actual_interlock_head != interlock_head
            or not reconciliations
        ):
            raise SafetyStateChangedError("live safety heads are not exact")
        reconciliation = reconciliations[-1]
        if (
            reconciliation.kind != "reconciliation.healthy"
            or reconciliation.payload.get("healthy") is not True
            or reconciliation.payload.get("reason_codes") != ()
            or reconciliation.payload.get("broker") != broker
            or reconciliation.occurred_at > instant
            or instant - reconciliation.occurred_at > timedelta(seconds=60)
            or _kill_active(kills, reconciliations)
            or _interlock_active(interlocks)
        ):
            raise SafetyStateChangedError("live safety state is not healthy")
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


_CALLBACK_HANDLE_TOKEN = object()
_APPROVAL_ROLE_VAULT: dict[str, _ApprovalRole] = {}
_RECONCILIATION_ROLE_VAULT: dict[str, _ReconciliationRole] = {}
_LIVE_ROLE_VAULT: dict[str, _LiveRole] = {}
_ROLE_VAULT_REFS: dict[str, int] = {}


def _release_role_vault(authority_id: str) -> None:
    remaining = _ROLE_VAULT_REFS.get(authority_id, 0) - 1
    if remaining > 0:
        _ROLE_VAULT_REFS[authority_id] = remaining
        return
    _ROLE_VAULT_REFS.pop(authority_id, None)
    _APPROVAL_ROLE_VAULT.pop(authority_id, None)
    _RECONCILIATION_ROLE_VAULT.pop(authority_id, None)
    _LIVE_ROLE_VAULT.pop(authority_id, None)


class _CallbackApprovalSafetyCapability:
    """Frozen approval-only handle containing fixed-purpose closure entry points."""

    __slots__ = ("_approval_issue", "_approval_read", "_approval_store_identity", "__weakref__")
    _approval_issue: Callable[..., tuple[str, str, str]]
    _approval_read: Callable[[str], tuple[EventRecord, ...]]
    _approval_store_identity: object

    def __init__(
        self,
        *,
        token: object,
        issue: Callable[..., tuple[str, str, str]],
        read: Callable[[str], tuple[EventRecord, ...]],
        store_identity: object,
    ) -> None:
        if token is not _CALLBACK_HANDLE_TOKEN or not callable(issue) or not callable(read):
            raise ValueError("approval safety handle cannot be constructed directly")
        object.__setattr__(self, "_approval_issue", issue)
        object.__setattr__(self, "_approval_read", read)
        object.__setattr__(self, "_approval_store_identity", store_identity)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("approval safety handle is immutable")

    @property
    def store_identity(self) -> object:
        return self._approval_store_identity

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
        return self._approval_issue(
            phrase=phrase,
            broker=broker,
            created_at=created_at,
            expires_at=expires_at,
            fingerprint=fingerprint,
            risk_decision_hash=risk_decision_hash,
        )

    def confirmation_events(self, confirmation_id: str) -> tuple[EventRecord, ...]:
        return self._approval_read(confirmation_id)


class _CallbackReconciliationSafetyCapability:
    """Frozen reconciliation-only handle with no generic signing or live route."""

    __slots__ = (
        "_reconciliation_clear",
        "_reconciliation_interlocks",
        "_reconciliation_kills",
        "_reconciliation_persist",
        "_reconciliation_read",
        "_reconciliation_store_identity",
        "__weakref__",
    )
    _reconciliation_clear: Callable[..., None]
    _reconciliation_interlocks: Callable[[], tuple[EventRecord, ...]]
    _reconciliation_kills: Callable[[], tuple[EventRecord, ...]]
    _reconciliation_persist: Callable[..., EventRecord]
    _reconciliation_read: Callable[[], tuple[EventRecord, ...]]
    _reconciliation_store_identity: object

    def __init__(
        self,
        *,
        token: object,
        persist: Callable[..., EventRecord],
        clear: Callable[..., None],
        read: Callable[[], tuple[EventRecord, ...]],
        kills: Callable[[], tuple[EventRecord, ...]],
        interlocks: Callable[[], tuple[EventRecord, ...]],
        store_identity: object,
    ) -> None:
        if token is not _CALLBACK_HANDLE_TOKEN or not all(
            callable(item) for item in (persist, clear, read, kills, interlocks)
        ):
            raise ValueError("reconciliation safety handle cannot be constructed directly")
        object.__setattr__(self, "_reconciliation_persist", persist)
        object.__setattr__(self, "_reconciliation_clear", clear)
        object.__setattr__(self, "_reconciliation_read", read)
        object.__setattr__(self, "_reconciliation_kills", kills)
        object.__setattr__(self, "_reconciliation_interlocks", interlocks)
        object.__setattr__(self, "_reconciliation_store_identity", store_identity)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("reconciliation safety handle is immutable")

    @property
    def store_identity(self) -> object:
        return self._reconciliation_store_identity

    def persist_report(
        self,
        *,
        broker: str,
        broker_hash: str,
        ledger_hash: str,
        reason_codes: tuple[str, ...],
        checked_at: datetime,
    ) -> EventRecord:
        return self._reconciliation_persist(
            broker=broker,
            broker_hash=broker_hash,
            ledger_hash=ledger_hash,
            reason_codes=reason_codes,
            checked_at=checked_at,
        )

    def clear_kill_switch(self, *, acknowledgement: str, now: datetime) -> None:
        self._reconciliation_clear(acknowledgement=acknowledgement, now=now)

    def reconciliation_events(self) -> tuple[EventRecord, ...]:
        return self._reconciliation_read()

    def kill_switch_events(self) -> tuple[EventRecord, ...]:
        return self._reconciliation_kills()

    def interlock_events(self) -> tuple[EventRecord, ...]:
        return self._reconciliation_interlocks()


class _CallbackLiveSafetyCapability:
    """Frozen live-only handle with fixed claim, acknowledgement, and UNKNOWN routes."""

    __slots__ = (
        "_live_acknowledge",
        "_live_claim",
        "_live_store_identity",
        "_live_unknown",
        "__weakref__",
    )
    _live_acknowledge: Callable[..., None]
    _live_claim: Callable[..., None]
    _live_store_identity: object
    _live_unknown: Callable[..., None]

    def __init__(
        self,
        *,
        token: object,
        claim: Callable[..., None],
        acknowledge: Callable[..., None],
        unknown: Callable[..., None],
        store_identity: object,
    ) -> None:
        if token is not _CALLBACK_HANDLE_TOKEN or not all(
            callable(item) for item in (claim, acknowledge, unknown)
        ):
            raise ValueError("live safety handle cannot be constructed directly")
        object.__setattr__(self, "_live_claim", claim)
        object.__setattr__(self, "_live_acknowledge", acknowledge)
        object.__setattr__(self, "_live_unknown", unknown)
        object.__setattr__(self, "_live_store_identity", store_identity)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("live safety handle is immutable")

    @property
    def store_identity(self) -> object:
        return self._live_store_identity

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
        self._live_claim(
            intent_id=intent_id,
            broker=broker,
            confirmation_id=confirmation_id,
            fingerprint=fingerprint,
            expires_at=expires_at,
            reconciliation_head=reconciliation_head,
            kill_switch_head=kill_switch_head,
            interlock_head=interlock_head,
            occurred_at=occurred_at,
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
        self._live_acknowledge(
            intent_id=intent_id,
            broker=broker,
            broker_order_id=broker_order_id,
            status=status,
            submission_id=submission_id,
            occurred_at=occurred_at,
        )

    def record_unknown(self, *, intent_id: str, submission_id: str, occurred_at: datetime) -> None:
        self._live_unknown(
            intent_id=intent_id,
            submission_id=submission_id,
            occurred_at=occurred_at,
        )


def _create_callback_safety_capabilities(
    *,
    audit_log: AuditLog,
    key: bytes,
    nonce_source: Callable[[], bytes],
) -> tuple[
    _CallbackApprovalSafetyCapability,
    _CallbackReconciliationSafetyCapability,
    _CallbackLiveSafetyCapability,
]:
    """Consume local key material once and return only three narrow immutable handles."""
    authority = _SafetyAuthority(
        audit_log=audit_log,
        authenticator=_SafetyMac(key=key, nonce_source=nonce_source),
    )
    approval_role = _ApprovalRole._from_repository(authority)
    reconciliation_role = _ReconciliationRole._from_repository(authority)
    live_role = _LiveRole._from_repository(authority)
    authority_id = uuid4().hex
    _APPROVAL_ROLE_VAULT[authority_id] = approval_role
    _RECONCILIATION_ROLE_VAULT[authority_id] = reconciliation_role
    _LIVE_ROLE_VAULT[authority_id] = live_role
    _ROLE_VAULT_REFS[authority_id] = 3

    def issue_confirmation(
        *,
        phrase: str,
        broker: str,
        created_at: datetime,
        expires_at: datetime,
        fingerprint: str,
        risk_decision_hash: str,
    ) -> tuple[str, str, str]:
        return _APPROVAL_ROLE_VAULT[authority_id].issue_confirmation(
            phrase=phrase,
            broker=broker,
            created_at=created_at,
            expires_at=expires_at,
            fingerprint=fingerprint,
            risk_decision_hash=risk_decision_hash,
        )

    def confirmation_events(confirmation_id: str) -> tuple[EventRecord, ...]:
        return _APPROVAL_ROLE_VAULT[authority_id].confirmation_events(confirmation_id)

    def persist_report(
        *,
        broker: str,
        broker_hash: str,
        ledger_hash: str,
        reason_codes: tuple[str, ...],
        checked_at: datetime,
    ) -> EventRecord:
        return _RECONCILIATION_ROLE_VAULT[authority_id].persist_report(
            broker=broker,
            broker_hash=broker_hash,
            ledger_hash=ledger_hash,
            reason_codes=reason_codes,
            checked_at=checked_at,
        )

    def clear_kill_switch(*, acknowledgement: str, now: datetime) -> None:
        _RECONCILIATION_ROLE_VAULT[authority_id].clear_kill_switch(
            acknowledgement=acknowledgement, now=now
        )

    def reconciliation_events() -> tuple[EventRecord, ...]:
        return _RECONCILIATION_ROLE_VAULT[authority_id].reconciliation_events()

    def kill_switch_events() -> tuple[EventRecord, ...]:
        return _RECONCILIATION_ROLE_VAULT[authority_id].kill_switch_events()

    def interlock_events() -> tuple[EventRecord, ...]:
        return _RECONCILIATION_ROLE_VAULT[authority_id].interlock_events()

    def claim_and_start(
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
        _LIVE_ROLE_VAULT[authority_id].claim_and_start(
            intent_id=intent_id,
            broker=broker,
            confirmation_id=confirmation_id,
            fingerprint=fingerprint,
            expires_at=expires_at,
            reconciliation_head=reconciliation_head,
            kill_switch_head=kill_switch_head,
            interlock_head=interlock_head,
            occurred_at=occurred_at,
        )

    def record_acknowledgement(
        *,
        intent_id: str,
        broker: str,
        broker_order_id: str,
        status: str,
        submission_id: str,
        occurred_at: datetime,
    ) -> None:
        _LIVE_ROLE_VAULT[authority_id].record_acknowledgement(
            intent_id=intent_id,
            broker=broker,
            broker_order_id=broker_order_id,
            status=status,
            submission_id=submission_id,
            occurred_at=occurred_at,
        )

    def record_unknown(*, intent_id: str, submission_id: str, occurred_at: datetime) -> None:
        _LIVE_ROLE_VAULT[authority_id].record_unknown(
            intent_id=intent_id,
            submission_id=submission_id,
            occurred_at=occurred_at,
        )

    identity = authority.event_store_identity
    approval = _CallbackApprovalSafetyCapability(
        token=_CALLBACK_HANDLE_TOKEN,
        issue=issue_confirmation,
        read=confirmation_events,
        store_identity=identity,
    )
    reconciliation = _CallbackReconciliationSafetyCapability(
        token=_CALLBACK_HANDLE_TOKEN,
        persist=persist_report,
        clear=clear_kill_switch,
        read=reconciliation_events,
        kills=kill_switch_events,
        interlocks=interlock_events,
        store_identity=identity,
    )
    live = _CallbackLiveSafetyCapability(
        token=_CALLBACK_HANDLE_TOKEN,
        claim=claim_and_start,
        acknowledge=record_acknowledgement,
        unknown=record_unknown,
        store_identity=identity,
    )
    finalize(approval, _release_role_vault, authority_id)
    finalize(reconciliation, _release_role_vault, authority_id)
    finalize(live, _release_role_vault, authority_id)
    return approval, reconciliation, live


class ApprovalSafetyCapability:
    """Factory-registered, approval-only opaque identity."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        pass

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("approval safety handle is immutable")

    def __copy__(self) -> ApprovalSafetyCapability:
        raise TypeError("approval safety handles cannot be copied")

    def __deepcopy__(self, memo: object) -> ApprovalSafetyCapability:
        del memo
        raise TypeError("approval safety handles cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("approval safety handles cannot be serialized")

    @property
    def store_identity(self) -> object:
        return _approval_binding(self).store_identity

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
        return _approval_binding(self).issue_confirmation(
            phrase=phrase,
            broker=broker,
            created_at=created_at,
            expires_at=expires_at,
            fingerprint=fingerprint,
            risk_decision_hash=risk_decision_hash,
        )

    def confirmation_events(self, confirmation_id: str) -> tuple[EventRecord, ...]:
        return _approval_binding(self).confirmation_events(confirmation_id)


class ReconciliationSafetyCapability:
    """Factory-registered, reconciliation-only opaque identity."""

    __slots__ = ("_reconciliation_layout", "__weakref__")

    def __init__(self) -> None:
        pass

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("reconciliation safety handle is immutable")

    def __copy__(self) -> ReconciliationSafetyCapability:
        raise TypeError("reconciliation safety handles cannot be copied")

    def __deepcopy__(self, memo: object) -> ReconciliationSafetyCapability:
        del memo
        raise TypeError("reconciliation safety handles cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("reconciliation safety handles cannot be serialized")

    @property
    def store_identity(self) -> object:
        return _reconciliation_binding(self).store_identity

    def persist_report(
        self,
        *,
        broker: str,
        broker_hash: str,
        ledger_hash: str,
        reason_codes: tuple[str, ...],
        checked_at: datetime,
    ) -> EventRecord:
        return _reconciliation_binding(self).persist_report(
            broker=broker,
            broker_hash=broker_hash,
            ledger_hash=ledger_hash,
            reason_codes=reason_codes,
            checked_at=checked_at,
        )

    def clear_kill_switch(self, *, acknowledgement: str, now: datetime) -> None:
        _reconciliation_binding(self).clear_kill_switch(
            acknowledgement=acknowledgement,
            now=now,
        )

    def reconciliation_events(self) -> tuple[EventRecord, ...]:
        return _reconciliation_binding(self).reconciliation_events()

    def kill_switch_events(self) -> tuple[EventRecord, ...]:
        return _reconciliation_binding(self).kill_switch_events()

    def interlock_events(self) -> tuple[EventRecord, ...]:
        return _reconciliation_binding(self).interlock_events()


class LiveSafetyCapability:
    """Factory-registered, live-only opaque identity."""

    __slots__ = ("_live_layout_a", "_live_layout_b", "__weakref__")

    def __init__(self) -> None:
        pass

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("live safety handle is immutable")

    def __copy__(self) -> LiveSafetyCapability:
        raise TypeError("live safety handles cannot be copied")

    def __deepcopy__(self, memo: object) -> LiveSafetyCapability:
        del memo
        raise TypeError("live safety handles cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("live safety handles cannot be serialized")

    @property
    def store_identity(self) -> object:
        return _live_binding(self).store_identity

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
        _live_binding(self).claim_and_start(
            intent_id=intent_id,
            broker=broker,
            confirmation_id=confirmation_id,
            fingerprint=fingerprint,
            expires_at=expires_at,
            reconciliation_head=reconciliation_head,
            kill_switch_head=kill_switch_head,
            interlock_head=interlock_head,
            occurred_at=occurred_at,
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
        _live_binding(self).record_acknowledgement(
            intent_id=intent_id,
            broker=broker,
            broker_order_id=broker_order_id,
            status=status,
            submission_id=submission_id,
            occurred_at=occurred_at,
        )

    def record_unknown(self, *, intent_id: str, submission_id: str, occurred_at: datetime) -> None:
        _live_binding(self).record_unknown(
            intent_id=intent_id,
            submission_id=submission_id,
            occurred_at=occurred_at,
        )


_APPROVAL_BINDINGS: WeakKeyDictionary[ApprovalSafetyCapability, _ApprovalRole] = WeakKeyDictionary()
_RECONCILIATION_BINDINGS: WeakKeyDictionary[ReconciliationSafetyCapability, _ReconciliationRole] = (
    WeakKeyDictionary()
)
_LIVE_BINDINGS: WeakKeyDictionary[LiveSafetyCapability, _LiveRole] = WeakKeyDictionary()


def _approval_binding(handle: ApprovalSafetyCapability) -> _ApprovalRole:
    binding = _APPROVAL_BINDINGS.get(handle)
    if type(binding) is not _ApprovalRole:
        raise SafetyIntegrityError("approval safety handle is not factory registered")
    return binding


def _reconciliation_binding(
    handle: ReconciliationSafetyCapability,
) -> _ReconciliationRole:
    binding = _RECONCILIATION_BINDINGS.get(handle)
    if type(binding) is not _ReconciliationRole:
        raise SafetyIntegrityError("reconciliation safety handle is not factory registered")
    return binding


def _live_binding(handle: LiveSafetyCapability) -> _LiveRole:
    binding = _LIVE_BINDINGS.get(handle)
    if type(binding) is not _LiveRole:
        raise SafetyIntegrityError("live safety handle is not factory registered")
    return binding


def create_safety_capabilities(
    *,
    audit_log: AuditLog,
    key: bytes,
    nonce_source: Callable[[], bytes],
) -> tuple[ApprovalSafetyCapability, ReconciliationSafetyCapability, LiveSafetyCapability]:
    """Consume key material once and register three zero-state opaque role identities."""
    authority = _SafetyAuthority(
        audit_log=audit_log,
        authenticator=_SafetyMac(key=key, nonce_source=nonce_source),
    )
    approval = ApprovalSafetyCapability()
    reconciliation = ReconciliationSafetyCapability()
    live = LiveSafetyCapability()
    _APPROVAL_BINDINGS[approval] = _ApprovalRole._from_repository(authority)
    _RECONCILIATION_BINDINGS[reconciliation] = _ReconciliationRole._from_repository(authority)
    _LIVE_BINDINGS[live] = _LiveRole._from_repository(authority)
    return approval, reconciliation, live


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


def _kill_active(rows: tuple[EventRecord, ...], reconciliations: tuple[EventRecord, ...]) -> bool:
    latest: EventRecord | None = None
    active = False
    for row in rows:
        if row.kind == "kill_switch.activated":
            reasons = row.payload.get("reason_codes")
            if type(reasons) is not tuple or not reasons:
                return True
            latest, active = row, True
        elif row.kind == "kill_switch.cleared":
            newer_healthy = any(
                candidate.kind == "reconciliation.healthy"
                and candidate.payload.get("healthy") is True
                and candidate.payload.get("reason_codes") == ()
                and latest is not None
                and latest.sequence < candidate.sequence < row.sequence
                for candidate in reconciliations
            )
            if (
                latest is None
                or row.payload.get("activation_event_id") != latest.event_id
                or row.payload.get("activation_sequence") != latest.sequence
                or latest.sequence >= row.sequence
                or not newer_healthy
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
