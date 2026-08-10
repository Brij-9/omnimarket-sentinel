"""Authenticated durable repository for live-safety authority events."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Final

from market_sentinel.operations.audit import AuditEvent, AuditLog
from market_sentinel.security import redact_mapping
from market_sentinel.storage.events import EventRecord

_MAC_DOMAIN: Final = b"omnimarket-sentinel:safety-event-mac:v1\x00"
_MAC_FIELD: Final = "safety_mac"
_VERSION_FIELD: Final = "safety_version"
_VERSION: Final = 1


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
class SafetyEvent:
    """One typed unsigned request; SafetyRepository alone adds authentication."""

    event_id: str
    kind: str
    aggregate_id: str
    payload: Mapping[str, object]
    occurred_at: datetime


class SafetyRepository:
    """Sign writes and reject any unauthenticated row during authoritative replay."""

    def __init__(self, *, audit_log: AuditLog, authenticator: SafetyAuthenticator) -> None:
        if type(audit_log) is not AuditLog or type(authenticator) is not SafetyAuthenticator:
            raise ValueError("safety repository requires exact durable authenticated capabilities")
        self._audit = audit_log
        self._authenticator = authenticator

    @property
    def audit_log(self) -> AuditLog:
        """Expose the underlying public log only for non-authoritative audit and diagnostics."""
        return self._audit

    @property
    def event_store_identity(self) -> object:
        """Opaque identity used to require one shared durable transaction boundary."""
        return self._audit.event_store

    def new_nonce(self) -> str:
        return self._authenticator.new_nonce()

    def record_many(self, batch: tuple[SafetyEvent, ...]) -> None:
        """Sign and atomically persist a nonempty safety batch."""
        self._audit.record_many(self._signed_batch(batch))

    def record_many_if_heads(
        self,
        batch: tuple[SafetyEvent, ...],
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

    def _signed_batch(self, batch: tuple[SafetyEvent, ...]) -> tuple[AuditEvent, ...]:
        if type(batch) is not tuple or not batch:
            raise ValueError("safety batch must be a nonempty tuple")
        signed: list[AuditEvent] = []
        for event in batch:
            if type(event) is not SafetyEvent:
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
