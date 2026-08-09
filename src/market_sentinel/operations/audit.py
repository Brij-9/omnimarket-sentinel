"""Safe facade for recording audit events."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from market_sentinel.domain.clock import Clock
from market_sentinel.security import redact_mapping
from market_sentinel.storage.events import EventAppend, EventStore


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """One first-class audit row with its authoritative occurrence time."""

    event_id: str
    kind: str
    aggregate_id: str
    payload: Mapping[str, object]
    occurred_at: datetime


class AuditLog:
    """Records redacted events using the application clock."""

    def __init__(self, store: EventStore, clock: Clock) -> None:
        if type(store) is not EventStore:
            raise ValueError("audit durability requires an exact EventStore")
        self._store = store
        self._clock = clock

    @property
    def event_store(self) -> EventStore:
        """Return the sealed durable capability used for both replay and continuation."""
        return self._store

    def record(
        self,
        event_id: str,
        kind: str,
        aggregate_id: str,
        payload: Mapping[str, object],
    ) -> None:
        """Redact and append an audit event at the current clock instant."""
        self._store.append(event_id, kind, aggregate_id, redact_mapping(payload), self._clock.now())

    def record_many(self, batch: tuple[AuditEvent, ...]) -> None:
        """Redact and persist separate supplied-time rows in one transaction."""
        if not isinstance(batch, tuple) or not batch:
            raise ValueError("audit batch must be a nonempty tuple")
        self._store.append_many(
            tuple(
                EventAppend(
                    event_id=event.event_id,
                    kind=event.kind,
                    aggregate_id=event.aggregate_id,
                    payload=redact_mapping(event.payload),
                    occurred_at=event.occurred_at,
                )
                for event in batch
            )
        )
