"""Safe facade for recording audit events."""

from collections.abc import Mapping

from market_sentinel.domain.clock import Clock
from market_sentinel.security import redact_mapping
from market_sentinel.storage.events import EventStore


class AuditLog:
    """Records redacted events using the application clock."""

    def __init__(self, store: EventStore, clock: Clock) -> None:
        self._store = store
        self._clock = clock

    def record(
        self,
        event_id: str,
        kind: str,
        aggregate_id: str,
        payload: Mapping[str, object],
    ) -> None:
        """Redact and append an audit event at the current clock instant."""
        self._store.append(event_id, kind, aggregate_id, redact_mapping(payload), self._clock.now())
