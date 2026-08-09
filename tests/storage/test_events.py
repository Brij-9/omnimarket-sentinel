"""Behavioral tests for the immutable event ledger."""

from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError

from market_sentinel.domain.clock import FrozenClock
from market_sentinel.operations.audit import AuditLog
from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventStore


def test_event_store_is_ordered_and_event_ids_are_immutable() -> None:
    """Duplicate event IDs must not overwrite the original event."""
    store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)

    store.append("evt-1", "risk.rejected", "intent-1", {"reason": "STALE_DATA"}, at)

    with pytest.raises(IntegrityError):
        store.append("evt-1", "risk.approved", "intent-1", {}, at)

    events = list(store.stream("intent-1"))
    assert [event.kind for event in events] == ["risk.rejected"]


def test_event_store_orders_same_timestamp_events_by_append_sequence() -> None:
    """Events sharing a timestamp retain their append order."""
    store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)

    store.append("evt-2", "second", "intent-1", {}, at)
    store.append("evt-1", "first", "intent-1", {}, at)

    assert [event.event_id for event in store.stream("intent-1")] == ["evt-2", "evt-1"]


def test_audit_log_redacts_payload_and_uses_its_clock() -> None:
    """Audit events must never persist credentials or a caller-selected time."""
    store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    audit = AuditLog(store, FrozenClock(at))

    audit.record(
        "evt-1",
        "broker.requested",
        "order-1",
        {"api_key": "credential", "details": {"access_token": "nested", "safe": "yes"}},
    )

    [event] = store.stream("order-1")
    assert event.occurred_at == at
    assert event.payload == {
        "api_key": "[REDACTED]",
        "details": {"access_token": "[REDACTED]", "safe": "yes"},
    }
