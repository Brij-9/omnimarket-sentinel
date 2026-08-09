"""Behavioral tests for the immutable event ledger."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest
from sqlalchemy import event
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

    store.append("evt-2", "risk.approved", "intent-1", {}, at)
    events = list(store.stream("intent-1"))
    assert [event.kind for event in events] == ["risk.rejected", "risk.approved"]
    assert [event.sequence for event in events] == [1, 2]


def test_event_store_orders_same_timestamp_events_by_append_sequence() -> None:
    """Events sharing a timestamp retain their append order."""
    store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)

    store.append("evt-2", "second", "intent-1", {}, at)
    store.append("evt-1", "first", "intent-1", {}, at)

    assert [event.event_id for event in store.stream("intent-1")] == ["evt-2", "evt-1"]


def test_event_store_allocates_unique_sequences_for_concurrent_appends(tmp_path: Path) -> None:
    """Distinct simultaneous appends must each receive an ordered sequence."""
    engine = create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'events.db'}")
    store = EventStore(engine)
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    sequence_reads = Barrier(2)

    def pause_after_reading_sequence(
        connection: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: object,
    ) -> None:
        del connection, cursor, parameters, context, executemany
        if "max(events.sequence)" in statement:
            sequence_reads.wait(timeout=5)

    event.listen(engine, "before_cursor_execute", pause_after_reading_sequence)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(store.append, f"evt-{index}", "recorded", "intent-1", {}, at)
                for index in range(2)
            ]
            for future in futures:
                future.result()
    finally:
        event.remove(engine, "before_cursor_execute", pause_after_reading_sequence)

    events = list(store.stream("intent-1"))
    assert [record.sequence for record in events] == [1, 2]
    assert {record.event_id for record in events} == {"evt-0", "evt-1"}


@pytest.mark.parametrize("invalid_number", [float("nan"), float("inf"), float("-inf")])
def test_event_store_rejects_non_finite_json_numbers(invalid_number: float) -> None:
    """Canonical JSON forbids NaN and both infinity values."""
    store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)

    with pytest.raises(ValueError):
        store.append("evt-1", "recorded", "intent-1", {"value": invalid_number}, at)

    assert list(store.stream("intent-1")) == []


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
