"""Behavioral tests for the immutable event ledger."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError

from market_sentinel.domain.clock import FrozenClock
from market_sentinel.operations.audit import AuditEvent, AuditLog
from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventAppend, EventHeadConflict, EventStore


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


def test_append_many_is_atomic_and_preserves_order_and_supplied_utc_times() -> None:
    """A bad later row must roll back every event and allocated sequence in the batch."""
    store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
    first_at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    second_at = datetime(2026, 8, 9, 10, 1, tzinfo=UTC)
    batch = (
        EventAppend("batch-1", "first", "aggregate", {"value": "one"}, first_at),
        EventAppend("batch-1", "duplicate", "aggregate", {"value": "two"}, second_at),
    )

    with pytest.raises(IntegrityError):
        store.append_many(batch)
    assert tuple(store.stream("aggregate")) == ()

    store.append_many(
        (
            EventAppend("batch-1", "first", "aggregate", {}, first_at),
            EventAppend("batch-2", "second", "aggregate", {}, second_at),
        )
    )
    rows = tuple(store.stream("aggregate"))
    assert [(row.event_id, row.sequence, row.occurred_at) for row in rows] == [
        ("batch-1", 1, first_at),
        ("batch-2", 2, second_at),
    ]


def test_audit_record_many_persists_first_class_redacted_rows_at_supplied_times() -> None:
    """A transactional audit group must not become one opaque surrogate batch event."""
    store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
    audit = AuditLog(store, FrozenClock(datetime(2030, 1, 1, tzinfo=UTC)))
    first_at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    second_at = datetime(2026, 8, 9, 10, 1, tzinfo=UTC)

    audit.record_many(
        (
            AuditEvent("audit-1", "transition", "order", {"api_key": "hidden"}, first_at),
            AuditEvent("audit-2", "fill", "order", {"safe": "yes"}, second_at),
        )
    )

    rows = tuple(store.stream("order"))
    assert [(row.kind, row.occurred_at) for row in rows] == [
        ("transition", first_at),
        ("fill", second_at),
    ]
    assert rows[0].payload["api_key"] == "[REDACTED]"


@pytest.mark.parametrize("payload_kind", ["huge_scalar", "deep", "cycle"])
def test_event_store_rejects_resource_hostile_payloads_before_json(
    payload_kind: str,
) -> None:
    """Payload bounds fail closed without recursive JSON allocation or mutation."""
    store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    if payload_kind == "huge_scalar":
        payload: dict[str, object] = {"value": "x" * 70_000}
    elif payload_kind == "deep":
        payload = {}
        cursor = payload
        for _ in range(40):
            nested: dict[str, object] = {}
            cursor["next"] = nested
            cursor = nested
    else:
        payload = {}
        payload["self"] = payload

    with pytest.raises(ValueError, match="payload"):
        store.append("hostile", "recorded", "aggregate", payload, at)

    assert tuple(store.stream("aggregate")) == ()


def test_audit_log_rejects_non_event_store_sink_and_exposes_sealed_store() -> None:
    """A wrapper around an arbitrary append_many callable is not durable provenance."""

    class _FakeSink:
        def append_many(self, batch: object) -> None:
            del batch

    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    with pytest.raises(ValueError, match="EventStore"):
        AuditLog(_FakeSink(), FrozenClock(at))  # type: ignore[arg-type]

    store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
    audit = AuditLog(store, FrozenClock(at))
    assert audit.event_store is store


def test_conditional_append_conflict_consumes_no_rows_or_sequence(tmp_path: Path) -> None:
    """A stale aggregate-head fence must roll back both its batch and sequence allocation."""
    url = f"sqlite+pysqlite:///{tmp_path / 'guarded.db'}"
    first = EventStore(create_engine_and_schema(url))
    second = EventStore(create_engine_and_schema(url))
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    first.append("healthy", "reconciliation.healthy", "reconciliation", {}, at)
    second.append("unhealthy", "reconciliation.unhealthy", "reconciliation", {}, at)

    with pytest.raises(EventHeadConflict):
        first.append_many_if_heads(
            (EventAppend("claim", "live.claimed", "intent", {}, at),),
            {"reconciliation": "healthy", "new-aggregate": None},
        )

    first.append("after", "recorded", "other", {}, at)
    assert tuple(first.stream("intent")) == ()
    assert [row.sequence for row in first.stream("other")] == [3]
