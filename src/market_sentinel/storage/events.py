"""Append-only repository for immutable audit events."""

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import cast

from sqlalchemy import select, update
from sqlalchemy.engine import Connection, Engine

from market_sentinel.storage.db import event_sequence, events

_MAX_PAYLOAD_DEPTH = 16
_MAX_PAYLOAD_NODES = 4_096
_MAX_COLLECTION_ITEMS = 512
_MAX_SCALAR_BYTES = 4_096
_MAX_TOTAL_SCALAR_BYTES = 65_536


@dataclass(frozen=True, slots=True)
class EventRecord:
    """A durable event returned from the ledger."""

    event_id: str
    kind: str
    aggregate_id: str
    payload: Mapping[str, object]
    occurred_at: datetime
    sequence: int


@dataclass(frozen=True, slots=True)
class EventAppend:
    """One validated event requested as part of an atomic append group."""

    event_id: str
    kind: str
    aggregate_id: str
    payload: Mapping[str, object]
    occurred_at: datetime


class EventHeadConflict(RuntimeError):
    """A guarded append observed a different aggregate head and wrote nothing."""


class EventStore:
    """Store events without mutation or deletion operations."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def append(
        self,
        event_id: str,
        kind: str,
        aggregate_id: str,
        payload: Mapping[str, object],
        occurred_at: datetime,
    ) -> None:
        """Append one event in a transaction, preserving the supplied UTC instant."""
        self.append_many((EventAppend(event_id, kind, aggregate_id, payload, occurred_at),))

    def append_many(self, batch: tuple[EventAppend, ...]) -> None:
        """Append first-class events atomically in caller order with contiguous sequences."""
        prepared = _prepare_batch(batch)
        with self._engine.begin() as connection:
            _append_prepared(connection, prepared)

    def append_many_if_heads(
        self,
        batch: tuple[EventAppend, ...],
        expected_heads: Mapping[str, str | None],
    ) -> None:
        """Append atomically only while every exact aggregate head matches the fence."""
        prepared = _prepare_batch(batch)
        if not isinstance(expected_heads, Mapping) or not expected_heads:
            raise ValueError("expected heads must be a nonempty mapping")
        if len(expected_heads) > _MAX_COLLECTION_ITEMS:
            raise ValueError("expected heads exceed the bounded mapping contract")
        normalized: dict[str, str | None] = {}
        for aggregate_id, event_id in expected_heads.items():
            if type(aggregate_id) is not str or not aggregate_id:
                raise ValueError("expected aggregate identity must be nonempty")
            if event_id is not None and (type(event_id) is not str or not event_id):
                raise ValueError("expected event identity must be nonempty or None")
            normalized[aggregate_id] = event_id
        with self._engine.begin() as connection:
            locked = connection.scalar(
                update(event_sequence)
                .where(event_sequence.c.counter_id == 1)
                .values(next_sequence=event_sequence.c.next_sequence)
                .returning(event_sequence.c.next_sequence)
            )
            if locked is None:
                raise RuntimeError("event sequence counter is not initialized")
            for aggregate_id, expected_event_id in normalized.items():
                actual_event_id = connection.scalar(
                    select(events.c.event_id)
                    .where(events.c.aggregate_id == aggregate_id)
                    .order_by(events.c.sequence.desc())
                    .limit(1)
                )
                if actual_event_id != expected_event_id:
                    raise EventHeadConflict("aggregate head changed")
            _append_prepared(connection, prepared)

    def stream(self, aggregate_id: str) -> Iterator[EventRecord]:
        """Yield aggregate events in authoritative database append order."""
        statement = (
            select(events).where(events.c.aggregate_id == aggregate_id).order_by(events.c.sequence)
        )
        with self._engine.connect() as connection:
            for row in connection.execute(statement).mappings():
                payload = json.loads(cast(str, row["payload_json"]))
                if not isinstance(payload, dict):
                    raise ValueError("event payload must be a JSON object")
                validate_event_payload(payload)
                yield EventRecord(
                    event_id=cast(str, row["event_id"]),
                    kind=cast(str, row["kind"]),
                    aggregate_id=cast(str, row["aggregate_id"]),
                    payload=_freeze_mapping(payload),
                    occurred_at=cast(datetime, row["occurred_at"]).astimezone(UTC),
                    sequence=cast(int, row["sequence"]),
                )


def _prepare_batch(batch: tuple[EventAppend, ...]) -> list[dict[str, object]]:
    """Validate and canonicalize one append group without touching durable state."""
    if not isinstance(batch, tuple) or not batch:
        raise ValueError("event batch must be a nonempty tuple")
    prepared: list[dict[str, object]] = []
    for item in batch:
        if not isinstance(item, EventAppend):
            raise ValueError("event batch must contain EventAppend records")
        if not item.event_id or not item.kind or not item.aggregate_id:
            raise ValueError("event identity fields must be nonempty")
        if item.occurred_at.tzinfo is None or item.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        validate_event_payload(item.payload)
        canonical_payload = json.dumps(
            dict(item.payload),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        prepared.append(
            {
                "event_id": item.event_id,
                "kind": item.kind,
                "aggregate_id": item.aggregate_id,
                "payload_json": canonical_payload,
                "occurred_at": item.occurred_at.astimezone(UTC),
            }
        )
    return prepared


def _append_prepared(connection: Connection, prepared: list[dict[str, object]]) -> None:
    final_sequence = cast(
        int | None,
        connection.scalar(
            update(event_sequence)
            .where(event_sequence.c.counter_id == 1)
            .values(next_sequence=event_sequence.c.next_sequence + len(prepared))
            .returning(event_sequence.c.next_sequence)
        ),
    )
    if final_sequence is None:
        raise RuntimeError("event sequence counter is not initialized")
    first_sequence = final_sequence - len(prepared) + 1
    for offset, values in enumerate(prepared):
        values["sequence"] = first_sequence + offset
    connection.execute(events.insert(), prepared)


def validate_event_payload(payload: Mapping[str, object]) -> None:
    """Iteratively reject resource-hostile values before canonical JSON work."""
    if not isinstance(payload, Mapping):
        raise ValueError("event payload must be a bounded mapping")
    stack: list[tuple[object, int]] = [(payload, 0)]
    seen_containers: set[int] = set()
    nodes = 0
    scalar_bytes = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_PAYLOAD_NODES or depth > _MAX_PAYLOAD_DEPTH:
            raise ValueError("event payload exceeds bounded canonical JSON contract")
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in seen_containers or len(value) > _MAX_COLLECTION_ITEMS:
                raise ValueError("event payload exceeds bounded canonical JSON contract")
            seen_containers.add(identity)
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("event payload keys must be strings")
                key_bytes = len(key.encode("utf-8"))
                if key_bytes > _MAX_SCALAR_BYTES:
                    raise ValueError("event payload exceeds bounded canonical JSON contract")
                scalar_bytes += key_bytes
                stack.append((item, depth + 1))
        elif isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in seen_containers or len(value) > _MAX_COLLECTION_ITEMS:
                raise ValueError("event payload exceeds bounded canonical JSON contract")
            seen_containers.add(identity)
            stack.extend((item, depth + 1) for item in value)
        elif value is None or type(value) is bool:
            scalar_bytes += 4
        elif type(value) is int:
            if value.bit_length() > 63:
                raise ValueError("event payload integer exceeds bounded range")
            scalar_bytes += len(str(value))
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("event payload float must be finite")
            scalar_bytes += len(repr(value))
        elif isinstance(value, str):
            item_bytes = len(value.encode("utf-8"))
            if item_bytes > _MAX_SCALAR_BYTES:
                raise ValueError("event payload exceeds bounded canonical JSON contract")
            scalar_bytes += item_bytes
        else:
            raise ValueError("event payload contains a non-JSON scalar")
        if scalar_bytes > _MAX_TOTAL_SCALAR_BYTES:
            raise ValueError("event payload exceeds bounded canonical JSON contract")


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: object) -> object:
    if isinstance(value, dict):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value
