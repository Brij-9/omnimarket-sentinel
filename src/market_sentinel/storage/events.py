"""Append-only repository for immutable audit events."""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import cast

from sqlalchemy import select, update
from sqlalchemy.engine import Engine

from market_sentinel.storage.db import event_sequence, events


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
        self.append_many(
            (EventAppend(event_id, kind, aggregate_id, payload, occurred_at),)
        )

    def append_many(self, batch: tuple[EventAppend, ...]) -> None:
        """Append first-class events atomically in caller order with contiguous sequences."""
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
        with self._engine.begin() as connection:
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

    def stream(self, aggregate_id: str) -> Iterator[EventRecord]:
        """Yield aggregate events in deterministic temporal order."""
        statement = (
            select(events)
            .where(events.c.aggregate_id == aggregate_id)
            .order_by(events.c.occurred_at, events.c.sequence)
        )
        with self._engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        for row in rows:
            payload = json.loads(cast(str, row["payload_json"]))
            if not isinstance(payload, dict):
                raise ValueError("event payload must be a JSON object")
            yield EventRecord(
                event_id=cast(str, row["event_id"]),
                kind=cast(str, row["kind"]),
                aggregate_id=cast(str, row["aggregate_id"]),
                payload=_freeze_mapping(payload),
                occurred_at=cast(datetime, row["occurred_at"]).astimezone(UTC),
                sequence=cast(int, row["sequence"]),
            )


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: object) -> object:
    if isinstance(value, dict):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value
