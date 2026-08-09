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
        if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        canonical_payload = json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._engine.begin() as connection:
            sequence = cast(
                int | None,
                connection.scalar(
                    update(event_sequence)
                    .where(event_sequence.c.counter_id == 1)
                    .values(next_sequence=event_sequence.c.next_sequence + 1)
                    .returning(event_sequence.c.next_sequence)
                ),
            )
            if sequence is None:
                raise RuntimeError("event sequence counter is not initialized")
            connection.execute(
                events.insert().values(
                    event_id=event_id,
                    kind=kind,
                    aggregate_id=aggregate_id,
                    payload_json=canonical_payload,
                    occurred_at=occurred_at.astimezone(UTC),
                    sequence=sequence,
                )
            )

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
