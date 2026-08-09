"""SQLite schema for the append-only event ledger."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Column, Index, Integer, MetaData, String, Table, create_engine
from sqlalchemy.engine import Dialect, Engine
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator[datetime]):
    """Persist aware datetimes as UTC ISO-8601 strings across SQLite connections."""

    impl = String(32)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> str | None:
        del dialect
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value.astimezone(UTC).isoformat()

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        del dialect
        if value is None:
            return None
        return datetime.fromisoformat(value).astimezone(UTC)


metadata = MetaData()

events = Table(
    "events",
    metadata,
    Column("event_id", String, primary_key=True),
    Column("aggregate_id", String, nullable=False),
    Column("kind", String, nullable=False),
    Column("occurred_at", UTCDateTime(), nullable=False),
    Column("payload_json", String, nullable=False),
    Column("sequence", Integer, nullable=False, unique=True),
    Index("ix_events_aggregate_id", "aggregate_id"),
)


def create_engine_and_schema(url: str) -> Engine:
    """Create an engine and install the event-ledger schema."""
    engine = create_engine(url)
    metadata.create_all(engine)
    return engine
