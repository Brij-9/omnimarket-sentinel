"""Clock abstractions for production and deterministic evaluation."""

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    def __init__(self, instant: datetime) -> None:
        if instant.tzinfo is None:
            raise ValueError("FrozenClock requires an aware datetime")
        self._instant = instant.astimezone(UTC)

    def now(self) -> datetime:
        return self._instant

    def advance(self, delta: timedelta) -> None:
        if delta.total_seconds() < 0:
            raise ValueError("FrozenClock cannot move backward")
        self._instant += delta
