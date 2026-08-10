"""Exchange-aware local scheduler with stale-run exclusion and durable audit."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from market_sentinel.domain.clock import Clock
from market_sentinel.operations.audit import AuditEvent, AuditLog


class ExchangeCalendar(Protocol):
    """Injected session decision; implementations must not perform hidden provider calls."""

    def is_session_open(self, instrument_id: str, instant: datetime) -> bool: ...


@dataclass(frozen=True, slots=True)
class ScheduledJob:
    """One exact due run with a bounded lateness window."""

    strategy_id: str
    instrument_id: str
    due_at: datetime
    max_lateness_seconds: int
    run: Callable[[], object]

    def __post_init__(self) -> None:
        if (
            type(self.strategy_id) is not str
            or not self.strategy_id
            or type(self.instrument_id) is not str
            or not self.instrument_id
            or type(self.due_at) is not datetime
            or self.due_at.tzinfo is None
            or self.due_at.utcoffset() is None
            or type(self.max_lateness_seconds) is not int
            or not 0 <= self.max_lateness_seconds <= 86_400
            or not callable(self.run)
        ):
            raise ValueError("scheduled job is malformed")


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Stable scheduler result without retained provider exceptions."""

    strategy_id: str
    instrument_id: str
    reason_code: str
    ran: bool


class Scheduler:
    """Run only current due jobs and exclude duplicate strategy/instrument workers."""

    def __init__(
        self,
        *,
        clock: Clock,
        calendar: ExchangeCalendar,
        audit: AuditLog,
        event_id_factory: Callable[[], str],
    ) -> None:
        if type(audit) is not AuditLog or not callable(event_id_factory):
            raise ValueError("scheduler requires durable audit and an event id factory")
        self._clock = clock
        self._calendar = calendar
        self._audit = audit
        self._event_id_factory = event_id_factory
        self._guard = threading.Lock()
        self._locks: dict[tuple[str, str], threading.Lock] = {}

    @property
    def audit(self) -> AuditLog:
        """Return the exact durable audit capability used by this scheduler."""
        return self._audit

    def run_due(self, job: ScheduledJob) -> RunOutcome:
        """Execute one due job without catch-up after its freshness window."""
        if type(job) is not ScheduledJob:
            raise ValueError("scheduler accepts exact ScheduledJob records")
        now = _aware_utc(self._clock.now())
        due_at = _aware_utc(job.due_at)
        if now < due_at:
            return self._outcome(job, "NOT_DUE", False)
        if (now - due_at).total_seconds() > job.max_lateness_seconds:
            return self._outcome(job, "MISSED_STALE", False)
        try:
            due_session_open = self._calendar.is_session_open(job.instrument_id, due_at)
            current_session_open = self._calendar.is_session_open(job.instrument_id, now)
        except Exception:
            session_open = False
            reason = "EXCHANGE_CALENDAR_UNAVAILABLE"
        else:
            session_open = (
                type(due_session_open) is bool
                and due_session_open
                and type(current_session_open) is bool
                and current_session_open
            )
            reason = "EXCHANGE_CLOSED"
        if not session_open:
            return self._outcome(job, reason, False)

        lock = self._lock_for(job)
        if not lock.acquire(blocking=False):
            return self._outcome(job, "ALREADY_RUNNING", False)
        try:
            aggregate = f"scheduler:{job.strategy_id}:{job.instrument_id}"
            try:
                self._audit.record(
                    self._event_id(),
                    "scheduler.started",
                    aggregate,
                    {
                        "strategy_id": job.strategy_id,
                        "instrument_id": job.instrument_id,
                        "due_at": due_at.isoformat(),
                    },
                )
            except Exception:
                return self._outcome(job, "AUDIT_PERSISTENCE_FAILED", False)
            try:
                job.run()
            except Exception:
                reason_code = "JOB_FAILED"
                health_kind = "scheduler.unhealthy"
            else:
                reason_code = "COMPLETED"
                health_kind = "scheduler.healthy"
            ended_at = _aware_utc(self._clock.now())
            try:
                self._audit.record_many(
                    (
                        AuditEvent(
                            self._event_id(),
                            "scheduler.ended",
                            aggregate,
                            {"reason_code": reason_code},
                            ended_at,
                        ),
                        AuditEvent(
                            self._event_id(),
                            health_kind,
                            aggregate,
                            {"healthy": reason_code == "COMPLETED"},
                            ended_at,
                        ),
                    )
                )
            except Exception:
                return self._outcome(job, "AUDIT_PERSISTENCE_FAILED", True)
            return self._outcome(job, reason_code, True)
        finally:
            lock.release()

    def _lock_for(self, job: ScheduledJob) -> threading.Lock:
        key = (job.strategy_id, job.instrument_id)
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def _event_id(self) -> str:
        event_id = self._event_id_factory()
        if type(event_id) is not str or not event_id:
            raise ValueError("scheduler event id is malformed")
        return event_id

    @staticmethod
    def _outcome(job: ScheduledJob, reason_code: str, ran: bool) -> RunOutcome:
        return RunOutcome(job.strategy_id, job.instrument_id, reason_code, ran)


def _aware_utc(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("scheduler clock must return an aware datetime")
    return value.astimezone(UTC)
