"""Durable exchange-aware scheduler with once-only due-run claims."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from market_sentinel.domain.clock import Clock
from market_sentinel.operations.audit import AuditEvent, AuditLog
from market_sentinel.storage.events import EventHeadConflict

_STRATEGY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_INSTRUMENT = re.compile(r"^[A-Z0-9][A-Z0-9._/-]{0,63}@[a-z0-9][a-z0-9-]{0,31}$")
_EVENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_RUN_DOMAIN = b"omnimarket-sentinel:scheduler-run:v1\x00"
_EXCLUSION_DOMAIN = b"omnimarket-sentinel:scheduler-exclusion:v1\x00"


class ExchangeCalendar(Protocol):
    """Injected read-only exchange session decision."""

    def is_session_open(self, instrument_id: str, instant: datetime) -> bool: ...


@dataclass(frozen=True, slots=True)
class ScheduledJob:
    """One bounded canonical due run."""

    strategy_id: str
    instrument_id: str
    due_at: datetime
    max_lateness_seconds: int
    run: Callable[[], object]

    def __post_init__(self) -> None:
        if (
            type(self.strategy_id) is not str
            or _STRATEGY.fullmatch(self.strategy_id) is None
            or type(self.instrument_id) is not str
            or _INSTRUMENT.fullmatch(self.instrument_id) is None
            or type(self.due_at) is not datetime
            or self.due_at.tzinfo is not UTC
            or type(self.max_lateness_seconds) is not int
            or not 0 <= self.max_lateness_seconds <= 86_400
            or not callable(self.run)
        ):
            raise ValueError("scheduled job is malformed")

    @property
    def aggregate_id(self) -> str:
        """Return the bounded collision-resistant identity of this exact due run."""
        return _run_aggregate(self)

    @property
    def exclusion_id(self) -> str:
        """Return the durable strategy/instrument exclusion identity."""
        return _exclusion_aggregate(self)


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Stable scheduler result without retained provider exceptions."""

    strategy_id: str
    instrument_id: str
    reason_code: str
    ran: bool


class Scheduler:
    """Use durable aggregate-head claims as the sole once-only run arbiter."""

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

    @property
    def audit(self) -> AuditLog:
        """Return the exact durable audit capability used by this scheduler."""
        return self._audit

    def run_due(self, job: ScheduledJob) -> RunOutcome:
        """Claim one exact due instant durably and never execute it more than once."""
        if type(job) is not ScheduledJob:
            raise ValueError("scheduler accepts exact ScheduledJob records")
        due_at = job.due_at
        aggregate = _run_aggregate(job)
        exclusion = _exclusion_aggregate(job)
        now = self._safe_now()
        if now is None:
            return self._claim_skip(job, aggregate, "CLOCK_UNAVAILABLE", due_at)
        if now < due_at:
            return self._outcome(job, "NOT_DUE", False)
        rejection = self._due_rejection(job, due_at, now)
        if rejection is not None:
            return self._claim_skip(job, aggregate, rejection, now)

        exclusion_state = self._exclusion_state(exclusion)
        if exclusion_state is None:
            return self._outcome(job, "AUDIT_PERSISTENCE_FAILED", False)
        exclusion_active, exclusion_head = exclusion_state
        if exclusion_active:
            return self._outcome(job, "ALREADY_RUNNING", False)
        start_ids = self._event_ids(2)
        if start_ids is None:
            return self._outcome(job, "AUDIT_PERSISTENCE_FAILED", False)
        started_id, exclusion_started_id = start_ids
        try:
            self._audit.record_many_if_heads(
                (
                    AuditEvent(
                        started_id,
                        "scheduler.started",
                        aggregate,
                        _job_payload(job, "STARTED"),
                        now,
                    ),
                    AuditEvent(
                        exclusion_started_id,
                        "scheduler.exclusion_started",
                        exclusion,
                        {"run_aggregate": aggregate},
                        now,
                    ),
                ),
                {aggregate: None, exclusion: exclusion_head},
            )
        except EventHeadConflict:
            return self._outcome(job, self._claim_conflict_reason(aggregate, exclusion), False)
        except Exception:
            return self._outcome(job, "AUDIT_PERSISTENCE_FAILED", False)

        immediate = self._safe_now()
        if immediate is None:
            return self._terminal(
                job,
                aggregate,
                started_id,
                exclusion,
                exclusion_started_id,
                "CLOCK_UNAVAILABLE",
                due_at,
                False,
            )
        rejection = self._due_rejection(job, due_at, immediate)
        if rejection is not None:
            return self._terminal(
                job,
                aggregate,
                started_id,
                exclusion,
                exclusion_started_id,
                rejection,
                immediate,
                False,
            )

        try:
            job.run()
        except Exception:
            reason = "JOB_FAILED"
        else:
            reason = "COMPLETED"
        ended_at = self._safe_now()
        if ended_at is None:
            reason = "CLOCK_UNAVAILABLE"
            ended_at = immediate
        return self._terminal(
            job,
            aggregate,
            started_id,
            exclusion,
            exclusion_started_id,
            reason,
            ended_at,
            True,
        )

    def _safe_now(self) -> datetime | None:
        try:
            value = self._clock.now()
        except Exception:
            return None
        if type(value) is not datetime or value.tzinfo is not UTC:
            return None
        return value

    def _due_rejection(
        self,
        job: ScheduledJob,
        due_at: datetime,
        now: datetime,
    ) -> str | None:
        if (now - due_at).total_seconds() > job.max_lateness_seconds:
            return "MISSED_STALE"
        try:
            due_open = self._calendar.is_session_open(job.instrument_id, due_at)
            current_open = self._calendar.is_session_open(job.instrument_id, now)
        except Exception:
            return "EXCHANGE_CALENDAR_UNAVAILABLE"
        if type(due_open) is not bool or type(current_open) is not bool:
            return "EXCHANGE_CALENDAR_UNAVAILABLE"
        if not due_open or not current_open:
            return "EXCHANGE_CLOSED"
        return None

    def _claim_skip(
        self,
        job: ScheduledJob,
        aggregate: str,
        reason: str,
        occurred_at: datetime,
    ) -> RunOutcome:
        event_ids = self._event_ids(2)
        if event_ids is None:
            return self._outcome(job, "AUDIT_PERSISTENCE_FAILED", False)
        try:
            self._audit.record_many_if_heads(
                (
                    AuditEvent(
                        event_ids[0],
                        "scheduler.skipped",
                        aggregate,
                        _job_payload(job, reason),
                        occurred_at,
                    ),
                    AuditEvent(
                        event_ids[1],
                        "scheduler.unhealthy",
                        aggregate,
                        {"healthy": False, "reason_code": reason},
                        occurred_at,
                    ),
                ),
                {aggregate: None},
            )
        except EventHeadConflict:
            return self._outcome(job, "ALREADY_CLAIMED", False)
        except Exception:
            return self._outcome(job, "AUDIT_PERSISTENCE_FAILED", False)
        return self._outcome(job, reason, False)

    def _terminal(
        self,
        job: ScheduledJob,
        aggregate: str,
        started_id: str,
        exclusion: str,
        exclusion_started_id: str,
        reason: str,
        occurred_at: datetime,
        ran: bool,
    ) -> RunOutcome:
        event_ids = self._event_ids(3)
        if event_ids is None:
            return self._outcome(job, "AUDIT_PERSISTENCE_FAILED", ran)
        completed = reason == "COMPLETED"
        terminal_kind = "scheduler.ended" if ran else "scheduler.skipped"
        try:
            self._audit.record_many_if_heads(
                (
                    AuditEvent(
                        event_ids[0],
                        terminal_kind,
                        aggregate,
                        {"reason_code": reason},
                        occurred_at,
                    ),
                    AuditEvent(
                        event_ids[1],
                        "scheduler.healthy" if completed else "scheduler.unhealthy",
                        aggregate,
                        {"healthy": completed, "reason_code": reason},
                        occurred_at,
                    ),
                    AuditEvent(
                        event_ids[2],
                        "scheduler.exclusion_released",
                        exclusion,
                        {"run_aggregate": aggregate, "reason_code": reason},
                        occurred_at,
                    ),
                ),
                {aggregate: started_id, exclusion: exclusion_started_id},
            )
        except Exception:
            return self._outcome(job, "AUDIT_PERSISTENCE_FAILED", ran)
        return self._outcome(job, reason, ran)

    def _exclusion_state(self, aggregate: str) -> tuple[bool, str | None] | None:
        try:
            rows = tuple(self._audit.event_store.stream(aggregate))
        except Exception:
            return None
        if not rows:
            return False, None
        for index, row in enumerate(rows):
            expected = (
                "scheduler.exclusion_started"
                if index % 2 == 0
                else "scheduler.exclusion_released"
            )
            if row.kind != expected or type(row.payload.get("run_aggregate")) is not str:
                return None
        return rows[-1].kind == "scheduler.exclusion_started", rows[-1].event_id

    def _claim_conflict_reason(self, aggregate: str, exclusion: str) -> str:
        try:
            if tuple(self._audit.event_store.stream(aggregate)):
                return "ALREADY_CLAIMED"
        except Exception:
            return "AUDIT_PERSISTENCE_FAILED"
        exclusion_state = self._exclusion_state(exclusion)
        if exclusion_state is None:
            return "AUDIT_PERSISTENCE_FAILED"
        return "ALREADY_RUNNING" if exclusion_state[0] else "AUDIT_PERSISTENCE_FAILED"

    def _event_id_or_none(self) -> str | None:
        try:
            event_id = self._event_id_factory()
        except Exception:
            return None
        if type(event_id) is not str or _EVENT_ID.fullmatch(event_id) is None:
            return None
        return event_id

    def _event_ids(self, count: int) -> tuple[str, ...] | None:
        values: list[str] = []
        for _ in range(count):
            event_id = self._event_id_or_none()
            if event_id is None or event_id in values:
                return None
            values.append(event_id)
        return tuple(values)

    @staticmethod
    def _outcome(job: ScheduledJob, reason_code: str, ran: bool) -> RunOutcome:
        return RunOutcome(job.strategy_id, job.instrument_id, reason_code, ran)


def _run_aggregate(job: ScheduledJob) -> str:
    payload = json.dumps(
        [job.strategy_id, job.instrument_id, job.due_at.isoformat()],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return "scheduler-run-" + hashlib.sha256(_RUN_DOMAIN + payload).hexdigest()


def _exclusion_aggregate(job: ScheduledJob) -> str:
    payload = json.dumps(
        [job.strategy_id, job.instrument_id],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return "scheduler-exclusion-" + hashlib.sha256(_EXCLUSION_DOMAIN + payload).hexdigest()


def _job_payload(job: ScheduledJob, reason: str) -> dict[str, object]:
    return {
        "strategy_id": job.strategy_id,
        "instrument_id": job.instrument_id,
        "due_at": job.due_at.isoformat(),
        "reason_code": reason,
    }
