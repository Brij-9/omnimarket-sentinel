"""Dashboard and scheduler behavior at the local operations boundary."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

import market_sentinel.operations.dashboard as dashboard_module
from market_sentinel.brokers.preflight import required_gate_names
from market_sentinel.domain.clock import FrozenClock
from market_sentinel.domain.enums import OrderStatus
from market_sentinel.domain.models import GateResult
from market_sentinel.operations.audit import AuditLog
from market_sentinel.operations.dashboard import (
    DashboardAspiration,
    DashboardBroker,
    DashboardOrder,
    DashboardPortfolio,
    DashboardPromotion,
    DashboardResearch,
    DashboardRisk,
    DashboardSafetyState,
    DashboardStatus,
    DashboardStrategy,
    DashboardValidationError,
    export_dashboard,
    safe_json_mapping,
)
from market_sentinel.operations.scheduler import ScheduledJob, Scheduler
from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventStore

AT = datetime(2026, 8, 9, 10, tzinfo=UTC)


def _status() -> DashboardStatus:
    return DashboardStatus(
        generated_at=AT,
        data_as_of=AT - timedelta(seconds=5),
        research=DashboardResearch("tauric-v1", True),
        strategies=(DashboardStrategy("intraday", "v2"),),
        promotion=DashboardPromotion("paper"),
        portfolio=DashboardPortfolio("USD", Decimal("10.000")),
        risk=DashboardRisk(
            Decimal("0.005"),
            Decimal("0.10"),
            Decimal("0.50"),
            Decimal("0.02"),
            Decimal("0.10"),
        ),
        brokers=(
            DashboardBroker(
                "alpaca",
                tuple(
                    GateResult(name=name, passed=True, reason_code="OK")
                    for name in sorted(required_gate_names("alpaca"))
                ),
            ),
        ),
        orders=(DashboardOrder("order-1", OrderStatus.PROPOSED),),
        kill_switches=(DashboardSafetyState(False, "OK"),),
        interlocks=(DashboardSafetyState(False, "OK"),),
        aspirational_target=DashboardAspiration(
            Decimal("10"),
            Decimal("10.000"),
            Decimal("1000000"),
            Decimal("100000"),
            Decimal("1"),
            Decimal("999990"),
            True,
        ),
    )


def test_dashboard_is_schema_v1_canonical_redacted_and_risk_separate(tmp_path: Path) -> None:
    """Secrets, unstable decimals, or target-derived risk must not enter the status file."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    with localcontext() as context:
        context.prec = 5
        export_dashboard(_status(), first)
    export_dashboard(_status(), second)

    text = first.read_text(encoding="utf-8")
    data = json.loads(text)
    assert text == second.read_text(encoding="utf-8")
    assert data["schema_version"] == 1
    assert data["portfolio"]["equity"] == "10"
    assert data["aspirational_target"]["required_multiple"] == "100000"
    assert data["aspirational_target"]["reporting_only"] is True
    assert data["risk"]["max_position_fraction"] == "0.1"
    assert data["brokers"][0]["gates"] == [
        {"name": name, "passed": True, "reason_code": "OK"}
        for name in sorted(required_gate_names("alpaca"))
    ]
    assert "api_key" not in text.lower()


def test_dashboard_rejects_provider_objects_callbacks_and_unsafe_paths(tmp_path: Path) -> None:
    """A provider handle or traversal destination must fail before filesystem mutation."""
    hostile = _status()
    object.__setattr__(hostile, "research", {"provider": object(), "callback": lambda: None})
    with pytest.raises(DashboardValidationError):
        export_dashboard(hostile, tmp_path / "status.json")
    with pytest.raises(DashboardValidationError):
        export_dashboard(_status(), Path("..") / "status.json")


def test_dashboard_rejects_custom_mapping_without_invoking_provider_methods(
    tmp_path: Path,
) -> None:
    """A Mapping-shaped provider object must not execute during redaction."""

    class _ProviderMapping(Mapping[str, object]):
        calls = 0

        def __getitem__(self, key: str) -> object:
            del key
            self.calls += 1
            raise RuntimeError("provider method called")

        def __iter__(self) -> Iterator[str]:
            self.calls += 1
            raise RuntimeError("provider method called")

        def __len__(self) -> int:
            self.calls += 1
            raise RuntimeError("provider method called")

    provider = _ProviderMapping()
    status = _status()
    object.__setattr__(status, "research", provider)

    with pytest.raises(DashboardValidationError):
        export_dashboard(status, tmp_path / "status.json")

    assert provider.calls == 0


def test_dashboard_rejects_unc_destination_before_path_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A UNC destination must fail before any filesystem operation can reach a network share."""
    destination = Path(r"\\server\share\status.json")

    def unexpected_resolve(path: Path, strict: bool = False) -> Path:
        del path, strict
        raise RuntimeError("filesystem resolution should not run")

    monkeypatch.setattr(Path, "resolve", unexpected_resolve)
    with pytest.raises(DashboardValidationError, match="DASHBOARD_PATH_INVALID"):
        export_dashboard(_status(), destination)


def test_dashboard_redacts_secret_shaped_values_even_under_innocent_keys(tmp_path: Path) -> None:
    """A bearer value must not escape merely because its field name looks harmless."""
    prepared = safe_json_mapping({"note": "Bearer live-value-that-must-not-leak"})
    text = json.dumps(prepared)
    assert "live-value-that-must-not-leak" not in text
    assert "[REDACTED]" in text


def test_dashboard_atomic_replace_preserves_old_file_and_cleans_temp_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed final replace must leave the prior dashboard intact and no sibling temp."""
    destination = tmp_path / "status.json"
    destination.write_text("old", encoding="utf-8")

    def fail_commit(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("simulated")

    monkeypatch.setattr(dashboard_module, "_commit_open_temp", fail_commit)
    with pytest.raises(DashboardValidationError, match="DASHBOARD_WRITE_FAILED") as failure:
        export_dashboard(_status(), destination)

    assert failure.value.__cause__ is None
    assert failure.value.__context__ is None
    assert destination.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".status.json.*.tmp")) == []


class _Calendar:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened

    def is_session_open(self, instrument_id: str, instant: datetime) -> bool:
        return self.opened and instrument_id == "AAPL@alpaca" and instant.tzinfo is not None


def _scheduler(tmp_path: Path, clock: FrozenClock) -> tuple[Scheduler, EventStore]:
    store = EventStore(create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'events.db'}"))
    ids = iter(f"scheduler-{index}" for index in range(20))
    return (
        Scheduler(
            clock=clock,
            calendar=_Calendar(),
            audit=AuditLog(store, clock),
            event_id_factory=lambda: next(ids),
        ),
        store,
    )


def test_scheduler_skips_future_closed_and_stale_jobs_without_running(tmp_path: Path) -> None:
    """Missed runs must never catch up using stale market data."""
    clock = FrozenClock(AT)
    scheduler, _ = _scheduler(tmp_path, clock)
    calls: list[str] = []
    future = ScheduledJob(
        "swing",
        "AAPL@alpaca",
        AT + timedelta(seconds=1),
        10,
        lambda: calls.append("future"),
    )
    stale = ScheduledJob(
        "swing",
        "AAPL@alpaca",
        AT - timedelta(seconds=11),
        10,
        lambda: calls.append("stale"),
    )

    assert scheduler.run_due(future).reason_code == "NOT_DUE"
    assert scheduler.run_due(stale).reason_code == "MISSED_STALE"
    assert calls == []


def test_scheduler_rechecks_exchange_session_at_current_run_time(tmp_path: Path) -> None:
    """A job due before the close must not run after the exchange closes moments later."""

    class _ClosingCalendar:
        def is_session_open(self, instrument_id: str, instant: datetime) -> bool:
            return instrument_id == "AAPL@alpaca" and instant <= AT

    clock = FrozenClock(AT + timedelta(seconds=5))
    store = EventStore(
        create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'closing-events.db'}")
    )
    scheduler = Scheduler(
        clock=clock,
        calendar=_ClosingCalendar(),
        audit=AuditLog(store, clock),
        event_id_factory=iter(("event-1", "event-2")).__next__,
    )
    calls: list[str] = []

    outcome = scheduler.run_due(
        ScheduledJob("swing", "AAPL@alpaca", AT, 10, lambda: calls.append("ran"))
    )

    assert outcome.reason_code == "EXCHANGE_CLOSED"
    assert calls == []

    closed = Scheduler(
        clock=clock,
        calendar=_Calendar(opened=False),
        audit=scheduler.audit,
        event_id_factory=iter(("closed-event-1", "closed-event-2")).__next__,
    )
    closed_job = ScheduledJob(
        "swing-closed", "AAPL@alpaca", AT, 10, lambda: calls.append("closed")
    )
    assert closed.run_due(closed_job).reason_code == "EXCHANGE_CLOSED"
    assert calls == []


def test_scheduler_persists_start_then_atomic_end_health_for_success(tmp_path: Path) -> None:
    """A completed run must have ordered durable lifecycle and healthy audit rows."""
    scheduler, store = _scheduler(tmp_path, FrozenClock(AT))
    job = ScheduledJob("swing", "AAPL@alpaca", AT, 10, lambda: "ok")
    outcome = scheduler.run_due(job)

    assert outcome.reason_code == "COMPLETED"
    rows = tuple(store.stream(job.aggregate_id))
    assert [row.kind for row in rows] == [
        "scheduler.started",
        "scheduler.ended",
        "scheduler.healthy",
    ]
    exclusion_rows = tuple(store.stream(job.exclusion_id))
    assert [row.kind for row in exclusion_rows] == [
        "scheduler.exclusion_started",
        "scheduler.exclusion_released",
    ]
    assert [row.sequence for row in rows] == [1, 3, 4]
    assert [row.sequence for row in exclusion_rows] == [2, 5]


def test_scheduler_excludes_same_strategy_instrument_while_running(tmp_path: Path) -> None:
    """Two workers must not execute the same strategy/instrument concurrently."""
    scheduler, _ = _scheduler(tmp_path, FrozenClock(AT))
    entered = threading.Event()
    release = threading.Event()

    def blocking_run() -> None:
        entered.set()
        release.wait(timeout=5)

    job = ScheduledJob("swing", "AAPL@alpaca", AT, 10, blocking_run)
    worker = threading.Thread(target=lambda: scheduler.run_due(job))
    worker.start()
    assert entered.wait(timeout=5)
    excluded = scheduler.run_due(job)
    release.set()
    worker.join(timeout=5)

    assert excluded.reason_code == "ALREADY_RUNNING"
