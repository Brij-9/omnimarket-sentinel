"""Regression tests for the first Task 15 independent safety review."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
import zipfile
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import count
from pathlib import Path
from threading import Lock, Thread

import pytest
from typer.testing import CliRunner

import market_sentinel.operations.dashboard as dashboard_module
from market_sentinel.brokers.preflight import PreflightReport, gate, required_gate_names
from market_sentinel.cli import (
    CONFIRMATION_PHRASE,
    BoundOrderRequest,
    OrderProposal,
    ServiceContainer,
    Task14CliLiveFacade,
    build_app,
)
from market_sentinel.domain.clock import FrozenClock
from market_sentinel.domain.enums import AssetClass, OrderStatus, OrderType
from market_sentinel.domain.models import BrokerOrder, GateResult, MarketSnapshot, OrderIntent
from market_sentinel.execution.approval import ApprovalService
from market_sentinel.execution.base import BrokerCapabilities
from market_sentinel.execution.live import LiveOrderError, LiveOrderService
from market_sentinel.execution.reconcile import (
    BrokerReconciliationSnapshot,
    Reconciler,
    ReconciliationReport,
)
from market_sentinel.execution.safety import create_safety_capabilities
from market_sentinel.operations.audit import AuditLog
from market_sentinel.operations.dashboard import (
    DashboardAspiration,
    DashboardBroker,
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
from market_sentinel.portfolio.ledger import PortfolioLedger
from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventStore
from tests.factories import snapshot

AT = datetime(2026, 8, 9, 10, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[2]


class _ReadyPreflight:
    def run(self, broker: str) -> PreflightReport:
        manifest = "ccxt-spot" if broker == "ccxt" else broker
        return PreflightReport(
            manifest,
            tuple(gate(name, True) for name in sorted(required_gate_names(manifest))),
        )


class _ForgedLiveService:
    def submit(
        self,
        request: BoundOrderRequest,
        confirmation_phrase: str,
    ) -> Mapping[str, object]:
        del confirmation_phrase
        return {"intent_id": request.intent_id, "status": "ACKNOWLEDGED"}


class _MappingProposal:
    def propose(self, request: BoundOrderRequest) -> Mapping[str, object]:
        return request.safe_mapping()


class _TypedProposal:
    def propose(self, request: BoundOrderRequest) -> OrderProposal:
        return OrderProposal.accept(request)


def _order_args(command: str) -> list[str]:
    arguments = [
        command,
        "--broker",
        "alpaca",
        "--intent-id",
        "intent-1",
        "--instrument",
        "AAPL@alpaca",
        "--side",
        "buy",
        "--quantity",
        "0.1",
        "--order-type",
        "limit",
        "--limit-price",
        "100",
        "--stop-loss",
        "98",
        "--take-profit",
        "104",
        "--time-in-force",
        "day",
        "--product",
        "cash",
        "--session",
        "regular",
        "--snapshot-hash",
        "a" * 64,
        "--created-at",
        "2026-08-09T10:00:00+00:00",
        "--expires-at",
        "2026-08-09T10:01:00+00:00",
        "--risk-approved-quantity",
        "0.1",
        "--risk-approved-notional",
        "10",
        "--portfolio-hash",
        "a" * 64,
        "--risk-decided-at",
        "2026-08-09T10:00:00+00:00",
        "--risk-expires-at",
        "2026-08-09T10:01:00+00:00",
    ]
    if command == "submit-confirmed-order":
        arguments.extend(("--confirm-real-money", CONFIRMATION_PHRASE))
    return arguments


def test_structural_live_service_cannot_report_live_success() -> None:
    """Only the concrete Task 14 facade may occupy live authority in the container."""
    with pytest.raises(ValueError, match="exact Task 14 facade"):
        ServiceContainer(
            preflight=_ReadyPreflight(),
            proposal=_TypedProposal(),
            live_submission=_ForgedLiveService(),
        )


def test_mapping_proposal_cannot_turn_caller_echo_into_approval() -> None:
    """A structural Mapping response must not be rendered as an accepted proposal."""
    result = CliRunner().invoke(
        build_app(lambda: ServiceContainer(proposal=_MappingProposal())),
        _order_args("propose-order"),
    )

    assert result.exit_code == 30
    assert result.stdout.strip() == '{"error":"PROPOSAL_RESPONSE_INVALID"}'


def test_side_aware_stop_limit_order_is_validated_by_domain_model() -> None:
    """A buy stop-limit below its trigger must fail before proposal service access."""
    arguments = _order_args("propose-order")
    type_index = arguments.index("--order-type") + 1
    arguments[type_index] = "stop_limit"
    arguments.extend(("--trigger-price", "101"))

    result = CliRunner().invoke(
        build_app(lambda: ServiceContainer(proposal=_MappingProposal())), arguments
    )

    assert result.exit_code == 22
    assert result.stdout.strip() == '{"error":"ORDER_PARAMETERS_INVALID"}'


class _LocalLiveBroker:
    """Offline provider-boundary fake behind the real Task 14 composition."""

    broker_name = "alpaca"

    def __init__(self) -> None:
        self.submit_calls = 0
        self.query_calls = 0
        self.submit_error: BaseException | None = None

    def preflight(self) -> PreflightReport:
        return _ready_report()

    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            broker="alpaca",
            supported_asset_classes=frozenset({AssetClass.EQUITY}),
            supported_order_types=frozenset(OrderType),
            supports_fractional_quantity=True,
            supports_notional_orders=True,
            supports_partial_fills=True,
            supports_shorting=False,
            supports_leverage=False,
            supports_derivatives=False,
            supports_cancel=True,
            is_paper=False,
        )

    def submit(self, intent: OrderIntent, market: MarketSnapshot) -> BrokerOrder:
        del market
        self.submit_calls += 1
        if self.submit_error is not None:
            raise self.submit_error
        return BrokerOrder(
            order_id="broker-order-1",
            client_order_id=intent.intent_id,
            broker="alpaca",
            instrument_id=intent.instrument_id,
            status=OrderStatus.ACKNOWLEDGED,
            requested_quantity=intent.quantity,
            requested_notional=intent.notional,
            filled_quantity=Decimal("0"),
            average_fill_price=None,
            submitted_at=AT,
            updated_at=AT,
        )

    def get_order_by_client_id(self, client_intent_id: str) -> BrokerOrder:
        del client_intent_id
        self.query_calls += 1
        raise RuntimeError("provider credential must not escape")


class _LiveHarness:
    def __init__(self, tmp_path: Path) -> None:
        engine = create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'live-cli.db'}")
        self.clock = FrozenClock(AT)
        self.store = EventStore(engine)
        nonces = count(1)
        nonce_lock = Lock()

        def next_nonce() -> bytes:
            with nonce_lock:
                return next(nonces).to_bytes(32, "big")

        approval_capability, reconciliation_capability, self.live_capability = (
            create_safety_capabilities(
                audit_log=AuditLog(self.store, self.clock),
                key=b"offline-task-15-live-cli-key-material",
                nonce_source=next_nonce,
            )
        )
        self.ledger = PortfolioLedger(starting_cash=Decimal("100"), currency="USD")
        self.approval = ApprovalService(
            clock=self.clock,
            safety_capability=approval_capability,
        )
        self.reconciler = Reconciler(
            safety_capability=reconciliation_capability,
            clock=self.clock,
        )
        self.report = self.healthy_report()
        self.broker = _LocalLiveBroker()
        self.live = LiveOrderService(
            broker=self.broker,
            approval_service=self.approval,
            reconciler=self.reconciler,
            safety_capability=self.live_capability,
            clock=self.clock,
            ledger=self.ledger,
        )
        self.request = self.order_request("intent-1")

    def healthy_report(self) -> ReconciliationReport:
        return self.reconciler.compare(
            BrokerReconciliationSnapshot(
                broker="alpaca",
                currency="USD",
                cash=Decimal("100"),
                positions=(),
                open_orders=(),
                observed_at=self.clock.now(),
            ),
            self.ledger,
            (),
        )

    def order_request(self, intent_id: str) -> BoundOrderRequest:
        position_hash = self.ledger.position_hash()
        return BoundOrderRequest(
            broker="alpaca",
            intent_id=intent_id,
            instrument_id="AAPL@alpaca",
            side="buy",
            quantity=Decimal("0.1"),
            notional=None,
            order_type="limit",
            limit_price=Decimal("100"),
            trigger_price=None,
            stop_loss=Decimal("98"),
            take_profit=Decimal("104"),
            time_in_force="day",
            product="cash",
            session="regular",
            snapshot_hash=position_hash,
            created_at=AT,
            expires_at=AT + timedelta(minutes=1),
            risk_approved_quantity=Decimal("0.1"),
            risk_approved_notional=Decimal("10"),
            portfolio_hash=position_hash,
            risk_decided_at=AT,
            risk_expires_at=AT + timedelta(minutes=1),
        )

    def facade(
        self,
        *,
        report: ReconciliationReport | None = None,
        preflight: PreflightReport | None = None,
    ) -> Task14CliLiveFacade:
        return Task14CliLiveFacade(
            approval_service=self.approval,
            live_order_service=self.live,
            snapshot=snapshot(),
            preflight=_ready_report() if preflight is None else preflight,
            reconciliation=self.report if report is None else report,
        )


def _ready_report() -> PreflightReport:
    return PreflightReport(
        "alpaca",
        tuple(gate(name, True) for name in sorted(required_gate_names("alpaca"))),
    )


def _live_cli_args(harness: _LiveHarness, *, phrase: str = CONFIRMATION_PHRASE) -> list[str]:
    arguments = _order_args("submit-confirmed-order")
    position_hash = harness.ledger.position_hash()
    arguments[arguments.index("--snapshot-hash") + 1] = position_hash
    arguments[arguments.index("--portfolio-hash") + 1] = position_hash
    arguments[arguments.index("--confirm-real-money") + 1] = phrase
    return arguments


def _live_cli_result(
    harness: _LiveHarness,
    *,
    facade: Task14CliLiveFacade | None = None,
    phrase: str = CONFIRMATION_PHRASE,
) -> object:
    return CliRunner().invoke(
        build_app(
            lambda: ServiceContainer(
                proposal=_TypedProposal(),
                live_submission=harness.facade() if facade is None else facade,
            )
        ),
        _live_cli_args(harness, phrase=phrase),
    )


def test_real_task14_cli_facade_submits_one_exact_typed_order(tmp_path: Path) -> None:
    """The only live-success path uses the real Task 14 gate stack and typed result."""
    harness = _LiveHarness(tmp_path)

    result = _live_cli_result(harness)

    assert result.exit_code == 0
    assert harness.broker.submit_calls == 1
    assert json.loads(result.stdout)["status"] == "acknowledged"
    assert harness.broker.query_calls == 0


@pytest.mark.parametrize("gate_name", ("confirmation", "preflight", "risk", "reconciliation"))
def test_real_task14_cli_facade_rejects_each_gate_before_broker_call(
    tmp_path: Path,
    gate_name: str,
) -> None:
    """Exact phrase, preflight, risk, and reconciliation gates all fail closed."""
    harness = _LiveHarness(tmp_path)
    phrase = CONFIRMATION_PHRASE
    facade = harness.facade()
    if gate_name == "confirmation":
        phrase = "almost-confirmed"
    elif gate_name == "preflight":
        facade = harness.facade(
            preflight=PreflightReport(
                "alpaca",
                tuple(
                    gate(name, False, "NOT_READY")
                    for name in sorted(required_gate_names("alpaca"))
                ),
            )
        )
    elif gate_name == "risk":
        harness.clock.advance(timedelta(minutes=2))
    else:
        facade = harness.facade(report=replace(harness.report, report_id="forged-report"))

    result = _live_cli_result(harness, facade=facade, phrase=phrase)

    assert result.exit_code in {21, 30}
    assert harness.broker.submit_calls == 0
    assert "provider credential" not in result.stdout


@pytest.mark.parametrize("gate_name", ("kill", "interlock"))
def test_real_task14_cli_facade_blocks_persistent_safety_gates_without_broker_call(
    tmp_path: Path,
    gate_name: str,
) -> None:
    """Persistent kill-switch and unresolved-interlock state cannot reach submit."""
    harness = _LiveHarness(tmp_path)
    if gate_name == "kill":
        harness.reconciler.compare(
            BrokerReconciliationSnapshot(
                broker="alpaca",
                currency="USD",
                cash=Decimal("99"),
                positions=(),
                open_orders=(),
                observed_at=AT,
            ),
            harness.ledger,
            (),
        )
    else:
        blocker = harness.order_request("interlock-blocker")
        blocker_proposal = OrderProposal.accept(blocker)
        confirmation = harness.approval.create(
            blocker_proposal.intent,
            blocker_proposal.risk_decision,
            phrase=CONFIRMATION_PHRASE,
            broker="alpaca",
        )
        fence = harness.reconciler.safety_fence(
            harness.report,
            broker="alpaca",
            ledger=harness.ledger,
        )
        harness.live_capability.claim_and_start(
            intent_id=blocker.intent_id,
            broker="alpaca",
            confirmation_id=confirmation.confirmation_id,
            fingerprint=confirmation.fingerprint,
            expires_at=confirmation.expires_at,
            reconciliation_head=fence.reconciliation_head,
            kill_switch_head=fence.kill_switch_head,
            interlock_head=fence.interlock_head,
            occurred_at=AT,
        )
    fresh_report = harness.healthy_report()

    result = _live_cli_result(harness, facade=harness.facade(report=fresh_report))

    assert result.exit_code == 30
    assert harness.broker.submit_calls == 0


def test_unknown_outcome_persists_and_blocks_next_cli_order_without_another_submit(
    tmp_path: Path,
) -> None:
    """A real Task 14 UNKNOWN result activates durable safety before the next CLI call."""
    harness = _LiveHarness(tmp_path)
    harness.broker.submit_error = TimeoutError("api_key=must-not-leak")
    first = OrderProposal.accept(harness.request)
    with pytest.raises(LiveOrderError, match="SUBMISSION_UNKNOWN"):
        harness.facade().submit(first, CONFIRMATION_PHRASE)
    assert harness.broker.submit_calls == 1
    harness.broker.submit_error = None
    harness.request = harness.order_request("intent-2")

    result = _live_cli_result(harness, facade=harness.facade(report=harness.healthy_report()))

    assert result.exit_code == 30
    assert harness.broker.submit_calls == 1
    assert "must-not-leak" not in result.stdout


class _Calendar:
    def is_session_open(self, instrument_id: str, instant: datetime) -> bool:
        return instrument_id == "AAPL@alpaca" and instant == AT


def _scheduler(store: EventStore, ids: count[int]) -> Scheduler:
    clock = FrozenClock(AT)
    return Scheduler(
        clock=clock,
        calendar=_Calendar(),
        audit=AuditLog(store, clock),
        event_id_factory=lambda: f"event-{next(ids)}",
    )


def test_scheduler_durable_claim_executes_once_across_instances_and_repeats(
    tmp_path: Path,
) -> None:
    """A due job must be once-only in durable state, not merely within one object lock."""
    database = tmp_path / "scheduler.db"
    first_store = EventStore(create_engine_and_schema(f"sqlite+pysqlite:///{database}"))
    second_store = EventStore(create_engine_and_schema(f"sqlite+pysqlite:///{database}"))
    ids = count(1)
    first = _scheduler(first_store, ids)
    second = _scheduler(second_store, ids)
    calls: list[str] = []
    job = ScheduledJob("swing", "AAPL@alpaca", AT, 10, lambda: calls.append("ran"))

    outcomes = (first.run_due(job), second.run_due(job), first.run_due(job))

    assert calls == ["ran"]
    assert [outcome.reason_code for outcome in outcomes] == [
        "COMPLETED",
        "ALREADY_CLAIMED",
        "ALREADY_CLAIMED",
    ]


def test_scheduler_excludes_different_due_instants_for_same_strategy_instrument(
    tmp_path: Path,
) -> None:
    """One active due run must exclude a later due instant across scheduler instances."""

    class _AlwaysOpen:
        def is_session_open(self, instrument_id: str, instant: datetime) -> bool:
            return instrument_id == "AAPL@alpaca" and instant.tzinfo is UTC

    database = tmp_path / "scheduler-exclusion.db"
    ids = count(1)
    clock = FrozenClock(AT + timedelta(seconds=1))

    def scheduler() -> Scheduler:
        store = EventStore(create_engine_and_schema(f"sqlite+pysqlite:///{database}"))
        return Scheduler(
            clock=clock,
            calendar=_AlwaysOpen(),
            audit=AuditLog(store, clock),
            event_id_factory=lambda: f"exclusion-event-{next(ids)}",
        )

    first = scheduler()
    second = scheduler()
    entered = Lock()
    release = Lock()
    entered.acquire()
    release.acquire()
    calls: list[str] = []

    def blocking() -> None:
        calls.append("first")
        entered.release()
        release.acquire()
        release.release()

    first_job = ScheduledJob("swing", "AAPL@alpaca", AT, 10, blocking)
    second_job = ScheduledJob(
        "swing",
        "AAPL@alpaca",
        AT + timedelta(seconds=1),
        10,
        lambda: calls.append("second"),
    )
    worker = Thread(target=lambda: first.run_due(first_job))
    worker.start()
    assert entered.acquire(timeout=5)

    outcome = second.run_due(second_job)
    release.release()
    worker.join(timeout=5)

    assert outcome.reason_code == "ALREADY_RUNNING"
    assert calls == ["first"]


def test_scheduler_bounds_identifiers_and_scrubs_clock_failures(tmp_path: Path) -> None:
    """Hostile identities and clock exceptions must not escape or grow scheduler state."""
    with pytest.raises(ValueError, match="malformed"):
        ScheduledJob("x" * 65, "AAPL@alpaca", AT, 10, lambda: None)

    class _FailingClock:
        def now(self) -> datetime:
            raise RuntimeError("clock-secret")

    store = EventStore(create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'clock.db'}"))
    clock_ids = count(1)
    scheduler = Scheduler(
        clock=_FailingClock(),
        calendar=_Calendar(),
        audit=AuditLog(store, FrozenClock(AT)),
        event_id_factory=lambda: f"event-clock-{next(clock_ids)}",
    )
    outcome = scheduler.run_due(ScheduledJob("swing", "AAPL@alpaca", AT, 10, lambda: None))

    assert outcome.reason_code == "CLOCK_UNAVAILABLE"
    assert "clock-secret" not in str(outcome)


def test_scheduler_marks_post_callback_clock_failure_unhealthy(tmp_path: Path) -> None:
    """A clock failure after a callback must not persist a false healthy terminal state."""

    class _EndingClock:
        def __init__(self) -> None:
            self.calls = 0

        def now(self) -> datetime:
            self.calls += 1
            if self.calls > 2:
                raise RuntimeError("post-callback-clock-secret")
            return AT

    store = EventStore(create_engine_and_schema(f"sqlite+pysqlite:///{tmp_path / 'ending.db'}"))
    ids = count(1)
    job = ScheduledJob("swing", "AAPL@alpaca", AT, 10, lambda: None)
    scheduler = Scheduler(
        clock=_EndingClock(),
        calendar=_Calendar(),
        audit=AuditLog(store, FrozenClock(AT)),
        event_id_factory=lambda: f"ending-event-{next(ids)}",
    )

    outcome = scheduler.run_due(job)

    assert outcome.reason_code == "CLOCK_UNAVAILABLE"
    assert outcome.ran is True
    assert [row.kind for row in store.stream(job.aggregate_id)][-1] == "scheduler.unhealthy"


def _dashboard_status(**changes: object) -> DashboardStatus:
    values: dict[str, object] = {
        "generated_at": AT,
        "data_as_of": AT,
        "research": DashboardResearch("research-v1", True),
        "strategies": (DashboardStrategy("swing", "v1"),),
        "promotion": DashboardPromotion("paper"),
        "portfolio": DashboardPortfolio("USD", Decimal("10")),
        "risk": DashboardRisk(
            Decimal("0.005"),
            Decimal("0.10"),
            Decimal("0.50"),
            Decimal("0.02"),
            Decimal("0.10"),
        ),
        "brokers": (
            DashboardBroker(
                "alpaca",
                tuple(
                    GateResult(name=name, passed=True, reason_code="OK")
                    for name in sorted(required_gate_names("alpaca"))
                ),
            ),
        ),
        "orders": (),
        "kill_switches": (DashboardSafetyState(False, "OK"),),
        "interlocks": (DashboardSafetyState(False, "OK"),),
        "aspirational_target": DashboardAspiration(
            Decimal("10"),
            Decimal("10"),
            Decimal("1000000"),
            Decimal("100000"),
            Decimal("1"),
            Decimal("999990"),
            True,
        ),
    }
    values.update(changes)
    return DashboardStatus(**values)  # type: ignore[arg-type]


def test_dashboard_requires_nonempty_explicit_safety_sections_and_arithmetic(
    tmp_path: Path,
) -> None:
    """Missing gates/switches or inconsistent aspiration arithmetic must fail closed."""
    with pytest.raises(DashboardValidationError):
        export_dashboard(
            _dashboard_status(brokers=(), kill_switches=(), interlocks=()),
            tmp_path / "empty.json",
        )
    with pytest.raises(DashboardValidationError):
        _dashboard_status(
            aspirational_target=DashboardAspiration(
                Decimal("10"),
                Decimal("10"),
                Decimal("1000000"),
                Decimal("2"),
                Decimal("1"),
                Decimal("999990"),
                True,
            )
        )


def test_dashboard_drops_camelcase_access_private_and_pem_secrets(tmp_path: Path) -> None:
    """Common credential key/value spellings must never survive schema conversion."""
    prepared = safe_json_mapping(
        {
            "accessKey": "ACCESS-VALUE",
            "privateKey": "PRIVATE-VALUE",
            "note": "-----BEGIN PRIVATE KEY----- secret -----END PRIVATE KEY-----",
            "first": "api_key=secret-token-123",
            "second": "password=hunter2",
            "third": "credential=live-value",
        }
    )
    text = json.dumps(prepared)
    for forbidden in (
        "accessKey",
        "ACCESS-VALUE",
        "privateKey",
        "PRIVATE-VALUE",
        "BEGIN PRIVATE KEY",
        "secret-token-123",
        "hunter2",
        "live-value",
    ):
        assert forbidden not in text


def test_dashboard_rejects_parent_symlink_and_preserves_target(tmp_path: Path) -> None:
    """A destination beneath a linked parent must never be opened or replaced."""
    real_parent = tmp_path / "real"
    linked_parent = tmp_path / "linked"
    real_parent.mkdir()
    try:
        os.symlink(real_parent, linked_parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    destination = linked_parent / "status.json"

    with pytest.raises(DashboardValidationError, match="DASHBOARD_PATH_INVALID"):
        export_dashboard(_dashboard_status(), destination)

    assert not (real_parent / "status.json").exists()


def test_dashboard_detects_parent_identity_race_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parent identity change after temp creation preserves the previous dashboard."""
    destination = tmp_path / "status.json"
    destination.write_text("old", encoding="utf-8")
    original_identity = dashboard_module._identity_tuple
    calls = 0

    def changing_identity(details: os.stat_result) -> tuple[int, int, int]:
        nonlocal calls
        calls += 1
        value = original_identity(details)
        return value if calls == 1 else (value[0], value[1] + 1, value[2])

    monkeypatch.setattr(dashboard_module, "_identity_tuple", changing_identity)
    with pytest.raises(DashboardValidationError, match="DASHBOARD_WRITE_FAILED"):
        export_dashboard(_dashboard_status(), destination)

    assert destination.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".status.json.*.tmp")) == []


def test_dashboard_blocks_parent_swap_inside_final_replace_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The validated parent must not be replaceable after the final path check."""
    parent = tmp_path / "dashboard"
    parked = tmp_path / "parked"
    parent.mkdir()
    destination = parent / "status.json"
    destination.write_text("old", encoding="utf-8")
    original_replace = os.replace

    def swap_parent_then_replace(
        source: str | os.PathLike[str],
        target: str | os.PathLike[str],
    ) -> None:
        source_path = Path(source)
        original_replace(parent, parked)
        parent.mkdir()
        original_replace(parked / source_path.name, parent / source_path.name)
        original_replace(parent / source_path.name, target)

    monkeypatch.setattr(os, "replace", swap_parent_then_replace)

    with pytest.raises(DashboardValidationError, match="DASHBOARD_WRITE_FAILED"):
        export_dashboard(_dashboard_status(), destination)

    assert destination.read_text(encoding="utf-8") == "old"
    assert not parked.exists()


def test_dashboard_rejects_mapped_network_destination_before_temp_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A network-classified drive must fail before any dashboard file is created."""
    monkeypatch.setattr(dashboard_module, "_is_network_path", lambda path: True)

    with pytest.raises(DashboardValidationError, match="DASHBOARD_PATH_INVALID"):
        export_dashboard(_dashboard_status(), tmp_path / "status.json")

    assert list(tmp_path.iterdir()) == []


def test_built_archives_include_both_powershell_wrappers(tmp_path: Path) -> None:
    """Installed artifacts must carry the local operational helpers."""
    output = tmp_path / "dist"
    result = subprocess.run(
        [
            os.sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--skip-dependency-check",
            "--outdir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    wheel = next(output.glob("*.whl"))
    sdist = next(output.glob("*.tar.gz"))
    expected = {"check-readiness.ps1", "export-dashboard-status.ps1"}
    with zipfile.ZipFile(wheel) as archive:
        assert expected <= {Path(name).name for name in archive.namelist()}
    with tarfile.open(sdist, "r:gz") as archive:
        assert expected <= {Path(name).name for name in archive.getnames()}
