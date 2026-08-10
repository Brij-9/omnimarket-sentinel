from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import NamedTuple
from unittest.mock import patch

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from market_sentinel.brokers.preflight import PreflightReport
from market_sentinel.domain.clock import FrozenClock
from market_sentinel.domain.enums import OrderStatus
from market_sentinel.domain.models import (
    BrokerOrder,
    GateResult,
    MarketSnapshot,
    OrderIntent,
    RiskDecision,
)
from market_sentinel.execution.approval import (
    CONFIRMATION_PHRASE,
    ApprovalService,
    OrderConfirmation,
)
from market_sentinel.execution.live import LiveOrderError, LiveOrderService
from market_sentinel.execution.reconcile import (
    BrokerReconciliationSnapshot,
    Reconciler,
    ReconciliationReport,
    SafetyFence,
)
from market_sentinel.operations.audit import AuditLog
from market_sentinel.portfolio.ledger import PortfolioLedger
from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventStore
from tests.factories import DEFAULT_INSTANT, intent, risk_decision, snapshot


class LocalBroker:
    """No-network broker double; only the provider boundary is replaced."""

    broker_name = "alpaca"

    def __init__(
        self,
        *,
        submit_error: BaseException | None = None,
        found: bool = False,
        malformed_ack: bool = False,
        submit_response: BrokerOrder | None = None,
    ) -> None:
        self.submit_error = submit_error
        self.found = found
        self.malformed_ack = malformed_ack
        self.submit_response = submit_response
        self.submit_calls = 0
        self.query_calls = 0
        self.preflight_calls = 0
        self.queried_ids: list[str] = []
        self.advance_clock: FrozenClock | None = None
        self._lock = Lock()

    def preflight(self) -> PreflightReport:
        self.preflight_calls += 1
        return PreflightReport(
            broker="alpaca",
            gates=(GateResult(name="LOCAL", passed=True, reason_code="OK"),),
        )

    def submit(self, order_intent: OrderIntent, market: MarketSnapshot) -> BrokerOrder:
        del market
        with self._lock:
            self.submit_calls += 1
        if self.submit_error is not None:
            raise self.submit_error
        if self.advance_clock is not None:
            self.advance_clock.advance(timedelta(seconds=1))
            advanced = self._order(order_intent.intent_id).model_copy(
                update={
                    "submitted_at": self.advance_clock.now(),
                    "updated_at": self.advance_clock.now(),
                }
            )
            return advanced
        if self.submit_response is not None:
            return self.submit_response
        order = self._order(order_intent.intent_id)
        if self.malformed_ack:
            return order.model_copy(update={"order_id": ""})
        return order

    def get_order_by_client_id(self, client_intent_id: str) -> BrokerOrder:
        with self._lock:
            self.query_calls += 1
            self.queried_ids.append(client_intent_id)
        if not self.found:
            raise RuntimeError("access_token=do-not-leak")
        return self._order(client_intent_id)

    @staticmethod
    def _order(client_id: str) -> BrokerOrder:
        return BrokerOrder(
            order_id="broker-order-1",
            client_order_id=client_id,
            broker="alpaca",
            instrument_id="AAPL@alpaca",
            status=OrderStatus.ACKNOWLEDGED,
            requested_quantity=Decimal("0.1"),
            filled_quantity=Decimal("0"),
            average_fill_price=None,
            submitted_at=DEFAULT_INSTANT,
            updated_at=DEFAULT_INSTANT,
        )


class Setup(NamedTuple):
    engine: Engine
    clock: FrozenClock
    store: EventStore
    reconciler: Reconciler
    service: LiveOrderService
    broker: LocalBroker
    intent: OrderIntent
    risk: RiskDecision
    confirmation: OrderConfirmation
    report: ReconciliationReport
    ready: PreflightReport


def _setup(path: Path | None = None, *, broker: LocalBroker | None = None) -> Setup:
    url = "sqlite+pysqlite:///:memory:" if path is None else f"sqlite+pysqlite:///{path}"
    engine = create_engine_and_schema(url)
    clock = FrozenClock(DEFAULT_INSTANT)
    store = EventStore(engine)
    audit = AuditLog(store, clock)
    ledger = PortfolioLedger(starting_cash=Decimal("100"), currency="USD")
    reconciler = Reconciler(audit_log=audit, clock=clock)
    report = reconciler.compare(
        BrokerReconciliationSnapshot(
            broker="alpaca",
            currency="USD",
            cash=Decimal("100"),
            positions=(),
            open_orders=(),
            observed_at=clock.now(),
        ),
        ledger,
        (),
    )
    order_intent = intent(
        quantity="0.1",
        notional=None,
        limit_price="100",
        stop_loss="98",
        take_profit="104",
        snapshot_hash=ledger.position_hash(),
    )
    risk = risk_decision(
        approved=True,
        reason_codes=(),
        approved_quantity="0.1",
        approved_notional="10",
        portfolio_hash=ledger.position_hash(),
        expires_at=DEFAULT_INSTANT + timedelta(minutes=1),
    )
    approval = ApprovalService(clock=clock)
    confirmation = approval.create(
        order_intent,
        risk,
        phrase=CONFIRMATION_PHRASE,
        broker="alpaca",
    )
    local_broker = broker or LocalBroker()
    service = LiveOrderService(
        broker=local_broker,
        approval_service=approval,
        reconciler=reconciler,
        audit_log=audit,
        clock=clock,
        ledger=ledger,
    )
    ready = PreflightReport(
        broker="alpaca",
        gates=(GateResult(name="LOCAL", passed=True, reason_code="OK"),),
    )
    return Setup(
        engine,
        clock,
        store,
        reconciler,
        service,
        local_broker,
        order_intent,
        risk,
        confirmation,
        report,
        ready,
    )


def _submit(
    service: LiveOrderService,
    order_intent: OrderIntent,
    risk: RiskDecision,
    confirmation: OrderConfirmation,
    report: ReconciliationReport,
    ready: PreflightReport,
) -> BrokerOrder:
    return service.submit_confirmed(
        intent=order_intent,
        risk_decision=risk,
        snapshot=snapshot(instrument_id=order_intent.instrument_id),
        confirmation=confirmation,
        preflight=ready,
        reconciliation=report,
    )


def test_submit_persists_claim_and_started_before_one_broker_call() -> None:
    """A successful path must leave durable claim/start evidence before one acknowledgement."""
    _engine, _clock, store, _r, service, broker, order_intent, risk, confirmation, report, ready = (
        _setup()
    )
    order = _submit(service, order_intent, risk, confirmation, report, ready)

    assert order.status is OrderStatus.ACKNOWLEDGED
    assert broker.submit_calls == 1
    assert broker.preflight_calls == 1
    kinds = [row.kind for row in store.stream(order_intent.intent_id)]
    assert kinds == ["live.confirmation_claimed", "live.submission_started", "live.acknowledged"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("preflight", "PREFLIGHT_NOT_READY"),
        ("risk_rejected", "RISK_NOT_APPROVED"),
        ("risk_stale", "RISK_STALE"),
        ("reconciliation", "RECONCILIATION_NOT_CURRENT"),
        ("confirmation", "CONFIRMATION_INVALID"),
        ("snapshot", "SNAPSHOT_INVALID"),
    ],
)
def test_each_missing_or_changed_gate_makes_zero_broker_calls(mutation: str, reason: str) -> None:
    """Removing any live gate must not reach the injected broker boundary."""
    _engine, clock, _store, _r, service, broker, order_intent, risk, confirmation, report, ready = (
        _setup()
    )
    market = snapshot(instrument_id=order_intent.instrument_id)
    if mutation == "preflight":
        ready = PreflightReport(
            "alpaca", (GateResult(name="LOCAL", passed=False, reason_code="NO"),)
        )
    elif mutation == "risk_rejected":
        risk = risk.model_copy(update={"approved": False, "reason_codes": ("NO",)})
    elif mutation == "risk_stale":
        clock.advance(timedelta(minutes=2))
    elif mutation == "reconciliation":
        report = report.__class__(
            report_id="forged",
            broker="alpaca",
            healthy=True,
            reason_codes=(),
            broker_hash=report.broker_hash,
            ledger_hash=report.ledger_hash,
            checked_at=report.checked_at,
            sequence=report.sequence,
        )
    elif mutation == "confirmation":
        confirmation = replace(confirmation, fingerprint="f" * 64)
    else:
        market = snapshot(instrument_id="MSFT@alpaca")

    with pytest.raises(LiveOrderError, match=reason):
        service.submit_confirmed(
            intent=order_intent,
            risk_decision=risk,
            snapshot=market,
            confirmation=confirmation,
            preflight=ready,
            reconciliation=report,
        )
    assert broker.submit_calls == 0


def test_claim_is_single_use_across_restart(tmp_path: Path) -> None:
    """Rebuilding every service object over the same SQLite store must not revive a claim."""
    db = tmp_path / "restart.db"
    (
        _engine,
        _clock,
        _store,
        _r,
        service,
        broker,
        order_intent,
        risk,
        confirmation,
        report,
        ready,
    ) = _setup(db)
    _submit(service, order_intent, risk, confirmation, report, ready)

    engine2 = create_engine_and_schema(f"sqlite+pysqlite:///{db}")
    clock2 = FrozenClock(DEFAULT_INSTANT)
    store2 = EventStore(engine2)
    audit2 = AuditLog(store2, clock2)
    ledger2 = PortfolioLedger(starting_cash=Decimal("100"), currency="USD")
    reconciler2 = Reconciler(audit_log=audit2, clock=clock2)
    restarted = LiveOrderService(
        broker=broker,
        approval_service=ApprovalService(clock=clock2),
        reconciler=reconciler2,
        audit_log=audit2,
        clock=clock2,
        ledger=ledger2,
    )
    with pytest.raises(LiveOrderError, match="CONFIRMATION_USED"):
        _submit(restarted, order_intent, risk, confirmation, report, ready)
    assert broker.submit_calls == 1


def test_concurrent_double_claim_allows_at_most_one_submit(tmp_path: Path) -> None:
    """The database uniqueness claim is the arbiter across independent service instances."""
    db = tmp_path / "concurrent.db"
    broker = LocalBroker()
    first = _setup(db, broker=broker)
    engine2 = create_engine_and_schema(f"sqlite+pysqlite:///{db}")
    clock2 = FrozenClock(DEFAULT_INSTANT)
    store2 = EventStore(engine2)
    audit2 = AuditLog(store2, clock2)
    ledger2 = PortfolioLedger(starting_cash=Decimal("100"), currency="USD")
    reconciler2 = Reconciler(audit_log=audit2, clock=clock2)
    service2 = LiveOrderService(
        broker=broker,
        approval_service=ApprovalService(clock=clock2),
        reconciler=reconciler2,
        audit_log=audit2,
        clock=clock2,
        ledger=ledger2,
    )
    _, _, _, _, service1, _, order_intent, risk, confirmation, report, ready = first

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_submit, service, order_intent, risk, confirmation, report, ready)
            for service in (service1, service2)
        ]
        outcomes: list[BrokerOrder | None] = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except LiveOrderError:
                outcomes.append(None)
    assert sum(item is not None for item in outcomes) == 1
    assert broker.submit_calls == 1


def test_audit_persistence_failure_makes_zero_broker_calls() -> None:
    """The claim/start transaction must commit before broker.submit can be invoked."""
    engine, _clock, _store, _r, service, broker, order_intent, risk, confirmation, report, ready = (
        _setup()
    )

    def fail_write(
        connection: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: object,
    ) -> None:
        del connection, cursor, parameters, context, executemany
        if statement.lstrip().upper().startswith(("INSERT", "UPDATE")):
            raise RuntimeError("database unavailable")

    event.listen(engine, "before_cursor_execute", fail_write)
    try:
        with pytest.raises(LiveOrderError, match="AUDIT_PERSISTENCE_FAILED"):
            _submit(service, order_intent, risk, confirmation, report, ready)
    finally:
        event.remove(engine, "before_cursor_execute", fail_write)
    assert broker.submit_calls == 0


def test_timeout_found_by_exact_client_id_is_acknowledged_without_resubmit() -> None:
    """An ambiguous submit may query exactly once but may never submit again."""
    broker = LocalBroker(submit_error=TimeoutError("secret provider text"), found=True)
    data = _setup(broker=broker)
    _engine, _clock, store, _r, service, _b, order_intent, risk, confirmation, report, ready = data
    order = _submit(service, order_intent, risk, confirmation, report, ready)

    assert order.client_order_id == order_intent.intent_id
    assert broker.submit_calls == 1
    assert broker.query_calls == 1
    assert broker.queried_ids == [order_intent.intent_id]
    assert [event.kind for event in store.stream(order_intent.intent_id)][-1] == "live.acknowledged"


def test_timeout_not_found_persists_unknown_and_kill_switch_without_secret_text() -> None:
    """An unresolved provider outcome must be durable UNKNOWN and stop all later submission."""
    broker = LocalBroker(submit_error=TimeoutError("api_key=secret"), found=False)
    data = _setup(broker=broker)
    (
        _engine,
        _clock,
        store,
        reconciler,
        service,
        _b,
        order_intent,
        risk,
        confirmation,
        report,
        ready,
    ) = data
    with pytest.raises(LiveOrderError, match="SUBMISSION_UNKNOWN") as captured:
        _submit(service, order_intent, risk, confirmation, report, ready)

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "api_key" not in str(captured.value)
    assert broker.submit_calls == 1
    assert broker.query_calls == 1
    assert reconciler.kill_switch_active() is True
    assert [event.kind for event in store.stream(order_intent.intent_id)][-1] == "live.unknown"


def test_persistent_kill_switch_blocks_even_after_a_new_healthy_report() -> None:
    """A healthy comparison cannot clear a prior activation as a submission side effect."""
    data = _setup()
    (
        _engine,
        _clock,
        _store,
        reconciler,
        service,
        broker,
        order_intent,
        risk,
        confirmation,
        _report,
        ready,
    ) = data
    ledger = PortfolioLedger(starting_cash=Decimal("100"), currency="USD")
    reconciler.compare(
        BrokerReconciliationSnapshot(
            broker="alpaca",
            currency="USD",
            cash=Decimal("99"),
            positions=(),
            open_orders=(),
            observed_at=DEFAULT_INSTANT,
        ),
        ledger,
        (),
    )
    healthy = reconciler.compare(
        BrokerReconciliationSnapshot(
            broker="alpaca",
            currency="USD",
            cash=Decimal("100"),
            positions=(),
            open_orders=(),
            observed_at=DEFAULT_INSTANT,
        ),
        ledger,
        (),
    )

    with pytest.raises(LiveOrderError, match="KILL_SWITCH_ACTIVE"):
        _submit(service, order_intent, risk, confirmation, healthy, ready)
    assert broker.submit_calls == 0


def test_report_from_economically_equal_different_currency_ledger_is_rejected() -> None:
    """Ledger provenance must bind currency, not only cash and position economics."""
    data = _setup()
    (
        _engine,
        _clock,
        _store,
        reconciler,
        service,
        broker,
        order_intent,
        risk,
        confirmation,
        _report,
        ready,
    ) = data
    euro_ledger = PortfolioLedger(starting_cash=Decimal("100"), currency="EUR")
    euro_report = reconciler.compare(
        BrokerReconciliationSnapshot(
            broker="alpaca",
            currency="EUR",
            cash=Decimal("100"),
            positions=(),
            open_orders=(),
            observed_at=DEFAULT_INSTANT,
        ),
        euro_ledger,
        (),
    )

    with pytest.raises(LiveOrderError, match="RECONCILIATION_NOT_CURRENT"):
        _submit(service, order_intent, risk, confirmation, euro_report, ready)
    assert broker.submit_calls == 0


def test_empty_preflight_is_rejected_even_when_caller_claims_ready() -> None:
    """Vacuous readiness cannot bypass a freshly obtained broker preflight."""
    data = _setup()
    (
        _engine,
        _clock,
        _store,
        _r,
        service,
        broker,
        order_intent,
        risk,
        confirmation,
        report,
        _ready,
    ) = data
    empty = PreflightReport(broker="alpaca", gates=())
    with pytest.raises(LiveOrderError, match="PREFLIGHT_NOT_READY"):
        _submit(service, order_intent, risk, confirmation, report, empty)
    assert broker.submit_calls == 0


def test_risk_freshness_is_measured_after_fresh_preflight_completes() -> None:
    """A slow preflight cannot preserve a risk decision that expired while it ran."""
    data = _setup()
    _engine, clock, _store, _r, service, broker, order_intent, risk, confirmation, report, ready = (
        data
    )
    original_preflight = broker.preflight

    def slow_preflight() -> PreflightReport:
        clock.advance(timedelta(minutes=2))
        return original_preflight()

    with (
        patch.object(broker, "preflight", side_effect=slow_preflight),
        pytest.raises(LiveOrderError, match="RISK_STALE"),
    ):
        _submit(service, order_intent, risk, confirmation, report, ready)
    assert broker.submit_calls == 0


def test_malformed_acknowledgement_queries_once_then_persists_unknown() -> None:
    """An incomplete provider order cannot be treated as a final acknowledgement."""
    broker = LocalBroker(malformed_ack=True, found=False)
    data = _setup(broker=broker)
    (
        _engine,
        _clock,
        _store,
        reconciler,
        service,
        _b,
        order_intent,
        risk,
        confirmation,
        report,
        ready,
    ) = data
    with pytest.raises(LiveOrderError, match="SUBMISSION_UNKNOWN"):
        _submit(service, order_intent, risk, confirmation, report, ready)
    assert broker.submit_calls == 1
    assert broker.query_calls == 1
    assert reconciler.kill_switch_active() is True


def test_impossible_filled_status_is_not_accepted_as_terminal_acknowledgement() -> None:
    """FILLED with zero fill quantity and no average price must remain ambiguous."""
    impossible = LocalBroker._order("intent-1").model_copy(update={"status": OrderStatus.FILLED})
    broker = LocalBroker(submit_response=impossible, found=False)
    data = _setup(broker=broker)
    (
        _engine,
        _clock,
        _store,
        reconciler,
        service,
        _b,
        order_intent,
        risk,
        confirmation,
        report,
        ready,
    ) = data

    with pytest.raises(LiveOrderError, match="SUBMISSION_UNKNOWN"):
        _submit(service, order_intent, risk, confirmation, report, ready)
    assert broker.query_calls == 1
    assert reconciler.kill_switch_active() is True


def test_acknowledgement_uses_clock_after_broker_submit_returns() -> None:
    """A normal provider timestamp after call start must not be misclassified as future."""
    broker = LocalBroker(found=False)
    data = _setup(broker=broker)
    _engine, clock, _store, _r, service, _b, order_intent, risk, confirmation, report, ready = data
    broker.advance_clock = clock

    order = _submit(service, order_intent, risk, confirmation, report, ready)

    assert order.status is OrderStatus.ACKNOWLEDGED
    assert broker.submit_calls == 1
    assert broker.query_calls == 0


def test_failed_unknown_persistence_leaves_restart_interlock_active(tmp_path: Path) -> None:
    """A crash-equivalent after started cannot permit unrelated live orders after restart."""
    db = tmp_path / "unknown-persistence.db"
    broker = LocalBroker(submit_error=TimeoutError("provider"), found=False)
    data = _setup(db, broker=broker)
    engine, clock, _store, _r, service, _b, order_intent, risk, confirmation, report, ready = data
    writes = 0

    def fail_after_claim(
        connection: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: object,
    ) -> None:
        nonlocal writes
        del connection, cursor, parameters, context, executemany
        if statement.lstrip().upper().startswith("INSERT"):
            writes += 1
            if writes > 1:
                raise RuntimeError("database unavailable")

    event.listen(engine, "before_cursor_execute", fail_after_claim)
    try:
        with pytest.raises(LiveOrderError, match="UNKNOWN_PERSISTENCE_FAILED"):
            _submit(service, order_intent, risk, confirmation, report, ready)
    finally:
        event.remove(engine, "before_cursor_execute", fail_after_claim)

    restarted_store = EventStore(create_engine_and_schema(f"sqlite+pysqlite:///{db}"))
    restarted = Reconciler(audit_log=AuditLog(restarted_store, clock), clock=clock)
    assert restarted.kill_switch_active() is True


def test_unhealthy_race_between_gate_read_and_claim_conflicts_before_submit(
    tmp_path: Path,
) -> None:
    """A later unhealthy head that wins the write lock must invalidate the stale safety fence."""
    db = tmp_path / "fence-race.db"
    data = _setup(db)
    (
        _engine,
        clock,
        _store,
        reconciler,
        service,
        broker,
        order_intent,
        risk,
        confirmation,
        report,
        ready,
    ) = data
    original_fence = reconciler.safety_fence
    second_store = EventStore(create_engine_and_schema(f"sqlite+pysqlite:///{db}"))
    second = Reconciler(audit_log=AuditLog(second_store, clock), clock=clock)

    def stale_fence(
        candidate: object,
        *,
        broker: str,
        ledger: PortfolioLedger,
    ) -> SafetyFence:
        fence = original_fence(candidate, broker=broker, ledger=ledger)
        second.compare(
            BrokerReconciliationSnapshot(
                broker="alpaca",
                currency="USD",
                cash=Decimal("99"),
                positions=(),
                open_orders=(),
                observed_at=DEFAULT_INSTANT,
            ),
            PortfolioLedger(starting_cash=Decimal("100"), currency="USD"),
            (),
        )
        return fence

    with (
        patch.object(reconciler, "safety_fence", side_effect=stale_fence),
        pytest.raises(LiveOrderError, match="SAFETY_STATE_CHANGED"),
    ):
        _submit(service, order_intent, risk, confirmation, report, ready)
    assert broker.submit_calls == 0
