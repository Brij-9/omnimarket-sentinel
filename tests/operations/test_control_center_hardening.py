"""Regression tests for the first Task 15 independent safety review."""

from __future__ import annotations

import copy
import gc
import json
import os
import pickle
import re
import subprocess
import tarfile
import urllib.parse as url_parse_module
import zipfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import count
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from types import MappingProxyType

import pytest
import typer
from sqlalchemy import func, select
from typer.testing import CliRunner

import market_sentinel.cli as cli_module
import market_sentinel.execution.live as live_module
import market_sentinel.operations.dashboard as dashboard_module
import market_sentinel.security as security_module
from market_sentinel.brokers.preflight import PreflightReport, gate, required_gate_names
from market_sentinel.cli import (
    CONFIRMATION_PHRASE,
    BoundOrderRequest,
    OrderProposal,
    ServiceContainer,
    Task14CliLiveFacade,
    build_app,
    create_task14_cli_live_facade,
)
from market_sentinel.domain.clock import FrozenClock
from market_sentinel.domain.enums import AssetClass, OrderStatus, OrderType
from market_sentinel.domain.models import (
    BrokerOrder,
    GateResult,
    MarketSnapshot,
    OrderIntent,
    RiskDecision,
)
from market_sentinel.execution.approval import ApprovalService
from market_sentinel.execution.base import BrokerCapabilities
from market_sentinel.execution.live import LiveOrderService
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
from market_sentinel.storage.db import create_engine_and_schema, events
from market_sentinel.storage.events import EventStore
from tests.factories import snapshot

AT = datetime(2026, 8, 9, 10, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[2]


def _require_posix_otmpfile(path: Path) -> None:
    """Skip integration only when the selected POSIX filesystem lacks O_TMPFILE."""
    flag = getattr(os, "O_TMPFILE", 0)
    if not flag:
        pytest.skip("POSIX O_TMPFILE is unavailable")
    try:
        descriptor = os.open(path, os.O_RDWR | flag, 0o600)
    except OSError:
        pytest.skip("tmp_path filesystem does not support O_TMPFILE")
    else:
        os.close(descriptor)


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


def test_unregistered_exact_live_facade_cannot_be_shadowed_or_substituted(
    tmp_path: Path,
) -> None:
    """An exact-class object.__new__ forgery must remain inert and container-ineligible."""
    harness = _LiveHarness(tmp_path)
    events_before = harness.event_count()
    forged = object.__new__(Task14CliLiveFacade)
    with pytest.raises(AttributeError):
        object.__setattr__(
            forged,
            "submit",
            lambda proposal, phrase: harness.broker.submit(proposal.intent, snapshot()),
        )

    result = _live_cli_result(harness, facade=forged)

    assert result.exit_code != 0
    assert harness.broker.submit_calls == 0
    assert harness.event_count() == events_before
    assert not hasattr(forged, "__dict__")


def test_live_container_revalidates_registry_after_frozen_state_tampering(
    tmp_path: Path,
) -> None:
    """Retrieval must reject an exact-class forgery substituted after construction."""
    harness = _LiveHarness(tmp_path)
    events_before = harness.event_count()
    container = ServiceContainer(
        proposal=_TypedProposal(), live_submission=harness.facade()
    )
    forged = object.__new__(Task14CliLiveFacade)
    object.__setattr__(container, "live_submission", forged)

    result = CliRunner().invoke(
        build_app(lambda: container),
        _live_cli_args(harness),
    )

    assert result.exit_code != 0
    assert harness.broker.submit_calls == 0
    assert harness.event_count() == events_before


def test_public_live_facade_constructor_is_unregistered_zero_state() -> None:
    """Direct construction must create only an immutable unregistered inert identity."""
    facade = Task14CliLiveFacade()

    assert not hasattr(facade, "__dict__")
    with pytest.raises(AttributeError):
        facade.submit = lambda proposal, phrase: None  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="registered"):
        ServiceContainer(live_submission=facade)


def test_genuine_live_facade_rejects_copy_deepcopy_and_pickle(tmp_path: Path) -> None:
    """Opaque live authority cannot be cloned or serialized into another identity."""
    facade = _LiveHarness(tmp_path).facade()

    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(facade)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(facade)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(facade)


def test_live_facade_factory_rejects_cross_store_reuse_and_gc_is_independent(
    tmp_path: Path,
) -> None:
    """Bindings require one safety store and one handle GC must not revoke another."""
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    first = _LiveHarness(first_path)
    second = _LiveHarness(second_path)
    factory = getattr(cli_module, "create_task14_cli_live_facade", None)
    assert callable(factory)
    with pytest.raises(ValueError, match="shared Task 14"):
        factory(
            approval_service=first.approval,
            live_order_service=second.live,
            snapshot=snapshot(),
            preflight=_ready_report(),
            reconciliation=second.report,
        )
    alternate_approval = ApprovalService(
        clock=first.clock,
        safety_capability=first.approval_capability,
    )
    with pytest.raises(ValueError, match="shared Task 14"):
        factory(
            approval_service=alternate_approval,
            live_order_service=first.live,
            snapshot=snapshot(),
            preflight=_ready_report(),
            reconciliation=first.report,
        )
    survivor = factory(
        approval_service=first.approval,
        live_order_service=first.live,
        snapshot=snapshot(),
        preflight=_ready_report(),
        reconciliation=first.report,
    )
    discarded = factory(
        approval_service=first.approval,
        live_order_service=first.live,
        snapshot=snapshot(),
        preflight=_ready_report(),
        reconciliation=first.report,
    )
    del discarded
    gc.collect()

    ServiceContainer(live_submission=survivor)


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

    def __init__(self, broker_name: str = "alpaca") -> None:
        self.broker_name = broker_name
        self.submit_calls = 0
        self.query_calls = 0
        self.submit_error: BaseException | None = None
        self.order_id = "broker-order-1"
        self.last_order: BrokerOrder | None = None
        self.response_mutation: tuple[str, str] | None = None
        self.submit_hook: Callable[[], None] | None = None
        self._post_submit_clock_reads = 0

    def preflight(self) -> PreflightReport:
        return _ready_report(self.broker_name)

    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            broker=self.broker_name,
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
        if self.submit_hook is not None:
            self.submit_hook()
        self.last_order = BrokerOrder(
            order_id=self.order_id,
            client_order_id=intent.intent_id,
            broker=self.broker_name,
            instrument_id=intent.instrument_id,
            status=OrderStatus.ACKNOWLEDGED,
            requested_quantity=intent.quantity,
            requested_notional=intent.notional,
            filled_quantity=Decimal("0"),
            average_fill_price=None,
            submitted_at=AT,
            updated_at=AT,
        )
        return self.last_order

    def mutate_response_after_validation(self, field: str, value: str) -> None:
        self.response_mutation = (field, value)

    def on_clock_read(self) -> None:
        if self.last_order is None or self.response_mutation is None:
            return
        self._post_submit_clock_reads += 1
        if self._post_submit_clock_reads == 2:
            field, value = self.response_mutation
            object.__setattr__(self.last_order, field, value)

    def get_order_by_client_id(self, client_intent_id: str) -> BrokerOrder:
        del client_intent_id
        self.query_calls += 1
        raise RuntimeError("provider credential must not escape")


class _LiveClock(FrozenClock):
    def __init__(self, instant: datetime, broker: _LocalLiveBroker) -> None:
        super().__init__(instant)
        self._broker = broker

    def now(self) -> datetime:
        self._broker.on_clock_read()
        return super().now()


class _LiveHarness:
    def __init__(
        self,
        tmp_path: Path,
        *,
        broker_name: str = "alpaca",
        external_broker: str = "alpaca",
        instrument_id: str = "AAPL@alpaca",
    ) -> None:
        self.engine = create_engine_and_schema(
            f"sqlite+pysqlite:///{tmp_path / 'live-cli.db'}"
        )
        self.broker_name = broker_name
        self.external_broker = external_broker
        self.instrument_id = instrument_id
        self.broker = _LocalLiveBroker(broker_name)
        self.clock = _LiveClock(AT, self.broker)
        self.store = EventStore(self.engine)
        nonces = count(1)
        nonce_lock = Lock()

        def next_nonce() -> bytes:
            with nonce_lock:
                return next(nonces).to_bytes(32, "big")

        self.approval_capability, reconciliation_capability, self.live_capability = (
            create_safety_capabilities(
                audit_log=AuditLog(self.store, self.clock),
                key=b"offline-task-15-live-cli-key-material",
                nonce_source=next_nonce,
            )
        )
        self.ledger = PortfolioLedger(starting_cash=Decimal("100"), currency="USD")
        self.approval = ApprovalService(
            clock=self.clock,
            safety_capability=self.approval_capability,
        )
        self.reconciler = Reconciler(
            safety_capability=reconciliation_capability,
            clock=self.clock,
        )
        self.report = self.healthy_report()
        self.live = LiveOrderService(
            broker=self.broker,
            approval_service=self.approval,
            reconciler=self.reconciler,
            safety_capability=self.live_capability,
            clock=self.clock,
            ledger=self.ledger,
        )
        self.request = self.order_request("intent-1")

    def event_count(self) -> int:
        with self.engine.connect() as connection:
            return int(connection.scalar(select(func.count()).select_from(events)) or 0)

    def audit_rows(self) -> tuple[tuple[str, str], ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(events.c.kind, events.c.payload_json).order_by(events.c.sequence)
            )
            return tuple((str(row.kind), str(row.payload_json)) for row in rows)

    def healthy_report(self) -> ReconciliationReport:
        return self.reconciler.compare(
            BrokerReconciliationSnapshot(
                broker=self.broker_name,
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
            broker=self.external_broker,
            intent_id=intent_id,
            instrument_id=self.instrument_id,
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
        return create_task14_cli_live_facade(
            approval_service=self.approval,
            live_order_service=self.live,
            snapshot=snapshot(instrument_id=self.instrument_id),
            preflight=_ready_report(self.broker_name) if preflight is None else preflight,
            reconciliation=self.report if report is None else report,
        )


def _ready_report(broker: str = "alpaca") -> PreflightReport:
    return PreflightReport(
        broker,
        tuple(gate(name, True) for name in sorted(required_gate_names(broker))),
    )


def _live_cli_args(harness: _LiveHarness, *, phrase: str = CONFIRMATION_PHRASE) -> list[str]:
    arguments = _order_args("submit-confirmed-order")
    arguments[arguments.index("--broker") + 1] = harness.external_broker
    arguments[arguments.index("--instrument") + 1] = harness.instrument_id
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


def test_live_command_rejects_without_executing_in_process_proposal_provider(
    tmp_path: Path,
) -> None:
    """Live authority never crosses an arbitrary same-process proposal callback."""
    harness = _LiveHarness(tmp_path)
    events_before = harness.event_count()
    calls = 0

    class _EffectfulProposal:
        def propose(self, request: BoundOrderRequest) -> OrderProposal:
            nonlocal calls
            calls += 1
            harness.broker.submit(
                request.domain_intent(),
                snapshot(instrument_id=harness.instrument_id),
            )
            return OrderProposal.accept(request)

    container = ServiceContainer(
        proposal=_EffectfulProposal(),
        live_submission=harness.facade(),
    )
    result = CliRunner().invoke(build_app(lambda: container), _live_cli_args(harness))

    assert result.exit_code == 30
    assert result.stdout.strip() == '{"error":"PROPOSAL_RESPONSE_INVALID"}'
    assert calls == 0
    assert harness.broker.submit_calls == 0
    assert harness.event_count() == events_before


def test_real_task14_cli_facade_normalizes_ccxt_external_broker_alias(tmp_path: Path) -> None:
    """External ccxt requests must bind to the internal ccxt-spot Task 14 identity."""
    harness = _LiveHarness(
        tmp_path,
        broker_name="ccxt-spot",
        external_broker="ccxt",
        instrument_id="BTC/USD@ccxt",
    )

    result = _live_cli_result(harness)

    assert result.exit_code == 0
    assert harness.broker.submit_calls == 1
    assert json.loads(result.stdout)["broker"] == "ccxt-spot"


def test_live_submit_captures_registry_validated_facade_across_proposal_call(
    tmp_path: Path,
) -> None:
    """A proposal service cannot swap the frozen container's live authority mid-command."""
    harness = _LiveHarness(tmp_path)
    events_before = harness.event_count()

    class _BrokerCallingFake:
        def submit(self, proposal: OrderProposal, phrase: str) -> BrokerOrder:
            del phrase
            return harness.broker.submit(
                proposal.intent,
                snapshot(instrument_id=harness.instrument_id),
            )

    class _SwappingProposal:
        container: ServiceContainer

        def propose(self, request: BoundOrderRequest) -> OrderProposal:
            object.__setattr__(self.container, "live_submission", _BrokerCallingFake())
            return OrderProposal.accept(request)

    proposal = _SwappingProposal()
    container = ServiceContainer(
        proposal=proposal,
        live_submission=harness.facade(),
    )
    proposal.container = container

    result = CliRunner().invoke(build_app(lambda: container), _live_cli_args(harness))

    assert result.exit_code != 0
    assert harness.broker.submit_calls == 0
    assert harness.event_count() == events_before


@pytest.mark.parametrize(
    "attack",
    (
        "class_submit",
        "module_resolver",
        "approval_create",
        "live_submit_confirmed",
        "proposal_reconstructor",
        "proposal_validator",
        "proposal_matcher",
        "broker_order_emitter",
    ),
)
def test_live_dispatch_is_fixed_before_proposal_can_mutate_dynamic_attributes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    """Proposal code cannot redirect live dispatch through a class or module lookup."""
    harness = _LiveHarness(tmp_path)
    events_before = harness.event_count()
    dynamic_calls: list[str] = []

    def class_submit(
        facade: Task14CliLiveFacade,
        proposal: OrderProposal,
        phrase: str,
    ) -> BrokerOrder:
        del facade, phrase
        return harness.broker.submit(
            proposal.intent,
            snapshot(instrument_id=harness.instrument_id),
        )

    class _FakeApproval:
        def create(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            return object()

    class _FakeLive:
        broker_name = "alpaca"

        def submit_confirmed(self, **kwargs: object) -> BrokerOrder:
            intent = kwargs["intent"]
            market = kwargs["snapshot"]
            assert type(intent) is OrderIntent
            assert type(market) is MarketSnapshot
            return harness.broker.submit(intent, market)

    class _FakeBinding:
        approval_service = _FakeApproval()
        live_order_service = _FakeLive()
        snapshot = snapshot()
        preflight = _ready_report()
        reconciliation = harness.report

    class _AttackingProposal:
        def propose(self, request: BoundOrderRequest) -> OrderProposal:
            if attack == "class_submit":
                monkeypatch.setattr(Task14CliLiveFacade, "submit", class_submit)
            elif attack == "module_resolver":
                monkeypatch.setattr(
                    cli_module,
                    "_live_facade_binding",
                    lambda facade: _FakeBinding(),
                )
            elif attack == "approval_create":
                def fake_create(*args: object, **kwargs: object) -> object:
                    del args, kwargs
                    dynamic_calls.append("approval_create")
                    raise RuntimeError("dynamic approval method must not run")

                monkeypatch.setattr(ApprovalService, "create", fake_create)
            elif attack == "live_submit_confirmed":
                def fake_submit(*args: object, **kwargs: object) -> BrokerOrder:
                    del args
                    dynamic_calls.append("live_submit_confirmed")
                    intent = kwargs["intent"]
                    market = kwargs["snapshot"]
                    assert type(intent) is OrderIntent
                    assert type(market) is MarketSnapshot
                    return harness.broker.submit(intent, market)

                monkeypatch.setattr(LiveOrderService, "submit_confirmed", fake_submit)
            elif attack == "proposal_reconstructor":
                def fake_reconstruct(value: object) -> OrderProposal:
                    assert type(value) is OrderProposal
                    dynamic_calls.append("proposal_reconstructor")
                    harness.broker.submit(
                        value.intent,
                        snapshot(instrument_id=harness.instrument_id),
                    )
                    return value

                monkeypatch.setattr(
                    cli_module,
                    "_reconstruct_order_proposal",
                    fake_reconstruct,
                )
            elif attack == "proposal_validator":
                def fake_validator(
                    value: object,
                    expected: BoundOrderRequest,
                ) -> OrderProposal:
                    del expected
                    assert type(value) is OrderProposal
                    dynamic_calls.append("proposal_validator")
                    harness.broker.submit(
                        value.intent,
                        snapshot(instrument_id=harness.instrument_id),
                    )
                    return value

                monkeypatch.setattr(cli_module, "_validated_proposal", fake_validator)
            elif attack == "proposal_matcher":
                def fake_matcher(value: object, expected: object) -> bool:
                    del expected
                    assert type(value) is OrderProposal
                    dynamic_calls.append("proposal_matcher")
                    harness.broker.submit(
                        value.intent,
                        snapshot(instrument_id=harness.instrument_id),
                    )
                    return True

                monkeypatch.setattr(cli_module, "_proposal_matches_request", fake_matcher)
            else:
                def fake_emitter(value: BrokerOrder) -> None:
                    del value
                    dynamic_calls.append("broker_order_emitter")

                monkeypatch.setattr(cli_module, "_emit_broker_order", fake_emitter)
            return OrderProposal.accept(request)

    container = ServiceContainer(
        proposal=_AttackingProposal(),
        live_submission=harness.facade(),
    )
    result = CliRunner().invoke(build_app(lambda: container), _live_cli_args(harness))

    assert result.exit_code != 0
    assert harness.broker.submit_calls == 0
    assert harness.event_count() == events_before
    assert dynamic_calls == []


def test_live_proposal_cannot_rebind_request_domain_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider code cannot redefine the request that was parsed before it ran."""
    harness = _LiveHarness(tmp_path)
    events_before = harness.event_count()

    class _RequestMethodAttack:
        def propose(self, request: BoundOrderRequest) -> OrderProposal:
            forged_intent = request.domain_intent().model_copy(
                update={"quantity": Decimal("0.2")}
            )
            forged_risk = request.domain_risk().model_copy(
                update={
                    "approved_quantity": Decimal("0.2"),
                    "approved_notional": Decimal("20"),
                }
            )
            forged_fingerprint = cli_module._proposal_fingerprint(
                request.broker,
                forged_intent,
                forged_risk,
            )
            monkeypatch.setattr(
                BoundOrderRequest,
                "domain_intent",
                lambda self: forged_intent,
            )
            monkeypatch.setattr(
                BoundOrderRequest,
                "domain_risk",
                lambda self: forged_risk,
            )
            monkeypatch.setattr(
                BoundOrderRequest,
                "fingerprint",
                lambda self: forged_fingerprint,
            )
            return OrderProposal(
                True,
                (),
                request.broker,
                forged_intent,
                forged_risk,
                forged_fingerprint,
            )

    container = ServiceContainer(
        proposal=_RequestMethodAttack(),
        live_submission=harness.facade(),
    )
    result = CliRunner().invoke(build_app(lambda: container), _live_cli_args(harness))

    assert result.exit_code != 0
    assert harness.broker.submit_calls == 0
    assert harness.event_count() == events_before


@pytest.mark.parametrize(
    "mutation",
    (
        "accepted_int",
        "accepted_string",
        "accepted_with_denial",
        "reasons_list",
        "reason_bool",
        "risk_approved_int",
        "risk_reasons_list",
        "intent_route_bool",
    ),
)
def test_live_submit_deep_reconstructs_mutated_proposal_before_authority(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Only a deeply reconstructed exact proposal may reach confirmation issuance."""
    harness = _LiveHarness(tmp_path)
    events_before = harness.event_count()

    class _MutatingProposal:
        def propose(self, request: BoundOrderRequest) -> OrderProposal:
            proposal = OrderProposal.accept(request)
            if mutation == "accepted_int":
                object.__setattr__(proposal, "accepted", 1)
            elif mutation == "accepted_string":
                object.__setattr__(proposal, "accepted", "true")
            elif mutation == "accepted_with_denial":
                object.__setattr__(proposal, "reason_codes", ("STRATEGY_DENIED",))
            elif mutation == "reasons_list":
                object.__setattr__(proposal, "reason_codes", [])
            elif mutation == "reason_bool":
                object.__setattr__(proposal, "reason_codes", (True,))
            elif mutation == "risk_approved_int":
                object.__setattr__(proposal.risk_decision, "approved", 1)
            elif mutation == "risk_reasons_list":
                object.__setattr__(proposal.risk_decision, "reason_codes", [])
            else:
                object.__setattr__(proposal.intent, "time_in_force", True)
            return proposal

    container = ServiceContainer(
        proposal=_MutatingProposal(),
        live_submission=harness.facade(),
    )
    result = CliRunner().invoke(build_app(lambda: container), _live_cli_args(harness))

    assert result.exit_code == 30
    assert result.stdout.strip() == '{"error":"PROPOSAL_RESPONSE_INVALID"}'
    assert harness.broker.submit_calls == 0
    assert harness.event_count() == events_before


@pytest.mark.parametrize("mutation", ("risk_reasons", "intent_string"))
def test_proposal_nested_bounds_run_before_model_dump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Oversized nested fields fail before Pydantic copies or hashes them."""
    proposal = OrderProposal.accept(_LiveHarness(tmp_path).request)
    dumps: list[str] = []
    original_intent_dump = OrderIntent.model_dump
    original_risk_dump = RiskDecision.model_dump

    def intent_dump(self: OrderIntent, *args: object, **kwargs: object) -> object:
        dumps.append("intent")
        return original_intent_dump(self, *args, **kwargs)

    def risk_dump(self: RiskDecision, *args: object, **kwargs: object) -> object:
        dumps.append("risk")
        return original_risk_dump(self, *args, **kwargs)

    monkeypatch.setattr(OrderIntent, "model_dump", intent_dump)
    monkeypatch.setattr(RiskDecision, "model_dump", risk_dump)
    if mutation == "risk_reasons":
        object.__setattr__(
            proposal.risk_decision,
            "reason_codes",
            ("OVERSIZED",) * 100_000,
        )
    else:
        object.__setattr__(proposal.intent, "time_in_force", "X" * 100_000)

    with pytest.raises(ValueError, match="malformed"):
        cli_module._reconstruct_order_proposal(proposal)

    assert dumps == []


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
    first_result = _live_cli_result(harness)
    assert first_result.exit_code == 30
    assert first_result.stdout.strip() == '{"error":"LIVE_SUBMISSION_REJECTED"}'
    assert harness.broker.submit_calls == 1
    harness.broker.submit_error = None
    harness.request = harness.order_request("intent-2")

    result = _live_cli_result(harness, facade=harness.facade(report=harness.healthy_report()))

    assert result.exit_code == 30
    assert harness.broker.submit_calls == 1
    assert "must-not-leak" not in result.stdout


@pytest.mark.parametrize(
    "secret_value",
    (
        "Authorization is Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==",
        "Authorization: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==",
        "Cookie is session=live-cookie-value",
        "Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==",
    ),
)
def test_live_broker_order_output_rejects_natural_language_credentials(
    tmp_path: Path,
    secret_value: str,
) -> None:
    """Broker-controlled identifiers cannot emit Basic auth or cookie material."""
    harness = _LiveHarness(tmp_path)
    harness.broker.order_id = secret_value

    result = _live_cli_result(harness)

    assert result.exit_code == 30
    assert result.stdout.strip() == '{"error":"LIVE_RESPONSE_INVALID"}'
    assert secret_value not in result.stdout


_ENCODED_BROKER_IDENTIFIERS = (
    ("order_id", "%42asic+QWxhZGRpbjpvcGVuIHNlc2FtZQ%3D%3D", "Basic QWxh"),
    (
        "client_order_id",
        "%41uthorization+was+%42earer+live-token-value-123456",
        "Authorization was Bearer",
    ),
    ("broker", "%43ookie%3A+session%3Dlive-cookie-value", "Cookie: session="),
    ("instrument_id", "%61pi%5Fkey%3Dsecret-token-123456", "api_key="),
    ("order_id", "%61ccess%4Bey%3Dlive-access-value-123456", "accessKey="),
    ("client_order_id", "%70rivate%4Bey%3Dlive-private-value-123456", "privateKey="),
    ("broker", "%63redential%3Dlive-credential-value-123456", "credential="),
    ("instrument_id", "%70assword%3Dhunter-two-secret-value", "password="),
    ("order_id", "%74oken%3Dlive-token-value-123456", "token="),
    (
        "client_order_id",
        "%2541uthorization%253A%2520Bearer%2520double-encoded-token-123456",
        "Authorization: Bearer",
    ),
    ("broker", "opaque%ZZprovider-value", "opaque"),
    (
        "instrument_id",
        "X" * (4096 - len("-%61pi%5Fkey%3Dmax-length-secret-123456"))
        + "-%61pi%5Fkey%3Dmax-length-secret-123456",
        "api_key=max-length",
    ),
    ("order_id", "交易-%54oKeN%3Dunicode-secret-value-123456", "ToKeN="),
    ("client_order_id", "é" * 2049, "ééé"),
    (
        "broker",
        "%25252541uthorization%2525253A%25252520Bearer%25252520ambiguous-token",
        "Authorization: Bearer",
    ),
    ("order_id", "\ud800", "\ud800"),
    ("order_id", "%61pi+key%3DOpaqueValue123456", "api key="),
    ("client_order_id", "%61pi%20key%3DOpaqueValue123456", "api key="),
    ("broker", "%61ccess+key%3DOpaqueValue123456", "access key="),
    ("instrument_id", "%70rivate+key%3DOpaqueValue123456", "private key="),
    ("order_id", "%73ession+id%3DOpaqueValue123456", "session id="),
    ("order_id", "%61pi%09key%3DOpaqueValue123456", "api\tkey="),
    ("client_order_id", "api  key=OpaqueValue123456", "api  key="),
    ("broker", "%61ccess%C2%A0key%3DOpaqueValue123456", "access\u00a0key="),
    ("instrument_id", "%70rivate%2Ekey%3DOpaqueValue123456", "private.key="),
    ("order_id", "%61pi%2Fkey%3DOpaqueValue123456", "api/key="),
)


@pytest.mark.parametrize(("field", "encoded_value", "decoded_marker"), _ENCODED_BROKER_IDENTIFIERS)
def test_real_task14_cli_rejects_encoded_credentials_after_truthful_acknowledgement(
    tmp_path: Path,
    field: str,
    encoded_value: str,
    decoded_marker: str,
) -> None:
    """Encoded provider text must not escape after the one truthful broker acknowledgement."""
    harness = _LiveHarness(tmp_path)
    if field == "order_id":
        harness.broker.order_id = encoded_value
    else:
        harness.broker.mutate_response_after_validation(field, encoded_value)

    result = _live_cli_result(harness)
    audit_rows = harness.audit_rows()
    audit_text = json.dumps(audit_rows, ensure_ascii=False)

    assert result.exit_code == 30
    assert result.stdout.strip() == '{"error":"LIVE_RESPONSE_INVALID"}'
    assert harness.broker.submit_calls == 1
    assert harness.broker.query_calls == 0
    assert "live.acknowledged" in {kind for kind, _payload in audit_rows}
    assert "live.submission_unknown" not in {kind for kind, _payload in audit_rows}
    for unsafe in (encoded_value, decoded_marker):
        assert unsafe not in result.stdout
        assert unsafe not in audit_text
        assert result.exception is not None
        error: BaseException | None = result.exception
        while error is not None:
            assert unsafe not in str(error)
            if error.__cause__ is not None:
                assert unsafe not in str(error.__cause__)
            error = error.__context__


def test_broker_identifier_secret_detection_is_shared_and_bounded() -> None:
    """CLI and dashboard must reject the same encoded, malformed, and bounded text."""
    for field in ("order_id", "client_order_id", "broker", "instrument_id"):
        for encoded_value, _decoded_marker in (
            (item[1], item[2]) for item in _ENCODED_BROKER_IDENTIFIERS
        ):
            order = BrokerOrder(
                order_id="broker-order-1",
                client_order_id="intent-1",
                broker="alpaca",
                instrument_id="AAPL@alpaca",
                status=OrderStatus.ACKNOWLEDGED,
                requested_quantity=Decimal("0.1"),
                requested_notional=None,
                filled_quantity=Decimal("0"),
                average_fill_price=None,
                submitted_at=AT,
                updated_at=AT,
            )
            object.__setattr__(order, field, encoded_value)
            emitted: list[Mapping[str, object]] = []

            with pytest.raises(typer.Exit):
                cli_module._emit_broker_order(
                    order,
                    secret_detector=cli_module._secret_shaped,
                    emit=emitted.append,
                )

            assert emitted == []
            assert dashboard_module._secret_value(encoded_value)


def test_live_acknowledgement_uses_pre_provider_captured_audit_redactor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broker-time module rebinding must not place a response credential in durable audit."""
    harness = _LiveHarness(tmp_path)
    credential = "api_key=OpaqueValue123456"
    harness.broker.order_id = credential
    harness.broker.submit_hook = lambda: monkeypatch.setattr(
        live_module,
        "redact_secret_text",
        lambda value: value,
    )

    result = _live_cli_result(harness)
    audit_rows = harness.audit_rows()
    audit_text = json.dumps(audit_rows, ensure_ascii=False)

    assert result.exit_code == 30
    assert result.stdout.strip() == '{"error":"LIVE_RESPONSE_INVALID"}'
    assert harness.broker.submit_calls == 1
    assert harness.broker.query_calls == 0
    assert "live.acknowledged" in {kind for kind, _payload in audit_rows}
    assert credential not in result.stdout
    assert credential not in audit_text


def test_live_facade_rejects_pre_entry_double_rebound_audit_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Container-time replacement of both audit globals must fail before any durable effect."""
    harness = _LiveHarness(tmp_path)
    events_before = harness.event_count()
    credential = "api_key=OpaqueValue123456"
    harness.broker.order_id = credential

    def unsafe_identity(value: str) -> str:
        return value

    def container_factory() -> ServiceContainer:
        harness.live._audit_redactor = unsafe_identity
        monkeypatch.setattr(live_module, "_AUDIT_REDACTOR", unsafe_identity)
        monkeypatch.setattr(live_module, "redact_secret_text", unsafe_identity)
        return ServiceContainer(live_submission=harness.facade())

    result = CliRunner().invoke(build_app(container_factory), _live_cli_args(harness))

    assert result.exit_code != 0
    assert credential not in result.stdout
    assert harness.broker.submit_calls == 0
    assert harness.broker.query_calls == 0
    assert harness.event_count() == events_before


def test_broker_cannot_rebind_nested_secret_detector_dependencies(
    tmp_path: Path,
) -> None:
    """Provider-time security-module mutation must not alter CLI or audit decisions."""
    assert live_module._AUDIT_REDACTOR is live_module.redact_secret_text
    harness = _LiveHarness(tmp_path)
    credential = "api_key=OpaqueValue123456"
    harness.broker.order_id = credential
    originals = (
        security_module._SECRET_TEXT,
        security_module._MALFORMED_PERCENT_ESCAPE,
        security_module.secret_text_present,
        security_module._REDACTED,
        url_parse_module.unquote,
    )
    nested_checks: list[bool] = []
    nested_redactions: list[str] = []

    def mutate_security_module() -> None:
        security_module._SECRET_TEXT = re.compile(r"(?!x)x")
        security_module._MALFORMED_PERCENT_ESCAPE = re.compile(r"(?!x)x")
        security_module.secret_text_present = lambda value: False
        security_module._REDACTED = credential
        try:
            url_parse_module.unquote = lambda value, *args, **kwargs: value
            try:
                nested_checks.extend(
                    (
                        cli_module._secret_shaped("%61pi%5Fkey%3DOpaqueValue123456"),
                        dashboard_module._secret_value("%61pi%5Fkey%3DOpaqueValue123456"),
                    )
                )
            finally:
                url_parse_module.unquote = originals[4]
            nested_redactions.append(harness.live._audit_redactor(credential))
        finally:
            (
                security_module._SECRET_TEXT,
                security_module._MALFORMED_PERCENT_ESCAPE,
                security_module.secret_text_present,
                security_module._REDACTED,
            ) = originals[:4]

    harness.broker.submit_hook = mutate_security_module
    try:
        result = _live_cli_result(harness)
    finally:
        (
            security_module._SECRET_TEXT,
            security_module._MALFORMED_PERCENT_ESCAPE,
            security_module.secret_text_present,
            security_module._REDACTED,
            url_parse_module.unquote,
        ) = originals
    audit_rows = harness.audit_rows()
    audit_text = json.dumps(audit_rows, ensure_ascii=False)

    assert result.exit_code == 30
    assert result.stdout.strip() == '{"error":"LIVE_RESPONSE_INVALID"}'
    assert harness.broker.submit_calls == 1
    assert harness.broker.query_calls == 0
    assert nested_checks == [True, True]
    assert nested_redactions == ["[REDACTED]"]
    assert "live.acknowledged" in {kind for kind, _payload in audit_rows}
    assert credential not in result.stdout
    assert credential not in audit_text


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


@pytest.mark.parametrize(
    "mutation",
    (
        "research",
        "strategy",
        "promotion",
        "portfolio",
        "risk",
        "broker",
        "order",
        "kill_switch",
        "interlock",
        "aspiration_reporting",
        "aspiration_arithmetic",
    ),
)
def test_dashboard_reconstructs_every_nested_record_before_export(
    tmp_path: Path,
    mutation: str,
) -> None:
    """object.__setattr__ must not bypass any nested schema-v1 validator."""
    status = _dashboard_status()
    if mutation == "research":
        object.__setattr__(status.research, "fresh", 1)
    elif mutation == "strategy":
        object.__setattr__(status.strategies[0], "version", "bad secret value")
    elif mutation == "promotion":
        object.__setattr__(status.promotion, "status", "live-unbounded")
    elif mutation == "portfolio":
        object.__setattr__(status.portfolio, "currency", "usd")
    elif mutation == "risk":
        object.__setattr__(status.risk, "max_position_fraction", Decimal("2"))
    elif mutation == "broker":
        object.__setattr__(status.brokers[0].gates[0], "reason_code", "NOT_OK")
    elif mutation == "order":
        mutated_order = dashboard_module.DashboardOrder("order-1", OrderStatus.PROPOSED)
        object.__setattr__(mutated_order, "status", "proposed")
        object.__setattr__(status, "orders", (mutated_order,))
    elif mutation == "kill_switch":
        object.__setattr__(status.kill_switches[0], "active", 1)
    elif mutation == "interlock":
        object.__setattr__(status.interlocks[0], "reason_code", "not_stable")
    elif mutation == "aspiration_reporting":
        object.__setattr__(status.aspirational_target, "reporting_only", False)
    else:
        object.__setattr__(status.aspirational_target, "required_multiple", Decimal("2"))
    destination = tmp_path / f"{mutation}.json"

    with pytest.raises(DashboardValidationError, match="DASHBOARD_STATUS_INVALID") as failure:
        export_dashboard(status, destination)

    assert failure.value.__cause__ is None
    assert failure.value.__context__ is None
    assert not destination.exists()


def test_dashboard_rejects_contradictory_broker_gate_coherence(tmp_path: Path) -> None:
    """Passed/non-OK and failed/OK gates must never enter dashboard output."""
    for passed, reason in ((True, "NOT_OK"), (False, "OK")):
        status = _dashboard_status()
        object.__setattr__(status.brokers[0].gates[0], "passed", passed)
        object.__setattr__(status.brokers[0].gates[0], "reason_code", reason)
        with pytest.raises(DashboardValidationError, match="DASHBOARD_STATUS_INVALID"):
            export_dashboard(status, tmp_path / f"coherence-{passed}.json")
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


def test_dashboard_reconstruction_bounds_collections_before_copying() -> None:
    """Schema reconstruction must reject oversized tuples before traversing them."""
    status = _dashboard_status()
    object.__setattr__(
        status,
        "strategies",
        (status.strategies[0],) * (dashboard_module._MAX_COLLECTION + 1),
    )

    with pytest.raises(DashboardValidationError, match="DASHBOARD_VALUE_BOUNDS_EXCEEDED"):
        dashboard_module._revalidated_dashboard_status(status)


def test_dashboard_reconstruction_bounds_nested_broker_gates_before_copying() -> None:
    """A tampered exact broker cannot force traversal of an oversized gate tuple."""
    status = _dashboard_status()
    object.__setattr__(
        status.brokers[0],
        "gates",
        (status.brokers[0].gates[0],) * (dashboard_module._MAX_COLLECTION + 1),
    )

    with pytest.raises(DashboardValidationError, match="DASHBOARD_VALUE_BOUNDS_EXCEEDED"):
        dashboard_module._revalidated_dashboard_status(status)


def test_dashboard_bounds_gate_names_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tampered exact gate must be rejected before an oversized name reaches set/hash."""
    status = _dashboard_status()
    object.__setattr__(status.brokers[0].gates[0], "name", "X" * 100_000)
    hash_attempts: list[int] = []
    builtin_set = set

    def bounded_set(values: object) -> set[object]:
        items = list(values)  # type: ignore[arg-type]
        hash_attempts.extend(len(item) for item in items if type(item) is str)
        return builtin_set(items)

    monkeypatch.setattr(dashboard_module, "set", bounded_set, raising=False)

    with pytest.raises(DashboardValidationError, match="DASHBOARD_BROKER_GATES_INVALID"):
        dashboard_module._revalidated_dashboard_status(status)

    assert hash_attempts == []


def test_dashboard_uses_captured_immutable_broker_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider-time mutation of the imported manifest helper cannot redefine safety."""
    status = _dashboard_status()
    monkeypatch.setattr(
        dashboard_module,
        "required_gate_names",
        lambda broker: {f"ATTACKER_{broker}"},
    )

    reconstructed = dashboard_module._revalidated_dashboard_status(status)

    assert reconstructed.brokers[0].gates == status.brokers[0].gates


def test_dashboard_rejects_rebound_captured_manifest_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider code cannot rebind the immutable manifest name to forge readiness."""
    status = _dashboard_status()
    object.__setattr__(
        status.brokers[0],
        "gates",
        (GateResult(name="X", passed=True, reason_code="OK"),),
    )
    forged = MappingProxyType(
        {
            "alpaca": frozenset({"X"}),
            "groww": frozenset({"X"}),
            "ccxt": frozenset({"X"}),
        }
    )
    monkeypatch.setattr(dashboard_module, "_CAPTURED_GATE_MANIFESTS", forged)

    with pytest.raises(DashboardValidationError, match="DASHBOARD_STATUS_INVALID"):
        export_dashboard(status, tmp_path / "forged-manifest.json")


def test_dashboard_rejects_rebound_manifest_lookup_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A helper rebind cannot redefine the immutable broker gate manifest."""
    status = _dashboard_status()
    object.__setattr__(
        status.brokers[0],
        "gates",
        (GateResult(name="X", passed=True, reason_code="OK"),),
    )
    monkeypatch.setattr(
        dashboard_module,
        "_captured_gate_manifest",
        lambda broker, manifests=None: frozenset({"X"}),
    )

    with pytest.raises(DashboardValidationError, match="DASHBOARD_STATUS_INVALID"):
        export_dashboard(status, tmp_path / "forged-helper.json")


def test_dashboard_aspiration_bounds_decimals_before_fraction_arithmetic() -> None:
    """Huge finite exponents must fail before constructing unbounded Fractions."""
    huge = Decimal("1E+5000")

    with pytest.raises(DashboardValidationError, match="DASHBOARD_DECIMAL_INVALID"):
        DashboardAspiration(
            Decimal("1"),
            Decimal("0"),
            huge,
            huge,
            Decimal("0"),
            huge,
            True,
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


def test_dashboard_redacts_standalone_basic_authorization_value() -> None:
    """A free-text Basic authorization scheme is never serialized."""
    prepared = safe_json_mapping(
        {"note": "Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ=="}
    )

    assert prepared == {"note": "[REDACTED]"}


def test_safe_json_converts_surrogate_failures_to_context_free_validation() -> None:
    """Hostile UTF-8 keys and values must never escape raw Unicode exceptions."""
    for payload in ({"safe": "\ud800"}, {"bad\ud800": "safe"}):
        with pytest.raises(DashboardValidationError) as failure:
            safe_json_mapping(payload)
        assert failure.value.__cause__ is None
        assert failure.value.__context__ is None


def test_safe_json_redacts_unassigned_provider_secret_free_text() -> None:
    """Secret-bearing provider text is unsafe even without an assignment delimiter."""
    prepared = safe_json_mapping({"note": "groww-real-secret-value-1234567890"})

    assert prepared == {"note": "[REDACTED]"}


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
    original_commit = dashboard_module._commit_open_temp
    parent_moved = False

    def swap_parent_then_commit(*args: object, **kwargs: object) -> object:
        nonlocal parent_moved
        original_replace(parent, parked)
        parent_moved = True
        parent.mkdir()
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(
        dashboard_module,
        "_commit_open_temp",
        swap_parent_then_commit,
    )

    with pytest.raises(DashboardValidationError, match="DASHBOARD_WRITE_FAILED"):
        export_dashboard(_dashboard_status(), destination)

    if parent_moved:
        assert not destination.exists()
        assert (parked / "status.json").read_text(encoding="utf-8") == "old"
        assert not list(parked.glob(".status.json.*.tmp"))
    else:
        assert destination.read_text(encoding="utf-8") == "old"
        assert not parked.exists()


@pytest.mark.parametrize("hook", ("after_write", "pre_replace"))
@pytest.mark.skipif(os.name != "nt", reason="named-temp replacement is Windows-only")
def test_dashboard_rejects_temp_name_swap_while_preserving_old_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hook: str,
) -> None:
    """A swapped temp directory entry must never place attacker bytes at destination."""
    destination = tmp_path / "status.json"
    destination.write_text("old", encoding="utf-8")
    parked = tmp_path / "parked-original.tmp"
    original_same_parent = dashboard_module._require_same_parent
    original_validated = dashboard_module._validated_destination
    same_parent_calls = 0
    validated_calls = 0

    def swap_temp() -> None:
        temporary = next(tmp_path.glob(".status.json.*.tmp"))
        os.replace(temporary, parked)
        temporary.write_text("attacker-content", encoding="utf-8")

    def same_parent(target: object) -> None:
        nonlocal same_parent_calls
        same_parent_calls += 1
        if hook == "after_write" and same_parent_calls == 3:
            swap_temp()
        original_same_parent(target)  # type: ignore[arg-type]

    def validated(target: object) -> object:
        nonlocal validated_calls
        validated_calls += 1
        if hook == "pre_replace" and validated_calls == 2:
            swap_temp()
        return original_validated(target)

    monkeypatch.setattr(dashboard_module, "_require_same_parent", same_parent)
    monkeypatch.setattr(dashboard_module, "_validated_destination", validated)

    with pytest.raises(DashboardValidationError, match="DASHBOARD_WRITE_FAILED"):
        export_dashboard(_dashboard_status(), destination)

    assert destination.read_text(encoding="utf-8") == "old"
    attacker_entry = next(tmp_path.glob(".status.json.*.tmp"))
    assert attacker_entry.read_text(encoding="utf-8") == "attacker-content"
    if parked.exists():
        assert "attacker-content" not in parked.read_text(encoding="utf-8")
    assert "attacker-content" not in destination.read_text(encoding="utf-8")


@pytest.mark.skipif(os.name != "nt", reason="handle-bound replacement is Windows-only")
def test_dashboard_commit_binds_open_temp_handle_after_final_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name swap after the last check must never become reader-visible at commit."""
    destination = tmp_path / "status.json"
    destination.write_text("old", encoding="utf-8")
    parked = tmp_path / "parked-open-temp.tmp"
    original_commit = dashboard_module._commit_open_temp
    original_replaced_check = dashboard_module._require_replaced_identity
    swapped = False
    reader_observations: list[str] = []

    def swap_after_final_check(*args: object, **kwargs: object) -> int:
        nonlocal swapped
        if not swapped:
            swapped = True
            temporary = next(tmp_path.glob(".status.json.*.tmp"))
            os.replace(temporary, parked)
            temporary.write_text("attacker-content", encoding="utf-8")
        return original_commit(*args, **kwargs)

    def observe_immediate_commit(
        descriptor: int,
        target: Path,
        expected: tuple[int, int, int],
        guard: object,
    ) -> None:
        try:
            reader_observations.append(target.read_text(encoding="utf-8"))
        except PermissionError:
            reader_observations.append("reader-blocked-by-open-commit-handle")
        original_replaced_check(
            descriptor,
            target,
            expected,
            guard,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(
        dashboard_module,
        "_commit_open_temp",
        swap_after_final_check,
    )
    monkeypatch.setattr(
        dashboard_module,
        "_require_replaced_identity",
        observe_immediate_commit,
    )

    export_dashboard(_dashboard_status(), destination)

    assert swapped
    assert reader_observations
    assert all(value != "attacker-content" for value in reader_observations)
    assert destination.read_text(encoding="utf-8") != "attacker-content"


def test_posix_commit_directly_links_held_fd_to_absent_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSIX create publishes the held fd directly with no intermediate source name."""
    temporary = tmp_path / "source.tmp"
    temporary.write_text("safe", encoding="utf-8")
    destination = tmp_path / "status.json"
    descriptor = os.open(temporary, os.O_RDONLY)
    operations: list[str] = []

    class _Guard:
        parent_fd = 91

        def lstat(self, path: Path) -> os.stat_result | None:
            del path
            operations.append("target-check")
            return None

        def replace(self, source: Path, target: Path) -> None:
            del source, target
            operations.append("replace")

        def unlink(self, path: Path) -> None:
            del path

    monkeypatch.setattr(dashboard_module.os, "name", "posix")
    monkeypatch.setattr(
        dashboard_module,
        "_link_open_file_at",
        lambda fd, parent_fd, name: operations.append(f"link:{name}"),
    )
    try:
        dashboard_module._commit_open_temp(
            descriptor,
            temporary,
            destination,
            _Guard(),  # type: ignore[arg-type]
        )
    finally:
        os.close(descriptor)

    assert operations == ["target-check", "link:status.json"]


def test_posix_commit_refuses_to_replace_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSIX safe mode preserves an existing target instead of name-renaming over it."""
    temporary = tmp_path / "source.tmp"
    temporary.write_text("safe", encoding="utf-8")
    destination = tmp_path / "status.json"
    destination.write_text("old", encoding="utf-8")
    descriptor = os.open(temporary, os.O_RDONLY)
    link_names: list[str] = []

    class _Guard:
        parent_fd = 91

        def lstat(self, path: Path) -> os.stat_result:
            return os.lstat(path)

    monkeypatch.setattr(dashboard_module.os, "name", "posix")
    monkeypatch.setattr(
        dashboard_module,
        "_link_open_file_at",
        lambda fd, parent_fd, name: link_names.append(name),
    )
    try:
        with pytest.raises(
            DashboardValidationError,
            match="DASHBOARD_HANDLE_REPLACE_UNAVAILABLE",
        ):
            dashboard_module._commit_open_temp(
                descriptor,
                temporary,
                destination,
                _Guard(),  # type: ignore[arg-type]
            )
    finally:
        os.close(descriptor)

    assert destination.read_text(encoding="utf-8") == "old"
    assert link_names == []


def test_posix_concurrent_absent_destination_has_exactly_one_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct no-replace links make concurrent create attempts one-winner only."""
    first = tmp_path / "first.tmp"
    second = tmp_path / "second.tmp"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    destination = tmp_path / "status.json"
    descriptors = (os.open(first, os.O_RDONLY), os.open(second, os.O_RDONLY))
    ready = Event()
    checked = 0
    checked_lock = Lock()
    published = False
    published_lock = Lock()
    outcomes: list[str] = []

    class _Guard:
        parent_fd = 91

        def lstat(self, path: Path) -> None:
            nonlocal checked
            del path
            with checked_lock:
                checked += 1
                if checked == 2:
                    ready.set()
            assert ready.wait(timeout=5)
            return None

    def exclusive_link(fd: int, parent_fd: int, name: str) -> None:
        nonlocal published
        del fd, parent_fd
        assert name == destination.name
        with published_lock:
            if published:
                raise FileExistsError(name)
            published = True

    def run(descriptor: int) -> None:
        try:
            dashboard_module._commit_open_temp(
                descriptor,
                first,
                destination,
                _Guard(),  # type: ignore[arg-type]
            )
        except (OSError, DashboardValidationError):
            outcomes.append("failed")
        else:
            outcomes.append("created")

    monkeypatch.setattr(dashboard_module.os, "name", "posix")
    monkeypatch.setattr(dashboard_module, "_link_open_file_at", exclusive_link)
    threads = [Thread(target=run, args=(descriptor,)) for descriptor in descriptors]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)

    assert sorted(outcomes) == ["created", "failed"]


@pytest.mark.skipif(os.name != "nt", reason="handle-bound replacement is Windows-only")
def test_dashboard_postcommit_verification_failure_never_rolls_back_exact_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning from exact handle commit is irreversible even if a postcheck fails."""
    destination = tmp_path / "status.json"
    destination.write_text("old", encoding="utf-8")

    calls = 0

    def fail_first_postcheck(*args: object, **kwargs: object) -> None:
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 1:
            raise DashboardValidationError("DASHBOARD_REPLACE_CHANGED")

    monkeypatch.setattr(
        dashboard_module,
        "_require_replaced_identity",
        fail_first_postcheck,
    )

    with pytest.raises(DashboardValidationError, match="DASHBOARD_WRITE_FAILED"):
        export_dashboard(_dashboard_status(), destination)

    committed = destination.read_text(encoding="utf-8")
    assert committed != "old"
    assert json.loads(committed)["schema_version"] == 1


@pytest.mark.skipif(os.name != "nt", reason="handle-bound replacement is Windows-only")
def test_dashboard_rollback_never_overwrites_a_newer_destination_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed writer must not restore stale backup over a changed destination inode."""
    destination = tmp_path / "status.json"
    destination.write_text("old", encoding="utf-8")
    external = tmp_path / "external.json"
    original_same_parent = dashboard_module._require_same_parent
    calls = 0

    def replace_after_commit(target: object) -> None:
        nonlocal calls
        calls += 1
        original_same_parent(target)  # type: ignore[arg-type]
        if calls == 4:
            external.write_text("newer-external-writer", encoding="utf-8")
            os.replace(external, destination)
            raise DashboardValidationError("DASHBOARD_PATH_CHANGED")

    monkeypatch.setattr(dashboard_module, "_require_same_parent", replace_after_commit)

    with pytest.raises(DashboardValidationError, match="DASHBOARD_WRITE_FAILED"):
        export_dashboard(_dashboard_status(), destination)

    assert destination.read_text(encoding="utf-8") == "newer-external-writer"


@pytest.mark.skipif(os.name != "nt", reason="handle-bound replacement is Windows-only")
def test_dashboard_committed_export_never_path_unlinks_generated_names(
    tmp_path: Path,
) -> None:
    """After an exact commit, cleanup never acts through mutable generated pathnames."""
    destination = tmp_path / "status.json"
    destination.write_text("old", encoding="utf-8")

    assert "unlink" not in dashboard_module._DirectoryGuard.__dict__
    assert "replace" not in dashboard_module._DirectoryGuard.__dict__
    assert "link" not in dashboard_module._DirectoryGuard.__dict__

    export_dashboard(_dashboard_status(), destination)

    assert json.loads(destination.read_text(encoding="utf-8"))["schema_version"] == 1


def test_dashboard_failure_cleanup_does_not_enumerate_destination_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure cleanup touches only generated names and never scans an unbounded directory."""
    destination = tmp_path / "status.json"
    destination.write_text("old", encoding="utf-8")
    observed: list[object] = []
    parent_checks = 0

    def scandir(path: object) -> Iterator[object]:
        observed.append(path)
        return iter(())

    def fail_after_temp(target: object) -> None:
        nonlocal parent_checks
        parent_checks += 1
        if parent_checks == 2:
            raise DashboardValidationError("DASHBOARD_PATH_CHANGED")
        original_same_parent(target)  # type: ignore[arg-type]

    original_same_parent = dashboard_module._require_same_parent
    monkeypatch.setattr(dashboard_module.os, "scandir", scandir)
    monkeypatch.setattr(dashboard_module, "_require_same_parent", fail_after_temp)

    with pytest.raises(DashboardValidationError, match="DASHBOARD_WRITE_FAILED"):
        export_dashboard(_dashboard_status(), destination)

    assert observed == []


def test_posix_open_file_link_uses_unprivileged_procfd_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unprivileged AT_EMPTY_PATH denial falls back to the exact procfd symlink."""
    calls: list[tuple[object, ...]] = []

    class _LinkAt:
        argtypes: object = None
        restype: object = None

        def __call__(self, *args: object) -> int:
            calls.append(args)
            return -1 if len(calls) == 1 else 0

    class _LibC:
        linkat = _LinkAt()

    monkeypatch.setattr(dashboard_module.ctypes, "CDLL", lambda *args, **kwargs: _LibC())

    dashboard_module._link_open_file_at(91, 92, "safe.commit")

    assert calls == [
        (91, b"", 92, b"safe.commit", 0x1000),
        (-100, b"/proc/self/fd/91", 92, b"safe.commit", 0x400),
    ]


@pytest.mark.skipif(os.name != "nt", reason="replacement serialization is Windows-only")
def test_dashboard_serializes_concurrent_writers_per_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writer B waits for A's commit and can never be rolled back to stale A/old data."""
    destination = tmp_path / "status.json"
    destination.write_text("old", encoding="utf-8")
    writer_a = _dashboard_status(research=DashboardResearch("writer-a", True))
    writer_b = _dashboard_status(research=DashboardResearch("writer-b", True))
    original_commit = dashboard_module._commit_open_temp
    entered = Event()
    release = Event()
    b_done = Event()
    blocked_a = False
    failures: list[BaseException] = []

    def block_a_before_commit(*args: object, **kwargs: object) -> int:
        nonlocal blocked_a
        if current_thread().name == "dashboard-writer-a" and not blocked_a:
            blocked_a = True
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test release timeout")
        return original_commit(*args, **kwargs)

    def run(status: DashboardStatus, done: Event | None = None) -> None:
        try:
            export_dashboard(status, destination)
        except BaseException as error:
            failures.append(error)
        finally:
            if done is not None:
                done.set()

    monkeypatch.setattr(
        dashboard_module,
        "_commit_open_temp",
        block_a_before_commit,
    )
    a_thread = Thread(
        target=run,
        args=(writer_a,),
        name="dashboard-writer-a",
    )
    b_thread = Thread(
        target=run,
        args=(writer_b, b_done),
        name="dashboard-writer-b",
    )
    a_thread.start()
    assert entered.wait(timeout=5)
    b_thread.start()
    b_finished_while_a_blocked = b_done.wait(timeout=0.25)
    release.set()
    a_thread.join(timeout=5)
    b_thread.join(timeout=5)

    assert not a_thread.is_alive()
    assert not b_thread.is_alive()
    assert not failures
    assert not b_finished_while_a_blocked
    assert json.loads(destination.read_text(encoding="utf-8"))["research"]["version"] == "writer-b"


def test_posix_temp_creation_uses_anonymous_held_parent_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSIX staging has no mutable directory entry before direct publication."""
    destination_path = tmp_path / "status.json"
    parent_identity = dashboard_module._identity_tuple(os.lstat(tmp_path))
    destination = dashboard_module._Destination(destination_path, parent_identity)
    observed: list[tuple[object, int, int, object]] = []

    def anonymous_open(
        path: object,
        flags: int,
        mode: int,
        *,
        dir_fd: object = None,
    ) -> int:
        observed.append((path, flags, mode, dir_fd))
        return 93

    monkeypatch.setattr(dashboard_module.os, "name", "posix")
    monkeypatch.setattr(dashboard_module.os, "O_TMPFILE", 0x400000, raising=False)
    monkeypatch.setattr(dashboard_module.os, "open", anonymous_open)

    descriptor, marker = dashboard_module._create_sibling_temp(
        destination,
        dashboard_module._DirectoryGuard(91),
    )

    assert descriptor == 93
    assert marker.name.endswith(".anonymous")
    assert observed == [(".", os.O_RDWR | 0x400000, 0o600, 91)]


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX O_TMPFILE integration",
)
def test_posix_export_end_to_end_directly_creates_fresh_destination(
    tmp_path: Path,
) -> None:
    """Real POSIX export publishes one fresh path without visible staging names."""
    _require_posix_otmpfile(tmp_path)
    destination = tmp_path / "status-v1.json"

    exported = export_dashboard(_dashboard_status(), destination)

    assert exported == destination
    assert json.loads(destination.read_text(encoding="utf-8"))["schema_version"] == 1
    assert not list(tmp_path.glob(".status-v1.json.*"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX create-once contract")
def test_posix_export_end_to_end_refuses_and_preserves_existing_target(
    tmp_path: Path,
) -> None:
    """Real POSIX safe mode never modifies an existing versioned destination."""
    destination = tmp_path / "status-v1.json"
    destination.write_text("old", encoding="utf-8")

    with pytest.raises(DashboardValidationError, match="DASHBOARD_WRITE_FAILED"):
        export_dashboard(_dashboard_status(), destination)

    assert destination.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".status-v1.json.*"))


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX O_TMPFILE integration",
)
def test_posix_export_end_to_end_concurrent_fresh_path_has_one_winner(
    tmp_path: Path,
) -> None:
    """Real held-parent locking plus no-replace link gives exactly one publisher."""
    _require_posix_otmpfile(tmp_path)
    destination = tmp_path / "status-v1.json"
    first = _dashboard_status(research=DashboardResearch("writer-a", True))
    second = _dashboard_status(research=DashboardResearch("writer-b", True))
    outcomes: list[str] = []
    outcome_lock = Lock()

    def run(status: DashboardStatus) -> None:
        try:
            export_dashboard(status, destination)
        except DashboardValidationError:
            outcome = "failed"
        else:
            outcome = "created"
        with outcome_lock:
            outcomes.append(outcome)

    threads = [Thread(target=run, args=(status,)) for status in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(outcomes) == ["created", "failed"]
    assert json.loads(destination.read_text(encoding="utf-8"))["research"]["version"] in {
        "writer-a",
        "writer-b",
    }
    assert not list(tmp_path.glob(".status-v1.json.*"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX O_TMPFILE fail-closed contract")
def test_posix_export_end_to_end_fails_closed_without_otmpfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No anonymous-staging primitive means no destination publication."""
    destination = tmp_path / "status-v1.json"
    monkeypatch.setattr(dashboard_module.os, "O_TMPFILE", 0, raising=False)

    with pytest.raises(DashboardValidationError, match="DASHBOARD_WRITE_FAILED"):
        export_dashboard(_dashboard_status(), destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".status-v1.json.*"))


def test_dashboard_json_failure_is_context_free_and_preserves_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON encoder failures must become stable validation without filesystem mutation."""
    destination = tmp_path / "status.json"
    destination.write_text("old", encoding="utf-8")

    def fail_json(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise UnicodeError("json-surrogate-secret")

    monkeypatch.setattr(dashboard_module.json, "dumps", fail_json)
    with pytest.raises(DashboardValidationError) as failure:
        export_dashboard(_dashboard_status(), destination)

    assert failure.value.__cause__ is None
    assert failure.value.__context__ is None
    assert destination.read_text(encoding="utf-8") == "old"


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
