"""Dependency-injected, fail-closed local control center."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Never, NoReturn, Protocol, SupportsIndex, cast
from weakref import WeakKeyDictionary

import typer

from market_sentinel.brokers.preflight import PreflightReport, gate, required_gate_names
from market_sentinel.domain.enums import OrderType, Side
from market_sentinel.domain.models import (
    BrokerOrder,
    GateResult,
    MarketSnapshot,
    OrderIntent,
    RiskDecision,
)
from market_sentinel.execution.approval import (
    ApprovalService,
    OrderConfirmation,
    risk_decision_hash,
)
from market_sentinel.execution.canonical import CanonicalEncodingError, canonical_decimal
from market_sentinel.execution.live import LiveOrderService
from market_sentinel.execution.reconcile import ReconciliationReport
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
    safe_json_mapping,
)
from market_sentinel.operations.dashboard import (
    export_dashboard as write_dashboard,
)
from market_sentinel.security import redact_secret_text, secret_text_present

CONFIRMATION_PHRASE = "I_CONFIRM_REAL_MONEY_ORDER"
_BROKERS = ("alpaca", "groww", "ccxt")
_INSTRUMENT = re.compile(r"^[A-Z0-9][A-Z0-9._/-]{0,31}@[a-z0-9][a-z0-9-]{0,31}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_ROUTE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_REASON = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_MAX_PROPOSAL_REASONS = 32
_MAX_PREFLIGHT_GATES = 64
_MAX_GATE_NAME_CHARS = 128
_MAX_GATE_REASON_CHARS = 64
_CAPTURED_PREFLIGHT_MANIFESTS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "alpaca": frozenset(required_gate_names("alpaca")),
        "groww": frozenset(required_gate_names("groww")),
        "ccxt-spot": frozenset(required_gate_names("ccxt-spot")),
    }
)
_APPROVAL_CREATE_IMPLEMENTATION = ApprovalService.create
_LIVE_SUBMIT_IMPLEMENTATION = LiveOrderService.submit_confirmed
class WorkflowService(Protocol):
    def run(self, request: WorkflowRequest) -> Mapping[str, object]: ...


class PreflightService(Protocol):
    def run(self, broker: str) -> PreflightReport: ...


class ProposalService(Protocol):
    def propose(self, request: BoundOrderRequest) -> OrderProposal: ...


@dataclass(frozen=True, slots=True)
class WorkflowRequest:
    workflow: str
    instrument_id: str
    as_of: datetime | None = None
    start_at: datetime | None = None
    end_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_instrument(self.instrument_id)
        for instant in (self.as_of, self.start_at, self.end_at):
            if instant is not None:
                _aware_utc(instant)
        if self.start_at is not None and self.end_at is not None and self.start_at >= self.end_at:
            raise ValueError("workflow interval is invalid")


@dataclass(frozen=True, slots=True)
class BoundOrderRequest:
    """Complete order and risk identity delegated unchanged to the live safety service."""

    broker: str
    intent_id: str
    instrument_id: str
    side: str
    quantity: Decimal | None
    notional: Decimal | None
    order_type: str
    limit_price: Decimal | None
    trigger_price: Decimal | None
    stop_loss: Decimal | None
    take_profit: Decimal | None
    time_in_force: str
    product: str
    session: str
    snapshot_hash: str
    created_at: datetime
    expires_at: datetime
    risk_approved_quantity: Decimal | None
    risk_approved_notional: Decimal | None
    portfolio_hash: str
    risk_decided_at: datetime
    risk_expires_at: datetime

    def __post_init__(self) -> None:
        if (
            self.broker not in _BROKERS
            or not _SAFE_ID.fullmatch(self.intent_id)
            or _secret_shaped(self.intent_id)
        ):
            raise ValueError("order identity is invalid")
        _require_instrument(self.instrument_id)
        if self.side not in {"buy", "sell"} or self.order_type not in {
            "market",
            "limit",
            "stop",
            "stop_limit",
        }:
            raise ValueError("order direction or type is invalid")
        _require_exact_size(self.quantity, self.notional)
        if self.risk_approved_quantity is None or self.risk_approved_notional is None:
            raise ValueError("risk decision requires exact quantity and notional")
        _positive_decimal(self.risk_approved_quantity)
        _positive_decimal(self.risk_approved_notional)
        if self.quantity is not None and self.quantity != self.risk_approved_quantity:
            raise ValueError("risk size is not exactly bound")
        if self.notional is not None and self.notional != self.risk_approved_notional:
            raise ValueError("risk size is not exactly bound")
        for value in (self.limit_price, self.trigger_price, self.stop_loss, self.take_profit):
            if value is not None:
                _positive_decimal(value)
        expected = {
            "market": (False, False),
            "limit": (True, False),
            "stop": (False, True),
            "stop_limit": (True, True),
        }[self.order_type]
        if (self.limit_price is not None, self.trigger_price is not None) != expected:
            raise ValueError("order price fields do not match order type")
        if not all(
            type(item) is str and _SAFE_ROUTE.fullmatch(item) and not _secret_shaped(item)
            for item in (self.time_in_force, self.product, self.session)
        ):
            raise ValueError("order routing fields are invalid")
        if not _HASH.fullmatch(self.snapshot_hash) or self.portfolio_hash != self.snapshot_hash:
            raise ValueError("portfolio identity is invalid")
        created = _aware_utc(self.created_at)
        expires = _aware_utc(self.expires_at)
        decided = _aware_utc(self.risk_decided_at)
        risk_expires = _aware_utc(self.risk_expires_at)
        if not (created < expires and decided < risk_expires and risk_expires <= expires):
            raise ValueError("order or risk expiry is invalid")
        venue = self.instrument_id.rsplit("@", 1)[1]
        expected_venues = {"ccxt": {"ccxt", "ccxt-spot"}}.get(self.broker, {self.broker})
        if venue not in expected_venues:
            raise ValueError("broker and instrument venue do not match")
        self.domain_intent()
        self.domain_risk()

    def domain_intent(self) -> OrderIntent:
        """Construct the exact Task 14 domain intent and run its validators."""
        return OrderIntent(
            intent_id=self.intent_id,
            instrument_id=self.instrument_id,
            side=Side(self.side),
            quantity=self.quantity,
            notional=self.notional,
            order_type=OrderType(self.order_type),
            limit_price=self.limit_price,
            trigger_price=self.trigger_price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            time_in_force=self.time_in_force,
            product=self.product,
            session=self.session,
            snapshot_hash=self.snapshot_hash,
            created_at=self.created_at,
            expires_at=self.expires_at,
        )

    def domain_risk(self) -> RiskDecision:
        """Construct the exact approved Task 14 risk record."""
        return RiskDecision(
            approved=True,
            reason_codes=(),
            approved_quantity=self.risk_approved_quantity,
            approved_notional=self.risk_approved_notional,
            portfolio_hash=self.portfolio_hash,
            decided_at=self.risk_decided_at,
            expires_at=self.risk_expires_at,
        )

    def fingerprint(self) -> str:
        """Hash the exact typed domain request with a domain-separated canonical form."""
        return _proposal_fingerprint(self.broker, self.domain_intent(), self.domain_risk())

    def safe_mapping(self) -> dict[str, object]:
        """Return every bound field in exact canonical, non-authorizing form."""
        return {
            "broker": self.broker,
            "intent_id": self.intent_id,
            "instrument": self.instrument_id,
            "side": self.side,
            "quantity": _decimal_text(self.quantity),
            "notional": _decimal_text(self.notional),
            "order_type": self.order_type,
            "limit_price": _decimal_text(self.limit_price),
            "trigger_price": _decimal_text(self.trigger_price),
            "stop_loss": _decimal_text(self.stop_loss),
            "take_profit": _decimal_text(self.take_profit),
            "time_in_force": self.time_in_force,
            "product": self.product,
            "session": self.session,
            "snapshot_hash": self.snapshot_hash,
            "created_at": _utc_text(self.created_at),
            "expires_at": _utc_text(self.expires_at),
            "risk_approved_quantity": _decimal_text(self.risk_approved_quantity),
            "risk_approved_notional": _decimal_text(self.risk_approved_notional),
            "portfolio_hash": self.portfolio_hash,
            "risk_decided_at": _utc_text(self.risk_decided_at),
            "risk_expires_at": _utc_text(self.risk_expires_at),
        }


@dataclass(frozen=True, slots=True)
class _ExpectedProposal:
    """Immutable request evidence captured before any injected provider runs."""

    broker: str
    intent: OrderIntent
    risk_decision: RiskDecision
    request_fingerprint: str
    safe_items: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class OrderProposal:
    """One exact typed proposal; a caller echo is never approval evidence."""

    accepted: bool
    reason_codes: tuple[str, ...]
    broker: str
    intent: OrderIntent
    risk_decision: RiskDecision
    request_fingerprint: str

    def __post_init__(self) -> None:
        if (
            type(self.accepted) is not bool
            or type(self.reason_codes) is not tuple
            or len(self.reason_codes) > _MAX_PROPOSAL_REASONS
            or not all(
                type(reason) is str
                and _REASON.fullmatch(reason)
                and not _secret_shaped(reason)
                for reason in self.reason_codes
            )
            or self.broker not in _BROKERS
            or type(self.intent) is not OrderIntent
            or type(self.risk_decision) is not RiskDecision
            or self.request_fingerprint
            != _proposal_fingerprint(self.broker, self.intent, self.risk_decision)
        ):
            raise ValueError("order proposal is malformed")
        if self.accepted:
            if (
                self.reason_codes
                or not self.risk_decision.approved
                or self.risk_decision.reason_codes
            ):
                raise ValueError("accepted proposal is inconsistent")
        elif not self.reason_codes:
            raise ValueError("denied proposal requires stable reasons")

    @classmethod
    def accept(cls, request: BoundOrderRequest) -> OrderProposal:
        """Create an accepted exact proposal from a validated request."""
        if type(request) is not BoundOrderRequest:
            raise ValueError("proposal request is malformed")
        intent = request.domain_intent()
        risk = request.domain_risk()
        return cls(True, (), request.broker, intent, risk, request.fingerprint())


class Task14CliLiveFacade:
    """Factory-registered zero-state marker for the exact Task 14 live bridge."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        pass

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("live CLI facade is immutable")

    def __copy__(self) -> Task14CliLiveFacade:
        raise TypeError("live CLI facades cannot be copied")

    def __deepcopy__(self, memo: object) -> Task14CliLiveFacade:
        del memo
        raise TypeError("live CLI facades cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("live CLI facades cannot be serialized")


@dataclass(frozen=True, slots=True)
class _Task14CliLiveBinding:
    approval_service: ApprovalService
    live_order_service: LiveOrderService
    snapshot: MarketSnapshot
    preflight: PreflightReport
    reconciliation: ReconciliationReport
    store_identity: object


_LIVE_FACADE_BINDINGS: WeakKeyDictionary[Task14CliLiveFacade, _Task14CliLiveBinding] = (
    WeakKeyDictionary()
)


class _LiveDispatch(Protocol):
    def __call__(
        self,
        proposal: OrderProposal,
        confirmation_phrase: str,
    ) -> BrokerOrder: ...


def _live_facade_binding(handle: Task14CliLiveFacade) -> _Task14CliLiveBinding:
    if type(handle) is not Task14CliLiveFacade:
        raise ValueError("live CLI facade is not factory registered")
    binding = _LIVE_FACADE_BINDINGS.get(handle)
    if (
        type(binding) is not _Task14CliLiveBinding
        or type(binding.approval_service) is not ApprovalService
        or type(binding.live_order_service) is not LiveOrderService
        or type(binding.snapshot) is not MarketSnapshot
        or type(binding.preflight) is not PreflightReport
        or type(binding.reconciliation) is not ReconciliationReport
        or binding.live_order_service._approval is not binding.approval_service
        or not binding.live_order_service.audit_boundary_intact
    ):
        raise ValueError("live CLI facade is not factory registered")
    try:
        approval_identity = binding.approval_service.safety_store_identity
        live_identity = binding.live_order_service.safety_store_identity
    except Exception:
        raise ValueError("live CLI facade safety binding is invalid") from None
    if (
        approval_identity is not binding.store_identity
        or live_identity is not binding.store_identity
    ):
        raise ValueError("live CLI facade safety binding is invalid")
    return binding


def _resolve_live_dispatch(
    handle: Task14CliLiveFacade,
) -> tuple[
    _Task14CliLiveBinding,
    _LiveDispatch,
    Callable[..., BrokerOrder],
    Callable[..., OrderConfirmation],
    Callable[..., BrokerOrder],
    Callable[[object], OrderProposal],
]:
    """Resolve one registered marker into a callback-private fixed Task 14 dispatch."""
    binding = _live_facade_binding(handle)
    snapshot = MarketSnapshot.model_validate(
        binding.snapshot.model_dump(mode="python", warnings="error"),
        strict=True,
    )
    preflight = _clone_preflight(binding.preflight)
    reconciliation = ReconciliationReport(
        report_id=binding.reconciliation.report_id,
        broker=binding.reconciliation.broker,
        healthy=binding.reconciliation.healthy,
        reason_codes=binding.reconciliation.reason_codes,
        broker_hash=binding.reconciliation.broker_hash,
        ledger_hash=binding.reconciliation.ledger_hash,
        checked_at=binding.reconciliation.checked_at,
        sequence=binding.reconciliation.sequence,
    )
    implementation = _submit_task14_order
    approval_service = binding.approval_service
    live_order_service = binding.live_order_service
    approval_create_implementation = _APPROVAL_CREATE_IMPLEMENTATION
    live_submit_implementation = _LIVE_SUBMIT_IMPLEMENTATION
    proposal_reconstructor = _reconstruct_order_proposal
    approval_create = approval_create_implementation.__get__(
        approval_service,
        ApprovalService,
    )
    live_submit = live_submit_implementation.__get__(
        live_order_service,
        LiveOrderService,
    )

    def dispatch(
        proposal: OrderProposal,
        confirmation_phrase: str,
        *,
        _implementation: Callable[..., BrokerOrder] = implementation,
        _approval_service: ApprovalService = approval_service,
        _live_order_service: LiveOrderService = live_order_service,
        _snapshot: MarketSnapshot = snapshot,
        _preflight: PreflightReport = preflight,
        _reconciliation: ReconciliationReport = reconciliation,
        _approval_create: Callable[..., OrderConfirmation] = approval_create,
        _live_submit: Callable[..., BrokerOrder] = live_submit,
        _proposal_reconstructor: Callable[[object], OrderProposal] = proposal_reconstructor,
    ) -> BrokerOrder:
        return _implementation(
            approval_service=_approval_service,
            live_order_service=_live_order_service,
            snapshot=_snapshot,
            preflight=_preflight,
            reconciliation=_reconciliation,
            approval_create=_approval_create,
            live_submit=_live_submit,
            proposal_reconstructor=_proposal_reconstructor,
            proposal=proposal,
            confirmation_phrase=confirmation_phrase,
        )

    return (
        binding,
        dispatch,
        implementation,
        approval_create_implementation,
        live_submit_implementation,
        proposal_reconstructor,
    )


def _submit_task14_order(
    *,
    approval_service: ApprovalService,
    live_order_service: LiveOrderService,
    snapshot: MarketSnapshot,
    preflight: PreflightReport,
    reconciliation: ReconciliationReport,
    approval_create: Callable[..., OrderConfirmation],
    live_submit: Callable[..., BrokerOrder],
    proposal_reconstructor: Callable[[object], OrderProposal],
    proposal: OrderProposal,
    confirmation_phrase: str,
) -> BrokerOrder:
    """Issue one confirmation and invoke only the captured exact Task 14 services."""
    exact = proposal_reconstructor(proposal)
    if not exact.accepted:
        raise ValueError("live order requires an accepted exact proposal")
    internal_broker = "ccxt-spot" if exact.broker == "ccxt" else exact.broker
    if internal_broker != live_order_service.broker_name:
        raise ValueError("proposal broker does not match live service")
    if snapshot.instrument_id != exact.intent.instrument_id:
        raise ValueError("proposal snapshot does not match live service")
    confirmation = approval_create(
        exact.intent,
        exact.risk_decision,
        phrase=confirmation_phrase,
        broker=internal_broker,
    )
    return live_submit(
        intent=exact.intent,
        risk_decision=exact.risk_decision,
        snapshot=snapshot,
        confirmation=confirmation,
        preflight=preflight,
        reconciliation=reconciliation,
    )


def create_task14_cli_live_facade(
    *,
    approval_service: ApprovalService,
    live_order_service: LiveOrderService,
    snapshot: MarketSnapshot,
    preflight: PreflightReport,
    reconciliation: ReconciliationReport,
) -> Task14CliLiveFacade:
    """Register one opaque facade around exact, shared Task 14 safety services."""
    if (
        type(approval_service) is not ApprovalService
        or type(live_order_service) is not LiveOrderService
        or type(snapshot) is not MarketSnapshot
        or type(preflight) is not PreflightReport
        or type(reconciliation) is not ReconciliationReport
        or live_order_service._approval is not approval_service
        or not live_order_service.audit_boundary_intact
    ):
        raise ValueError("live CLI facade requires exact shared Task 14 services")
    try:
        store_identity = approval_service.safety_store_identity
        live_store_identity = live_order_service.safety_store_identity
    except Exception:
        raise ValueError("live CLI facade requires exact shared Task 14 services") from None
    if store_identity is not live_store_identity:
        raise ValueError("live CLI facade requires exact shared Task 14 services")
    expected_broker = live_order_service.broker_name
    external_broker = "ccxt" if expected_broker == "ccxt-spot" else expected_broker
    if expected_broker not in _BROKERS and expected_broker != "ccxt-spot":
        raise ValueError("live CLI facade broker is invalid")
    if _validated_preflight(preflight, external_broker) is None:
        raise ValueError("live CLI facade preflight is invalid")
    handle = Task14CliLiveFacade()
    _LIVE_FACADE_BINDINGS[handle] = _Task14CliLiveBinding(
        approval_service,
        live_order_service,
        snapshot,
        preflight,
        reconciliation,
        store_identity,
    )
    return handle


@dataclass(frozen=True, slots=True)
class ServiceContainer:
    """All potentially effectful behavior supplied at one explicit local boundary."""

    research: WorkflowService | None = None
    backtest: WorkflowService | None = None
    paper: WorkflowService | None = None
    preflight: PreflightService | None = None
    proposal: ProposalService | None = None
    live_submission: Task14CliLiveFacade | None = None
    dashboard: Callable[[str], DashboardStatus] | None = None

    def __post_init__(self) -> None:
        if self.live_submission is not None:
            try:
                _live_facade_binding(self.live_submission)
            except (TypeError, ValueError):
                raise ValueError(
                    "live submission requires the exact Task 14 facade that is factory registered"
                ) from None


@dataclass(frozen=True, slots=True)
class _FixtureWorkflowService:
    """Expose only the three packaged, sanitized, no-network fixture workflows."""

    workflow: str

    def run(self, request: WorkflowRequest) -> Mapping[str, object]:
        from market_sentinel.operations.fixture_pipeline import FixturePipelineRunner

        specifications = {
            "RELIANCE@groww": (
                "india",
                datetime(2026, 8, 10, 3, 45, tzinfo=UTC),
                datetime(2026, 8, 10, 3, 48, 30, tzinfo=UTC),
            ),
            "AAPL@alpaca": (
                "us",
                datetime(2026, 8, 10, 13, 30, tzinfo=UTC),
                datetime(2026, 8, 10, 13, 33, 30, tzinfo=UTC),
            ),
            "BTC-USDT@ccxt-spot": (
                "crypto",
                datetime(2026, 8, 10, 0, 0, tzinfo=UTC),
                datetime(2026, 8, 10, 0, 20, 30, tzinfo=UTC),
            ),
        }
        if request.workflow != self.workflow or request.instrument_id not in specifications:
            raise ValueError("fixture workflow request is not allowlisted")
        market, start, cutoff = specifications[request.instrument_id]
        if self.workflow == "backtest":
            if request.start_at != start or request.end_at != cutoff or request.as_of is not None:
                raise ValueError("fixture backtest interval is not exact")
        elif (
            request.as_of != cutoff
            or request.start_at is not None
            or request.end_at is not None
        ):
            raise ValueError("fixture point-in-time request is not exact")

        result = FixturePipelineRunner().run(
            market,
            requested_as_of=cutoff,
            expected_instrument_id=request.instrument_id,
        )
        common: dict[str, object] = {
            "instrument": result.instrument_id,
            "live_ready": False,
            "market": result.market,
            "workflow": self.workflow,
        }
        if self.workflow == "research":
            common.update(
                {
                    "as_of": result.research_packet.as_of.isoformat(),
                    "evidence_count": len(result.research_packet.evidence),
                    "model_id": result.research_packet.model_id,
                    "risk_approved": result.risk_decision.approved,
                }
            )
            return common
        if self.workflow == "backtest":
            metrics = result.backtest.metrics
            common.update(
                {
                    "after_cost": result.backtest.after_cost,
                    "benchmark_excess_return": (
                        None
                        if metrics.benchmark_excess_return is None
                        else str(metrics.benchmark_excess_return)
                    ),
                    "fees": str(result.backtest.total_fees),
                    "maximum_drawdown": str(metrics.maximum_drawdown),
                    "robustness_stressed_return": str(
                        result.backtest.robustness_stressed_return
                    ),
                }
            )
            return common
        common.update(
            {
                "reconciliation_healthy": result.reconciliation.healthy,
                "status": result.paper_order.status.value,
            }
        )
        return common


class _OfflinePreflight:
    def run(self, broker: str) -> PreflightReport:
        normalized = "ccxt-spot" if broker == "ccxt" else broker
        return PreflightReport(
            normalized,
            tuple(
                gate(name, False, "GATE_NOT_SATISFIED")
                for name in sorted(_CAPTURED_PREFLIGHT_MANIFESTS[normalized])
            ),
        )


def _default_dashboard(broker: str) -> DashboardStatus:
    now = datetime.now(UTC)
    manifest_broker = "ccxt-spot" if broker == "ccxt" else broker
    return DashboardStatus(
        generated_at=now,
        data_as_of=now,
        research=DashboardResearch(version="unavailable", fresh=False),
        strategies=(DashboardStrategy("unavailable", "unavailable"),),
        promotion=DashboardPromotion("not_promoted"),
        portfolio=DashboardPortfolio("USD", Decimal("10")),
        risk=DashboardRisk(
            Decimal("0.005"),
            Decimal("0.10"),
            Decimal("0.50"),
            Decimal("0.02"),
            Decimal("0.10"),
        ),
        brokers=(
            DashboardBroker(
                broker,
                tuple(
                    GateResult(name=name, passed=False, reason_code="GATE_NOT_SATISFIED")
                    for name in sorted(_CAPTURED_PREFLIGHT_MANIFESTS[manifest_broker])
                ),
            ),
        ),
        orders=(),
        kill_switches=(DashboardSafetyState(True, "LIVE_SERVICE_UNAVAILABLE"),),
        interlocks=(DashboardSafetyState(True, "LIVE_SERVICE_UNAVAILABLE"),),
        aspirational_target=DashboardAspiration(
            Decimal("10"),
            Decimal("10"),
            Decimal("1000000"),
            Decimal("100000"),
            Decimal("1"),
            Decimal("999990"),
            True,
        ),
    )


def _default_container() -> ServiceContainer:
    return ServiceContainer(
        research=_FixtureWorkflowService("research"),
        backtest=_FixtureWorkflowService("backtest"),
        paper=_FixtureWorkflowService("paper-run"),
        preflight=_OfflinePreflight(),
        dashboard=_default_dashboard,
    )


def build_app(
    container_factory: Callable[[], ServiceContainer] | None = None,
) -> typer.Typer:
    """Build a CLI whose effectful services are supplied explicitly."""
    factory = _default_container if container_factory is None else container_factory
    facade_type = Task14CliLiveFacade
    facade_registry = _LIVE_FACADE_BINDINGS
    binding_resolver = _live_facade_binding
    dispatch_resolver = _resolve_live_dispatch
    dispatch_implementation = _submit_task14_order
    approval_service_type = ApprovalService
    live_order_service_type = LiveOrderService
    approval_create_implementation = _APPROVAL_CREATE_IMPLEMENTATION
    live_submit_implementation = _LIVE_SUBMIT_IMPLEMENTATION
    proposal_reconstructor = _reconstruct_order_proposal
    proposal_validator = _validated_proposal
    proposal_matcher = _proposal_matches_request
    broker_order_emitter = _emit_broker_order
    secret_detector = _secret_shaped
    audit_redactor_anchor = redact_secret_text
    mapping_emitter = _emit
    preflight_manifests = _CAPTURED_PREFLIGHT_MANIFESTS
    preflight_validator = _validated_preflight
    preflight_gate_validator = _valid_gate_fields
    application = typer.Typer(no_args_is_help=True, add_completion=False)

    @application.command("status", hidden=True)
    def status() -> None:
        """Preserve the original inert compatibility probe without exposing a workflow."""
        typer.echo("mode=research live_ready=false")

    @application.command("research")
    def research(instrument: str = typer.Option(...), as_of: str = typer.Option(...)) -> None:
        request = _workflow_request("research", instrument, as_of=as_of)
        _run_workflow(factory, "research", request)

    @application.command("backtest")
    def backtest(
        instrument: str = typer.Option(...),
        start: str = typer.Option(...),
        end: str = typer.Option(...),
    ) -> None:
        request = _workflow_request("backtest", instrument, start=start, end=end)
        _run_workflow(factory, "backtest", request)

    @application.command("paper-run")
    def paper_run(instrument: str = typer.Option(...), as_of: str = typer.Option(...)) -> None:
        request = _workflow_request("paper-run", instrument, as_of=as_of)
        _run_workflow(factory, "paper", request)

    @application.command("live-preflight")
    def live_preflight(broker: str = typer.Option(...)) -> None:
        normalized = _require_broker(broker)
        container = _get_container(factory)
        if container.preflight is None:
            _error("PREFLIGHT_SERVICE_UNAVAILABLE", 20)
        preflight_failed = False
        try:
            report = container.preflight.run(normalized)
        except Exception:
            preflight_failed = True
            report = None
        if preflight_failed:
            _error("PREFLIGHT_UNAVAILABLE", 20)
        if (
            _CAPTURED_PREFLIGHT_MANIFESTS is not preflight_manifests
            or _validated_preflight is not preflight_validator
            or _valid_gate_fields is not preflight_gate_validator
        ):
            _error("PREFLIGHT_INVALID", 20)
        manifest = preflight_validator(
            report,
            normalized,
            manifests=preflight_manifests,
            gate_validator=preflight_gate_validator,
        )
        if manifest is None:
            _error("PREFLIGHT_INVALID", 20)
        missing_gates, ready = manifest
        output = {
            "broker": normalized,
            "missing_gates": missing_gates,
            "ready": ready,
        }
        _emit(output)
        if not ready:
            raise typer.Exit(20)

    @application.command("propose-order")
    def propose_order(
        broker: str | None = typer.Option(None),
        intent_id: str | None = typer.Option(None),
        instrument: str | None = typer.Option(None),
        side: str | None = typer.Option(None),
        quantity: str | None = typer.Option(None),
        notional: str | None = typer.Option(None),
        order_type: str | None = typer.Option(None),
        limit_price: str | None = typer.Option(None),
        trigger_price: str | None = typer.Option(None),
        stop_loss: str | None = typer.Option(None),
        take_profit: str | None = typer.Option(None),
        time_in_force: str | None = typer.Option(None),
        product: str | None = typer.Option(None),
        session: str | None = typer.Option(None),
        snapshot_hash: str | None = typer.Option(None),
        created_at: str | None = typer.Option(None),
        expires_at: str | None = typer.Option(None),
        risk_approved_quantity: str | None = typer.Option(None),
        risk_approved_notional: str | None = typer.Option(None),
        portfolio_hash: str | None = typer.Option(None),
        risk_decided_at: str | None = typer.Option(None),
        risk_expires_at: str | None = typer.Option(None),
    ) -> None:
        request = _order_request(locals())
        expected = _expected_proposal(request)
        container = _get_container(factory)
        if container.proposal is None:
            _error("PROPOSAL_SERVICE_UNAVAILABLE", 30)
        proposal_failed = False
        try:
            result = container.proposal.propose(request)
        except Exception:
            proposal_failed = True
            result = None
        if proposal_failed:
            _error("PROPOSAL_FAILED", 30)
        if (
            _reconstruct_order_proposal is not proposal_reconstructor
            or _validated_proposal is not proposal_validator
            or _proposal_matches_request is not proposal_matcher
        ):
            _error("PROPOSAL_RESPONSE_INVALID", 30)
        proposal = proposal_validator(
            result,
            expected,
            reconstructor=proposal_reconstructor,
            matcher=proposal_matcher,
        )
        if proposal is None:
            _error("PROPOSAL_RESPONSE_INVALID", 30)
        if not proposal.accepted:
            _emit({"accepted": False, "reason_codes": list(proposal.reason_codes)})
            raise typer.Exit(30)
        _emit(_proposal_mapping(proposal, expected))

    @application.command("submit-confirmed-order")
    def submit_confirmed_order(
        broker: str | None = typer.Option(None),
        intent_id: str | None = typer.Option(None),
        instrument: str | None = typer.Option(None),
        side: str | None = typer.Option(None),
        quantity: str | None = typer.Option(None),
        notional: str | None = typer.Option(None),
        order_type: str | None = typer.Option(None),
        limit_price: str | None = typer.Option(None),
        trigger_price: str | None = typer.Option(None),
        stop_loss: str | None = typer.Option(None),
        take_profit: str | None = typer.Option(None),
        time_in_force: str | None = typer.Option(None),
        product: str | None = typer.Option(None),
        session: str | None = typer.Option(None),
        snapshot_hash: str | None = typer.Option(None),
        created_at: str | None = typer.Option(None),
        expires_at: str | None = typer.Option(None),
        risk_approved_quantity: str | None = typer.Option(None),
        risk_approved_notional: str | None = typer.Option(None),
        portfolio_hash: str | None = typer.Option(None),
        risk_decided_at: str | None = typer.Option(None),
        risk_expires_at: str | None = typer.Option(None),
        confirm_real_money: str | None = typer.Option(None),
    ) -> None:
        if confirm_real_money != CONFIRMATION_PHRASE:
            _emit({"error": "CONFIRMATION_PHRASE_REQUIRED", "required_phrase": CONFIRMATION_PHRASE})
            raise typer.Exit(21)
        request = _order_request(locals())
        expected = _expected_proposal(request)
        exact_proposal = OrderProposal(
            True,
            (),
            expected.broker,
            expected.intent,
            expected.risk_decision,
            expected.request_fingerprint,
        )
        container = _get_container(factory)
        if container.proposal is not None:
            _error("PROPOSAL_RESPONSE_INVALID", 30)
        live_facade = container.live_submission
        if live_facade is None:
            _error("LIVE_SERVICE_UNAVAILABLE", 30)
        try:
            (
                binding,
                live_dispatch,
                resolved_implementation,
                resolved_approval_create,
                resolved_live_submit,
                resolved_proposal_reconstructor,
            ) = dispatch_resolver(live_facade)
        except Exception:
            _error("LIVE_SERVICE_INVALID", 30)
        approval_service = binding.approval_service
        live_order_service = binding.live_order_service
        snapshot = binding.snapshot
        preflight = binding.preflight
        reconciliation = binding.reconciliation
        store_identity = binding.store_identity
        def binding_is_intact() -> bool:
            try:
                return bool(
                container.live_submission is live_facade
                and Task14CliLiveFacade is facade_type
                and "submit" not in facade_type.__dict__
                and _LIVE_FACADE_BINDINGS is facade_registry
                and _live_facade_binding is binding_resolver
                and _resolve_live_dispatch is dispatch_resolver
                and _submit_task14_order is dispatch_implementation
                and resolved_implementation is dispatch_implementation
                and ApprovalService is approval_service_type
                and LiveOrderService is live_order_service_type
                and _APPROVAL_CREATE_IMPLEMENTATION is approval_create_implementation
                and _LIVE_SUBMIT_IMPLEMENTATION is live_submit_implementation
                and approval_service_type.create is approval_create_implementation
                and live_order_service_type.submit_confirmed is live_submit_implementation
                and resolved_approval_create is approval_create_implementation
                and resolved_live_submit is live_submit_implementation
                and _reconstruct_order_proposal is proposal_reconstructor
                and _validated_proposal is proposal_validator
                and _proposal_matches_request is proposal_matcher
                and resolved_proposal_reconstructor is proposal_reconstructor
                and _emit_broker_order is broker_order_emitter
                and _secret_shaped is secret_detector
                and _emit is mapping_emitter
                and facade_registry.get(live_facade) is binding
                and binding.approval_service is approval_service
                and binding.live_order_service is live_order_service
                and binding.snapshot is snapshot
                and binding.preflight is preflight
                and binding.reconciliation is reconciliation
                and binding.store_identity is store_identity
                and live_order_service._approval is approval_service
                and live_order_service._audit_redactor is audit_redactor_anchor
                and approval_service.safety_store_identity is store_identity
                and live_order_service.safety_store_identity is store_identity
                )
            except Exception:
                return False

        if not binding_is_intact():
            _error("LIVE_SERVICE_INVALID", 30)
        submission_failed = False
        try:
            result = live_dispatch(exact_proposal, confirm_real_money)
        except Exception:
            submission_failed = True
            result = None
        if submission_failed:
            _error("LIVE_SUBMISSION_REJECTED", 30)
        if type(result) is not BrokerOrder:
            _error("LIVE_RESPONSE_INVALID", 30)
        if not binding_is_intact():
            _error("LIVE_RESPONSE_INVALID", 30)
        broker_order_emitter(
            result,
            secret_detector=secret_detector,
            emit=mapping_emitter,
        )

    @application.command("export-dashboard")
    def export_dashboard(
        broker: Annotated[str, typer.Option()],
        path: Annotated[Path, typer.Option()],
    ) -> None:
        normalized = _require_broker(broker)
        container = _get_container(factory)
        if container.dashboard is None:
            _error("DASHBOARD_SERVICE_UNAVAILABLE", 40)
        export_failed = False
        try:
            snapshot = container.dashboard(normalized)
            written = write_dashboard(snapshot, path)
        except Exception:
            export_failed = True
            written = None
        if export_failed or written is None:
            _error("DASHBOARD_EXPORT_FAILED", 40)
        _emit({"broker": normalized, "path": str(written), "schema_version": 1})

    return application


def _workflow_request(
    workflow: str,
    instrument: str,
    *,
    as_of: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> WorkflowRequest:
    invalid = False
    try:
        request = WorkflowRequest(
            workflow,
            instrument,
            as_of=None if as_of is None else _parse_datetime(as_of),
            start_at=None if start is None else _parse_datetime(start),
            end_at=None if end is None else _parse_datetime(end),
        )
    except (TypeError, ValueError):
        invalid = True
        request = None
    if invalid or request is None:
        _error("WORKFLOW_PARAMETERS_INVALID", 10)
    return request


def _run_workflow(
    factory: Callable[[], ServiceContainer],
    service_name: str,
    request: WorkflowRequest,
) -> None:
    container = _get_container(factory)
    service = getattr(container, service_name)
    if service is None:
        _error("WORKFLOW_SERVICE_UNAVAILABLE", 10)
    workflow_failed = False
    try:
        result = service.run(request)
    except Exception:
        workflow_failed = True
        result = None
    if workflow_failed:
        _error("WORKFLOW_FAILED", 10)
    _emit_service_result(result)


def _order_request(arguments: Mapping[str, object]) -> BoundOrderRequest:
    invalid = False
    try:
        required = {
            name: _required_text(arguments.get(name))
            for name in (
                "broker",
                "intent_id",
                "instrument",
                "side",
                "order_type",
                "time_in_force",
                "product",
                "session",
                "snapshot_hash",
                "created_at",
                "expires_at",
                "portfolio_hash",
                "risk_decided_at",
                "risk_expires_at",
            )
        }
        request = BoundOrderRequest(
            broker=_require_broker(required["broker"]),
            intent_id=required["intent_id"],
            instrument_id=required["instrument"],
            side=required["side"],
            quantity=_parse_optional_decimal(arguments.get("quantity")),
            notional=_parse_optional_decimal(arguments.get("notional")),
            order_type=required["order_type"],
            limit_price=_parse_optional_decimal(arguments.get("limit_price")),
            trigger_price=_parse_optional_decimal(arguments.get("trigger_price")),
            stop_loss=_parse_optional_decimal(arguments.get("stop_loss")),
            take_profit=_parse_optional_decimal(arguments.get("take_profit")),
            time_in_force=required["time_in_force"],
            product=required["product"],
            session=required["session"],
            snapshot_hash=required["snapshot_hash"],
            created_at=_parse_datetime(required["created_at"]),
            expires_at=_parse_datetime(required["expires_at"]),
            risk_approved_quantity=_parse_optional_decimal(arguments.get("risk_approved_quantity")),
            risk_approved_notional=_parse_optional_decimal(arguments.get("risk_approved_notional")),
            portfolio_hash=required["portfolio_hash"],
            risk_decided_at=_parse_datetime(required["risk_decided_at"]),
            risk_expires_at=_parse_datetime(required["risk_expires_at"]),
        )
    except (CanonicalEncodingError, InvalidOperation, TypeError, ValueError):
        invalid = True
        request = None
    if invalid or request is None:
        _error("ORDER_PARAMETERS_INVALID", 22)
    return request


def _get_container(factory: Callable[[], ServiceContainer]) -> ServiceContainer:
    unavailable = False
    try:
        container = factory()
    except Exception:
        unavailable = True
        container = None
    if unavailable:
        _error("SERVICE_CONTAINER_UNAVAILABLE", 50)
    if type(container) is not ServiceContainer:
        _error("SERVICE_CONTAINER_INVALID", 50)
    if container.live_submission is not None:
        try:
            _live_facade_binding(container.live_submission)
        except (TypeError, ValueError):
            _error("SERVICE_CONTAINER_INVALID", 50)
    return container


def _emit_service_result(result: object) -> None:
    if type(result) not in {dict, MappingProxyType}:
        _error("SERVICE_RESPONSE_INVALID", 50)
    invalid = False
    try:
        prepared = safe_json_mapping(cast(Mapping[str, object], result))
    except Exception:
        invalid = True
        prepared = None
    if invalid or prepared is None:
        _error("SERVICE_RESPONSE_INVALID", 50)
    _emit(prepared)


def _emit(payload: Mapping[str, object]) -> None:
    failed = False
    try:
        text = json.dumps(
            dict(payload), allow_nan=False, separators=(",", ":"), sort_keys=True
        )
    except Exception:
        failed = True
        text = ""
    if failed:
        typer.echo('{"error":"OUTPUT_SERIALIZATION_FAILED"}')
        raise typer.Exit(50)
    typer.echo(text)


def _error(reason_code: str, status: int) -> Never:
    _emit({"error": reason_code})
    raise typer.Exit(status)


def _require_broker(value: object) -> str:
    if type(value) is not str or value not in _BROKERS:
        _error("BROKER_INVALID", 2)
    return value


def _reconstruct_order_proposal(value: object) -> OrderProposal:
    """Deeply reconstruct one exact proposal without trusting frozen object state."""
    if (
        type(value) is not OrderProposal
        or type(value.accepted) is not bool
        or type(value.reason_codes) is not tuple
        or len(value.reason_codes) > _MAX_PROPOSAL_REASONS
        or any(
            type(reason) is not str
            or len(reason) > 64
            or _REASON.fullmatch(reason) is None
            or _secret_shaped(reason)
            for reason in value.reason_codes
        )
        or value.accepted is not (not value.reason_codes)
        or type(value.broker) is not str
        or value.broker not in _BROKERS
        or type(value.intent) is not OrderIntent
        or type(value.risk_decision) is not RiskDecision
        or type(value.request_fingerprint) is not str
        or _HASH.fullmatch(value.request_fingerprint) is None
    ):
        raise ValueError("order proposal is malformed")
    intent_strings = (
        value.intent.intent_id,
        value.intent.instrument_id,
        value.intent.time_in_force,
        value.intent.product,
        value.intent.session,
        value.intent.snapshot_hash,
    )
    risk_reasons = value.risk_decision.reason_codes
    if (
        any(not _bounded_model_text(item, 256) for item in intent_strings)
        or type(value.risk_decision.approved) is not bool
        or type(risk_reasons) is not tuple
        or len(risk_reasons) > _MAX_PROPOSAL_REASONS
        or any(
            not _bounded_model_text(reason, 64)
            or _REASON.fullmatch(reason) is None
            for reason in risk_reasons
        )
        or not _bounded_model_text(value.risk_decision.portfolio_hash, 128)
    ):
        raise ValueError("order proposal is malformed")
    intent = OrderIntent.model_validate(
        value.intent.model_dump(mode="python", warnings="error"),
        strict=True,
    )
    risk_decision = RiskDecision.model_validate(
        value.risk_decision.model_dump(mode="python", warnings="error"),
        strict=True,
    )
    return OrderProposal(
        accepted=value.accepted,
        reason_codes=tuple(value.reason_codes),
        broker=value.broker,
        intent=intent,
        risk_decision=risk_decision,
        request_fingerprint=value.request_fingerprint,
    )


def _bounded_model_text(value: object, limit: int) -> bool:
    if type(value) is not str or not value or len(value) > limit:
        return False
    try:
        return len(value.encode("utf-8")) <= limit
    except UnicodeError:
        return False


def _validated_proposal(
    value: object,
    expected: _ExpectedProposal,
    *,
    reconstructor: Callable[[object], OrderProposal],
    matcher: Callable[[OrderProposal, _ExpectedProposal], bool],
) -> OrderProposal | None:
    try:
        proposal = reconstructor(value)
        matches = matcher(proposal, expected)
    except Exception:
        return None
    return proposal if matches else None


def _proposal_matches_request(proposal: OrderProposal, expected: _ExpectedProposal) -> bool:
    return bool(
        proposal.broker == expected.broker
        and proposal.intent == expected.intent
        and proposal.risk_decision == expected.risk_decision
        and proposal.request_fingerprint == expected.request_fingerprint
    )


def _expected_proposal(request: BoundOrderRequest) -> _ExpectedProposal:
    """Freeze every request-derived value before calling injected proposal code."""
    return _ExpectedProposal(
        broker=request.broker,
        intent=request.domain_intent(),
        risk_decision=request.domain_risk(),
        request_fingerprint=request.fingerprint(),
        safe_items=tuple(request.safe_mapping().items()),
    )


def _clone_preflight(report: PreflightReport) -> PreflightReport:
    if (
        type(report) is not PreflightReport
        or type(report.broker) is not str
        or type(report.gates) is not tuple
        or len(report.gates) > _MAX_PREFLIGHT_GATES
    ):
        raise ValueError("preflight report is malformed")
    gates: list[GateResult] = []
    for item in report.gates:
        if not _valid_gate_fields(item):
            raise ValueError("preflight report is malformed")
        gates.append(
            GateResult(
                name=item.name,
                passed=item.passed,
                reason_code=item.reason_code,
            )
        )
    return PreflightReport(report.broker, tuple(gates))


def _proposal_mapping(
    proposal: OrderProposal,
    expected: _ExpectedProposal,
) -> dict[str, object]:
    mapping = dict(expected.safe_items)
    mapping.update(
        {
            "accepted": True,
            "reason_codes": [],
            "request_fingerprint": proposal.request_fingerprint,
        }
    )
    if any(
        _secret_shaped(value)
        for value in mapping.values()
        if type(value) is str
    ):
        _error("PROPOSAL_RESPONSE_INVALID", 30)
    return mapping


def _emit_broker_order(
    order: BrokerOrder,
    *,
    secret_detector: Callable[[object], bool],
    emit: Callable[[Mapping[str, object]], None],
) -> None:
    if any(
        secret_detector(value)
        for value in (order.order_id, order.client_order_id, order.broker, order.instrument_id)
    ):
        _error("LIVE_RESPONSE_INVALID", 30)
    emit(
        {
            "order_id": order.order_id,
            "client_order_id": order.client_order_id,
            "broker": order.broker,
            "instrument": order.instrument_id,
            "status": order.status.value,
            "requested_quantity": _decimal_text(order.requested_quantity),
            "requested_notional": _decimal_text(order.requested_notional),
            "filled_quantity": _decimal_text(order.filled_quantity),
            "average_fill_price": _decimal_text(order.average_fill_price),
            "submitted_at": _utc_text(order.submitted_at),
            "updated_at": _utc_text(order.updated_at),
        }
    )


def _proposal_fingerprint(broker: str, intent: OrderIntent, risk: RiskDecision) -> str:
    if broker not in _BROKERS or type(intent) is not OrderIntent or type(risk) is not RiskDecision:
        raise ValueError("proposal fingerprint input is malformed")
    payload = {
        "broker": broker,
        "intent": {
            "intent_id": intent.intent_id,
            "instrument_id": intent.instrument_id,
            "side": intent.side.value,
            "quantity": _decimal_text(intent.quantity),
            "notional": _decimal_text(intent.notional),
            "order_type": intent.order_type.value,
            "limit_price": _decimal_text(intent.limit_price),
            "trigger_price": _decimal_text(intent.trigger_price),
            "stop_loss": _decimal_text(intent.stop_loss),
            "take_profit": _decimal_text(intent.take_profit),
            "time_in_force": intent.time_in_force,
            "product": intent.product,
            "session": intent.session,
            "snapshot_hash": intent.snapshot_hash,
            "created_at": _utc_text(intent.created_at),
            "expires_at": _utc_text(intent.expires_at),
        },
        "risk_decision_hash": risk_decision_hash(risk),
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(
        b"omnimarket-sentinel:cli-order-proposal:v1\x00" + encoded.encode("ascii")
    ).hexdigest()


_secret_shaped = secret_text_present


def _validated_preflight(
    report: object,
    broker: str,
    *,
    manifests: Mapping[str, frozenset[str]] = _CAPTURED_PREFLIGHT_MANIFESTS,
    gate_validator: Callable[[object], bool] | None = None,
) -> tuple[list[str], bool] | None:
    validator = _valid_gate_fields if gate_validator is None else gate_validator
    expected_broker = "ccxt-spot" if broker == "ccxt" else broker
    if (
        type(report) is not PreflightReport
        or report.broker != expected_broker
        or type(report.gates) is not tuple
    ):
        return None
    required = manifests.get(expected_broker)
    if required is None:
        return None
    if (
        len(report.gates) > _MAX_PREFLIGHT_GATES
        or len(report.gates) != len(required)
    ):
        return None
    names: list[str] = []
    missing: list[str] = []
    for gate_result in report.gates:
        if not validator(gate_result):
            return None
        names.append(gate_result.name)
        if not gate_result.passed:
            missing.append(gate_result.name)
    if set(names) != required:
        return None
    return sorted(missing), not missing


def _valid_gate_fields(value: object) -> bool:
    if type(value) is not GateResult:
        return False
    name = value.name
    reason = value.reason_code
    if (
        type(name) is not str
        or not name
        or len(name) > _MAX_GATE_NAME_CHARS
        or type(reason) is not str
        or len(reason) > _MAX_GATE_REASON_CHARS
        or type(value.passed) is not bool
    ):
        return False
    try:
        if len(name.encode("utf-8")) > _MAX_GATE_NAME_CHARS:
            return False
        if len(reason.encode("utf-8")) > _MAX_GATE_REASON_CHARS:
            return False
    except UnicodeError:
        return False
    return bool(
        _REASON.fullmatch(reason)
        and (not value.passed or reason == "OK")
        and (value.passed or reason != "OK")
    )


def _require_instrument(value: object) -> None:
    if type(value) is not str or not _INSTRUMENT.fullmatch(value):
        raise ValueError("instrument identity is invalid")


def _required_text(value: object) -> str:
    if type(value) is not str or not value or len(value) > 256:
        raise ValueError("required parameter is absent")
    return value


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _aware_utc(parsed)


def _aware_utc(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be aware")
    return value.astimezone(UTC)


def _parse_optional_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    if type(value) is not str or not value or len(value) > 4_096:
        raise ValueError("decimal parameter is invalid")
    decimal = Decimal(value)
    _positive_decimal(decimal)
    canonical_decimal(decimal)
    return decimal


def _positive_decimal(value: Decimal) -> None:
    if type(value) is not Decimal or not value.is_finite() or value <= 0:
        raise ValueError("numeric parameter must be finite and positive")


def _require_exact_size(quantity: Decimal | None, notional: Decimal | None) -> None:
    if (quantity is None) == (notional is None):
        raise ValueError("exactly one size is required")
    _positive_decimal(quantity if quantity is not None else notional)  # type: ignore[arg-type]


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else canonical_decimal(value)


def _utc_text(value: datetime) -> str:
    return _aware_utc(value).isoformat().replace("+00:00", "Z")


app = build_app()

if __name__ == "__main__":
    app()
