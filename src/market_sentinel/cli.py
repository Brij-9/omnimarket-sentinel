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
from typing import Annotated, Never, Protocol, cast

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
from market_sentinel.execution.approval import ApprovalService, risk_decision_hash
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
    DashboardValidationError,
    safe_json_mapping,
)
from market_sentinel.operations.dashboard import (
    export_dashboard as write_dashboard,
)

CONFIRMATION_PHRASE = "I_CONFIRM_REAL_MONEY_ORDER"
_BROKERS = ("alpaca", "groww", "ccxt")
_INSTRUMENT = re.compile(r"^[A-Z0-9][A-Z0-9._/-]{0,31}@[a-z0-9][a-z0-9-]{0,31}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_ROUTE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_REASON = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_SECRET_TEXT = re.compile(
    r"(?i)(?:bearer\s+|-----BEGIN|private.?key|access.?key|api.?key|"
    r"secret|password|credential|gh[pousr]_|sk-[a-z0-9_-]{8,})"
)


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
            or not all(
                type(reason) is str and _REASON.fullmatch(reason)
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
    """Exact CLI bridge that can only call the real authenticated Task 14 service."""

    def __init__(
        self,
        *,
        approval_service: ApprovalService,
        live_order_service: LiveOrderService,
        snapshot: MarketSnapshot,
        preflight: PreflightReport,
        reconciliation: ReconciliationReport,
    ) -> None:
        if (
            type(approval_service) is not ApprovalService
            or type(live_order_service) is not LiveOrderService
            or type(snapshot) is not MarketSnapshot
            or type(preflight) is not PreflightReport
            or type(reconciliation) is not ReconciliationReport
            or approval_service.safety_store_identity
            is not live_order_service.safety_store_identity
        ):
            raise ValueError("live CLI facade requires exact shared Task 14 services")
        self._approval = approval_service
        self._live = live_order_service
        self._snapshot = snapshot
        self._preflight = preflight
        self._reconciliation = reconciliation

    def submit(self, proposal: OrderProposal, confirmation_phrase: str) -> BrokerOrder:
        """Durably issue confirmation, then delegate every live gate to Task 14."""
        if type(proposal) is not OrderProposal or not proposal.accepted:
            raise ValueError("live order requires an accepted exact proposal")
        intent = OrderIntent.model_validate(proposal.intent.model_dump(mode="python"))
        risk_decision = RiskDecision.model_validate(
            proposal.risk_decision.model_dump(mode="python")
        )
        if intent != proposal.intent or risk_decision != proposal.risk_decision:
            raise ValueError("live proposal did not survive exact domain revalidation")
        if proposal.broker != self._live.broker_name:
            raise ValueError("proposal broker does not match live service")
        if self._snapshot.instrument_id != intent.instrument_id:
            raise ValueError("proposal snapshot does not match live service")
        confirmation = self._approval.create(
            intent,
            risk_decision,
            phrase=confirmation_phrase,
            broker=proposal.broker,
        )
        return self._live.submit_confirmed(
            intent=intent,
            risk_decision=risk_decision,
            snapshot=self._snapshot,
            confirmation=confirmation,
            preflight=self._preflight,
            reconciliation=self._reconciliation,
        )


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
        if (
            self.live_submission is not None
            and type(self.live_submission) is not Task14CliLiveFacade
        ):
            raise ValueError("live submission requires the exact Task 14 facade")


class _OfflinePreflight:
    def run(self, broker: str) -> PreflightReport:
        normalized = "ccxt-spot" if broker == "ccxt" else broker
        return PreflightReport(
            normalized,
            tuple(
                gate(name, False, "GATE_NOT_SATISFIED")
                for name in sorted(required_gate_names(normalized))
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
                    for name in sorted(required_gate_names(manifest_broker))
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
    return ServiceContainer(preflight=_OfflinePreflight(), dashboard=_default_dashboard)


def build_app(
    container_factory: Callable[[], ServiceContainer] | None = None,
) -> typer.Typer:
    """Build a CLI whose effectful services are supplied explicitly."""
    factory = _default_container if container_factory is None else container_factory
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
        manifest = _validated_preflight(report, normalized)
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
        if type(result) is not OrderProposal:
            _error("PROPOSAL_RESPONSE_INVALID", 30)
        if not _proposal_matches_request(result, request):
            _error("PROPOSAL_RESPONSE_INVALID", 30)
        if not result.accepted:
            _emit({"accepted": False, "reason_codes": list(result.reason_codes)})
            raise typer.Exit(30)
        _emit(_proposal_mapping(result, request))

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
        container = _get_container(factory)
        if container.live_submission is None:
            _error("LIVE_SERVICE_UNAVAILABLE", 30)
        if type(container.live_submission) is not Task14CliLiveFacade:
            _error("LIVE_SERVICE_INVALID", 30)
        if container.proposal is None:
            _error("PROPOSAL_SERVICE_UNAVAILABLE", 30)
        proposal_failed = False
        try:
            proposal = container.proposal.propose(request)
        except Exception:
            proposal_failed = True
            proposal = None
        if proposal_failed:
            _error("PROPOSAL_FAILED", 30)
        if (
            type(proposal) is not OrderProposal
            or not proposal.accepted
            or not _proposal_matches_request(proposal, request)
        ):
            _error("PROPOSAL_RESPONSE_INVALID", 30)
        submission_failed = False
        try:
            result = container.live_submission.submit(proposal, confirm_real_money)
        except Exception:
            submission_failed = True
            result = None
        if submission_failed:
            _error("LIVE_SUBMISSION_REJECTED", 30)
        if type(result) is not BrokerOrder:
            _error("LIVE_RESPONSE_INVALID", 30)
        _emit_broker_order(result)

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
    return container


def _emit_service_result(result: object) -> None:
    if not isinstance(result, Mapping):
        _error("SERVICE_RESPONSE_INVALID", 50)
    invalid = False
    try:
        prepared = safe_json_mapping(cast(Mapping[str, object], result))
    except DashboardValidationError:
        invalid = True
        prepared = None
    if invalid or prepared is None:
        _error("SERVICE_RESPONSE_INVALID", 50)
    _emit(prepared)


def _emit(payload: Mapping[str, object]) -> None:
    typer.echo(json.dumps(dict(payload), allow_nan=False, separators=(",", ":"), sort_keys=True))


def _error(reason_code: str, status: int) -> Never:
    _emit({"error": reason_code})
    raise typer.Exit(status)


def _require_broker(value: object) -> str:
    if type(value) is not str or value not in _BROKERS:
        _error("BROKER_INVALID", 2)
    return value


def _proposal_matches_request(proposal: OrderProposal, request: BoundOrderRequest) -> bool:
    try:
        return (
            proposal.broker == request.broker
            and proposal.intent == request.domain_intent()
            and proposal.risk_decision == request.domain_risk()
            and proposal.request_fingerprint == request.fingerprint()
        )
    except (TypeError, ValueError, CanonicalEncodingError):
        return False


def _proposal_mapping(
    proposal: OrderProposal,
    request: BoundOrderRequest,
) -> dict[str, object]:
    mapping = request.safe_mapping()
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


def _emit_broker_order(order: BrokerOrder) -> None:
    if any(
        _secret_shaped(value)
        for value in (order.order_id, order.client_order_id, order.broker, order.instrument_id)
    ):
        _error("LIVE_RESPONSE_INVALID", 30)
    _emit(
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


def _secret_shaped(value: object) -> bool:
    return type(value) is str and _SECRET_TEXT.search(value) is not None


def _validated_preflight(
    report: object,
    broker: str,
) -> tuple[list[str], bool] | None:
    expected_broker = "ccxt-spot" if broker == "ccxt" else broker
    if (
        type(report) is not PreflightReport
        or report.broker != expected_broker
        or type(report.gates) is not tuple
    ):
        return None
    required = required_gate_names(expected_broker)
    names: list[str] = []
    missing: list[str] = []
    for gate_result in report.gates:
        if (
            type(gate_result) is not GateResult
            or type(gate_result.name) is not str
            or not gate_result.name
            or type(gate_result.passed) is not bool
            or type(gate_result.reason_code) is not str
            or not gate_result.reason_code
        ):
            return None
        names.append(gate_result.name)
        if not gate_result.passed:
            missing.append(gate_result.name)
    if len(names) != len(required) or set(names) != required:
        return None
    return sorted(missing), not missing


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
