"""Dependency-injected, fail-closed local control center."""

from __future__ import annotations

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
from market_sentinel.domain.models import GateResult
from market_sentinel.execution.canonical import CanonicalEncodingError, canonical_decimal
from market_sentinel.operations.dashboard import (
    DashboardStatus,
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


class WorkflowService(Protocol):
    def run(self, request: WorkflowRequest) -> Mapping[str, object]: ...


class PreflightService(Protocol):
    def run(self, broker: str) -> PreflightReport: ...


class ProposalService(Protocol):
    def propose(self, request: BoundOrderRequest) -> Mapping[str, object]: ...


class LiveSubmissionService(Protocol):
    """Injected facade that must own the authenticated Task 14 live flow."""

    def submit(
        self,
        request: BoundOrderRequest,
        confirmation_phrase: str,
    ) -> Mapping[str, object]: ...


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
        if self.broker not in _BROKERS or not self.intent_id or len(self.intent_id) > 128:
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
        _require_exact_size(self.risk_approved_quantity, self.risk_approved_notional)
        if (
            self.quantity != self.risk_approved_quantity
            or self.notional != self.risk_approved_notional
        ):
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
            type(item) is str and 0 < len(item) <= 64
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
class ServiceContainer:
    """All potentially effectful behavior supplied at one explicit local boundary."""

    research: WorkflowService | None = None
    backtest: WorkflowService | None = None
    paper: WorkflowService | None = None
    preflight: PreflightService | None = None
    proposal: ProposalService | None = None
    live_submission: LiveSubmissionService | None = None
    dashboard: Callable[[str], DashboardStatus] | None = None


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
        research={"status": "unavailable", "fresh": False},
        strategies=(),
        promotion={"status": "not_promoted"},
        portfolio={"currency": "USD", "equity": Decimal("10")},
        risk={
            "max_trade_risk_fraction": Decimal("0.005"),
            "max_position_fraction": Decimal("0.10"),
            "max_gross_exposure_fraction": Decimal("0.50"),
            "max_daily_loss_fraction": Decimal("0.02"),
            "max_drawdown_fraction": Decimal("0.10"),
        },
        brokers=(
            {
                "name": broker,
                "missing_gates": tuple(sorted(required_gate_names(manifest_broker))),
                "ready": False,
            },
        ),
        orders=(),
        kill_switches=({"active": True, "reason_code": "LIVE_SERVICE_UNAVAILABLE"},),
        interlocks=({"active": True, "reason_code": "LIVE_SERVICE_UNAVAILABLE"},),
        aspirational_target={
            "starting_capital": Decimal("10"),
            "current_equity": Decimal("10"),
            "target": Decimal("1000000"),
            "required_multiple": Decimal("100000"),
            "reporting_only": True,
        },
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
        if not isinstance(result, Mapping):
            _error("PROPOSAL_RESPONSE_INVALID", 30)
        _emit(request.safe_mapping())

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
        submission_failed = False
        try:
            result = container.live_submission.submit(request, confirm_real_money)
        except Exception:
            submission_failed = True
            result = None
        if submission_failed:
            _error("LIVE_SUBMISSION_REJECTED", 30)
        _emit_service_result(result)

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
