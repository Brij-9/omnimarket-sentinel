"""Behavioral tests for the dependency-injected command surface."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import click
from typer.main import get_command
from typer.testing import CliRunner

from market_sentinel.brokers.preflight import PreflightReport, gate, required_gate_names
from market_sentinel.cli import (
    CONFIRMATION_PHRASE,
    BoundOrderRequest,
    ServiceContainer,
    WorkflowRequest,
    build_app,
)
from market_sentinel.operations.dashboard import DashboardStatus

COMMANDS = {
    "research",
    "backtest",
    "paper-run",
    "live-preflight",
    "propose-order",
    "submit-confirmed-order",
    "export-dashboard",
}


class _Workflow:
    def run(self, request: WorkflowRequest) -> Mapping[str, object]:
        return {"instrument": request.instrument_id, "workflow": request.workflow}


class _FailingWorkflow:
    def run(self, request: WorkflowRequest) -> Mapping[str, object]:
        del request
        raise RuntimeError("provider-secret-must-not-survive")


class _Preflight:
    def run(self, broker: str) -> PreflightReport:
        normalized = "ccxt-spot" if broker == "ccxt" else broker
        gates = tuple(
            gate(name, name.endswith("MODE"))
            for name in sorted(required_gate_names(normalized))
        )
        return PreflightReport(normalized, gates)


class _Proposal:
    def propose(self, request: BoundOrderRequest) -> Mapping[str, object]:
        return request.safe_mapping()


class _PartialProposal:
    def propose(self, request: BoundOrderRequest) -> Mapping[str, object]:
        del request
        return {"status": "partial"}


class _MalformedPreflight:
    def run(self, broker: str) -> PreflightReport:
        return PreflightReport(broker, (gate("MADE_UP_GATE", True),))


class _HostilePreflight:
    def run(self, broker: str) -> PreflightReport:
        return PreflightReport(broker, (object(),))  # type: ignore[arg-type]


class _Submission:
    def __init__(self) -> None:
        self.accepted = 0
        self.confirmation_phrase: str | None = None

    def submit(
        self, request: BoundOrderRequest, confirmation_phrase: str
    ) -> Mapping[str, object]:
        self.accepted += 1
        self.confirmation_phrase = confirmation_phrase
        return {"intent_id": request.intent_id, "status": "ACKNOWLEDGED"}


def _dashboard_status() -> DashboardStatus:
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    return DashboardStatus(
        generated_at=at,
        data_as_of=at,
        research={"version": "research-v1", "fresh": True},
        strategies=({"id": "swing", "version": "v1"},),
        promotion={"status": "paper"},
        portfolio={"currency": "USD", "equity": Decimal("10")},
        risk={"max_position_fraction": Decimal("0.10")},
        brokers=({"name": "alpaca", "missing_gates": ("MARKET_SENTINEL_MODE",)},),
        orders=(),
        kill_switches=({"active": False},),
        interlocks=({"active": False},),
        aspirational_target={
            "starting_capital": Decimal("10"),
            "target": Decimal("1000000"),
            "required_multiple": Decimal("100000"),
            "reporting_only": True,
        },
    )


def _container(submission: _Submission | None = None) -> ServiceContainer:
    return ServiceContainer(
        research=_Workflow(),
        backtest=_Workflow(),
        paper=_Workflow(),
        preflight=_Preflight(),
        proposal=_Proposal(),
        live_submission=submission,
        dashboard=lambda broker: _dashboard_status(),
    )


def test_help_exposes_exactly_seven_safe_commands_and_no_secret_options() -> None:
    """Adding a hidden credential option or accidental eighth command must be caught."""
    app = build_app(lambda: _container())
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    actual_commands = {command.name for command in app.registered_commands if not command.hidden}
    assert actual_commands == COMMANDS
    lowered = result.stdout.lower()
    forbidden_options = ("--api-key", "--secret", "--token", "--password", "--credential")
    for forbidden in forbidden_options:
        assert forbidden not in lowered
    stack = [get_command(app)]
    while stack:
        command = stack.pop()
        option_tokens = " ".join(
            token
            for parameter in command.params
            if isinstance(parameter, click.Option)
            for token in (*parameter.opts, *parameter.secondary_opts)
        ).lower()
        assert all(forbidden not in option_tokens for forbidden in forbidden_options)
        if isinstance(command, click.Group):
            stack.extend(command.commands.values())


def test_offline_preflight_lists_only_sorted_missing_gate_names_and_fails_closed() -> None:
    """Readiness output must not expose reason details or treat partial gates as ready."""
    result = CliRunner().invoke(
        build_app(lambda: _container()), ["live-preflight", "--broker", "alpaca"]
    )

    assert result.exit_code == 20
    payload = json.loads(result.stdout)
    assert payload == {
        "broker": "alpaca",
        "missing_gates": sorted(required_gate_names("alpaca") - {"MARKET_SENTINEL_MODE"}),
        "ready": False,
    }


def test_preflight_rejects_unknown_or_incomplete_gate_manifests() -> None:
    """A one-gate report must not masquerade as broker readiness."""
    container = _container()
    malformed = ServiceContainer(
        research=container.research,
        backtest=container.backtest,
        paper=container.paper,
        preflight=_MalformedPreflight(),
        proposal=container.proposal,
        dashboard=container.dashboard,
    )

    result = CliRunner().invoke(
        build_app(lambda: malformed), ["live-preflight", "--broker", "alpaca"]
    )

    assert result.exit_code == 20
    assert result.stdout.strip() == '{"error":"PREFLIGHT_INVALID"}'

    hostile = ServiceContainer(preflight=_HostilePreflight())
    hostile_result = CliRunner().invoke(
        build_app(lambda: hostile), ["live-preflight", "--broker", "alpaca"]
    )
    assert hostile_result.exit_code == 20
    assert hostile_result.stdout.strip() == '{"error":"PREFLIGHT_INVALID"}'
    assert "traceback" not in hostile_result.stdout.lower()


def test_workflows_require_canonical_instrument_and_aware_dates_without_traceback() -> None:
    """Malformed instruments and naive dates must stop before injected work runs."""
    app = build_app(lambda: _container())
    malformed = CliRunner().invoke(
        app,
        ["research", "--instrument", "AAPL", "--as-of", "2026-08-09T10:00:00"],
    )
    valid = CliRunner().invoke(
        app,
        [
            "research",
            "--instrument",
            "AAPL@alpaca",
            "--as-of",
            "2026-08-09T10:00:00+00:00",
        ],
    )

    assert malformed.exit_code != 0
    assert "traceback" not in malformed.stdout.lower()
    assert valid.exit_code == 0
    assert json.loads(valid.stdout) == {"instrument": "AAPL@alpaca", "workflow": "research"}


def test_service_failure_has_stable_output_and_zero_provider_exception_context() -> None:
    """A provider exception must not remain reachable from the CLI failure object."""
    container = _container()
    failing = ServiceContainer(
        research=_FailingWorkflow(),
        backtest=container.backtest,
        paper=container.paper,
        preflight=container.preflight,
        proposal=container.proposal,
        dashboard=container.dashboard,
    )
    result = CliRunner().invoke(
        build_app(lambda: failing),
        [
            "research",
            "--instrument",
            "AAPL@alpaca",
            "--as-of",
            "2026-08-09T10:00:00+00:00",
        ],
    )

    assert result.exit_code == 10
    assert result.stdout.strip() == '{"error":"WORKFLOW_FAILED"}'
    exception = result.exception
    while exception is not None:
        assert "provider-secret-must-not-survive" not in str(exception)
        assert "provider-secret-must-not-survive" not in repr(exception)
        exception = exception.__cause__ or exception.__context__


def test_submit_requires_all_bound_fields_and_exact_phrase_before_service() -> None:
    """Missing approval identity or an inexact phrase must never reach live authority."""
    submission = _Submission()
    app = build_app(lambda: _container(submission))
    incomplete = CliRunner().invoke(
        app,
        [
            "submit-confirmed-order",
            "--broker",
            "alpaca",
            "--intent-id",
            "intent-1",
            "--instrument",
            "AAPL@alpaca",
            "--confirm-real-money",
            "almost",
        ],
    )

    assert incomplete.exit_code != 0
    assert submission.accepted == 0
    assert CONFIRMATION_PHRASE in incomplete.stdout
    assert "traceback" not in incomplete.stdout.lower()


def test_complete_submit_delegates_one_exact_notional_request() -> None:
    """Changing or omitting a bound field must not produce an acknowledged live request."""
    submission = _Submission()
    app = build_app(lambda: _container(submission))
    result = CliRunner().invoke(app, _complete_submit_args())

    assert result.exit_code == 0
    assert submission.accepted == 1
    assert submission.confirmation_phrase == CONFIRMATION_PHRASE
    assert json.loads(result.stdout) == {"intent_id": "intent-1", "status": "ACKNOWLEDGED"}


def test_proposal_output_is_complete_even_if_service_returns_partial_metadata() -> None:
    """The approval surface must emit every exact bound field, never a partial service shape."""
    container = _container()
    partial = ServiceContainer(
        research=container.research,
        backtest=container.backtest,
        paper=container.paper,
        preflight=container.preflight,
        proposal=_PartialProposal(),
        dashboard=container.dashboard,
    )
    result = CliRunner().invoke(build_app(lambda: partial), _complete_proposal_args())

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert set(payload) == {
        "broker",
        "intent_id",
        "instrument",
        "side",
        "quantity",
        "notional",
        "order_type",
        "limit_price",
        "trigger_price",
        "stop_loss",
        "take_profit",
        "time_in_force",
        "product",
        "session",
        "snapshot_hash",
        "created_at",
        "expires_at",
        "risk_approved_quantity",
        "risk_approved_notional",
        "portfolio_hash",
        "risk_decided_at",
        "risk_expires_at",
    }


def test_default_container_fails_before_live_submission_without_external_calls() -> None:
    """The installed command must remain inert when live services are absent."""
    result = CliRunner().invoke(build_app(), _complete_submit_args())

    assert result.exit_code == 30
    assert result.stdout.strip() == '{"error":"LIVE_SERVICE_UNAVAILABLE"}'
    assert result.exception is not None


def test_dashboard_provider_failure_is_scrubbed_without_exception_context(tmp_path: Path) -> None:
    """An unexpected dashboard provider failure must remain a stable local error."""

    def fail_dashboard(broker: str) -> DashboardStatus:
        del broker
        raise RuntimeError("dashboard-provider-secret")

    result = CliRunner().invoke(
        build_app(lambda: ServiceContainer(dashboard=fail_dashboard)),
        [
            "export-dashboard",
            "--broker",
            "alpaca",
            "--path",
            str(tmp_path / "status.json"),
        ],
    )

    assert result.exit_code == 40
    assert result.stdout.strip() == '{"error":"DASHBOARD_EXPORT_FAILED"}'
    exception = result.exception
    while exception is not None:
        assert "dashboard-provider-secret" not in str(exception)
        assert "dashboard-provider-secret" not in repr(exception)
        exception = exception.__cause__ or exception.__context__


def _complete_submit_args() -> list[str]:
    return _complete_proposal_args("submit-confirmed-order") + [
        "--confirm-real-money",
        CONFIRMATION_PHRASE,
    ]


def _complete_proposal_args(command: str = "propose-order") -> list[str]:
    return [
        command,
        "--broker",
        "alpaca",
        "--intent-id",
        "intent-1",
        "--instrument",
        "AAPL@alpaca",
        "--side",
        "buy",
        "--notional",
        "10",
        "--order-type",
        "market",
        "--stop-loss",
        "95",
        "--take-profit",
        "110",
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
        "--risk-approved-notional",
        "10",
        "--portfolio-hash",
        "a" * 64,
        "--risk-decided-at",
        "2026-08-09T10:00:00+00:00",
        "--risk-expires-at",
        "2026-08-09T10:01:00+00:00",
    ]
