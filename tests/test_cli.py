"""Behavioral tests for the dependency-injected command surface."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import click
import pytest
from typer.main import get_command
from typer.testing import CliRunner

import market_sentinel.cli as cli_module
from market_sentinel.brokers.preflight import PreflightReport, gate, required_gate_names
from market_sentinel.cli import (
    CONFIRMATION_PHRASE,
    BoundOrderRequest,
    OrderProposal,
    ServiceContainer,
    WorkflowRequest,
    build_app,
)
from market_sentinel.domain.models import GateResult
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
)

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


class _SecretShapedWorkflow:
    def run(self, request: WorkflowRequest) -> Mapping[str, object]:
        del request
        return {
            "assignment": "api_key=secret-token-123",
            "header": "Authorization: Bearer header-value",
            "query": "https://local.invalid/?credential=live-value",
            "encoded": "https://local.invalid/?api%5Fkey=encoded-value",
            "provider_text": "groww-real-secret-value-1234567890",
        }


class _SurrogateWorkflow:
    def run(self, request: WorkflowRequest) -> Mapping[str, object]:
        del request
        return {"innocent": "\ud800"}


class _Preflight:
    def run(self, broker: str) -> PreflightReport:
        normalized = "ccxt-spot" if broker == "ccxt" else broker
        gates = tuple(
            gate(name, name.endswith("MODE"))
            for name in sorted(required_gate_names(normalized))
        )
        return PreflightReport(normalized, gates)


class _Proposal:
    def propose(self, request: BoundOrderRequest) -> OrderProposal:
        return OrderProposal.accept(request)


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
        research=DashboardResearch("research-v1", True),
        strategies=(DashboardStrategy("swing", "v1"),),
        promotion=DashboardPromotion("paper"),
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
                "alpaca",
                tuple(
                    GateResult(
                        name=name,
                        passed=name == "MARKET_SENTINEL_MODE",
                        reason_code="OK" if name == "MARKET_SENTINEL_MODE" else "NOT_READY",
                    )
                    for name in sorted(required_gate_names("alpaca"))
                ),
            ),
        ),
        orders=(),
        kill_switches=(DashboardSafetyState(False, "OK"),),
        interlocks=(DashboardSafetyState(False, "OK"),),
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


def test_preflight_rejects_pass_reason_contradictions() -> None:
    """Only passed/OK and failed/non-OK gate pairs are coherent readiness evidence."""

    class _Contradictory:
        def __init__(self, *, passed: bool, reason: str) -> None:
            self.passed = passed
            self.reason = reason

        def run(self, broker: str) -> PreflightReport:
            return PreflightReport(
                broker,
                tuple(
                    GateResult(name=name, passed=self.passed, reason_code=self.reason)
                    for name in sorted(required_gate_names(broker))
                ),
            )

    for passed, reason in ((True, "NOT_OK"), (False, "OK")):
        result = CliRunner().invoke(
            build_app(
                lambda passed=passed, reason=reason: ServiceContainer(
                    preflight=_Contradictory(passed=passed, reason=reason)
                )
            ),
            ["live-preflight", "--broker", "alpaca"],
        )
        assert result.exit_code == 20
        assert result.stdout.strip() == '{"error":"PREFLIGHT_INVALID"}'


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


def test_workflow_output_redacts_assignment_query_header_and_encoded_secrets() -> None:
    """Free-form provider text must pass through the same normalized output sanitizer."""
    result = CliRunner().invoke(
        build_app(lambda: ServiceContainer(research=_SecretShapedWorkflow())),
        [
            "research",
            "--instrument",
            "AAPL@alpaca",
            "--as-of",
            "2026-08-09T10:00:00+00:00",
        ],
    )

    assert result.exit_code == 0
    assert result.exception is None
    assert json.loads(result.stdout) == {
        "assignment": "[REDACTED]",
        "encoded": "[REDACTED]",
        "header": "[REDACTED]",
        "provider_text": "[REDACTED]",
        "query": "[REDACTED]",
    }
    for forbidden in (
        "secret-token-123",
        "header-value",
        "live-value",
        "encoded-value",
        "groww-real-secret-value-1234567890",
    ):
        assert forbidden not in result.stdout


def test_workflow_surrogate_output_fails_with_zero_exception_context() -> None:
    """Invalid UTF-8 provider output becomes one stable secret-free CLI failure."""
    result = CliRunner().invoke(
        build_app(lambda: ServiceContainer(research=_SurrogateWorkflow())),
        [
            "research",
            "--instrument",
            "AAPL@alpaca",
            "--as-of",
            "2026-08-09T10:00:00+00:00",
        ],
    )

    assert result.exit_code == 50
    assert result.stdout.strip() == '{"error":"SERVICE_RESPONSE_INVALID"}'
    exception = result.exception
    while exception is not None:
        assert "surrogate" not in str(exception).lower()
        exception = exception.__cause__ or exception.__context__


def test_cli_json_failure_has_stable_output_and_zero_exception_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A JSON encoder failure must not escape through any workflow command."""

    def fail_json(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise UnicodeError("json-provider-secret")

    monkeypatch.setattr(cli_module.json, "dumps", fail_json)
    result = CliRunner().invoke(
        build_app(lambda: ServiceContainer(research=_Workflow())),
        [
            "research",
            "--instrument",
            "AAPL@alpaca",
            "--as-of",
            "2026-08-09T10:00:00+00:00",
        ],
    )

    assert result.exit_code == 50
    assert result.stdout.strip() == '{"error":"OUTPUT_SERIALIZATION_FAILED"}'
    exception = result.exception
    while exception is not None:
        assert "json-provider-secret" not in str(exception)
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


def test_structural_submitter_cannot_acknowledge_an_exact_notional_request() -> None:
    """A structurally compatible submitter cannot replace the concrete Task 14 facade."""
    submission = _Submission()
    app = build_app(lambda: _container(submission))
    result = CliRunner().invoke(app, _complete_submit_args())

    assert result.exit_code == 50
    assert submission.accepted == 0
    assert submission.confirmation_phrase is None
    assert result.stdout.strip() == '{"error":"SERVICE_CONTAINER_UNAVAILABLE"}'


def test_proposal_rejects_partial_mapping_metadata() -> None:
    """The approval surface rejects a Mapping instead of echoing it as approval."""
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

    assert result.exit_code == 30
    assert result.stdout.strip() == '{"error":"PROPOSAL_RESPONSE_INVALID"}'


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
        "--risk-approved-quantity",
        "0.1",
        "--portfolio-hash",
        "a" * 64,
        "--risk-decided-at",
        "2026-08-09T10:00:00+00:00",
        "--risk-expires-at",
        "2026-08-09T10:01:00+00:00",
    ]
