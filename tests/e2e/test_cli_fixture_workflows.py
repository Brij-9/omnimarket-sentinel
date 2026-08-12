"""Installed-default CLI evidence for the three sanitized fixture markets."""

from __future__ import annotations

import json
import socket

import pytest
from typer.testing import CliRunner

from market_sentinel.cli import build_app

_MARKETS = (
    (
        "india",
        "RELIANCE@groww",
        "2026-08-10T03:45:00+00:00",
        "2026-08-10T03:48:30+00:00",
    ),
    (
        "us",
        "AAPL@alpaca",
        "2026-08-10T13:30:00+00:00",
        "2026-08-10T13:33:30+00:00",
    ),
    (
        "crypto",
        "BTC-USDT@ccxt-spot",
        "2026-08-10T00:00:00+00:00",
        "2026-08-10T00:20:30+00:00",
    ),
)


def _deny_network(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    attempts: list[tuple[object, ...]] = []

    def blocked(*args: object, **kwargs: object) -> None:
        del kwargs
        attempts.append(args)
        raise AssertionError("fixture CLI attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    return attempts


@pytest.mark.e2e
@pytest.mark.parametrize(("market", "instrument", "start", "cutoff"), _MARKETS)
def test_default_cli_research_runs_sanitized_typed_pipeline_without_network(
    monkeypatch: pytest.MonkeyPatch,
    market: str,
    instrument: str,
    start: str,
    cutoff: str,
) -> None:
    """The installed CLI must expose real offline research evidence, not a stub."""
    del start
    attempts = _deny_network(monkeypatch)
    result = CliRunner().invoke(
        build_app(),
        ["research", "--instrument", instrument, "--as-of", cutoff],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["workflow"] == "research"
    assert payload["market"] == market
    assert payload["instrument"] == instrument
    assert payload["evidence_count"] > 0
    assert payload["risk_approved"] is True
    assert payload["live_ready"] is False
    assert attempts == []


@pytest.mark.e2e
@pytest.mark.parametrize(("market", "instrument", "start", "cutoff"), _MARKETS)
def test_default_cli_backtest_reports_after_cost_evidence_without_network(
    monkeypatch: pytest.MonkeyPatch,
    market: str,
    instrument: str,
    start: str,
    cutoff: str,
) -> None:
    """Every representative market must produce the same bounded evidence schema."""
    attempts = _deny_network(monkeypatch)
    result = CliRunner().invoke(
        build_app(),
        [
            "backtest",
            "--instrument",
            instrument,
            "--start",
            start,
            "--end",
            cutoff,
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["workflow"] == "backtest"
    assert payload["market"] == market
    assert payload["instrument"] == instrument
    assert payload["after_cost"] is True
    assert payload["fees"] != "0"
    assert "benchmark_excess_return" in payload
    assert "maximum_drawdown" in payload
    assert "robustness_stressed_return" in payload
    assert payload["live_ready"] is False
    assert attempts == []


@pytest.mark.e2e
@pytest.mark.parametrize(("market", "instrument", "start", "cutoff"), _MARKETS)
def test_default_cli_paper_run_fills_and_reconciles_without_network(
    monkeypatch: pytest.MonkeyPatch,
    market: str,
    instrument: str,
    start: str,
    cutoff: str,
) -> None:
    """Paper CLI execution must remain in memory and end in healthy reconciliation."""
    del start
    attempts = _deny_network(monkeypatch)
    result = CliRunner().invoke(
        build_app(),
        ["paper-run", "--instrument", instrument, "--as-of", cutoff],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "instrument": instrument,
        "live_ready": False,
        "market": market,
        "reconciliation_healthy": True,
        "status": "filled",
        "workflow": "paper-run",
    }
    assert attempts == []


@pytest.mark.e2e
def test_default_cli_rejects_unknown_fixture_instrument_without_traceback() -> None:
    """The safe fixture service is an allowlist, not a universal market claim."""
    result = CliRunner().invoke(
        build_app(),
        [
            "research",
            "--instrument",
            "MSFT@alpaca",
            "--as-of",
            "2026-08-10T13:33:30+00:00",
        ],
    )

    assert result.exit_code == 10
    assert result.stdout.strip() == '{"error":"WORKFLOW_FAILED"}'
    assert "traceback" not in result.stdout.lower()


@pytest.mark.e2e
@pytest.mark.parametrize(
    "arguments",
    (
        (
            "research",
            "--instrument",
            "AAPL@alpaca",
            "--as-of",
            "2026-08-10T13:33:29+00:00",
        ),
        (
            "paper-run",
            "--instrument",
            "RELIANCE@groww",
            "--as-of",
            "2026-08-10T03:48:31+00:00",
        ),
        (
            "backtest",
            "--instrument",
            "BTC-USDT@ccxt-spot",
            "--start",
            "2026-08-10T00:00:01+00:00",
            "--end",
            "2026-08-10T00:20:30+00:00",
        ),
        (
            "backtest",
            "--instrument",
            "AAPL@alpaca",
            "--start",
            "2026-08-10T13:30:00+00:00",
            "--end",
            "2026-08-10T13:33:31+00:00",
        ),
    ),
)
def test_default_cli_rejects_nonexact_fixture_times_without_network(
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
) -> None:
    """Packaged evidence is valid only at its exact fixed point-in-time boundary."""
    attempts = _deny_network(monkeypatch)
    result = CliRunner().invoke(build_app(), list(arguments))

    assert result.exit_code == 10
    assert result.stdout.strip() == '{"error":"WORKFLOW_FAILED"}'
    assert attempts == []
