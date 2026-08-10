import pytest

from market_sentinel.brokers.alpaca import AlpacaBroker
from market_sentinel.brokers.ccxt_spot import CcxtSpotBroker
from market_sentinel.brokers.groww import GrowwBroker
from market_sentinel.brokers.preflight import PreflightReport, required_gate_names
from market_sentinel.config import Settings
from market_sentinel.domain import GateResult


def test_alpaca_preflight_lists_missing_gates_without_values() -> None:
    report = AlpacaBroker.from_settings(Settings(_env_file=None)).preflight()  # type: ignore[call-arg]

    assert report.ready is False
    assert "ALPACA_LIVE_TRADING_ENABLED" in report.missing_gate_names
    rendered = report.safe_summary()
    assert "secret" not in rendered.lower()
    assert "api_key" not in rendered.lower()


def test_preflight_readiness_is_derived_from_every_gate() -> None:
    report = PreflightReport(
        broker="test-broker",
        gates=(
            GateResult(name="LOCAL_GATE", passed=True, reason_code="OK"),
            GateResult(name="REMOTE_GATE", passed=False, reason_code="UNAVAILABLE"),
        ),
    )

    assert report.ready is False
    assert report.missing_gate_names == ("REMOTE_GATE",)
    assert report.safe_summary() == "broker=test-broker missing_gates=REMOTE_GATE"


def test_each_live_adapter_has_a_fixed_nonempty_exact_gate_manifest() -> None:
    """A caller-chosen gate name cannot substitute for the adapter's canonical gates."""
    for broker in ("alpaca", "groww", "ccxt-spot"):
        manifest = required_gate_names(broker)
        assert type(manifest) is frozenset
        assert manifest
        assert "LOCAL" not in manifest
    with pytest.raises(ValueError):
        required_gate_names("unknown")


def test_every_adapter_report_matches_its_manifest_even_when_not_ready() -> None:
    """Failure paths retain all required gates so missing evidence is explicit and exact."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    reports = (
        AlpacaBroker.from_settings(settings).preflight(),
        GrowwBroker.from_settings(settings).preflight(),
        CcxtSpotBroker.from_settings(settings).preflight(),
    )
    for report in reports:
        assert {item.name for item in report.gates} == required_gate_names(report.broker)
