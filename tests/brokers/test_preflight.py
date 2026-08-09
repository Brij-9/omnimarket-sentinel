from market_sentinel.brokers.alpaca import AlpacaBroker
from market_sentinel.brokers.preflight import PreflightReport
from market_sentinel.config import Settings
from market_sentinel.domain import GateResult


def test_alpaca_preflight_lists_missing_gates_without_values() -> None:
    report = AlpacaBroker.from_settings(Settings(_env_file=None)).preflight()

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
