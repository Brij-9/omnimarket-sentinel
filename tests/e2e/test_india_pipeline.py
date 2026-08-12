"""Offline end-to-end evidence for the representative India fixture."""

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from market_sentinel.domain.enums import AssetClass, OrderStatus, OrderType
from market_sentinel.operations import fixture_pipeline as e2e_runner
from market_sentinel.operations.fixture_pipeline import (
    FixturePipelineRunner,
    FixtureRequestError,
    run_fixture_pipeline,
)


@pytest.mark.e2e
def test_india_fixture_researches_risks_papers_and_reconciles() -> None:
    """Skipping a real local stage would leave the final evidence chain incomplete."""
    result = run_fixture_pipeline("india")

    assert result.instrument_id == "RELIANCE@groww"
    assert len(result.normalized_bars) == result.raw_bar_count
    assert result.account.currency == "INR"
    assert result.account.cash == Decimal("10")
    assert result.broker_capabilities.broker == "groww"
    assert result.broker_capabilities.supported_asset_classes == frozenset({AssetClass.EQUITY})
    assert OrderType.LIMIT in result.broker_capabilities.supported_order_types
    assert result.broker_capabilities.supports_fractional_quantity is False
    assert result.research_packet.evidence
    assert all(
        evidence.uri.startswith("https://fixtures.invalid/")
        and evidence.published_at <= result.research_packet.as_of <= result.cutoff
        for evidence in result.research_packet.evidence
    )
    assert result.signal.invalidation_price < result.signal.entry_price < result.signal.take_profit
    assert result.risk_decision.approved is True
    assert result.risk_decision.reason_codes == ()
    assert result.backtest.after_cost is True
    assert result.backtest.metrics.total_return is not None
    assert result.backtest.costs.fee_bps > Decimal("0")
    assert result.backtest.total_fees > Decimal("0")
    assert result.backtest.metrics.benchmark_excess_return is not None
    assert result.backtest.metrics.maximum_drawdown >= Decimal("0")
    assert result.backtest.robustness_stressed_return is not None
    assert result.backtest.evidence_sufficiency_reason_codes
    assert result.paper_order.status is OrderStatus.FILLED
    assert result.paper_fill.order_id == result.paper_order.order_id
    assert result.ledger_state.positions[0].instrument_id == result.instrument_id
    assert result.reconciliation.healthy is True
    assert result.audit_kinds[-1] == "reconciliation.healthy"
    assert result.dashboard_payload["schema_version"] == 1


@pytest.mark.e2e
def test_fixture_runner_enforces_no_lookahead_freshness_and_request_identity() -> None:
    """Replay identity, time, or instrument drift must fail before a paper order exists."""
    runner = FixturePipelineRunner()
    first = runner.run("india", request_id="india-request-1")

    with pytest.raises(FixtureRequestError, match="DUPLICATE_FIXTURE_REQUEST"):
        runner.run("india", request_id="india-request-1")
    with pytest.raises(FixtureRequestError, match="FIXTURE_LOOKAHEAD_REJECTED"):
        runner.run(
            "india",
            request_id="india-request-2",
            requested_as_of=first.cutoff - timedelta(microseconds=1),
        )
    with pytest.raises(FixtureRequestError, match="FIXTURE_TIME_INVALID"):
        runner.run(
            "india",
            request_id="india-request-3",
            requested_as_of=datetime(2026, 8, 10, 4, 0),
        )
    fresh = runner.run(
        "india",
        request_id="india-request-4",
        requested_as_of=(
            first.cutoff + first.maximum_request_age - timedelta(microseconds=1)
        ),
    )
    assert fresh.reconciliation.healthy is True
    with pytest.raises(FixtureRequestError, match="STALE_FIXTURE_REQUEST"):
        runner.run(
            "india",
            request_id="india-request-5",
            requested_as_of=first.cutoff + first.maximum_request_age,
        )
    with pytest.raises(FixtureRequestError, match="FIXTURE_INSTRUMENT_MISMATCH"):
        runner.run(
            "india",
            request_id="india-request-6",
            expected_instrument_id="AAPL@alpaca",
        )


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("section", "key", "value", "reason"),
    (
        (
            "broker_capabilities",
            "unexpected",
            True,
            "FIXTURE_CAPABILITIES_INVALID",
        ),
        ("account", "unexpected", True, "FIXTURE_ACCOUNT_INVALID"),
        ("account", "cash", 10, "FIXTURE_ACCOUNT_INVALID"),
        ("account", "equity", 10, "FIXTURE_ACCOUNT_INVALID"),
        ("account", "peak_equity", 10, "FIXTURE_ACCOUNT_INVALID"),
        ("account", "gross_exposure", 0, "FIXTURE_ACCOUNT_INVALID"),
        ("account", "daily_pnl", 0, "FIXTURE_ACCOUNT_INVALID"),
        ("account", "realized_pnl", 0, "FIXTURE_ACCOUNT_INVALID"),
        ("account", "currency", 7, "FIXTURE_ACCOUNT_INVALID"),
    ),
)
def test_fixture_capability_and_account_sections_reject_non_exact_json_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    key: str,
    value: object,
    reason: str,
) -> None:
    """Unknown keys or scalar coercion must not alter typed fixture evidence."""
    loaded = json.loads(
        Path("tests/fixtures/e2e_markets.json").read_text(encoding="utf-8")
    )
    assert type(loaded) is dict
    market_fixture = loaded["india"]
    assert type(market_fixture) is dict
    target = market_fixture[section]
    assert type(target) is dict
    target[key] = value
    malformed = tmp_path / "malformed-e2e-markets.json"
    malformed.write_text(json.dumps(loaded), encoding="utf-8")
    monkeypatch.setattr(e2e_runner, "_FIXTURE_PATH", malformed)

    with pytest.raises(FixtureRequestError, match=reason):
        run_fixture_pipeline("india")
