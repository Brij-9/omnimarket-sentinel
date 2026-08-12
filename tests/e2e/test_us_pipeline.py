"""Offline end-to-end evidence for the representative US fixture."""

import pytest

from market_sentinel.domain.enums import AssetClass, OrderStatus, OrderType
from market_sentinel.operations.fixture_pipeline import run_fixture_pipeline


@pytest.mark.e2e
def test_us_fixture_is_deterministic_and_uses_typed_point_in_time_evidence() -> None:
    """A hidden clock, graph, or provider dependency would change identical fixture results."""
    first = run_fixture_pipeline("us")
    second = run_fixture_pipeline("us")

    assert first.instrument_id == "AAPL@alpaca"
    assert first.account.currency == "USD"
    assert first.broker_capabilities.broker == "alpaca"
    assert first.broker_capabilities.supported_asset_classes == frozenset(
        {AssetClass.EQUITY}
    )
    assert OrderType.STOP_LIMIT in first.broker_capabilities.supported_order_types
    assert first.broker_capabilities.supports_fractional_quantity is True
    assert first.broker_capabilities.supports_notional_orders is True
    assert first.research_packet == second.research_packet
    assert first.signal == second.signal
    assert first.risk_decision == second.risk_decision
    assert first.backtest == second.backtest
    assert first.paper_fill == second.paper_fill
    assert first.reconciliation.healthy is True
    assert first.expected_open_orders == ()
    assert first.broker_snapshot.open_orders == ()
    assert first.paper_order.status is OrderStatus.FILLED
    assert first.research_packet.model_id == "tauric-fixture-no-graph"
    assert first.research_packet.prompt_version == "fixture-v1"
    assert first.research_packet.configuration_hash == second.research_packet.configuration_hash
    assert first.dashboard.research.fresh is True
    assert first.dashboard.aspirational_target.reporting_only is True


@pytest.mark.e2e
def test_us_fixture_pipeline_performs_no_network_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An accidental provider call must fail this otherwise complete offline run."""
    import socket

    def reject_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("network connection attempted")

    monkeypatch.setattr(socket.socket, "connect", reject_network)

    result = run_fixture_pipeline("us")

    assert result.reconciliation.healthy is True


@pytest.mark.e2e
def test_us_fixture_reconciliation_detects_independent_broker_position_mismatch() -> None:
    """Broker state drift must not be hidden by deriving the source from the ledger."""
    result = run_fixture_pipeline("us", simulate_reconciliation_mismatch=True)

    assert result.broker_snapshot.positions[0].quantity != (
        result.ledger_state.positions[0].quantity
    )
    assert result.reconciliation.healthy is False
    assert result.reconciliation.reason_codes == ("POSITION_QUANTITY_MISMATCH",)
    assert result.kill_switch_active is True
    assert result.audit_kinds[-1] == "reconciliation.unhealthy"


@pytest.mark.e2e
def test_us_fixture_reconciliation_detects_unexpected_broker_open_order() -> None:
    """A terminal internal order must expect no opens independently of broker state."""
    result = run_fixture_pipeline(
        "us",
        simulate_broker_open_order_mismatch=True,
    )

    assert result.paper_order.status is OrderStatus.FILLED
    assert result.expected_open_orders == ()
    assert len(result.broker_snapshot.open_orders) == 1
    assert result.reconciliation.healthy is False
    assert result.reconciliation.reason_codes == ("ORDER_UNKNOWN",)
    assert result.kill_switch_active is True
