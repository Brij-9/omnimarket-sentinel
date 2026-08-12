"""Offline end-to-end evidence for the representative crypto spot fixture."""

import json
from pathlib import Path

import pytest

from market_sentinel.domain.enums import AssetClass, OrderStatus, OrderType, SignalDirection
from market_sentinel.operations.fixture_pipeline import run_fixture_pipeline
from market_sentinel.security import secret_text_present


@pytest.mark.e2e
def test_crypto_fixture_remains_spot_long_unlevered_and_reconciled() -> None:
    """The representative crypto path must not introduce leverage, derivatives, or shorts."""
    result = run_fixture_pipeline("crypto")

    assert result.instrument.asset_class is AssetClass.CRYPTO_SPOT
    assert result.account.currency == "USDT"
    assert result.broker_capabilities.broker == "ccxt-spot"
    assert result.broker_capabilities.supported_asset_classes == frozenset(
        {AssetClass.CRYPTO_SPOT}
    )
    assert result.broker_capabilities.supported_order_types == frozenset(
        {OrderType.MARKET, OrderType.LIMIT}
    )
    assert result.broker_capabilities.supports_leverage is False
    assert result.broker_capabilities.supports_derivatives is False
    assert result.broker_capabilities.supports_shorting is False
    assert result.signal.direction is SignalDirection.LONG
    assert result.strategy_spot_only is True
    assert result.strategy_leverage_allowed is False
    assert result.intent.product == "cash"
    assert result.paper_order.status is OrderStatus.FILLED
    assert result.reconciliation.healthy is True
    assert result.live_order is None


@pytest.mark.e2e
def test_all_e2e_fixture_strings_are_sanitized() -> None:
    """A secret-shaped fixture value would make local evidence unsafe to publish."""
    fixture = json.loads(
        Path("tests/fixtures/e2e_markets.json").read_text(encoding="utf-8")
    )
    pending: list[object] = [fixture]
    strings: list[str] = []
    while pending:
        value = pending.pop()
        if type(value) is dict:
            pending.extend(value.keys())
            pending.extend(value.values())
        elif type(value) is list:
            pending.extend(value)
        elif type(value) is str:
            strings.append(value)

    assert strings
    assert all(not secret_text_present(value) for value in strings)
