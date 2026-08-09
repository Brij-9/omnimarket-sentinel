"""Tests for fail-safe environment configuration."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from market_sentinel.config import Settings
from market_sentinel.domain.enums import OperatingMode


def test_target_is_reporting_only() -> None:
    """Changing reporting values cannot alter the risk defaults."""
    settings = Settings(_env_file=None)

    assert settings.starting_capital == Decimal("10")
    assert settings.aspirational_target == Decimal("1000000")
    assert settings.required_multiple == Decimal("100000")
    assert settings.risk.max_position_fraction == Decimal("0.10")


def test_live_mode_cannot_relax_risk_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live environment cannot raise an approved exposure limit."""
    monkeypatch.setenv("MARKET_SENTINEL_MODE", "live-small")
    monkeypatch.setenv("MARKET_SENTINEL_MAX_DRAWDOWN", "0.20")

    settings = Settings(_env_file=None)

    assert settings.risk.max_drawdown_fraction == Decimal("0.10")


def test_live_mode_clamps_each_configured_risk_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every live risk cap rejects a looser environment value."""
    monkeypatch.setenv("MARKET_SENTINEL_MODE", OperatingMode.LIVE_SMALL.value)
    monkeypatch.setenv("MARKET_SENTINEL_MAX_TRADE_RISK", "0.01")
    monkeypatch.setenv("MARKET_SENTINEL_MAX_POSITION", "0.20")
    monkeypatch.setenv("MARKET_SENTINEL_MAX_GROSS_EXPOSURE", "0.75")
    monkeypatch.setenv("MARKET_SENTINEL_MAX_DAILY_LOSS", "0.03")
    monkeypatch.setenv("MARKET_SENTINEL_MAX_DRAWDOWN", "0.20")

    settings = Settings(_env_file=None)

    assert settings.risk.max_trade_risk_fraction == Decimal("0.005")
    assert settings.risk.max_position_fraction == Decimal("0.10")
    assert settings.risk.max_gross_exposure_fraction == Decimal("0.50")
    assert settings.risk.max_daily_loss_fraction == Decimal("0.02")
    assert settings.risk.max_drawdown_fraction == Decimal("0.10")


def test_target_progress_is_reporting_math() -> None:
    """Target progress reports hand-calculated progress without changing risk."""
    settings = Settings(_env_file=None)

    progress = settings.target_progress(Decimal("25"))

    assert progress.starting_capital == Decimal("10")
    assert progress.current_equity == Decimal("25")
    assert progress.aspirational_target == Decimal("1000000")
    assert progress.required_multiple == Decimal("100000")
    assert progress.achieved_multiple == Decimal("2.5")
    assert progress.remaining_gap == Decimal("999975")
    assert settings.risk.max_position_fraction == Decimal("0.10")


def test_broker_gate_uses_the_unprefixed_environment_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broker gates read their documented names instead of double-prefixed variants."""
    monkeypatch.setenv("ALPACA_LIVE_TRADING_ENABLED", "true")
    monkeypatch.setenv("GROWW_STATIC_IP_ALLOWLISTED", "true")
    monkeypatch.setenv("CCXT_WITHDRAWALS_DISABLED_CONFIRMED", "true")

    settings = Settings(_env_file=None)

    assert settings.alpaca_live_trading_enabled is True
    assert settings.groww_static_ip_allowlisted is True
    assert settings.ccxt_withdrawals_disabled_confirmed is True


def test_zero_starting_capital_is_rejected() -> None:
    """Zero capital cannot produce undefined target-multiple reporting."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, starting_capital=Decimal("0"))
