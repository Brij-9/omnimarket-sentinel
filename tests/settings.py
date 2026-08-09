"""Sanitized live-mode settings fixtures for offline broker adapter tests."""

from __future__ import annotations

from typing import Any

from market_sentinel.config import Settings
from market_sentinel.domain.enums import OperatingMode


def groww_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "mode": OperatingMode.LIVE_SMALL,
        "primary_broker": "groww",
        "INDIA_LIVE_TRADING_ENABLED": True,
        "INDIA_ALGO_COMPLIANCE_VERIFIED": True,
        "GROWW_REAL_API_ENABLED": True,
        "GROWW_API_SUBSCRIPTION_ACTIVE": True,
        "GROWW_PROTECTED_ORDER_CLIENT": True,
        "GROWW_STATIC_OUTBOUND_IP": "198.51.100.9",
        "GROWW_STATIC_IP_ALLOWLISTED": True,
        "GROWW_ALGO_ID": "approved-test-algo",
        "GROWW_ACCESS_TOKEN": "test-token",
    }
    aliases = {"static_ip_allowlisted": "GROWW_STATIC_IP_ALLOWLISTED"}
    values.update({aliases.get(name, name): value for name, value in overrides.items()})
    return Settings(**values)


def alpaca_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "mode": OperatingMode.LIVE_SMALL,
        "ALPACA_LIVE_TRADING_ENABLED": True,
        "ALPACA_REAL_API_ENABLED": True,
        "ALPACA_TRADING_ENDPOINT": "https://api.alpaca.markets",
        "ALPACA_ACCOUNT_ID": "account-test-id",
        "ALPACA_KEY_ID": "test-key-id",
        "ALPACA_SECRET_KEY": "test-secret-value",
    }
    aliases = {"trading_endpoint": "ALPACA_TRADING_ENDPOINT"}
    values.update({aliases.get(name, name): value for name, value in overrides.items()})
    return Settings(**values)


def ccxt_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "mode": OperatingMode.LIVE_SMALL,
        "CCXT_LIVE_TRADING_ENABLED": True,
        "CCXT_REAL_API_ENABLED": True,
        "CCXT_EXCHANGE_ID": "testexchange",
        "CCXT_SANDBOX": True,
        "CCXT_API_KEY": "test-key-id",
        "CCXT_SECRET": "test-secret-value",
        "CCXT_WITHDRAWALS_DISABLED_CONFIRMED": True,
        "CCXT_IP_RESTRICTED_CONFIRMED": True,
    }
    aliases = {"sandbox": "CCXT_SANDBOX"}
    values.update({aliases.get(name, name): value for name, value in overrides.items()})
    return Settings(**values)
