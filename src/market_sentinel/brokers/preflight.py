"""Fail-closed readiness records shared by live broker adapters."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from market_sentinel.domain import GateResult

REQUIRED_PREFLIGHT_GATES: Final = MappingProxyType(
    {
        "alpaca": frozenset(
            {
                "MARKET_SENTINEL_MODE",
                "ALPACA_LIVE_TRADING_ENABLED",
                "ALPACA_REAL_API_ENABLED",
                "ALPACA_LIVE_ENDPOINT",
                "ALPACA_ACCOUNT_ID_PRESENT",
                "ALPACA_LOCAL_CREDENTIALS_PRESENT",
                "ALPACA_ACCOUNT_ID_MATCHED",
                "ALPACA_ACCOUNT_ACTIVE",
                "ALPACA_ACCOUNT_UNBLOCKED",
                "ALPACA_SUFFICIENT_BUYING_POWER",
            }
        ),
        "groww": frozenset(
            {
                "GROWW_PRIMARY_BROKER",
                "MARKET_SENTINEL_MODE",
                "INDIA_LIVE_TRADING_ENABLED",
                "INDIA_ALGO_COMPLIANCE_VERIFIED",
                "GROWW_REAL_API_ENABLED",
                "GROWW_API_SUBSCRIPTION_ACTIVE",
                "GROWW_PROTECTED_ORDER_CLIENT",
                "GROWW_STATIC_OUTBOUND_IPV4",
                "GROWW_STATIC_IP_ALLOWLISTED",
                "GROWW_BROKER_APPROVED_ALGO_ID",
                "GROWW_LOCAL_CREDENTIALS_PRESENT",
                "GROWW_AUTH_SESSION_FRESH",
                "GROWW_READ_ONLY_PROFILE_ACCESS",
                "GROWW_PROFILE_ACTIVE",
                "GROWW_REGULAR_SESSION_SUPPORTED",
                "GROWW_PROTECTED_ORDERS_SUPPORTED",
            }
        ),
        "ccxt-spot": frozenset(
            {
                "MARKET_SENTINEL_MODE",
                "CCXT_LIVE_TRADING_ENABLED",
                "CCXT_REAL_API_ENABLED",
                "CCXT_EXCHANGE_ID_CONFIGURED",
                "CCXT_SPOT_ONLY",
                "CCXT_LOCAL_CREDENTIALS_PRESENT",
                "CCXT_WITHDRAWALS_DISABLED_CONFIRMED",
                "CCXT_IP_RESTRICTED_CONFIRMED",
                "CCXT_NO_SANDBOX_ACKNOWLEDGED",
                "CCXT_EXCHANGE_CONFIGURED",
                "CCXT_SPOT_MARKETS_AVAILABLE",
                "CCXT_CREATE_ORDER_SUPPORTED",
            }
        ),
    }
)


def required_gate_names(broker: object) -> frozenset[str]:
    """Return the immutable exact readiness manifest for one known live adapter."""
    if type(broker) is not str or broker not in REQUIRED_PREFLIGHT_GATES:
        raise ValueError("unknown live broker preflight manifest")
    return REQUIRED_PREFLIGHT_GATES[broker]


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """A derived, secret-free broker readiness result."""

    broker: str
    gates: tuple[GateResult, ...]

    @property
    def ready(self) -> bool:
        return all(gate.passed for gate in self.gates)

    @property
    def missing_gate_names(self) -> tuple[str, ...]:
        return tuple(gate.name for gate in self.gates if not gate.passed)

    def safe_summary(self) -> str:
        missing = ",".join(name for name in self.missing_gate_names if not _credential_term(name))
        return f"broker={self.broker} missing_gates={missing or 'none'}"


def gate(name: str, passed: bool, reason_code: str | None = None) -> GateResult:
    """Create a gate result without ever retaining configuration values."""
    return GateResult(
        name=name,
        passed=passed,
        reason_code="OK" if passed else (reason_code or "GATE_NOT_SATISFIED"),
    )


def _credential_term(value: str) -> bool:
    lowered = value.lower()
    return "secret" in lowered or "api_key" in lowered or "access_token" in lowered
