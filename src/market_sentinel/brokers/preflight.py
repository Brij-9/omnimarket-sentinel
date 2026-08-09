"""Fail-closed readiness records shared by live broker adapters."""

from __future__ import annotations

from dataclasses import dataclass

from market_sentinel.domain import GateResult


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
