"""Explicit schema-v1 dashboard conversion and hardened atomic export."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import os
import re
import secrets
import stat
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast

from market_sentinel.brokers.preflight import required_gate_names
from market_sentinel.domain.enums import OrderStatus
from market_sentinel.domain.models import GateResult
from market_sentinel.execution.canonical import CanonicalEncodingError, canonical_decimal
from market_sentinel.security import secret_text_present

_MAX_DEPTH = 12
_MAX_NODES = 4_096
_MAX_COLLECTION = 512
_MAX_STRING_BYTES = 4_096
_MAX_PATH_CHARS = 2_048
_MAX_GATE_NAME_BYTES = 128
_MAX_GATE_REASON_BYTES = 64
_REPARSE_POINT = 0x400
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@-]{0,127}$")
_CURRENCY = re.compile(r"^[A-Z]{3,8}$")
_REASON = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_OUTPUT_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_PROMOTIONS = frozenset({"not_promoted", "research", "backtest", "paper", "live-small"})
_CAPTURED_GATE_MANIFESTS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "alpaca": frozenset(required_gate_names("alpaca")),
        "groww": frozenset(required_gate_names("groww")),
        "ccxt": frozenset(required_gate_names("ccxt-spot")),
    }
)
_SECRET_KEY_PARTS = (
    "apikey",
    "secret",
    "token",
    "password",
    "credential",
    "authorization",
    "authenticator",
    "privatekey",
    "accesskey",
    "callback",
    "exception",
)
_REDACTED = "[REDACTED]"


class DashboardValidationError(ValueError):
    """A dashboard snapshot or destination failed its bounded safe contract."""


class _FileLockModule(Protocol):
    LOCK_EX: int
    LOCK_UN: int

    def flock(self, descriptor: int, operation: int, /) -> None: ...


def _captured_gate_manifest(
    broker: str,
    manifests: Mapping[str, frozenset[str]] = _CAPTURED_GATE_MANIFESTS,
) -> frozenset[str]:
    try:
        return manifests[broker]
    except KeyError:
        raise DashboardValidationError("DASHBOARD_BROKER_INVALID") from None


@dataclass(frozen=True, slots=True)
class DashboardResearch:
    version: str
    fresh: bool

    def __post_init__(self) -> None:
        _version(self.version)
        _exact_bool(self.fresh)


@dataclass(frozen=True, slots=True)
class DashboardStrategy:
    strategy_id: str
    version: str

    def __post_init__(self) -> None:
        _identity(self.strategy_id)
        _version(self.version)


@dataclass(frozen=True, slots=True)
class DashboardPromotion:
    status: str

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in _PROMOTIONS:
            raise DashboardValidationError("DASHBOARD_PROMOTION_INVALID")


@dataclass(frozen=True, slots=True)
class DashboardPortfolio:
    currency: str
    equity: Decimal

    def __post_init__(self) -> None:
        if type(self.currency) is not str or _CURRENCY.fullmatch(self.currency) is None:
            raise DashboardValidationError("DASHBOARD_PORTFOLIO_INVALID")
        _nonnegative_decimal(self.equity)


@dataclass(frozen=True, slots=True)
class DashboardRisk:
    max_trade_risk_fraction: Decimal
    max_position_fraction: Decimal
    max_gross_exposure_fraction: Decimal
    max_daily_loss_fraction: Decimal
    max_drawdown_fraction: Decimal

    def __post_init__(self) -> None:
        for value in (
            self.max_trade_risk_fraction,
            self.max_position_fraction,
            self.max_gross_exposure_fraction,
            self.max_daily_loss_fraction,
            self.max_drawdown_fraction,
        ):
            decimal = _positive_decimal(value)
            if decimal > 1:
                raise DashboardValidationError("DASHBOARD_RISK_INVALID")


@dataclass(frozen=True, slots=True)
class DashboardBroker:
    name: str
    gates: tuple[GateResult, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in {"alpaca", "groww", "ccxt"}:
            raise DashboardValidationError("DASHBOARD_BROKER_INVALID")
        try:
            required = _CAPTURED_GATE_MANIFESTS[self.name]
        except KeyError:
            raise DashboardValidationError("DASHBOARD_BROKER_INVALID") from None
        if type(self.gates) is not tuple or len(self.gates) != len(required):
            raise DashboardValidationError("DASHBOARD_BROKER_GATES_INVALID")
        names: list[str] = []
        for gate in self.gates:
            name, _, _ = _validated_dashboard_gate(gate)
            names.append(name)
        if set(names) != required:
            raise DashboardValidationError("DASHBOARD_BROKER_GATES_INVALID")


@dataclass(frozen=True, slots=True)
class DashboardOrder:
    order_id: str
    status: OrderStatus

    def __post_init__(self) -> None:
        _identity(self.order_id)
        if type(self.status) is not OrderStatus:
            raise DashboardValidationError("DASHBOARD_ORDER_INVALID")


@dataclass(frozen=True, slots=True)
class DashboardSafetyState:
    active: bool
    reason_code: str

    def __post_init__(self) -> None:
        _exact_bool(self.active)
        if type(self.reason_code) is not str or _REASON.fullmatch(self.reason_code) is None:
            raise DashboardValidationError("DASHBOARD_SAFETY_STATE_INVALID")


@dataclass(frozen=True, slots=True)
class DashboardAspiration:
    starting_capital: Decimal
    current_equity: Decimal
    target: Decimal
    required_multiple: Decimal
    achieved_multiple: Decimal
    remaining_gap: Decimal
    reporting_only: bool

    def __post_init__(self) -> None:
        starting = _positive_decimal(self.starting_capital)
        current = _nonnegative_decimal(self.current_equity)
        target = _positive_decimal(self.target)
        required = _positive_decimal(self.required_multiple)
        achieved = _nonnegative_decimal(self.achieved_multiple)
        remaining = _finite_decimal(self.remaining_gap)
        _exact_bool(self.reporting_only)
        if (
            self.reporting_only is not True
            or Fraction(required) * Fraction(starting) != Fraction(target)
            or Fraction(achieved) * Fraction(starting) != Fraction(current)
            or Fraction(target) - Fraction(current) != Fraction(remaining)
        ):
            raise DashboardValidationError("DASHBOARD_ASPIRATION_INVALID")


@dataclass(frozen=True, slots=True)
class DashboardStatus:
    """Exact required dashboard schema v1 before safe JSON conversion."""

    generated_at: datetime
    data_as_of: datetime
    research: DashboardResearch
    strategies: tuple[DashboardStrategy, ...]
    promotion: DashboardPromotion
    portfolio: DashboardPortfolio
    risk: DashboardRisk
    brokers: tuple[DashboardBroker, ...]
    orders: tuple[DashboardOrder, ...]
    kill_switches: tuple[DashboardSafetyState, ...]
    interlocks: tuple[DashboardSafetyState, ...]
    aspirational_target: DashboardAspiration

    def __post_init__(self) -> None:
        generated = _utc(self.generated_at)
        data_as_of = _utc(self.data_as_of)
        if data_as_of > generated:
            raise DashboardValidationError("DASHBOARD_FRESHNESS_INVALID")
        if type(self.research) is not DashboardResearch:
            raise DashboardValidationError("DASHBOARD_RESEARCH_INVALID")
        _nonempty_exact_tuple(self.strategies, DashboardStrategy, "DASHBOARD_STRATEGY_INVALID")
        if type(self.promotion) is not DashboardPromotion:
            raise DashboardValidationError("DASHBOARD_PROMOTION_INVALID")
        if type(self.portfolio) is not DashboardPortfolio or type(self.risk) is not DashboardRisk:
            raise DashboardValidationError("DASHBOARD_PORTFOLIO_INVALID")
        _nonempty_exact_tuple(self.brokers, DashboardBroker, "DASHBOARD_BROKER_INVALID")
        if type(self.orders) is tuple and len(self.orders) > _MAX_COLLECTION:
            raise DashboardValidationError("DASHBOARD_VALUE_BOUNDS_EXCEEDED")
        if type(self.orders) is not tuple or not all(
            type(item) is DashboardOrder for item in self.orders
        ):
            raise DashboardValidationError("DASHBOARD_ORDER_INVALID")
        _nonempty_exact_tuple(
            self.kill_switches, DashboardSafetyState, "DASHBOARD_KILL_SWITCH_INVALID"
        )
        _nonempty_exact_tuple(
            self.interlocks, DashboardSafetyState, "DASHBOARD_INTERLOCK_INVALID"
        )
        if (
            type(self.aspirational_target) is not DashboardAspiration
            or self.portfolio.equity != self.aspirational_target.current_equity
        ):
            raise DashboardValidationError("DASHBOARD_ASPIRATION_INVALID")


@dataclass(slots=True)
class _Budget:
    nodes: int = 0


@dataclass(frozen=True, slots=True)
class _Destination:
    path: Path
    parent_identity: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class _DirectoryGuard:
    parent_fd: int | None

    def lstat(self, path: Path) -> os.stat_result | None:
        if self.parent_fd is None:
            return _safe_lstat(path)
        try:
            return os.stat(path.name, dir_fd=self.parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except OSError:
            raise DashboardValidationError("DASHBOARD_PATH_CHANGED") from None


def safe_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Return bounded built-in JSON data with credential fields removed."""
    invalid = False
    try:
        if type(value) not in (dict, MappingProxyType):
            raise DashboardValidationError("DASHBOARD_VALUE_INVALID")
        budget = _Budget()
        prepared = _safe_value(value, depth=0, budget=budget)
        if type(prepared) is not dict:
            raise DashboardValidationError("DASHBOARD_VALUE_INVALID")
    except Exception:
        invalid = True
        prepared = None
    if invalid or type(prepared) is not dict:
        raise DashboardValidationError("DASHBOARD_VALUE_INVALID")
    return prepared


def export_dashboard(
    status: DashboardStatus,
    destination: Path,
    *,
    _gate_manifests: Mapping[str, frozenset[str]] = _CAPTURED_GATE_MANIFESTS,
) -> Path:
    """Publish schema v1 by Windows handle-replace or POSIX absent-path direct link."""
    if type(status) is not DashboardStatus:
        raise DashboardValidationError("DASHBOARD_STATUS_INVALID")
    invalid = False
    try:
        if _CAPTURED_GATE_MANIFESTS is not _gate_manifests:
            raise DashboardValidationError("DASHBOARD_BROKER_GATES_INVALID")
        validated = _revalidated_dashboard_status(
            status,
            gate_manifests=_gate_manifests,
        )
        prepared = safe_json_mapping(_dashboard_payload(validated))
    except Exception:
        invalid = True
        prepared = None
    if invalid or prepared is None:
        raise DashboardValidationError("DASHBOARD_STATUS_INVALID")
    serialization_failed = False
    try:
        text = json.dumps(
            prepared,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"
        text.encode("utf-8")
    except Exception:
        serialization_failed = True
        text = ""
    if serialization_failed:
        raise DashboardValidationError("DASHBOARD_SERIALIZATION_INVALID")
    target = _validated_destination(destination)
    temporary: Path | None = None
    descriptor: int | None = None
    temp_identity: tuple[int, int, int] | None = None
    replaced = False
    write_failed = False
    try:
        with _locked_destination(target) as guard:
            try:
                descriptor, temporary = _create_sibling_temp(target, guard)
                temp_identity = _identity_tuple(os.fstat(descriptor))
                _require_temp_identity(descriptor, temporary, temp_identity, guard)
                _require_same_parent(target)
                stream = os.fdopen(
                    descriptor,
                    "w",
                    encoding="utf-8",
                    newline="\n",
                    closefd=False,
                )
                with stream:
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                _require_same_parent(target)
                _require_temp_identity(descriptor, temporary, temp_identity, guard)
                refreshed = _validated_destination(target.path)
                if refreshed.parent_identity != target.parent_identity:
                    raise DashboardValidationError("DASHBOARD_PATH_CHANGED")
                _require_temp_identity(descriptor, temporary, temp_identity, guard)
                if os.name != "nt" and guard.lstat(target.path) is not None:
                    raise DashboardValidationError("DASHBOARD_HANDLE_REPLACE_UNAVAILABLE")
                descriptor = _commit_open_temp(
                    descriptor,
                    temporary,
                    target.path,
                    guard,
                )
                replaced = True
                # Both platform commit primitives bind the exact open descriptor.
                # Returning from that syscall is the irreversible commit point.
                _require_replaced_identity(descriptor, target.path, temp_identity, guard)
                os.close(descriptor)
                descriptor = None
                _require_same_parent(target)
                temporary = None
            except (OSError, UnicodeError, ValueError):
                if descriptor is not None:
                    if os.name == "nt" and not replaced:
                        with suppress(OSError, ValueError):
                            _delete_windows_open_file(descriptor)
                    with suppress(OSError):
                        os.close(descriptor)
                    descriptor = None
                raise
            finally:
                if descriptor is not None:
                    with suppress(OSError):
                        os.close(descriptor)
                    descriptor = None
    except (OSError, UnicodeError, ValueError):
        write_failed = True
    if write_failed:
        raise DashboardValidationError("DASHBOARD_WRITE_FAILED")
    return target.path


def _revalidated_dashboard_status(
    status: object,
    *,
    gate_manifests: Mapping[str, frozenset[str]] = _CAPTURED_GATE_MANIFESTS,
) -> DashboardStatus:
    """Rebuild every schema-v1 record so frozen-object tampering is never trusted."""
    if type(status) is not DashboardStatus:
        raise DashboardValidationError("DASHBOARD_STATUS_INVALID")
    if type(status.research) is not DashboardResearch:
        raise DashboardValidationError("DASHBOARD_RESEARCH_INVALID")
    research = DashboardResearch(status.research.version, status.research.fresh)
    if type(status.strategies) is tuple and len(status.strategies) > _MAX_COLLECTION:
        raise DashboardValidationError("DASHBOARD_VALUE_BOUNDS_EXCEEDED")
    if type(status.strategies) is not tuple:
        raise DashboardValidationError("DASHBOARD_STRATEGY_INVALID")
    strategies: list[DashboardStrategy] = []
    for item in status.strategies:
        if type(item) is not DashboardStrategy:
            raise DashboardValidationError("DASHBOARD_STRATEGY_INVALID")
        strategies.append(DashboardStrategy(item.strategy_id, item.version))
    if type(status.promotion) is not DashboardPromotion:
        raise DashboardValidationError("DASHBOARD_PROMOTION_INVALID")
    promotion = DashboardPromotion(status.promotion.status)
    if type(status.portfolio) is not DashboardPortfolio:
        raise DashboardValidationError("DASHBOARD_PORTFOLIO_INVALID")
    portfolio = DashboardPortfolio(status.portfolio.currency, status.portfolio.equity)
    if type(status.risk) is not DashboardRisk:
        raise DashboardValidationError("DASHBOARD_RISK_INVALID")
    risk = DashboardRisk(
        status.risk.max_trade_risk_fraction,
        status.risk.max_position_fraction,
        status.risk.max_gross_exposure_fraction,
        status.risk.max_daily_loss_fraction,
        status.risk.max_drawdown_fraction,
    )
    if type(status.brokers) is tuple and len(status.brokers) > _MAX_COLLECTION:
        raise DashboardValidationError("DASHBOARD_VALUE_BOUNDS_EXCEEDED")
    if type(status.brokers) is not tuple:
        raise DashboardValidationError("DASHBOARD_BROKER_INVALID")
    brokers: list[DashboardBroker] = []
    for broker in status.brokers:
        if type(broker) is not DashboardBroker or type(broker.gates) is not tuple:
            raise DashboardValidationError("DASHBOARD_BROKER_INVALID")
        if len(broker.gates) > _MAX_COLLECTION:
            raise DashboardValidationError("DASHBOARD_VALUE_BOUNDS_EXCEEDED")
        if type(broker.name) is not str or broker.name not in {
            "alpaca",
            "groww",
            "ccxt",
        }:
            raise DashboardValidationError("DASHBOARD_BROKER_INVALID")
        try:
            required = gate_manifests[broker.name]
        except KeyError:
            raise DashboardValidationError("DASHBOARD_BROKER_INVALID") from None
        if len(broker.gates) != len(required):
            raise DashboardValidationError("DASHBOARD_BROKER_GATES_INVALID")
        gates: list[GateResult] = []
        for gate in broker.gates:
            name, passed, reason = _validated_dashboard_gate(gate)
            gates.append(
                GateResult(
                    name=name,
                    passed=passed,
                    reason_code=reason,
                )
            )
        brokers.append(DashboardBroker(broker.name, tuple(gates)))
    if type(status.orders) is tuple and len(status.orders) > _MAX_COLLECTION:
        raise DashboardValidationError("DASHBOARD_VALUE_BOUNDS_EXCEEDED")
    if type(status.orders) is not tuple:
        raise DashboardValidationError("DASHBOARD_ORDER_INVALID")
    orders: list[DashboardOrder] = []
    for order in status.orders:
        if type(order) is not DashboardOrder or type(order.status) is not OrderStatus:
            raise DashboardValidationError("DASHBOARD_ORDER_INVALID")
        orders.append(DashboardOrder(order.order_id, order.status))
    kill_switches = _revalidated_safety_states(
        status.kill_switches, "DASHBOARD_KILL_SWITCH_INVALID"
    )
    interlocks = _revalidated_safety_states(
        status.interlocks, "DASHBOARD_INTERLOCK_INVALID"
    )
    aspiration_value = status.aspirational_target
    if type(aspiration_value) is not DashboardAspiration:
        raise DashboardValidationError("DASHBOARD_ASPIRATION_INVALID")
    aspiration = DashboardAspiration(
        aspiration_value.starting_capital,
        aspiration_value.current_equity,
        aspiration_value.target,
        aspiration_value.required_multiple,
        aspiration_value.achieved_multiple,
        aspiration_value.remaining_gap,
        aspiration_value.reporting_only,
    )
    return DashboardStatus(
        generated_at=status.generated_at,
        data_as_of=status.data_as_of,
        research=research,
        strategies=tuple(strategies),
        promotion=promotion,
        portfolio=portfolio,
        risk=risk,
        brokers=tuple(brokers),
        orders=tuple(orders),
        kill_switches=kill_switches,
        interlocks=interlocks,
        aspirational_target=aspiration,
    )


def _revalidated_safety_states(
    value: object,
    reason: str,
) -> tuple[DashboardSafetyState, ...]:
    if type(value) is tuple and len(value) > _MAX_COLLECTION:
        raise DashboardValidationError("DASHBOARD_VALUE_BOUNDS_EXCEEDED")
    if type(value) is not tuple:
        raise DashboardValidationError(reason)
    result: list[DashboardSafetyState] = []
    for item in value:
        if type(item) is not DashboardSafetyState:
            raise DashboardValidationError(reason)
        result.append(DashboardSafetyState(item.active, item.reason_code))
    return tuple(result)


def _validated_dashboard_gate(value: object) -> tuple[str, bool, str]:
    if type(value) is not GateResult:
        raise DashboardValidationError("DASHBOARD_BROKER_GATES_INVALID")
    name = value.name
    passed = value.passed
    reason = value.reason_code
    if (
        type(name) is not str
        or not name
        or len(name) > _MAX_GATE_NAME_BYTES
        or type(passed) is not bool
        or type(reason) is not str
        or not reason
        or len(reason) > _MAX_GATE_REASON_BYTES
    ):
        raise DashboardValidationError("DASHBOARD_BROKER_GATES_INVALID")
    try:
        bounded = (
            len(name.encode("utf-8")) <= _MAX_GATE_NAME_BYTES
            and len(reason.encode("utf-8")) <= _MAX_GATE_REASON_BYTES
        )
    except UnicodeError:
        bounded = False
    if (
        not bounded
        or _REASON.fullmatch(reason) is None
        or (passed and reason != "OK")
        or (not passed and reason == "OK")
    ):
        raise DashboardValidationError("DASHBOARD_BROKER_GATES_INVALID")
    return name, passed, reason


def _dashboard_payload(status: DashboardStatus) -> dict[str, object]:
    generated = status.generated_at
    data_as_of = status.data_as_of
    return {
        "schema_version": 1,
        "generated_at": _time_text(generated),
        "data_as_of": _time_text(data_as_of),
        "freshness_age_seconds": int((generated - data_as_of).total_seconds()),
        "research": {"version": status.research.version, "fresh": status.research.fresh},
        "strategies": [
            {"id": item.strategy_id, "version": item.version} for item in status.strategies
        ],
        "promotion": {"status": status.promotion.status},
        "portfolio": {
            "currency": status.portfolio.currency,
            "equity": status.portfolio.equity,
        },
        "risk": {
            "max_trade_risk_fraction": status.risk.max_trade_risk_fraction,
            "max_position_fraction": status.risk.max_position_fraction,
            "max_gross_exposure_fraction": status.risk.max_gross_exposure_fraction,
            "max_daily_loss_fraction": status.risk.max_daily_loss_fraction,
            "max_drawdown_fraction": status.risk.max_drawdown_fraction,
        },
        "brokers": [
            {
                "name": broker.name,
                "ready": all(gate.passed for gate in broker.gates),
                "missing_gates": sorted(gate.name for gate in broker.gates if not gate.passed),
                "gates": [
                    {
                        "name": gate.name,
                        "passed": gate.passed,
                        "reason_code": gate.reason_code,
                    }
                    for gate in sorted(broker.gates, key=lambda item: item.name)
                ],
            }
            for broker in status.brokers
        ],
        "orders": [
            {"id": order.order_id, "status": order.status.value} for order in status.orders
        ],
        "kill_switches": [
            {"active": item.active, "reason_code": item.reason_code}
            for item in status.kill_switches
        ],
        "interlocks": [
            {"active": item.active, "reason_code": item.reason_code}
            for item in status.interlocks
        ],
        "aspirational_target": {
            "starting_capital": status.aspirational_target.starting_capital,
            "current_equity": status.aspirational_target.current_equity,
            "target": status.aspirational_target.target,
            "required_multiple": status.aspirational_target.required_multiple,
            "achieved_multiple": status.aspirational_target.achieved_multiple,
            "remaining_gap": status.aspirational_target.remaining_gap,
            "reporting_only": status.aspirational_target.reporting_only,
        },
    }


def _safe_value(value: object, *, depth: int, budget: _Budget) -> object:
    budget.nodes += 1
    if budget.nodes > _MAX_NODES or depth > _MAX_DEPTH:
        raise DashboardValidationError("DASHBOARD_VALUE_BOUNDS_EXCEEDED")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if value.bit_length() > 63:
            raise DashboardValidationError("DASHBOARD_INTEGER_INVALID")
        return value
    if type(value) is Decimal:
        try:
            return canonical_decimal(value)
        except CanonicalEncodingError:
            raise DashboardValidationError("DASHBOARD_DECIMAL_INVALID") from None
    if type(value) is str:
        if len(value) > _MAX_STRING_BYTES:
            raise DashboardValidationError("DASHBOARD_STRING_INVALID")
        if len(value.encode("utf-8")) > _MAX_STRING_BYTES or any(
            ord(character) < 32 and character not in "\t\n\r" for character in value
        ):
            raise DashboardValidationError("DASHBOARD_STRING_INVALID")
        return _REDACTED if _secret_value(value) else value
    if type(value) in (dict, MappingProxyType):
        assert isinstance(value, Mapping)
        if len(value) > _MAX_COLLECTION:
            raise DashboardValidationError("DASHBOARD_VALUE_BOUNDS_EXCEEDED")
        result: dict[str, object] = {}
        for key, item in value.items():
            if (
                type(key) is not str
                or len(key) > 64
                or _OUTPUT_KEY.fullmatch(key) is None
                or len(key.encode("utf-8")) > _MAX_STRING_BYTES
            ):
                raise DashboardValidationError("DASHBOARD_KEY_INVALID")
            normalized = "".join(character for character in key.casefold() if character.isalnum())
            if any(part in normalized for part in _SECRET_KEY_PARTS):
                continue
            result[key] = _safe_value(item, depth=depth + 1, budget=budget)
        return result
    if type(value) in (list, tuple):
        assert isinstance(value, Sequence)
        if len(value) > _MAX_COLLECTION:
            raise DashboardValidationError("DASHBOARD_VALUE_BOUNDS_EXCEEDED")
        return [_safe_value(item, depth=depth + 1, budget=budget) for item in value]
    raise DashboardValidationError("DASHBOARD_VALUE_INVALID")


def _validated_destination(destination: object) -> _Destination:
    if type(destination) is not type(Path()):
        raise DashboardValidationError("DASHBOARD_PATH_INVALID")
    assert isinstance(destination, Path)
    raw = str(destination)
    if (
        not raw
        or len(raw) > _MAX_PATH_CHARS
        or destination.suffix.casefold() != ".json"
        or raw.startswith(("\\\\", "//"))
        or any(ord(character) < 32 for character in raw)
        or (not destination.is_absolute() and ".." in destination.parts)
    ):
        raise DashboardValidationError("DASHBOARD_PATH_INVALID")
    lexical = Path(os.path.abspath(destination))
    if str(lexical).startswith(("\\\\", "//")) or _is_network_path(lexical):
        raise DashboardValidationError("DASHBOARD_PATH_INVALID")
    _inspect_existing_components(lexical)
    parent = lexical.parent
    parent_stat = _safe_lstat(parent)
    if parent_stat is None or not stat.S_ISDIR(parent_stat.st_mode) or _is_reparse(parent_stat):
        raise DashboardValidationError("DASHBOARD_PATH_INVALID")
    target_stat = _safe_lstat(lexical)
    if target_stat is not None and (
        not stat.S_ISREG(target_stat.st_mode) or _is_reparse(target_stat)
    ):
        raise DashboardValidationError("DASHBOARD_PATH_INVALID")
    return _Destination(lexical, _identity_tuple(parent_stat))


def _inspect_existing_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        details = _safe_lstat(current)
        if details is not None and (_is_reparse(details) or stat.S_ISLNK(details.st_mode)):
            raise DashboardValidationError("DASHBOARD_PATH_INVALID")


def _safe_lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise DashboardValidationError("DASHBOARD_PATH_INVALID") from None


def _is_reparse(details: os.stat_result) -> bool:
    attributes = getattr(details, "st_file_attributes", 0)
    return type(attributes) is int and bool(attributes & _REPARSE_POINT)


def _is_network_path(path: Path) -> bool:
    if os.name != "nt":
        return False
    root = path.anchor
    if not root:
        return True
    try:
        kernel32 = vars(ctypes)["WinDLL"]("kernel32", use_last_error=True)
        get_drive_type = cast(Callable[[str], int], kernel32.GetDriveTypeW)
        drive_type = get_drive_type(root)
    except Exception:
        return True
    return drive_type in {0, 1, 4}


def _identity_tuple(details: os.stat_result) -> tuple[int, int, int]:
    return (
        int(details.st_dev),
        int(details.st_ino),
        int(getattr(details, "st_file_attributes", 0)),
    )


def _require_same_parent(destination: _Destination) -> None:
    _inspect_existing_components(destination.path)
    details = _safe_lstat(destination.path.parent)
    if (
        details is None
        or _identity_tuple(details) != destination.parent_identity
        or _is_reparse(details)
    ):
        raise DashboardValidationError("DASHBOARD_PATH_CHANGED")


def _require_regular_local_file(path: Path) -> None:
    details = _safe_lstat(path)
    if details is None or not stat.S_ISREG(details.st_mode) or _is_reparse(details):
        raise DashboardValidationError("DASHBOARD_PATH_INVALID")


def _require_temp_identity(
    descriptor: int,
    path: Path,
    expected: tuple[int, int, int],
    guard: _DirectoryGuard,
) -> None:
    handle_details = os.fstat(descriptor)
    if _is_anonymous_posix_temp(path, guard):
        if (
            _identity_tuple(handle_details) != expected
            or not stat.S_ISREG(handle_details.st_mode)
            or _is_reparse(handle_details)
        ):
            raise DashboardValidationError("DASHBOARD_TEMP_CHANGED")
        return
    entry_details = guard.lstat(path)
    if (
        _identity_tuple(handle_details) != expected
        or entry_details is None
        or not stat.S_ISREG(entry_details.st_mode)
        or _is_reparse(entry_details)
        or _identity_tuple(entry_details) != expected
    ):
        raise DashboardValidationError("DASHBOARD_TEMP_CHANGED")


def _is_anonymous_posix_temp(path: Path, guard: _DirectoryGuard) -> bool:
    return (
        os.name != "nt"
        and guard.parent_fd is not None
        and path.name.endswith(".anonymous")
    )


def _require_replaced_identity(
    descriptor: int | None,
    destination: Path,
    expected: tuple[int, int, int],
    guard: _DirectoryGuard,
) -> None:
    entry_details = guard.lstat(destination)
    if (
        entry_details is None
        or not stat.S_ISREG(entry_details.st_mode)
        or _is_reparse(entry_details)
        or _identity_tuple(entry_details) != expected
    ):
        raise DashboardValidationError("DASHBOARD_REPLACE_CHANGED")
    if descriptor is not None and _identity_tuple(os.fstat(descriptor)) != expected:
        raise DashboardValidationError("DASHBOARD_REPLACE_CHANGED")


def _commit_open_temp(
    descriptor: int,
    temporary: Path,
    destination: Path,
    guard: _DirectoryGuard,
) -> int:
    """Commit the exact open file, never the last occupant of its temporary name."""
    if os.name == "nt":
        _rename_windows_open_file(descriptor, destination)
        return descriptor
    if guard.parent_fd is None:
        raise DashboardValidationError("DASHBOARD_HANDLE_COMMIT_UNAVAILABLE")
    if guard.lstat(destination) is not None:
        raise DashboardValidationError("DASHBOARD_HANDLE_REPLACE_UNAVAILABLE")
    _link_open_file_at(descriptor, guard.parent_fd, destination.name)
    return descriptor


def _link_open_file_at(descriptor: int, parent_fd: int, name: str) -> None:
    """Bind an exact open POSIX file into its held parent, or fail closed."""
    if (
        type(name) is not str
        or not name
        or "/" in name
        or "\\" in name
        or len(name) > 255
    ):
        raise DashboardValidationError("DASHBOARD_HANDLE_COMMIT_INVALID")
    try:
        encoded = name.encode("utf-8")
        libc = ctypes.CDLL(None, use_errno=True)
        linkat = libc.linkat
        linkat.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        )
        linkat.restype = ctypes.c_int
        linked = linkat(descriptor, b"", parent_fd, encoded, 0x1000)
        if linked != 0:
            procfd = f"/proc/self/fd/{descriptor}".encode("ascii")
            linked = linkat(-100, procfd, parent_fd, encoded, 0x400)
    except Exception:
        raise DashboardValidationError("DASHBOARD_HANDLE_COMMIT_UNAVAILABLE") from None
    if linked != 0:
        raise DashboardValidationError("DASHBOARD_HANDLE_COMMIT_UNAVAILABLE")


def _rename_windows_open_file(descriptor: int, destination: Path) -> None:
    import msvcrt
    from ctypes import wintypes

    filename = str(destination)
    encoded_filename = filename.encode("utf-16-le")

    class _FileRenameInfo(ctypes.Structure):
        _fields_ = (
            ("replace_if_exists", wintypes.BOOLEAN),
            ("root_directory", wintypes.HANDLE),
            ("filename_length", wintypes.DWORD),
            ("filename", wintypes.WCHAR * 1),
        )

    filename_offset = _FileRenameInfo.filename.offset
    information_size = filename_offset + len(encoded_filename) + ctypes.sizeof(
        wintypes.WCHAR
    )
    information_buffer = ctypes.create_string_buffer(information_size)
    information = _FileRenameInfo.from_buffer(information_buffer)
    information.replace_if_exists = 1
    information.root_directory = None
    information.filename_length = len(encoded_filename)
    ctypes.memmove(
        ctypes.addressof(information_buffer) + filename_offset,
        encoded_filename,
        len(encoded_filename),
    )
    kernel32 = vars(ctypes)["WinDLL"]("kernel32", use_last_error=True)
    prototype = vars(ctypes)["WINFUNCTYPE"](
        wintypes.BOOL,
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    set_information = prototype(("SetFileInformationByHandle", kernel32))
    succeeded = set_information(
        vars(msvcrt)["get_osfhandle"](descriptor),
        3,
        information_buffer,
        information_size,
    )
    if not succeeded:
        raise DashboardValidationError("DASHBOARD_HANDLE_COMMIT_FAILED")


def _delete_windows_open_file(descriptor: int) -> None:
    """Mark the exact still-open Windows temporary for deletion on close."""
    import msvcrt
    from ctypes import wintypes

    class _FileDispositionInfo(ctypes.Structure):
        _fields_ = (("delete_file", wintypes.BOOLEAN),)

    information = _FileDispositionInfo(True)
    kernel32 = vars(ctypes)["WinDLL"]("kernel32", use_last_error=True)
    prototype = vars(ctypes)["WINFUNCTYPE"](
        wintypes.BOOL,
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    set_information = prototype(("SetFileInformationByHandle", kernel32))
    succeeded = set_information(
        vars(msvcrt)["get_osfhandle"](descriptor),
        4,
        ctypes.byref(information),
        ctypes.sizeof(information),
    )
    if not succeeded:
        raise DashboardValidationError("DASHBOARD_TEMP_CLEANUP_UNAVAILABLE")


@contextmanager
def _locked_destination(destination: _Destination) -> Iterator[_DirectoryGuard]:
    if os.name != "nt":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(destination.path.parent, flags)
        lock_descriptor: int | None = None
        try:
            if _identity_tuple(os.fstat(descriptor)) != destination.parent_identity:
                raise DashboardValidationError("DASHBOARD_PATH_CHANGED")
            lock_descriptor = _lock_posix_destination(descriptor, destination.path.name)
            _require_same_parent(destination)
            yield _DirectoryGuard(descriptor)
        finally:
            if lock_descriptor is not None:
                _unlock_posix_destination(lock_descriptor)
            os.close(descriptor)
        return

    with _windows_destination_mutex(destination.path):
        handles: list[int] = []
        try:
            for component in _directory_components(destination.path.parent):
                handles.append(_lock_windows_directory(component))
            _require_same_parent(destination)
            yield _DirectoryGuard(None)
        finally:
            for handle in reversed(handles):
                _close_windows_handle(handle)


def _lock_posix_destination(parent_fd: int, name: str) -> int:
    fcntl = cast(_FileLockModule, importlib.import_module("fcntl"))

    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    lock_name = f".market-sentinel-dashboard-{digest}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_name, flags, 0o600, dir_fd=parent_fd)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or _is_reparse(details):
            raise DashboardValidationError("DASHBOARD_LOCK_INVALID")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _unlock_posix_destination(descriptor: int) -> None:
    fcntl = cast(_FileLockModule, importlib.import_module("fcntl"))

    with suppress(OSError):
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


@contextmanager
def _windows_destination_mutex(destination: Path) -> Iterator[None]:
    from ctypes import wintypes

    digest = hashlib.sha256(str(destination).casefold().encode("utf-8")).hexdigest()
    name = f"Local\\OmnimarketSentinelDashboard-{digest}"
    kernel32 = vars(ctypes)["WinDLL"]("kernel32", use_last_error=True)
    create_prototype = vars(ctypes)["WINFUNCTYPE"](
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    )
    create_mutex = create_prototype(("CreateMutexW", kernel32))
    handle = create_mutex(None, False, name)
    if handle is None:
        raise DashboardValidationError("DASHBOARD_LOCK_UNAVAILABLE")
    wait_prototype = vars(ctypes)["WINFUNCTYPE"](
        wintypes.DWORD, wintypes.HANDLE, wintypes.DWORD
    )
    wait = wait_prototype(("WaitForSingleObject", kernel32))
    release_prototype = vars(ctypes)["WINFUNCTYPE"](
        wintypes.BOOL, wintypes.HANDLE
    )
    release = release_prototype(("ReleaseMutex", kernel32))
    acquired = wait(handle, 30_000)
    if acquired not in {0, 0x00000080}:
        _close_windows_handle(int(handle))
        raise DashboardValidationError("DASHBOARD_LOCK_UNAVAILABLE")
    try:
        yield
    finally:
        release(handle)
        _close_windows_handle(int(handle))


def _create_sibling_temp(
    destination: _Destination,
    guard: _DirectoryGuard,
) -> tuple[int, Path]:
    if guard.parent_fd is None:
        if os.name == "nt":
            return _create_windows_sibling_temp(destination)
        raise DashboardValidationError("DASHBOARD_TEMP_UNAVAILABLE")
    anonymous_flag = getattr(os, "O_TMPFILE", 0)
    if not anonymous_flag:
        raise DashboardValidationError("DASHBOARD_TEMP_UNAVAILABLE")
    try:
        descriptor = os.open(
            ".",
            os.O_RDWR | anonymous_flag,
            0o600,
            dir_fd=guard.parent_fd,
        )
    except OSError:
        raise DashboardValidationError("DASHBOARD_TEMP_UNAVAILABLE") from None
    marker = destination.path.parent / (
        f".{destination.path.name}.{secrets.token_hex(12)}.anonymous"
    )
    return descriptor, marker


def _create_windows_sibling_temp(destination: _Destination) -> tuple[int, Path]:
    import msvcrt

    kernel32 = vars(ctypes)["WinDLL"]("kernel32", use_last_error=True)
    prototype = vars(ctypes)["WINFUNCTYPE"](
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    create_file = prototype(("CreateFileW", kernel32))
    invalid = ctypes.c_void_p(-1).value
    for _ in range(16):
        path = destination.path.parent / (
            f".{destination.path.name}.{secrets.token_hex(12)}.tmp"
        )
        handle = create_file(
            str(path),
            0x80000000 | 0x40000000 | 0x00010000,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            1,
            0x00000080 | 0x00000100,
            None,
        )
        if handle is not None and handle != invalid:
            try:
                descriptor = vars(msvcrt)["open_osfhandle"](
                    int(handle), os.O_RDWR | getattr(os, "O_BINARY", 0)
                )
            except OSError:
                _close_windows_handle(int(handle))
                raise
            return descriptor, path
        if vars(ctypes)["get_last_error"]() != 80:
            raise DashboardValidationError("DASHBOARD_TEMP_UNAVAILABLE")
    raise DashboardValidationError("DASHBOARD_TEMP_UNAVAILABLE")


def _directory_components(parent: Path) -> tuple[Path, ...]:
    components: list[Path] = []
    current = Path(parent.anchor)
    components.append(current)
    for part in parent.parts[1:]:
        current /= part
        components.append(current)
    return tuple(components)


def _lock_windows_directory(path: Path) -> int:
    kernel32 = vars(ctypes)["WinDLL"]("kernel32", use_last_error=True)
    prototype = vars(ctypes)["WINFUNCTYPE"](
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    create_file = prototype(("CreateFileW", kernel32))
    handle = create_file(
        str(path),
        0x80000000,
        0x00000001 | 0x00000002,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle is None or handle == invalid:
        raise DashboardValidationError("DASHBOARD_PATH_LOCK_FAILED")
    return int(handle)


def _close_windows_handle(handle: int) -> None:
    kernel32 = vars(ctypes)["WinDLL"]("kernel32", use_last_error=True)
    prototype = vars(ctypes)["WINFUNCTYPE"](ctypes.c_int, ctypes.c_void_p)
    close_handle = prototype(("CloseHandle", kernel32))
    close_handle(handle)


def _utc(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is not UTC:
        raise DashboardValidationError("DASHBOARD_TIMESTAMP_INVALID")
    return value


def _time_text(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _exact_bool(value: object) -> bool:
    if type(value) is not bool:
        raise DashboardValidationError("DASHBOARD_BOOLEAN_INVALID")
    return value


def _finite_decimal(value: object) -> Decimal:
    if type(value) is not Decimal or not value.is_finite():
        raise DashboardValidationError("DASHBOARD_DECIMAL_INVALID")
    try:
        canonical_decimal(value)
    except CanonicalEncodingError:
        raise DashboardValidationError("DASHBOARD_DECIMAL_INVALID") from None
    return value


def _positive_decimal(value: object) -> Decimal:
    decimal = _finite_decimal(value)
    if decimal <= 0:
        raise DashboardValidationError("DASHBOARD_DECIMAL_INVALID")
    return decimal


def _nonnegative_decimal(value: object) -> Decimal:
    decimal = _finite_decimal(value)
    if decimal < 0:
        raise DashboardValidationError("DASHBOARD_DECIMAL_INVALID")
    return decimal


def _identity(value: object) -> str:
    if (
        type(value) is not str
        or len(value) > 128
        or _IDENTITY.fullmatch(value) is None
        or _secret_value(value)
    ):
        raise DashboardValidationError("DASHBOARD_IDENTITY_INVALID")
    return value


def _version(value: object) -> str:
    if (
        type(value) is not str
        or len(value) > 64
        or _VERSION.fullmatch(value) is None
        or _secret_value(value)
    ):
        raise DashboardValidationError("DASHBOARD_VERSION_INVALID")
    return value


def _nonempty_exact_tuple(value: object, item_type: type[object], reason: str) -> None:
    if type(value) is tuple and len(value) > _MAX_COLLECTION:
        raise DashboardValidationError("DASHBOARD_VALUE_BOUNDS_EXCEEDED")
    if type(value) is not tuple or not value or not all(type(item) is item_type for item in value):
        raise DashboardValidationError(reason)


_secret_value = secret_text_present
