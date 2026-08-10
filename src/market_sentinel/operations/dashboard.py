"""Explicit schema-v1 dashboard conversion and hardened atomic export."""

from __future__ import annotations

import ctypes
import json
import os
import re
import secrets
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import cast
from urllib.parse import unquote_plus

from market_sentinel.brokers.preflight import required_gate_names
from market_sentinel.domain.enums import OrderStatus
from market_sentinel.domain.models import GateResult
from market_sentinel.execution.canonical import CanonicalEncodingError, canonical_decimal

_MAX_DEPTH = 12
_MAX_NODES = 4_096
_MAX_COLLECTION = 512
_MAX_STRING_BYTES = 4_096
_MAX_PATH_CHARS = 2_048
_REPARSE_POINT = 0x400
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@-]{0,127}$")
_CURRENCY = re.compile(r"^[A-Z]{3,8}$")
_REASON = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_PROMOTIONS = frozenset({"not_promoted", "research", "backtest", "paper", "live-small"})
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
_SECRET_VALUE = re.compile(
    r"(?is)(?:\bbearer\s+\S+|\bsk-[a-z0-9_-]{8,}|\bgh[pousr]_[a-z0-9]{20,}|"
    r"\bAKIA[A-Z0-9]{16}\b|-----BEGIN[^-]*(?:PRIVATE|SECRET)[^-]*-----|"
    r"[a-z0-9_-]{20,}\.[a-z0-9_-]{20,}\.[a-z0-9_-]{20,}|"
    r"\b(?:api[_-]?key|access[_-]?(?:key|token)|private[_-]?key|client[_-]?secret|"
    r"refresh[_-]?token|id[_-]?token|secret|token|password|credential|authorization|"
    r"cookie|set[_-]?cookie|session(?:id)?)\b[\"']?\s*[:=]\s*[\"']?[^\s&;,\"'}]+)"
)
_REDACTED = "[REDACTED]"


class DashboardValidationError(ValueError):
    """A dashboard snapshot or destination failed its bounded safe contract."""


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
        manifest_name = "ccxt-spot" if self.name == "ccxt" else self.name
        required = required_gate_names(manifest_name)
        if type(self.gates) is not tuple or len(self.gates) != len(required):
            raise DashboardValidationError("DASHBOARD_BROKER_GATES_INVALID")
        names: list[str] = []
        for gate in self.gates:
            if (
                type(gate) is not GateResult
                or type(gate.name) is not str
                or type(gate.passed) is not bool
                or type(gate.reason_code) is not str
                or _REASON.fullmatch(gate.reason_code) is None
            ):
                raise DashboardValidationError("DASHBOARD_BROKER_GATES_INVALID")
            names.append(gate.name)
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

    def replace(self, source: Path, destination: Path) -> None:
        if self.parent_fd is None:
            os.replace(source, destination)
            return
        os.replace(
            source.name,
            destination.name,
            src_dir_fd=self.parent_fd,
            dst_dir_fd=self.parent_fd,
        )

    def unlink(self, path: Path) -> None:
        if self.parent_fd is None:
            path.unlink(missing_ok=True)
            return
        try:
            os.unlink(path.name, dir_fd=self.parent_fd)
        except FileNotFoundError:
            return


def safe_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Return bounded built-in JSON data with credential fields removed."""
    if type(value) not in (dict, MappingProxyType):
        raise DashboardValidationError("DASHBOARD_VALUE_INVALID")
    budget = _Budget()
    prepared = _safe_value(value, depth=0, budget=budget)
    if type(prepared) is not dict:
        raise DashboardValidationError("DASHBOARD_VALUE_INVALID")
    return prepared


def export_dashboard(status: DashboardStatus, destination: Path) -> Path:
    """Convert exact schema v1, then atomically replace after repeated path checks."""
    if type(status) is not DashboardStatus:
        raise DashboardValidationError("DASHBOARD_STATUS_INVALID")
    invalid = False
    try:
        status.__post_init__()
        prepared = safe_json_mapping(_dashboard_payload(status))
    except Exception:
        invalid = True
        prepared = None
    if invalid or prepared is None:
        raise DashboardValidationError("DASHBOARD_STATUS_INVALID")
    target = _validated_destination(destination)
    text = json.dumps(
        prepared,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    temporary: Path | None = None
    descriptor: int | None = None
    write_failed = False
    try:
        with _locked_destination(target) as guard:
            try:
                descriptor, temporary = _create_sibling_temp(target, guard)
                _require_regular_local_file(temporary)
                _require_same_parent(target)
                stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
                descriptor = None
                with stream:
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                _require_same_parent(target)
                refreshed = _validated_destination(target.path)
                if refreshed.parent_identity != target.parent_identity:
                    raise DashboardValidationError("DASHBOARD_PATH_CHANGED")
                guard.replace(temporary, target.path)
                temporary = None
            except (OSError, UnicodeError, ValueError):
                if descriptor is not None:
                    with suppress(OSError):
                        os.close(descriptor)
                if temporary is not None:
                    with suppress(OSError):
                        guard.unlink(temporary)
                raise
    except (OSError, UnicodeError, ValueError):
        write_failed = True
    if write_failed:
        raise DashboardValidationError("DASHBOARD_WRITE_FAILED")
    return target.path


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
            if type(key) is not str or len(key.encode("utf-8")) > _MAX_STRING_BYTES:
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
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
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


@contextmanager
def _locked_destination(destination: _Destination) -> Iterator[_DirectoryGuard]:
    if os.name != "nt":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(destination.path.parent, flags)
        try:
            if _identity_tuple(os.fstat(descriptor)) != destination.parent_identity:
                raise DashboardValidationError("DASHBOARD_PATH_CHANGED")
            _require_same_parent(destination)
            yield _DirectoryGuard(descriptor)
        finally:
            os.close(descriptor)
        return

    handles: list[int] = []
    try:
        for component in _directory_components(destination.path.parent):
            handles.append(_lock_windows_directory(component))
        _require_same_parent(destination)
        yield _DirectoryGuard(None)
    finally:
        for handle in reversed(handles):
            _close_windows_handle(handle)


def _create_sibling_temp(
    destination: _Destination,
    guard: _DirectoryGuard,
) -> tuple[int, Path]:
    if guard.parent_fd is None:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.path.name}.",
            suffix=".tmp",
            dir=destination.path.parent,
        )
        return descriptor, Path(name)
    for _ in range(16):
        name = f".{destination.path.name}.{secrets.token_hex(12)}.tmp"
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=guard.parent_fd,
            )
        except FileExistsError:
            continue
        return descriptor, destination.path.parent / name
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
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    prototype = ctypes.WINFUNCTYPE(
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
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    prototype = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
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
        or _IDENTITY.fullmatch(value) is None
        or _secret_value(value)
    ):
        raise DashboardValidationError("DASHBOARD_IDENTITY_INVALID")
    return value


def _version(value: object) -> str:
    if (
        type(value) is not str
        or _VERSION.fullmatch(value) is None
        or _secret_value(value)
    ):
        raise DashboardValidationError("DASHBOARD_VERSION_INVALID")
    return value


def _nonempty_exact_tuple(value: object, item_type: type[object], reason: str) -> None:
    if type(value) is not tuple or not value or not all(type(item) is item_type for item in value):
        raise DashboardValidationError(reason)


def _secret_value(value: str) -> bool:
    return _SECRET_VALUE.search(value) is not None or _SECRET_VALUE.search(
        unquote_plus(value)
    ) is not None
