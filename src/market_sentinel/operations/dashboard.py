"""Deterministic, redacted, atomic local dashboard export."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType

from market_sentinel.execution.canonical import CanonicalEncodingError, canonical_decimal

_MAX_DEPTH = 12
_MAX_NODES = 4_096
_MAX_COLLECTION = 512
_MAX_STRING_BYTES = 4_096
_MAX_PATH_CHARS = 2_048
_FORBIDDEN_KEY_PARTS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "credential",
    "authorization",
    "authenticator",
    "private_key",
    "callback",
    "exception",
)
_SECRET_VALUE = re.compile(
    r"(?i)(?:\bbearer\s+\S+|\bsk-[a-z0-9_-]{12,}|\bgh[pousr]_[a-z0-9]{20,}|"
    r"\bAKIA[A-Z0-9]{16}\b|[a-z0-9_-]{20,}\.[a-z0-9_-]{20,}\.[a-z0-9_-]{20,})"
)
_REDACTED = "[REDACTED]"


class DashboardValidationError(ValueError):
    """A dashboard snapshot or destination failed its bounded safe contract."""


@dataclass(frozen=True, slots=True)
class DashboardStatus:
    """Explicit local status fields permitted in dashboard schema v1."""

    generated_at: datetime
    data_as_of: datetime
    research: Mapping[str, object]
    strategies: tuple[Mapping[str, object], ...]
    promotion: Mapping[str, object]
    portfolio: Mapping[str, object]
    risk: Mapping[str, object]
    brokers: tuple[Mapping[str, object], ...]
    orders: tuple[Mapping[str, object], ...]
    kill_switches: tuple[Mapping[str, object], ...]
    interlocks: tuple[Mapping[str, object], ...]
    aspirational_target: Mapping[str, object]


@dataclass(slots=True)
class _Budget:
    nodes: int = 0


def safe_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Return a bounded JSON mapping with secret-bearing fields removed."""
    if type(value) not in (dict, MappingProxyType):
        raise DashboardValidationError("DASHBOARD_VALUE_INVALID")
    budget = _Budget()
    prepared = _safe_value(value, depth=0, budget=budget)
    if type(prepared) is not dict:
        raise DashboardValidationError("DASHBOARD_VALUE_INVALID")
    return prepared


def export_dashboard(status: DashboardStatus, destination: Path) -> Path:
    """Write schema v1 through a flushed sibling temporary file and atomic replace."""
    if type(status) is not DashboardStatus:
        raise DashboardValidationError("DASHBOARD_STATUS_INVALID")
    generated = _aware_utc(status.generated_at)
    data_as_of = _aware_utc(status.data_as_of)
    if data_as_of > generated:
        raise DashboardValidationError("DASHBOARD_FRESHNESS_INVALID")
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at": generated,
        "data_as_of": data_as_of,
        "freshness_age_seconds": int((generated - data_as_of).total_seconds()),
        "research": status.research,
        "strategies": status.strategies,
        "promotion": status.promotion,
        "portfolio": status.portfolio,
        "risk": status.risk,
        "brokers": status.brokers,
        "orders": status.orders,
        "kill_switches": status.kill_switches,
        "interlocks": status.interlocks,
        "aspirational_target": status.aspirational_target,
    }
    prepared = safe_json_mapping(payload)
    aspiration = prepared.get("aspirational_target")
    if not isinstance(aspiration, dict) or aspiration.get("reporting_only") is not True:
        raise DashboardValidationError("ASPIRATION_MUST_BE_REPORTING_ONLY")
    target = _validated_destination(destination)
    text = json.dumps(
        prepared,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    temporary: Path | None = None
    write_failed = False
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    except (OSError, UnicodeError, ValueError):
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        write_failed = True
    if write_failed:
        raise DashboardValidationError("DASHBOARD_WRITE_FAILED")
    return target


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
    if type(value) is datetime:
        return _aware_utc(value).isoformat().replace("+00:00", "Z")
    if type(value) is str:
        if len(value.encode("utf-8")) > _MAX_STRING_BYTES or any(
            ord(character) < 32 and character not in "\t\n\r" for character in value
        ):
            raise DashboardValidationError("DASHBOARD_STRING_INVALID")
        if _SECRET_VALUE.search(value):
            return _REDACTED
        return value
    if type(value) in (dict, MappingProxyType):
        assert isinstance(value, Mapping)
        if len(value) > _MAX_COLLECTION:
            raise DashboardValidationError("DASHBOARD_VALUE_BOUNDS_EXCEEDED")
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or len(key.encode("utf-8")) > _MAX_STRING_BYTES:
                raise DashboardValidationError("DASHBOARD_KEY_INVALID")
            normalized = key.casefold().replace("-", "_")
            if any(part in normalized for part in _FORBIDDEN_KEY_PARTS):
                continue
            result[key] = _safe_value(item, depth=depth + 1, budget=budget)
        return result
    if type(value) in (list, tuple):
        assert isinstance(value, Sequence)
        if len(value) > _MAX_COLLECTION:
            raise DashboardValidationError("DASHBOARD_VALUE_BOUNDS_EXCEEDED")
        return [_safe_value(item, depth=depth + 1, budget=budget) for item in value]
    raise DashboardValidationError("DASHBOARD_VALUE_INVALID")


def _aware_utc(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise DashboardValidationError("DASHBOARD_TIMESTAMP_INVALID")
    return value.astimezone(UTC)


def _validated_destination(destination: object) -> Path:
    if type(destination) is not type(Path()):
        raise DashboardValidationError("DASHBOARD_PATH_INVALID")
    assert isinstance(destination, Path)
    raw = str(destination)
    if (
        not raw
        or len(raw) > _MAX_PATH_CHARS
        or raw.startswith(("\\\\", "//"))
        or destination.suffix.casefold() != ".json"
        or any(ord(character) < 32 for character in raw)
        or (not destination.is_absolute() and ".." in destination.parts)
    ):
        raise DashboardValidationError("DASHBOARD_PATH_INVALID")
    target = destination.resolve(strict=False)
    if destination.is_symlink() or not target.parent.exists() or not target.parent.is_dir():
        raise DashboardValidationError("DASHBOARD_PATH_INVALID")
    return target
