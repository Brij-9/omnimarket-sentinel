"""Helpers for removing credentials from structured output."""

from collections.abc import Mapping
from typing import cast

_REDACTED = "[REDACTED]"
_SECRET_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "secret_key",
        "access_token",
        "password",
        "totp",
        "private_key",
    }
)


def redact_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Return a copy of *value* with nested secret-bearing fields redacted."""
    return {
        key: _REDACTED if _is_secret_key(key) else _redact_value(item)
        for key, item in value.items()
    }


def _is_secret_key(key: str) -> bool:
    normalized = key.lower()
    return normalized in _SECRET_KEYS or normalized.endswith(("_secret", "_token"))


def _redact_value(value: object) -> object:
    if isinstance(value, Mapping):
        return redact_mapping(cast(Mapping[str, object], value))
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    return value
