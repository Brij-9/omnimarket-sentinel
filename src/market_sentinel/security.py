"""Bounded helpers for detecting and removing credentials from output."""

import re
from collections.abc import Callable, Mapping
from typing import cast

_REDACTED = "[REDACTED]"
_MAX_SECRET_TEXT_CHARS = 4_096
_MAX_SECRET_TEXT_UTF8_BYTES = 4_096
_MAX_URL_DECODE_ROUNDS = 3
_MALFORMED_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_SECRET_TEXT = re.compile(
    r"(?is)(?:\b(?:basic|bearer)\s+\S+|\bsk-[a-z0-9_-]{8,}|"
    r"\bgh[pousr]_[a-z0-9]{20,}|"
    r"\b(?:authorization|proxy[\s._/-]*authorization)\s+(?:is|was)\s+"
    r"(?:basic|bearer)\s+\S+|"
    r"\bAKIA[A-Z0-9]{16}\b|-----BEGIN[^-]*(?:PRIVATE|SECRET)[^-]*-----|"
    r"[a-z0-9_-]{20,}\.[a-z0-9_-]{20,}\.[a-z0-9_-]{20,}|"
    r"\b(?:api[\s._/-]*key|access[\s._/-]*(?:key|token)|"
    r"private[\s._/-]*key|client[\s._/-]*secret|refresh[\s._/-]*token|"
    r"id[\s._/-]*token|secret|token|password|credential|authorization|cookie|"
    r"set[\s._/-]*cookie|session(?:[\s._/-]*id)?)"
    r"\b[\"']?\s*[:=]\s*[\"']?[^\s&;,\"'}]+|"
    r"\b(?:api[\s._/-]*key|access[\s._/-]*(?:key|token)|"
    r"private[\s._/-]*key|client[\s._/-]*secret|refresh[\s._/-]*token|"
    r"id[\s._/-]*token|secret|token|password|credential|authorization|cookie|"
    r"set[\s._/-]*cookie|session(?:[\s._/-]*id)?)"
    r"\s+(?:is|was)\s+(?:basic\s+|bearer\s+)?[^\s&;,\"'}]+|"
    r"\b(?:[a-z0-9]+[-_])?(?:real[-_])?(?:secret|token|password|credential|"
    r"private[\s._/-]*key|access[\s._/-]*key)(?:[-_](?:value|key|token))?[-_:]"
    r"[a-z0-9][a-z0-9_-]{7,}\b)"
)
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


def _make_secret_text_detector() -> Callable[[object], bool]:
    secret_pattern = _SECRET_TEXT
    malformed_pattern = _MALFORMED_PERCENT_ESCAPE
    max_chars = _MAX_SECRET_TEXT_CHARS
    max_bytes = _MAX_SECRET_TEXT_UTF8_BYTES
    decode_rounds = _MAX_URL_DECODE_ROUNDS
    buffer_factory = bytearray
    hex_values = {
        code: value
        for value, code in enumerate(b"0123456789")
    } | {
        code: value + 10
        for value, code in enumerate(b"abcdef")
    } | {
        code: value + 10
        for value, code in enumerate(b"ABCDEF")
    }

    def bounded_utf8(value: str) -> bool:
        if len(value) > max_chars:
            return False
        try:
            return len(value.encode("utf-8")) <= max_bytes
        except UnicodeEncodeError:
            return False

    def strict_url_decode(value: str) -> str:
        raw = value.replace("+", " ").encode("utf-8")
        decoded = buffer_factory()
        offset = 0
        while offset < len(raw):
            current = raw[offset]
            if current != 0x25:
                decoded.append(current)
                offset += 1
                continue
            if offset + 2 >= len(raw):
                raise ValueError("malformed percent escape")
            high = hex_values.get(raw[offset + 1])
            low = hex_values.get(raw[offset + 2])
            if high is None or low is None:
                raise ValueError("malformed percent escape")
            decoded.append((high << 4) | low)
            offset += 3
        return decoded.decode("utf-8", errors="strict")

    def detect(value: object) -> bool:
        """Fail closed on bounded raw or repeatedly URL-decoded credential text."""
        if type(value) is not str or not bounded_utf8(value):
            return True
        current = value
        for _round in range(decode_rounds):
            if secret_pattern.search(current) is not None:
                return True
            if malformed_pattern.search(current) is not None:
                return True
            try:
                decoded = strict_url_decode(current)
            except (UnicodeDecodeError, UnicodeEncodeError, ValueError):
                return True
            if not bounded_utf8(decoded):
                return True
            if decoded == current:
                return False
            current = decoded
        if secret_pattern.search(current) is not None:
            return True
        if malformed_pattern.search(current) is not None:
            return True
        try:
            decoded = strict_url_decode(current)
        except (UnicodeDecodeError, UnicodeEncodeError, ValueError):
            return True
        return not bounded_utf8(decoded) or decoded != current

    return detect


def _make_secret_text_redactor(
    detector: Callable[[object], bool],
) -> Callable[[str], str]:
    marker = _REDACTED

    def redact(value: str) -> str:
        """Return a fixed audit-safe surrogate for possible credential text."""
        return marker if detector(value) else value

    return redact


secret_text_present = _make_secret_text_detector()
redact_secret_text = _make_secret_text_redactor(secret_text_present)


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
