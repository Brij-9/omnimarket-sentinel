"""Bounded helpers for detecting and removing credentials from output."""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeGuard

_REDACTED = "[REDACTED]"
_MAX_SECRET_TEXT_CHARS = 4_096
_MAX_SECRET_TEXT_UTF8_BYTES = 4_096
_MAX_URL_DECODE_ROUNDS = 3
_MALFORMED_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_SECRET_TEXT = re.compile(
    r"(?is)(?:\b(?:basic|bearer)\s+\S+|\bsk-[a-z0-9_-]{8,}|"
    r"\bgh[pousr]_[a-z0-9]{20,}|"
    r"\bAKIA[A-Z0-9]{16}\b|-----BEGIN[^-]*(?:PRIVATE|SECRET)[^-]*-----|"
    r"\b(?:[a-z0-9]+[-_])?(?:real[-_])?(?:secret|token|password|credential|"
    r"private[\s._/-]*key|access[\s._/-]*key)(?:[-_](?:value|key|token))?[-_:]"
    r"[a-z0-9][a-z0-9_-]{7,}\b)"
)
_CANONICAL_SENSITIVE_LABELS = frozenset(
    {
        "authorization",
        "auth",
        "basic",
        "bearer",
        "api_key",
        "access_key",
        "secret_key",
        "access_token",
        "private_key",
        "private_token",
        "session_key",
        "session_token",
        "client_secret",
        "refresh_token",
        "id_token",
        "password",
        "credential",
        "cookie",
        "secret",
        "token",
        "jwt",
        "totp",
        "otp",
        "passphrase",
        "passwd",
        "pwd",
    }
)
_SENSITIVE_LABEL_TOKENS = frozenset(
    token for label in _CANONICAL_SENSITIVE_LABELS for token in label.split("_")
)


@dataclass(frozen=True, slots=True)
class SecretTextScan:
    """One bounded credential scan result with deterministic work accounting."""

    detected: bool
    steps: int


def _make_security_scanners() -> tuple[
    Callable[[object], bool], Callable[[object], tuple[bool, int]]
]:
    secret_pattern = _SECRET_TEXT
    malformed_pattern = _MALFORMED_PERCENT_ESCAPE
    sensitive_label_tokens = tuple(sorted(_SENSITIVE_LABEL_TOKENS))
    canonical_sequences = tuple(
        sorted(tuple(label.split("_")) for label in _CANONICAL_SENSITIVE_LABELS)
    )
    strong_single_tokens = frozenset(
        sequence[0] for sequence in canonical_sequences if len(sequence) == 1
    )
    compound_pairs = frozenset(
        sequence for sequence in canonical_sequences if len(sequence) == 2
    )
    flattened_compounds = frozenset(
        "".join(sequence) for sequence in compound_pairs
    )
    assignment_boundaries = frozenset("&;,\n\r?")
    max_chars = _MAX_SECRET_TEXT_CHARS
    max_bytes = _MAX_SECRET_TEXT_UTF8_BYTES
    decode_rounds = _MAX_URL_DECODE_ROUNDS
    buffer_factory = bytearray
    len_fn = len
    range_fn = range
    enumerate_fn = enumerate
    type_fn = type
    str_type = str
    tuple_type = tuple
    unicode_decode_error = UnicodeDecodeError
    unicode_encode_error = UnicodeEncodeError
    value_error_type = ValueError
    decode_errors = (unicode_decode_error, unicode_encode_error, value_error_type)
    hex_values = (
        {code: value for value, code in enumerate(b"0123456789")}
        | {code: value + 10 for value, code in enumerate(b"abcdef")}
        | {code: value + 10 for value, code in enumerate(b"ABCDEF")}
    )

    def is_exact_str(value: object) -> TypeGuard[str]:
        return type_fn(value) is str_type

    def bounded_utf8(value: str) -> tuple[bool, int]:
        steps = len_fn(value) + 1
        if len_fn(value) > max_chars:
            return False, steps
        try:
            return len_fn(value.encode("utf-8")) <= max_bytes, steps
        except unicode_encode_error:
            return False, steps

    def strict_url_decode(value: str) -> tuple[str, int]:
        raw = value.replace("+", " ").encode("utf-8")
        decoded = buffer_factory()
        offset = 0
        steps = 0
        while offset < len_fn(raw):
            steps += 1
            current = raw[offset]
            if current != 0x25:
                decoded.append(current)
                offset += 1
                continue
            if offset + 2 >= len_fn(raw):
                raise value_error_type("malformed percent escape")
            high = hex_values.get(raw[offset + 1])
            low = hex_values.get(raw[offset + 2])
            if high is None or low is None:
                raise value_error_type("malformed percent escape")
            decoded.append((high << 4) | low)
            offset += 3
        return decoded.decode("utf-8", errors="strict"), steps

    def segmentable_sensitive_token(token: str) -> tuple[bool, int]:
        reachable_offsets = {0}
        steps = 0
        for offset in range_fn(len_fn(token)):
            if offset not in reachable_offsets:
                steps += 1
                continue
            for sensitive in sensitive_label_tokens:
                steps += 1
                if token.startswith(sensitive, offset):
                    reachable_offsets.add(offset + len_fn(sensitive))
        return len_fn(token) in reachable_offsets, steps

    def tokenize_decoded_label(value: str) -> tuple[tuple[str, ...], int]:
        """Tokenize one already-bounded label in one forward pass."""
        offset = 0
        steps = 0
        tokens: list[str] = []
        while offset < len_fn(value):
            steps += 1
            if not value[offset].isalnum():
                offset += 1
                continue
            token_start = offset
            offset += 1
            while offset < len_fn(value) and value[offset].isalnum():
                steps += 1
                offset += 1
            tokens.append(value[token_start:offset].casefold())
        return tuple_type(tokens), steps

    def structured_label_tokens_are_sensitive(
        tokens: tuple[str, ...],
    ) -> tuple[bool, int]:
        """Match complete canonical labels anywhere without promoting weak parts."""
        steps = 1
        if tokens == ("key",):
            return True, steps
        for offset, token in enumerate_fn(tokens):
            steps += 1
            if token in strong_single_tokens or token in flattened_compounds:
                return True, steps
            if (
                offset + 1 < len_fn(tokens)
                and (token, tokens[offset + 1]) in compound_pairs
            ):
                return True, steps
        return False, steps

    def classify_structured_decoded_label(value: str) -> tuple[bool, int]:
        tokens, steps = tokenize_decoded_label(value)
        detected, policy_steps = structured_label_tokens_are_sensitive(tokens)
        return detected, steps + policy_steps

    def classify_assignment_decoded_label(value: str) -> tuple[bool, int]:
        tokens, steps = tokenize_decoded_label(value)
        detected, policy_steps = structured_label_tokens_are_sensitive(tokens)
        steps += policy_steps
        if detected or not tokens:
            return detected, steps
        for token in tokens:
            token_detected, token_steps = segmentable_sensitive_token(token)
            steps += token_steps
            if not token_detected:
                return False, steps
        return True, steps

    def scan_sensitive_label(value: object) -> bool:
        """Apply the bounded decoded label grammar used by text and mappings."""
        if not is_exact_str(value):
            return True
        bounded, _steps = bounded_utf8(value)
        if not bounded:
            return True
        current = value
        for _round in range_fn(decode_rounds):
            detected, _label_steps = classify_structured_decoded_label(current)
            if detected:
                return True
            if malformed_pattern.search(current) is not None:
                return True
            try:
                decoded, _decode_steps = strict_url_decode(current)
            except decode_errors:
                return True
            bounded, _bound_steps = bounded_utf8(decoded)
            if not bounded:
                return True
            if decoded == current:
                return False
            current = decoded
        detected, _label_steps = classify_structured_decoded_label(current)
        if detected or malformed_pattern.search(current) is not None:
            return True
        try:
            decoded, _decode_steps = strict_url_decode(current)
        except decode_errors:
            return True
        bounded, _bound_steps = bounded_utf8(decoded)
        return not bounded or decoded != current

    def sensitive_assignment(value: str) -> tuple[bool, int]:
        candidate_start = 0
        offset = 0
        steps = 0
        while offset < len_fn(value):
            steps += 1
            character = value[offset]
            if character.isalnum():
                token_start = offset
                offset += 1
                while offset < len_fn(value) and value[offset].isalnum():
                    steps += 1
                    offset += 1
                token = value[token_start:offset].casefold()
                natural_operator = (
                    token in {"is", "was"}
                    and token_start > candidate_start
                    and value[token_start - 1].isspace()
                    and offset < len_fn(value)
                    and value[offset].isspace()
                )
                if natural_operator:
                    detected, label_steps = classify_assignment_decoded_label(
                        value[candidate_start:token_start]
                    )
                    steps += label_steps
                    if not detected:
                        label_end = token_start
                        while label_end > candidate_start and not value[
                            label_end - 1
                        ].isalnum():
                            steps += 1
                            label_end -= 1
                        label_start = label_end
                        while label_start > candidate_start and value[
                            label_start - 1
                        ].isalnum():
                            steps += 1
                            label_start -= 1
                        detected, label_steps = classify_assignment_decoded_label(
                            value[label_start:label_end]
                        )
                        steps += label_steps
                    if detected and offset + 1 < len_fn(value):
                        return True, steps
                    candidate_start = offset
                continue
            if character in "=:":
                detected, label_steps = classify_assignment_decoded_label(
                    value[candidate_start:offset]
                )
                steps += label_steps
                if detected and offset + 1 < len_fn(value):
                    return True, steps
                candidate_start = offset + 1
            elif character in assignment_boundaries:
                candidate_start = offset + 1
            offset += 1
        return False, steps

    def jwt_like_value(value: str) -> tuple[bool, int]:
        """Recognize three long base64url segments in one deterministic pass."""
        qualified_segments = 0
        segment_length = 0
        steps = 0
        for character in value:
            steps += 1
            is_base64url = (
                "a" <= character <= "z"
                or "A" <= character <= "Z"
                or "0" <= character <= "9"
                or character in "_-"
            )
            if is_base64url:
                segment_length += 1
                continue
            if character == ".":
                if segment_length >= 20:
                    if qualified_segments >= 2:
                        return True, steps
                    qualified_segments += 1
                else:
                    qualified_segments = 0
                segment_length = 0
                continue
            if qualified_segments >= 2 and segment_length >= 20:
                return True, steps
            qualified_segments = 0
            segment_length = 0
        return qualified_segments >= 2 and segment_length >= 20, steps

    def scan_decoded_text(value: str) -> tuple[bool, int]:
        steps = len_fn(value) + 1
        if secret_pattern.search(value) is not None:
            return True, steps
        assignment_detected, assignment_steps = sensitive_assignment(value)
        steps += assignment_steps
        if assignment_detected:
            return True, steps
        jwt_detected, jwt_steps = jwt_like_value(value)
        steps += jwt_steps
        return jwt_detected, steps

    def scan_text(value: object) -> tuple[bool, int]:
        """Fail closed on bounded raw or repeatedly decoded credential text."""
        if not is_exact_str(value):
            return True, 1
        bounded, steps = bounded_utf8(value)
        if not bounded:
            return True, steps
        current = value
        for _round in range_fn(decode_rounds):
            detected, scan_steps = scan_decoded_text(current)
            steps += scan_steps
            if detected or malformed_pattern.search(current) is not None:
                return True, steps
            try:
                decoded, decode_steps = strict_url_decode(current)
            except decode_errors:
                return True, steps + len_fn(current) + 1
            steps += decode_steps
            bounded, bound_steps = bounded_utf8(decoded)
            steps += bound_steps
            if not bounded:
                return True, steps
            if decoded == current:
                return False, steps
            current = decoded
        detected, scan_steps = scan_decoded_text(current)
        steps += scan_steps
        if detected or malformed_pattern.search(current) is not None:
            return True, steps
        try:
            decoded, decode_steps = strict_url_decode(current)
        except decode_errors:
            return True, steps + len_fn(current) + 1
        steps += decode_steps
        bounded, bound_steps = bounded_utf8(decoded)
        steps += bound_steps
        return not bounded or decoded != current, steps

    return scan_sensitive_label, scan_text


_SENSITIVE_LABEL_PRESENT, _SCAN_SECRET_TEXT = _make_security_scanners()


def _make_secret_text_detector(
    scanner: Callable[[object], tuple[bool, int]],
) -> Callable[[object], bool]:
    def detect(value: object) -> bool:
        detected, _steps = scanner(value)
        return detected

    return detect


def _make_public_secret_scanner(
    scanner: Callable[[object], tuple[bool, int]],
    result_type: type[SecretTextScan],
) -> Callable[[object], SecretTextScan]:
    def scan(value: object) -> SecretTextScan:
        detected, steps = scanner(value)
        return result_type(detected, steps)

    return scan


def _make_secret_text_redactor(
    detector: Callable[[object], bool],
) -> Callable[[str], str]:
    marker = _REDACTED

    def redact(value: str) -> str:
        """Return a fixed audit-safe surrogate for possible credential text."""
        return marker if detector(value) else value

    return redact


def _make_mapping_redactor(
    label_detector: Callable[[object], bool],
) -> Callable[[Mapping[str, object]], dict[str, object]]:
    marker = _REDACTED
    mapping_abc = Mapping
    type_fn = type
    isinstance_fn = isinstance
    dict_type = dict
    list_type = list
    tuple_type = tuple
    value_error_type = ValueError
    proxy_type: type[Mapping[object, object]] = type_fn(MappingProxyType({}))
    unsupported_container_families = (mapping_abc, list_type, tuple_type)

    def redact_value(value: Any) -> object:
        value_type = type_fn(value)
        if value_type is dict_type or value_type is proxy_type:
            return redact(value)
        if value_type is list_type:
            return [redact_value(item) for item in value]
        if value_type is tuple_type:
            return tuple_type(redact_value(item) for item in value)
        if isinstance_fn(value, unsupported_container_families):
            raise value_error_type("audit payload container type is unsupported")
        return value

    def redact(value: Mapping[str, object]) -> dict[str, object]:
        """Return a copy of *value* with nested secret-bearing fields redacted."""
        if type_fn(value) is not dict_type and type_fn(value) is not proxy_type:
            raise value_error_type("audit payload mapping type is unsupported")
        return dict_type(
            (
                (key, marker if label_detector(key) else redact_value(item))
                for key, item in value.items()
            )
        )

    return redact


scan_secret_text = _make_public_secret_scanner(_SCAN_SECRET_TEXT, SecretTextScan)
secret_text_present = _make_secret_text_detector(_SCAN_SECRET_TEXT)
redact_secret_text = _make_secret_text_redactor(secret_text_present)
redact_mapping = _make_mapping_redactor(_SENSITIVE_LABEL_PRESENT)
