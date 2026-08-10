"""Tests for structured secret redaction."""

from time import monotonic
from types import MappingProxyType

import pytest

from market_sentinel import security as security_module
from market_sentinel.security import redact_mapping

_CANONICAL_LABELS = (
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
)
_LABEL_SEPARATORS = ("", " ", "  ", "\t", "\u00a0", "_", "-", ".", "/", ":")
_BENIGN_NEAR_MISSES = ("monkey", "rapid", "author", "clientele", "keynote")
_SECURITY_GLOBAL_REBINDINGS = (
    ("len", lambda value: 0),
    ("type", lambda value: object),
    ("str", bytes),
    ("range", lambda *args: ()),
    ("tuple", lambda value=(): ()),
    ("list", str),
    ("dict", str),
    ("enumerate", lambda *args: ()),
    ("isinstance", lambda *args: False),
    ("UnicodeDecodeError", RuntimeError),
    ("UnicodeEncodeError", RuntimeError),
    ("ValueError", RuntimeError),
)


def _percent_plus_encode(value: str) -> str:
    encoded: list[str] = []
    for octet in value.encode("utf-8"):
        if 0x30 <= octet <= 0x39 or 0x41 <= octet <= 0x5A or 0x61 <= octet <= 0x7A:
            encoded.append(chr(octet))
        elif octet == 0x20:
            encoded.append("+")
        else:
            encoded.append(f"%{octet:02X}")
    return "".join(encoded)


def _encoded_rounds(value: str, depth: int) -> str:
    for _round in range(depth):
        value = _percent_plus_encode(value)
    return value


def test_nested_secrets_are_redacted() -> None:
    """Known secret fields are hidden at every dictionary depth."""
    value = {
        "Authorization": "Bearer abc",
        "broker": {"api_key": "secret", "account_id": "A1"},
    }

    assert redact_mapping(value) == {
        "Authorization": "[REDACTED]",
        "broker": {"api_key": "[REDACTED]", "account_id": "A1"},
    }


def test_suffix_secrets_in_lists_and_tuples_are_redacted_without_mutation() -> None:
    """Nested containers are copied while token and secret suffixes remain hidden."""
    value: dict[str, object] = {
        "events": [
            {"refresh_token": "list-token", "name": "safe"},
            ("unchanged", {"client_secret": "tuple-secret"}),
        ],
    }

    result = redact_mapping(value)

    assert result == {
        "events": [
            {"refresh_token": "[REDACTED]", "name": "safe"},
            ("unchanged", {"client_secret": "[REDACTED]"}),
        ],
    }
    assert value == {
        "events": [
            {"refresh_token": "list-token", "name": "safe"},
            ("unchanged", {"client_secret": "tuple-secret"}),
        ],
    }


def test_redaction_key_matching_is_case_insensitive() -> None:
    """Credentials remain hidden regardless of header or field capitalization."""
    assert redact_mapping({"PASSWORD": "p", "ToTp": "123456", "public": "ok"}) == {
        "PASSWORD": "[REDACTED]",
        "ToTp": "[REDACTED]",
        "public": "ok",
    }


@pytest.mark.parametrize(
    "label",
    (
        "api-key",
        "api.key",
        "api/key",
        "api key",
        "apikey",
        "session-token",
        "sessiontoken",
        "secret_key",
        "private_token",
        "totp",
        "key",
    ),
)
def test_alternate_structured_secret_labels_use_the_assignment_grammar(label: str) -> None:
    """A mapping key cannot bypass a spelling that free-text assignments reject."""
    assert security_module.secret_text_present(f"{label}=OpaqueValue123456")
    assert redact_mapping({label: "OpaqueValue123456"}) == {label: "[REDACTED]"}


def test_every_canonical_label_form_is_shared_by_mapping_and_text_detection() -> None:
    """One vocabulary and classifier cover compounds, separators, and decoding depths."""
    assert frozenset(_CANONICAL_LABELS) == security_module._CANONICAL_SENSITIVE_LABELS
    for canonical_label in _CANONICAL_LABELS:
        tokens = canonical_label.split("_")
        for separator in _LABEL_SEPARATORS:
            raw_label = separator.join(tokens)
            for encoding_depth in range(4):
                label = _encoded_rounds(raw_label, encoding_depth)
                assert security_module.secret_text_present(
                    f"{label}=OpaqueValue123456"
                ), (canonical_label, separator, encoding_depth, "text")
                assert redact_mapping({label: "OpaqueValue123456"}) == {
                    label: "[REDACTED]"
                }, (canonical_label, separator, encoding_depth, "mapping")


@pytest.mark.parametrize("label", _BENIGN_NEAR_MISSES)
@pytest.mark.parametrize("encoding_depth", range(4))
def test_benign_near_miss_labels_remain_visible_in_text_and_mappings(
    label: str,
    encoding_depth: int,
) -> None:
    """Sensitive substrings inside unrelated whole tokens are not credentials."""
    encoded = _encoded_rounds(label, encoding_depth)
    assert not security_module.secret_text_present(f"{encoded}=ordinary")
    assert redact_mapping({encoded: "ordinary"}) == {encoded: "ordinary"}


@pytest.mark.parametrize(
    "hostile_label",
    (
        "x" * 4_097,
        "\u0800" * 1_366,
        "api%25252525key",
        "%",
        "\ud800",
    ),
)
def test_structured_label_bounds_and_ambiguous_decoding_fail_closed(
    hostile_label: str,
) -> None:
    """Unbounded, malformed, or over-encoded keys never reach tokenization or audit."""
    assert redact_mapping({hostile_label: "OpaqueValue123456"}) == {
        hostile_label: "[REDACTED]"
    }


@pytest.mark.parametrize("size", (512, 1_024, 2_048, 4_095))
@pytest.mark.parametrize("kind", ("benign", "jwt_like"))
def test_secret_scanner_reports_deterministic_linear_work(size: int, kind: str) -> None:
    """Manual JWT scanning has a deterministic work budget proportional to input size."""
    if kind == "benign":
        value = ("broker-" + "a-" * size)[:size]
        expected = False
    else:
        segment = max(20, (size - 2) // 3)
        value = ("a" * segment + "." + "b" * segment + "." + "c" * segment)[:size]
        expected = True

    first = security_module.scan_secret_text(value)
    second = security_module.scan_secret_text(value)

    assert first == second
    assert first.detected is expected
    assert first.steps <= 20 * len(value) + 128


def test_secret_scanner_linear_work_has_a_secondary_wall_clock_smoke_bound() -> None:
    """Maximum benign and JWT-like inputs complete within a modest secondary budget."""
    values = (
        ("broker-" + "a-" * 4_095)[:4_095],
        "a" * 1_364 + "." + "b" * 1_364 + "." + "c" * 1_365,
    )
    started = monotonic()

    for value in values:
        security_module.scan_secret_text(value)

    assert monotonic() - started < 1.0


def test_secret_scanner_detects_three_qualifying_segments_before_a_fourth() -> None:
    """A fourth segment cannot erase the already-complete three-segment JWT prefix."""
    jwt_prefix = "a" * 20 + "." + "b" * 20 + "." + "c" * 20

    assert security_module.scan_secret_text(jwt_prefix + ".short").detected


def test_security_truth_is_unchanged_by_public_scan_result_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider cannot replace a diagnostic result class to disable redaction."""
    unsafe = "api_key=OpaqueValue123456"

    class ForgedSafeResult:
        def __init__(self, detected: bool, steps: int) -> None:
            del detected
            self.detected = False
            self.steps = steps

    monkeypatch.setattr(security_module, "SecretTextScan", ForgedSafeResult)

    assert security_module.secret_text_present(unsafe)
    assert security_module.redact_secret_text(unsafe) == "[REDACTED]"
    assert security_module.scan_secret_text(unsafe).detected


def test_nested_mapping_redaction_is_unchanged_by_mapping_abc_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider cannot replace the module ABC to bypass nested audit redaction."""
    payload = {"outer": {"api_key": "OpaqueValue123456"}}
    monkeypatch.setattr(security_module, "Mapping", str)
    monkeypatch.setattr(
        security_module,
        "cast",
        lambda annotation, value: value,
        raising=False,
    )

    assert redact_mapping(payload) == {
        "outer": {"api_key": "[REDACTED]"},
    }


def test_prefixed_canonical_secret_suffixes_share_mapping_and_text_classification() -> None:
    """Benign prefixes cannot hide a complete canonical label in text or mappings."""
    for prefix in ("broker", "github", "x", "prefix"):
        for canonical_label in _CANONICAL_LABELS:
            tokens = canonical_label.split("_")
            for separator in (" ", "_", "-", ".", "/"):
                raw_label = separator.join((prefix, *tokens))
                for encoding_depth in range(4):
                    label = _encoded_rounds(raw_label, encoding_depth)
                    assert security_module.secret_text_present(
                        f"{label}=OpaqueValue123456"
                    ), (prefix, canonical_label, separator, encoding_depth, "text")
                    assert redact_mapping({label: "OpaqueValue123456"}) == {
                        label: "[REDACTED]"
                    }, (prefix, canonical_label, separator, encoding_depth, "mapping")


def test_public_identifier_compounds_are_not_structured_credentials() -> None:
    """Weak label constituents cannot corrupt durable public identity fields."""
    identifiers = {
        "session_id": "session-1",
        "client_intent_id": "intent-1",
        "account_id": "account-1",
        "instrument_id": "AAPL@alpaca",
        "order_id": "order-1",
        "event_id": "event-1",
    }

    assert redact_mapping(identifiers) == identifiers
    assert security_module.secret_text_present("session_id=OpaqueValue123456")


@pytest.mark.parametrize(("name", "replacement"), _SECURITY_GLOBAL_REBINDINGS)
def test_captured_security_boundaries_ignore_module_builtin_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    replacement: object,
) -> None:
    """Every runtime builtin used by captured security closures is immutable locally."""
    values = (
        "api_key=OpaqueValue123456",
        "session_id=OpaqueValue123456",
        "broker-order-1",
        "%252561pi%25255Fkey%25253DOpaqueValue123456",
        "%",
        "%FF",
        "\ud800",
        "a" * 20 + "." + "b" * 20 + "." + "c" * 20,
    )
    mapping = {
        "outer": [
            {"api_key": "OpaqueValue123456"},
            ({"session-token": "OpaqueValue123456"},),
            MappingProxyType({"private.token": "OpaqueValue123456"}),
        ]
    }
    expected_scans = tuple(security_module.scan_secret_text(value) for value in values)
    expected_redaction = redact_mapping(mapping)
    monkeypatch.setattr(security_module, name, replacement, raising=False)

    assert tuple(security_module.scan_secret_text(value) for value in values) == expected_scans
    assert redact_mapping(mapping) == expected_redaction
    assert security_module.secret_text_present(values[0])
    assert security_module.redact_secret_text(values[0]) == "[REDACTED]"


@pytest.mark.parametrize(
    "hostile",
    (
        type("HostileList", (list,), {})(({"api_key": "OpaqueValue123456"},)),
        type("HostileDict", (dict,), {})({"api_key": "OpaqueValue123456"}),
        type("HostileTuple", (tuple,), {})(({"api_key": "OpaqueValue123456"},)),
    ),
)
def test_unsupported_nested_container_subclasses_fail_closed(hostile: object) -> None:
    """A serializable container subclass is rejected instead of treated as a scalar."""
    with pytest.raises(ValueError, match="mapping type|container type"):
        redact_mapping({"outer": hostile})
