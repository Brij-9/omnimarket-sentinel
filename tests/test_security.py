"""Tests for structured secret redaction."""

from market_sentinel.security import redact_mapping


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
