"""Tests for the fail-closed market-data freshness gate."""

from datetime import UTC, datetime, timedelta

from market_sentinel.data.freshness import FreshnessGate
from tests.factories import snapshot


def test_freshness_gate_uses_provider_source_time_not_receipt_time() -> None:
    """Using observed_at would approve data that was received recently but is stale at source."""
    now = datetime(2026, 8, 9, 12, tzinfo=UTC)
    market = snapshot(
        observed_at=now,
        source_at=now - timedelta(minutes=2),
        max_age_seconds=60,
    )

    result = FreshnessGate().check(market, now)

    assert result.name == "market_data_fresh"
    assert result.passed is False
    assert result.reason_code == "STALE_DATA"


def test_freshness_gate_accepts_source_data_at_its_maximum_age() -> None:
    """Changing the stale comparison to >= would reject data at the documented age limit."""
    now = datetime(2026, 8, 9, 12, tzinfo=UTC)
    market = snapshot(source_at=now - timedelta(seconds=60), max_age_seconds=60)

    result = FreshnessGate().check(market, now)

    assert result.name == "market_data_fresh"
    assert result.passed is True
    assert result.reason_code == "FRESH_DATA"


def test_freshness_gate_never_exposes_provider_credentials_or_headers() -> None:
    """Provider diagnostics in a gate result could expose authentication material."""
    now = datetime(2026, 8, 9, 12, tzinfo=UTC)
    credential = "Bearer secret-provider-token"
    market = snapshot(
        source_at=now - timedelta(minutes=2),
        max_age_seconds=60,
        provider=credential,
    )

    result = FreshnessGate().check(market, now)

    assert credential not in str(result)
    assert "Authorization" not in str(result)
