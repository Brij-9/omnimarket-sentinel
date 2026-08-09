"""Fail-closed freshness checks for normalized market snapshots."""

from datetime import UTC, datetime

from market_sentinel.domain import GateResult, MarketSnapshot


class FreshnessGate:
    """Reports whether a snapshot remains usable at the supplied evaluation time."""

    def check(self, snapshot: MarketSnapshot, now: datetime) -> GateResult:
        """Use the provider source timestamp, never local receipt time, for freshness."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        if snapshot.is_stale(now.astimezone(UTC)):
            return GateResult(
                name="market_data_fresh",
                passed=False,
                reason_code="STALE_DATA",
            )
        return GateResult(
            name="market_data_fresh",
            passed=True,
            reason_code="FRESH_DATA",
        )
