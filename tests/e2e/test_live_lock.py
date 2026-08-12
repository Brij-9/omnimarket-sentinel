"""End-to-end proof that absent local readiness cannot reach live submit."""

import pytest

from market_sentinel.operations.fixture_pipeline import run_fixture_pipeline


@pytest.mark.e2e
def test_live_submission_is_locked_without_credentials_or_real_local_gates() -> None:
    """An incomplete preflight must reject before the injected adapter submit boundary."""
    result = run_fixture_pipeline("us", request_live=True)

    assert result.live_order is None
    assert result.live_rejection is not None
    assert result.live_rejection.reason_codes == ("PREFLIGHT_NOT_READY",)
    assert result.live_submit_calls == 0
    assert result.live_query_calls == 0
    assert result.live_flags_enabled is False
