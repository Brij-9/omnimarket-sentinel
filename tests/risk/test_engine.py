"""Executable fail-closed risk-control contracts."""

from datetime import UTC, datetime, timedelta

import pytest

from market_sentinel.domain.enums import AssetClass, Side
from market_sentinel.domain.models import RiskDecision
from market_sentinel.risk.engine import PositionSizer, RiskEngine
from market_sentinel.risk.policy import RiskPolicy
from tests.factories import instrument, intent, portfolio, signal, snapshot

NOW = datetime(2026, 8, 9, 10, tzinfo=UTC)


def test_stale_snapshot_is_rejected_even_when_order_is_small() -> None:
    decision = RiskEngine.safe_defaults().assess(
        intent=intent(notional="1", stop_loss="9"),
        instrument=instrument(minimum_notional="1"),
        market=snapshot(source_at=NOW - timedelta(minutes=5), max_age_seconds=60),
        portfolio=portfolio(equity="10"),
        now=NOW,
    )

    assert decision.approved is False
    assert "STALE_DATA" in decision.reason_codes
    assert decision.approved_quantity is None
    assert decision.approved_notional is None


def test_below_minimum_notional_is_not_scaled_up() -> None:
    decision = RiskEngine.safe_defaults().assess(
        intent=intent(notional="0.50", stop_loss="9"),
        instrument=instrument(minimum_notional="1"),
        market=snapshot(),
        portfolio=portfolio(equity="10"),
        now=NOW,
    )

    assert decision.approved is False
    assert decision.approved_notional is None
    assert "BELOW_MINIMUM_NOTIONAL" in decision.reason_codes


@pytest.mark.parametrize(
    "expected",
    [
        "KILL_SWITCH_ACTIVE",
        "EXPIRED_INTENT",
        "STALE_DATA",
        "PORTFOLIO_HASH_MISMATCH",
        "MISSING_PROTECTIVE_EXIT",
        "DRAWDOWN_LIMIT",
        "DAILY_LOSS_LIMIT",
        "LEVERAGE_FORBIDDEN",
        "SHORT_FORBIDDEN",
        "DERIVATIVE_FORBIDDEN",
        "POSITION_LIMIT",
        "GROSS_EXPOSURE_LIMIT",
        "BELOW_MINIMUM_NOTIONAL",
        "INVALID_PRECISION",
    ],
)
def test_reason_codes_are_stable_and_fail_closed(expected: str) -> None:
    """A broken gate must reject and must never carry executable values."""
    policy = RiskPolicy.safe_defaults()
    base_intent = intent(notional="1", limit_price="1", stop_loss="0.5", take_profit="2")
    base_market = snapshot()
    base_portfolio = portfolio(equity="100", peak_equity="100")
    engine = RiskEngine(policy=policy)

    if expected == "KILL_SWITCH_ACTIVE":
        engine = RiskEngine(policy=policy, kill_switch=True)
    elif expected == "EXPIRED_INTENT":
        base_intent = intent(
            notional="1", limit_price="1", stop_loss="0.5", take_profit="2", expires_at=NOW
        )
    elif expected == "STALE_DATA":
        base_market = snapshot(source_at=NOW - timedelta(seconds=61))
    elif expected == "PORTFOLIO_HASH_MISMATCH":
        engine = RiskEngine(policy=policy, expected_portfolio_hash="expected")
    elif expected == "MISSING_PROTECTIVE_EXIT":
        base_intent = intent(notional="1", limit_price="1", stop_loss=None, take_profit=None)
    elif expected == "DRAWDOWN_LIMIT":
        base_portfolio = portfolio(equity="89", peak_equity="100")
    elif expected == "DAILY_LOSS_LIMIT":
        base_portfolio = portfolio(equity="100", peak_equity="100", daily_pnl="-2")
    elif expected == "LEVERAGE_FORBIDDEN":
        base_intent = intent(
            notional="1", limit_price="1", stop_loss="0.5", take_profit="2", product="margin"
        )
    elif expected == "SHORT_FORBIDDEN":
        base_intent = intent(
            notional="1", limit_price="1", stop_loss="2", take_profit="0.5", side=Side.SELL
        )
    elif expected == "DERIVATIVE_FORBIDDEN":
        base_market = snapshot()
    elif expected == "POSITION_LIMIT":
        base_intent = intent(notional="11", limit_price="1", stop_loss="0.5", take_profit="2")
    elif expected == "GROSS_EXPOSURE_LIMIT":
        base_intent = intent(notional="41", limit_price="1", stop_loss="0.5", take_profit="2")
        base_portfolio = portfolio(equity="100", peak_equity="100", gross_exposure="10")
    elif expected == "BELOW_MINIMUM_NOTIONAL":
        base_intent = intent(notional="0.5", limit_price="1", stop_loss="0.5", take_profit="2")
    elif expected == "INVALID_PRECISION":
        base_intent = intent(
            quantity="1.005", notional=None, limit_price="1", stop_loss="0.5", take_profit="2"
        )

    tested_instrument = instrument(
        asset_class=AssetClass.FUTURE if expected == "DERIVATIVE_FORBIDDEN" else AssetClass.EQUITY,
        quantity_step="0.01",
    )
    decision = engine.assess(
        intent=base_intent,
        instrument=tested_instrument,
        market=base_market,
        portfolio=base_portfolio,
        now=NOW,
    )

    assert decision.approved is False
    assert expected in decision.reason_codes
    assert decision.approved_quantity is None
    assert decision.approved_notional is None


def test_reasons_follow_the_documented_order_when_multiple_gates_fail() -> None:
    decision = RiskEngine.safe_defaults(kill_switch=True).assess(
        intent=intent(
            notional="0.5",
            limit_price="1",
            stop_loss=None,
            take_profit=None,
            expires_at=NOW,
            product="margin",
        ),
        instrument=instrument(asset_class=AssetClass.FUTURE, quantity_step="0.01"),
        market=snapshot(source_at=NOW - timedelta(seconds=61)),
        portfolio=portfolio(equity="89", peak_equity="100", gross_exposure="50", daily_pnl="-2"),
        now=NOW,
    )

    assert decision.reason_codes == (
        "KILL_SWITCH_ACTIVE",
        "EXPIRED_INTENT",
        "STALE_DATA",
        "MISSING_PROTECTIVE_EXIT",
        "DRAWDOWN_LIMIT",
        "DAILY_LOSS_LIMIT",
        "LEVERAGE_FORBIDDEN",
        "DERIVATIVE_FORBIDDEN",
        "GROSS_EXPOSURE_LIMIT",
        "BELOW_MINIMUM_NOTIONAL",
    )


def test_position_sizer_rounds_down_and_rejects_below_venue_minimum() -> None:
    sizer = PositionSizer(policy=RiskPolicy.safe_defaults())
    sized = sizer.create_intent(
        signal=signal(entry_price="100", invalidation_price="99", take_profit="102"),
        instrument=instrument(quantity_step="0.01", minimum_notional="1.01"),
        portfolio=portfolio(equity="10"),
        snapshot_hash="portfolio-hash",
        now=NOW,
    )

    assert isinstance(sized, RiskDecision)
    assert sized.reason_codes == ("BELOW_MINIMUM_NOTIONAL",)
