"""Regression tests for non-bypassable risk-engine safety boundaries."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from market_sentinel.domain.models import Position
from market_sentinel.risk.engine import PositionSizer, RiskEngine, portfolio_hash
from market_sentinel.risk.policy import RiskPolicy
from tests.factories import instrument, intent, portfolio, signal, snapshot

NOW = datetime(2026, 8, 9, 10, tzinfo=UTC)


def _valid_portfolio() -> object:
    return portfolio(cash="100", equity="100", peak_equity="100", gross_exposure="0")


def _valid_intent() -> object:
    account = _valid_portfolio()
    return intent(
        notional="1",
        limit_price="1",
        stop_loss="0.5",
        take_profit="2",
        snapshot_hash=portfolio_hash(account),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_trade_risk_fraction", Decimal("0.006")),
        ("max_position_fraction", Decimal("0.11")),
        ("max_gross_exposure_fraction", Decimal("0.51")),
        ("max_daily_loss_fraction", Decimal("0.021")),
        ("max_drawdown_fraction", Decimal("0.101")),
    ],
)
def test_direct_policy_constructor_cannot_relax_safe_caps(field: str, value: Decimal) -> None:
    """A caller cannot bypass fixed safe limits by constructing a policy directly."""
    values = {
        "max_trade_risk_fraction": Decimal("0.005"),
        "max_position_fraction": Decimal("0.10"),
        "max_gross_exposure_fraction": Decimal("0.50"),
        "max_daily_loss_fraction": Decimal("0.02"),
        "max_drawdown_fraction": Decimal("0.10"),
    }
    values[field] = value

    with pytest.raises(ValueError, match="safe"):
        RiskPolicy(**values)


def test_direct_policy_constructor_cannot_extend_decision_lifetime() -> None:
    with pytest.raises(ValueError, match="60"):
        RiskPolicy(
            max_trade_risk_fraction=Decimal("0.005"),
            max_position_fraction=Decimal("0.10"),
            max_gross_exposure_fraction=Decimal("0.50"),
            max_daily_loss_fraction=Decimal("0.02"),
            max_drawdown_fraction=Decimal("0.10"),
            decision_ttl=timedelta(seconds=61),
        )


def test_direct_intent_over_stop_risk_is_rejected() -> None:
    account = _valid_portfolio()
    decision = RiskEngine.safe_defaults().assess(
        intent=intent(
            quantity="2",
            notional=None,
            limit_price="1",
            stop_loss="0.5",
            take_profit="2",
            snapshot_hash=portfolio_hash(account),
        ),
        instrument=instrument(quantity_step="1"),
        market=snapshot(),
        portfolio=account,
        now=NOW,
    )

    assert decision.approved is False
    assert "POSITION_LIMIT" in decision.reason_codes


def test_hash_is_required_and_is_derived_from_the_actual_portfolio() -> None:
    account = _valid_portfolio()
    decision = RiskEngine.safe_defaults().assess(
        intent=intent(notional="1", limit_price="1", stop_loss="0.5", take_profit="2"),
        instrument=instrument(),
        market=snapshot(),
        portfolio=account,
        now=NOW,
    )

    assert decision.approved is False
    assert "PORTFOLIO_HASH_MISMATCH" in decision.reason_codes


def test_hash_changes_when_each_risk_relevant_snapshot_field_changes() -> None:
    base = _valid_portfolio()
    changed = portfolio(cash="99", equity="99", peak_equity="100", daily_pnl="-1")

    assert portfolio_hash(base) != portfolio_hash(changed)


@pytest.mark.parametrize(
    "mutate,reason",
    [
        ("bad_intent_id", "INVALID_PRECISION"),
        ("bad_market_id", "INVALID_PRECISION"),
        ("created_after_now", "EXPIRED_INTENT"),
        ("duration_too_long", "EXPIRED_INTENT"),
        ("future_source", "STALE_DATA"),
        ("future_portfolio", "STALE_DATA"),
        ("bad_exit_shape", "MISSING_PROTECTIVE_EXIT"),
        ("bad_exit_tick", "INVALID_PRECISION"),
    ],
)
def test_identity_temporal_and_exit_validation_fail_closed(mutate: str, reason: str) -> None:
    account = _valid_portfolio()
    order = _valid_intent()
    market = snapshot()
    venue = instrument(price_tick="0.1")
    if mutate == "bad_intent_id":
        order = intent(
            instrument_id="other@venue",
            notional="1",
            limit_price="1",
            stop_loss="0.5",
            take_profit="2",
            snapshot_hash=portfolio_hash(account),
        )
    elif mutate == "bad_market_id":
        market = snapshot(instrument_id="other@venue")
    elif mutate == "created_after_now":
        order = intent(
            notional="1",
            limit_price="1",
            stop_loss="0.5",
            take_profit="2",
            created_at=NOW + timedelta(seconds=1),
            snapshot_hash=portfolio_hash(account),
        )
    elif mutate == "duration_too_long":
        order = intent(
            notional="1",
            limit_price="1",
            stop_loss="0.5",
            take_profit="2",
            expires_at=NOW + timedelta(seconds=61),
            snapshot_hash=portfolio_hash(account),
        )
    elif mutate == "future_source":
        market = snapshot(source_at=NOW + timedelta(seconds=1))
    elif mutate == "future_portfolio":
        account = portfolio(
            cash="100", equity="100", peak_equity="100", observed_at=NOW + timedelta(seconds=1)
        )
        order = intent(
            notional="1",
            limit_price="1",
            stop_loss="0.5",
            take_profit="2",
            snapshot_hash=portfolio_hash(account),
        )
    elif mutate == "bad_exit_shape":
        order = intent(
            notional="1",
            limit_price="1",
            stop_loss="1.5",
            take_profit="2",
            snapshot_hash=portfolio_hash(account),
        )
    else:
        order = intent(
            notional="1",
            limit_price="1",
            stop_loss="0.55",
            take_profit="2",
            snapshot_hash=portfolio_hash(account),
        )

    decision = RiskEngine.safe_defaults().assess(
        intent=order, instrument=venue, market=market, portfolio=account, now=NOW
    )

    assert decision.approved is False
    assert reason in decision.reason_codes


def test_malformed_portfolio_arithmetic_is_rejected_by_sizer_and_engine() -> None:
    malformed = portfolio(
        cash="100",
        equity="100",
        peak_equity="100",
        gross_exposure="0",
        positions=(
            Position(
                instrument_id="AAPL@alpaca",
                quantity=Decimal("1"),
                average_price=Decimal("10"),
                market_price=Decimal("10"),
                unrealized_pnl=Decimal("0"),
            ),
        ),
    )
    order = intent(
        notional="1",
        limit_price="1",
        stop_loss="0.5",
        take_profit="2",
        snapshot_hash=portfolio_hash(malformed),
    )
    engine_decision = RiskEngine.safe_defaults().assess(
        intent=order, instrument=instrument(), market=snapshot(), portfolio=malformed, now=NOW
    )
    sizer_result = PositionSizer(policy=RiskPolicy.safe_defaults()).create_intent(
        signal=signal(),
        instrument=instrument(),
        portfolio=malformed,
        snapshot_hash=portfolio_hash(malformed),
        now=NOW,
    )

    assert "INVALID_PRECISION" in engine_decision.reason_codes
    assert getattr(sizer_result, "reason_codes", ()) == ("INVALID_PRECISION",)


def test_cash_buy_without_cash_is_rejected_as_leverage() -> None:
    account = portfolio(
        cash="0",
        equity="100",
        peak_equity="100",
        gross_exposure="100",
        positions=(
            Position(
                instrument_id="held@alpaca",
                quantity=Decimal("100"),
                average_price=Decimal("1"),
                market_price=Decimal("1"),
                unrealized_pnl=Decimal("0"),
            ),
        ),
    )
    decision = RiskEngine.safe_defaults().assess(
        intent=intent(
            notional="1",
            limit_price="1",
            stop_loss="0.5",
            take_profit="2",
            snapshot_hash=portfolio_hash(account),
        ),
        instrument=instrument(),
        market=snapshot(),
        portfolio=account,
        now=NOW,
    )

    assert "LEVERAGE_FORBIDDEN" in decision.reason_codes
