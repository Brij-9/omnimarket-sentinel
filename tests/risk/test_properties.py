"""Properties that keep risk limits unbypassable."""

from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from market_sentinel.risk.engine import PositionSizer, RiskEngine
from market_sentinel.risk.policy import RiskPolicy
from tests.factories import instrument, intent, portfolio, signal, snapshot

NOW = datetime(2026, 8, 9, 10, tzinfo=UTC)


@given(
    equity=st.decimals(min_value="100", max_value="100000", places=2),
    entry=st.decimals(min_value="10", max_value="1000", places=2),
    stop_distance=st.decimals(min_value="0.01", max_value="9", places=2),
    step=st.sampled_from([Decimal("0.0001"), Decimal("0.001"), Decimal("0.01")]),
)
def test_approved_size_never_exceeds_position_or_stop_risk_limits(
    equity: Decimal, entry: Decimal, stop_distance: Decimal, step: Decimal
) -> None:
    sized = PositionSizer(policy=RiskPolicy.safe_defaults()).create_intent(
        signal=signal(
            entry_price=entry,
            invalidation_price=entry - stop_distance,
            take_profit=entry + stop_distance,
        ),
        instrument=instrument(quantity_step=step, minimum_notional="0.01"),
        portfolio=portfolio(equity=equity, peak_equity=equity),
        snapshot_hash="hash",
        now=NOW,
    )
    if hasattr(sized, "approved"):
        return

    quantity = sized.quantity
    assert quantity is not None
    approved_notional = quantity * entry
    assert approved_notional <= equity * Decimal("0.10")
    assert quantity * stop_distance <= equity * Decimal("0.005")


@given(
    notional=st.decimals(min_value="1", max_value="10", places=2),
    extra_loss=st.decimals(min_value="2", max_value="20", places=2),
)
def test_adding_a_rejection_condition_never_turns_rejection_into_approval(
    notional: Decimal, extra_loss: Decimal
) -> None:
    baseline = RiskEngine.safe_defaults().assess(
        intent=intent(notional=notional, limit_price="1", stop_loss="0.5", take_profit="2"),
        instrument=instrument(),
        market=snapshot(),
        portfolio=portfolio(equity="100", peak_equity="100", gross_exposure="60"),
        now=NOW,
    )
    with_extra_rejection = RiskEngine.safe_defaults().assess(
        intent=intent(
            notional=notional, limit_price="1", stop_loss="0.5", take_profit="2", product="margin"
        ),
        instrument=instrument(),
        market=snapshot(),
        portfolio=portfolio(
            equity="100", peak_equity="100", gross_exposure="60", daily_pnl=-extra_loss
        ),
        now=NOW,
    )

    assert baseline.approved is False
    assert with_extra_rejection.approved is False
