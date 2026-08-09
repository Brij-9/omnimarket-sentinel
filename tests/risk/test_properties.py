"""Properties that keep risk limits unbypassable."""

from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from market_sentinel.domain.models import OrderIntent
from market_sentinel.risk.engine import PositionSizer, RiskEngine, portfolio_hash
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
    account = portfolio(cash=equity, equity=equity, peak_equity=equity)
    sized = PositionSizer(policy=RiskPolicy.safe_defaults()).create_intent(
        signal=signal(
            entry_price=entry,
            invalidation_price=entry - stop_distance,
            take_profit=entry + stop_distance,
        ),
        instrument=instrument(quantity_step=step, minimum_notional="0.01"),
        portfolio=account,
        snapshot_hash=portfolio_hash(account),
        now=NOW,
    )
    assert isinstance(sized, OrderIntent)

    quantity = sized.quantity
    assert quantity is not None
    approved_notional = quantity * entry
    assert approved_notional <= equity * Decimal("0.10")
    assert quantity * stop_distance <= equity * Decimal("0.005")


@given(
    notional=st.decimals(min_value="0.01", max_value="1", places=2),
)
def test_adding_a_rejection_condition_cannot_leave_an_approval_approved(
    notional: Decimal,
) -> None:
    account = portfolio(cash="100", equity="100", peak_equity="100")
    accepted_intent = intent(
        notional=notional,
        limit_price="1",
        stop_loss="0.5",
        take_profit="2",
        snapshot_hash=portfolio_hash(account),
    )
    baseline = RiskEngine.safe_defaults().assess(
        intent=accepted_intent,
        instrument=instrument(minimum_notional="0.01"),
        market=snapshot(),
        portfolio=account,
        now=NOW,
    )
    with_extra_rejection = RiskEngine.safe_defaults().assess(
        intent=accepted_intent.model_copy(update={"product": "margin"}),
        instrument=instrument(minimum_notional="0.01"),
        market=snapshot(),
        portfolio=account,
        now=NOW,
    )

    assert baseline.approved is True
    assert with_extra_rejection.approved is False
