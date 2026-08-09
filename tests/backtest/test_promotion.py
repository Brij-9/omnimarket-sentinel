"""Exact, stable promotion-gate tests."""

from dataclasses import replace
from decimal import Decimal
from typing import cast

import pytest

from market_sentinel.backtest.metrics import PerformanceMetrics
from market_sentinel.backtest.promotion import (
    PaperEvidence,
    PromotionEvaluator,
    PromotionStatus,
)
from market_sentinel.domain.enums import Horizon


def _metrics(**changes: object) -> PerformanceMetrics:
    values: dict[str, object] = {
        "total_return": Decimal("0.01"),
        "benchmark_excess_return": Decimal("0.001"),
        "maximum_drawdown": Decimal("0.10"),
        "annualized_sharpe": Decimal("1"),
        "annualized_sortino": Decimal("1"),
        "profit_factor": Decimal("1.10"),
        "turnover": Decimal("1"),
        "exposure": Decimal("0.5"),
        "hit_rate": Decimal("0.5"),
        "completed_trade_count": 100,
        "periods_per_year": 252,
    }
    values.update(changes)
    return PerformanceMetrics(**values)  # type: ignore[arg-type]


def test_backtest_exact_thresholds_are_eligible() -> None:
    """Changing inclusive drawdown/PF/count thresholds would reject boundary evidence."""
    decision = PromotionEvaluator().evaluate_backtest(
        metrics=_metrics(),
        stressed_total_return=Decimal("0.000001"),
        horizon=Horizon.INTRADAY,
    )

    assert decision.status is PromotionStatus.ELIGIBLE
    assert decision.reason_codes == ()


@pytest.mark.parametrize(
    ("change", "value", "reason"),
    [
        ("total_return", Decimal("0"), "NON_POSITIVE_OOS_AFTER_COST_RETURN"),
        ("benchmark_excess_return", Decimal("0"), "NON_POSITIVE_BENCHMARK_EXCESS"),
        ("maximum_drawdown", Decimal("0.1000001"), "MAX_DRAWDOWN_EXCEEDED"),
        ("profit_factor", Decimal("1.099999"), "PROFIT_FACTOR_BELOW_MINIMUM"),
    ],
)
def test_backtest_failed_performance_gate_is_rejected(
    change: str, value: Decimal, reason: str
) -> None:
    """Weakening any exact performance threshold would promote rejected OOS evidence."""
    decision = PromotionEvaluator().evaluate_backtest(
        metrics=_metrics(**{change: value}),
        stressed_total_return=Decimal("0.01"),
        horizon=Horizon.INTRADAY,
    )

    assert decision.status is PromotionStatus.PROMOTION_REJECTED
    assert decision.reason_codes == (reason,)


def test_backtest_2x_cost_failure_is_rejected() -> None:
    """Considering only the base-cost run would promote a cost-fragile result."""
    decision = PromotionEvaluator().evaluate_backtest(
        metrics=_metrics(),
        stressed_total_return=Decimal("0"),
        horizon=Horizon.INTRADAY,
    )

    assert decision.status is PromotionStatus.PROMOTION_REJECTED
    assert decision.reason_codes == ("NON_POSITIVE_STRESSED_RETURN",)


@pytest.mark.parametrize(
    ("horizon", "count"),
    [(Horizon.INTRADAY, 99), (Horizon.SWING, 29), (Horizon.SWING, 0)],
)
def test_backtest_small_or_no_trade_sample_is_insufficient_not_rejected(
    horizon: Horizon, count: int
) -> None:
    """A smaller sample has missing evidence rather than negative evidence."""
    decision = PromotionEvaluator().evaluate_backtest(
        metrics=_metrics(total_return=Decimal("-1"), completed_trade_count=count),
        stressed_total_return=Decimal("-1"),
        horizon=horizon,
    )

    assert decision.status is PromotionStatus.INSUFFICIENT_EVIDENCE
    assert decision.reason_codes == ("INSUFFICIENT_COMPLETED_TRADES",)


def _paper(horizon: Horizon = Horizon.INTRADAY) -> PaperEvidence:
    return PaperEvidence(
        horizon=horizon,
        trading_days=20,
        calendar_days=90,
        completed_trade_count=100,
        after_cost_return=Decimal("0.01"),
        risk_breach=False,
        unexplained_reconciliation_event=False,
        realized_slippage_bps=Decimal("5"),
        stressed_slippage_bps=Decimal("5"),
    )


def test_live_promotion_exact_intraday_and_swing_thresholds_are_eligible() -> None:
    """The exact required duration, trades, and slippage must be sufficient."""
    intraday = PromotionEvaluator().evaluate_paper(evidence=_paper())
    swing = PromotionEvaluator().evaluate_paper(
        evidence=replace(
            _paper(Horizon.SWING),
            trading_days=0,
            calendar_days=90,
            completed_trade_count=30,
        )
    )

    assert intraday.status is PromotionStatus.ELIGIBLE
    assert swing.status is PromotionStatus.ELIGIBLE


def test_live_sample_deficits_are_reported_before_performance_is_inferred() -> None:
    """Poor metrics from an incomplete window cannot turn missing evidence into rejection."""
    decision = PromotionEvaluator().evaluate_paper(
        evidence=replace(
            _paper(),
            trading_days=19,
            completed_trade_count=99,
            after_cost_return=Decimal("-1"),
            risk_breach=True,
        )
    )

    assert decision.status is PromotionStatus.INSUFFICIENT_EVIDENCE
    assert decision.reason_codes == (
        "INSUFFICIENT_PAPER_DURATION",
        "INSUFFICIENT_COMPLETED_TRADES",
    )


def test_live_rejections_have_stable_order_and_slippage_boundary() -> None:
    """Passing a breached or unreconciled account due to reason ordering would weaken safety."""
    decision = PromotionEvaluator().evaluate_paper(
        evidence=replace(
            _paper(),
            after_cost_return=Decimal("0"),
            risk_breach=True,
            unexplained_reconciliation_event=True,
            realized_slippage_bps=Decimal("5.0001"),
        )
    )

    assert decision.status is PromotionStatus.PROMOTION_REJECTED
    assert decision.reason_codes == (
        "NON_POSITIVE_PAPER_AFTER_COST_RETURN",
        "RISK_BREACH",
        "UNEXPLAINED_RECONCILIATION_EVENT",
        "REALIZED_SLIPPAGE_EXCEEDS_STRESS",
    )


def test_backtest_rejects_non_horizon_value_before_selecting_thresholds() -> None:
    """An unknown string must not silently receive the lower swing trade threshold."""
    with pytest.raises(ValueError, match="Horizon"):
        PromotionEvaluator().evaluate_backtest(
            metrics=_metrics(completed_trade_count=30),
            stressed_total_return=Decimal("0.01"),
            horizon=cast("Horizon", "unknown"),
        )


def test_paper_evidence_rejects_non_horizon_value() -> None:
    """Invalid runtime horizons must fail at the evidence boundary, not branch as swing."""
    with pytest.raises(ValueError, match="Horizon"):
        replace(_paper(), horizon=cast("Horizon", "swing"))
