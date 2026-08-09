"""Hand-calculated tests for after-cost performance evidence."""

from decimal import Decimal

import pytest

from market_sentinel.backtest.engine import BacktestEngine, CostModel
from market_sentinel.backtest.metrics import (
    calculate_performance_metrics,
    calculate_result_metrics,
    chronological_split,
    walk_forward_splits,
)
from tests.backtest.test_engine import _bars, _ProtectiveExitStrategy
from tests.factories import instrument


def test_metrics_match_hand_calculated_returns_drawdown_and_ratios() -> None:
    """Wrong return chaining, drawdown peak, or annualization would change these literals."""
    metrics = calculate_performance_metrics(
        equity_curve=(Decimal("100"), Decimal("110"), Decimal("110")),
        benchmark_curve=(Decimal("100"), Decimal("105"), Decimal("105")),
        completed_trade_pnls=(Decimal("4"), Decimal("-2"), Decimal("1")),
        traded_notional=Decimal("220"),
        exposed_periods=2,
        periods_per_year=4,
    )

    assert metrics.total_return == Decimal("0.1")
    assert metrics.benchmark_excess_return == Decimal("0.05")
    assert metrics.maximum_drawdown == Decimal("0")
    assert metrics.annualized_sharpe == Decimal("2")
    assert metrics.annualized_sortino is None
    assert metrics.profit_factor == Decimal("2.5")
    assert metrics.turnover == Decimal("2.0625")
    assert metrics.exposure == Decimal("0.6666666666666666666666666667")
    assert metrics.hit_rate == Decimal("0.6666666666666666666666666667")
    assert metrics.completed_trade_count == 3
    assert metrics.periods_per_year == 4


def test_metrics_handle_flat_curve_and_no_trades_without_dividing_by_zero() -> None:
    """Empty trade evidence and zero volatility must not fabricate ratios."""
    metrics = calculate_performance_metrics(
        equity_curve=(Decimal("10"), Decimal("10")),
        benchmark_curve=(Decimal("10"), Decimal("10")),
        completed_trade_pnls=(),
        traded_notional=Decimal("0"),
        exposed_periods=0,
        periods_per_year=252,
    )

    assert metrics.total_return == Decimal("0")
    assert metrics.annualized_sharpe is None
    assert metrics.annualized_sortino is None
    assert metrics.profit_factor is None
    assert metrics.hit_rate == Decimal("0")
    assert metrics.completed_trade_count == 0


@pytest.mark.parametrize("periods_per_year", [0, -1])
def test_metrics_require_explicit_positive_annualization_frequency(
    periods_per_year: int,
) -> None:
    """Silently guessing or accepting an invalid frequency would make ratios incomparable."""
    with pytest.raises(ValueError, match="periods_per_year"):
        calculate_performance_metrics(
            equity_curve=(Decimal("10"), Decimal("11")),
            benchmark_curve=(Decimal("10"), Decimal("11")),
            completed_trade_pnls=(),
            traded_notional=Decimal("0"),
            exposed_periods=0,
            periods_per_year=periods_per_year,
        )


def test_chronological_helpers_leave_each_test_window_untouched() -> None:
    """Overlapping training with a reported test window would contaminate OOS evidence."""
    split = chronological_split(tuple(range(10)), train_size=6, validation_size=2)
    windows = walk_forward_splits(
        tuple(range(12)), train_size=4, validation_size=2, test_size=2, step=4
    )

    assert split.train == (0, 1, 2, 3, 4, 5)
    assert split.validation == (6, 7)
    assert split.test == (8, 9)
    assert tuple(window.test for window in windows) == (
        (6, 7),
        (10, 11),
    )
    assert all(set(window.train + window.validation).isdisjoint(window.test) for window in windows)


def test_walk_forward_rejects_stride_that_recycles_prior_test_as_validation() -> None:
    """A step smaller than validation plus test reuses OOS evidence in the next fold."""
    with pytest.raises(ValueError, match="step"):
        walk_forward_splits(
            tuple(range(12)),
            train_size=4,
            validation_size=2,
            test_size=2,
            step=2,
        )


def test_walk_forward_oos_observations_are_globally_disjoint_by_identity() -> None:
    """No object may appear in validation/test evidence for more than one fold."""
    observations = tuple(object() for _ in range(16))
    windows = walk_forward_splits(
        observations,
        train_size=4,
        validation_size=2,
        test_size=2,
    )
    oos_ids = [id(item) for window in windows for item in window.validation + window.test]

    assert len(oos_ids) == len(set(oos_ids))


def test_sortino_uses_target_semideviation_over_every_period() -> None:
    """Excluding nonnegative periods from the denominator understates downside deviation."""
    metrics = calculate_performance_metrics(
        equity_curve=(Decimal("100"), Decimal("110"), Decimal("104.5")),
        benchmark_curve=(Decimal("100"), Decimal("100"), Decimal("100")),
        completed_trade_pnls=(),
        traded_notional=Decimal("0"),
        exposed_periods=0,
        periods_per_year=4,
    )

    assert metrics.annualized_sortino == Decimal("1.414213562373095048801688724")


def test_result_metrics_use_after_cost_equity_and_completed_trades() -> None:
    """Reconstructing metrics from gross prices would lose the ledger's cost evidence."""
    bars = list(_bars("10", "10", "10", "11"))
    bars[2] = bars[2].model_copy(update={"high": Decimal("12")})
    result = BacktestEngine(costs=CostModel(fee_bps=Decimal("10"))).run(
        instrument=instrument(minimum_notional="0.01"),
        bars=tuple(bars),
        strategy=_ProtectiveExitStrategy(),
        initial_cash=Decimal("10"),
    )

    metrics = calculate_result_metrics(result, periods_per_year=252)

    assert metrics.total_return == result.ending_equity / result.initial_cash - Decimal("1")
    assert metrics.completed_trade_count == 1
    assert metrics.profit_factor == Decimal("Infinity")
