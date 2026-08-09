"""After-cost metrics and chronological out-of-sample split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from market_sentinel.backtest.engine import BacktestResult

_ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Immutable metrics with the annualization assumption carried alongside values."""

    total_return: Decimal
    benchmark_excess_return: Decimal
    maximum_drawdown: Decimal
    annualized_sharpe: Decimal | None
    annualized_sortino: Decimal | None
    profit_factor: Decimal | None
    turnover: Decimal
    exposure: Decimal
    hit_rate: Decimal
    completed_trade_count: int
    periods_per_year: int


@dataclass(frozen=True, slots=True)
class ChronologicalSplit[T]:
    """Non-overlapping ordered train, validation, and untouched test samples."""

    train: tuple[T, ...]
    validation: tuple[T, ...]
    test: tuple[T, ...]


def calculate_performance_metrics(
    *,
    equity_curve: tuple[Decimal, ...],
    benchmark_curve: tuple[Decimal, ...],
    completed_trade_pnls: tuple[Decimal, ...],
    traded_notional: Decimal,
    exposed_periods: int,
    periods_per_year: int,
) -> PerformanceMetrics:
    """Calculate after-cost evidence without inferring missing trade observations."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    _validate_curve(equity_curve, "equity_curve")
    _validate_curve(benchmark_curve, "benchmark_curve")
    if len(equity_curve) != len(benchmark_curve):
        raise ValueError("equity and benchmark curves must have equal lengths")
    if not isinstance(traded_notional, Decimal) or not traded_notional.is_finite():
        raise ValueError("traded_notional must be a finite Decimal")
    if traded_notional < _ZERO:
        raise ValueError("traded_notional must be nonnegative")
    if not 0 <= exposed_periods <= len(equity_curve):
        raise ValueError("exposed_periods must fall within the equity curve")
    if any(not isinstance(pnl, Decimal) or not pnl.is_finite() for pnl in completed_trade_pnls):
        raise ValueError("completed trade PnLs must be finite Decimals")

    total_return = equity_curve[-1] / equity_curve[0] - Decimal("1")
    benchmark_return = benchmark_curve[-1] / benchmark_curve[0] - Decimal("1")
    period_returns = tuple(
        current / previous - Decimal("1")
        for previous, current in zip(equity_curve, equity_curve[1:], strict=False)
    )
    sharpe, sortino = _annualized_ratios(period_returns, periods_per_year)
    equity_sum = sum(equity_curve, _ZERO)
    positives = sum((pnl for pnl in completed_trade_pnls if pnl > _ZERO), _ZERO)
    losses = -sum((pnl for pnl in completed_trade_pnls if pnl < _ZERO), _ZERO)
    if not completed_trade_pnls:
        profit_factor: Decimal | None = None
    elif losses == _ZERO:
        profit_factor = Decimal("Infinity") if positives > _ZERO else None
    else:
        profit_factor = positives / losses
    wins = sum(1 for pnl in completed_trade_pnls if pnl > _ZERO)
    trade_count = len(completed_trade_pnls)
    return PerformanceMetrics(
        total_return=total_return,
        benchmark_excess_return=total_return - benchmark_return,
        maximum_drawdown=_maximum_drawdown(equity_curve),
        annualized_sharpe=sharpe,
        annualized_sortino=sortino,
        profit_factor=profit_factor,
        turnover=traded_notional * Decimal(len(equity_curve)) / equity_sum,
        exposure=Decimal(exposed_periods) / Decimal(len(equity_curve)),
        hit_rate=Decimal(wins) / Decimal(trade_count) if trade_count else _ZERO,
        completed_trade_count=trade_count,
        periods_per_year=periods_per_year,
    )


def calculate_result_metrics(
    result: BacktestResult, *, periods_per_year: int
) -> PerformanceMetrics:
    """Calculate metrics directly from one immutable after-cost result artifact."""
    return calculate_performance_metrics(
        equity_curve=tuple(point.value for point in result.equity_curve),
        benchmark_curve=tuple(point.value for point in result.benchmark_curve),
        completed_trade_pnls=tuple(trade.net_pnl for trade in result.completed_trades),
        traded_notional=result.traded_notional,
        exposed_periods=result.exposed_periods,
        periods_per_year=periods_per_year,
    )


def chronological_split[T](
    items: tuple[T, ...], *, train_size: int, validation_size: int
) -> ChronologicalSplit[T]:
    """Split once, assigning every remaining item only to the untouched test sample."""
    if train_size <= 0 or validation_size <= 0 or train_size + validation_size >= len(items):
        raise ValueError("split sizes must leave nonempty train, validation, and test samples")
    validation_end = train_size + validation_size
    return ChronologicalSplit(
        train=items[:train_size],
        validation=items[train_size:validation_end],
        test=items[validation_end:],
    )


def walk_forward_splits[T](
    items: tuple[T, ...],
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    step: int | None = None,
) -> tuple[ChronologicalSplit[T], ...]:
    """Return expanding-train windows whose validation/test observations never overlap."""
    if min(train_size, validation_size, test_size) <= 0:
        raise ValueError("walk-forward sizes must be positive")
    stride = test_size if step is None else step
    if stride <= 0:
        raise ValueError("step must be positive")
    windows: list[ChronologicalSplit[T]] = []
    train_end = train_size
    while train_end + validation_size + test_size <= len(items):
        validation_end = train_end + validation_size
        test_end = validation_end + test_size
        windows.append(
            ChronologicalSplit(
                train=items[:train_end],
                validation=items[train_end:validation_end],
                test=items[validation_end:test_end],
            )
        )
        train_end += stride
    return tuple(windows)


def _validate_curve(curve: tuple[Decimal, ...], name: str) -> None:
    if not curve or any(
        not isinstance(value, Decimal) or not value.is_finite() or value <= _ZERO
        for value in curve
    ):
        raise ValueError(f"{name} must contain positive finite Decimals")


def _maximum_drawdown(curve: tuple[Decimal, ...]) -> Decimal:
    peak = curve[0]
    maximum = _ZERO
    for value in curve:
        peak = max(peak, value)
        maximum = max(maximum, (peak - value) / peak)
    return maximum


def _annualized_ratios(
    returns: tuple[Decimal, ...], periods_per_year: int
) -> tuple[Decimal | None, Decimal | None]:
    if not returns:
        return None, None
    mean = sum(returns, _ZERO) / Decimal(len(returns))
    variance = sum(((value - mean) ** 2 for value in returns), _ZERO) / Decimal(len(returns))
    standard_deviation = variance.sqrt()
    scale = Decimal(periods_per_year).sqrt()
    sharpe = mean / standard_deviation * scale if standard_deviation > _ZERO else None
    negative = tuple(value for value in returns if value < _ZERO)
    if not negative:
        sortino = None
    else:
        downside_deviation = (
            sum((value**2 for value in negative), _ZERO) / Decimal(len(negative))
        ).sqrt()
        sortino = mean / downside_deviation * scale if downside_deviation > _ZERO else None
    return sharpe, sortino
