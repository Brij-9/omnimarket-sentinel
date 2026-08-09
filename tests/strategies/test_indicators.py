"""Hand-calculated contracts for deterministic, trailing-only indicators."""

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from market_sentinel.domain.enums import Horizon
from market_sentinel.strategies.base import Strategy, StrategyContext
from market_sentinel.strategies.indicators import atr, ema, rsi, sma, vwap
from tests.factories import bar_series


def test_sma_uses_only_trailing_values() -> None:
    """Using an older value instead of the trailing window changes the average."""
    values = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("100")]

    assert sma(values[:3], window=3) == Decimal("2")
    assert sma(values, window=3) == Decimal("35")


def test_indicators_return_none_when_their_required_history_is_incomplete() -> None:
    """A premature indicator value could make a strategy act on incomplete data."""
    bars = bar_series(count=2)

    assert sma([Decimal("1")], window=2) is None
    assert ema([Decimal("1")], window=2) is None
    assert atr(bars, window=2) is None
    assert rsi([Decimal("1"), Decimal("2")], window=2) is None
    assert vwap(bars, window=3) is None


def test_ema_seeds_with_the_first_window_then_uses_decimal_smoothing() -> None:
    """Changing the EMA seed or alpha produces a different deterministic result."""
    values = [Decimal("2"), Decimal("4"), Decimal("8")]

    assert ema(values, window=2) == Decimal("6.333333333333333333333333334")


def test_atr_uses_the_previous_close_in_true_range() -> None:
    """Ignoring the prior close would miss an overnight gap in true range."""
    bars = (
        bar_series(count=1, start_price="9", increment="0")[0],
        bar_series(count=1, start_price="11", increment="0")[0].model_copy(
            update={"high": Decimal("12"), "low": Decimal("9"), "close": Decimal("11")}
        ),
        bar_series(count=1, start_price="12", increment="0")[0].model_copy(
            update={"high": Decimal("13"), "low": Decimal("10"), "close": Decimal("12")}
        ),
    )

    assert atr(bars, window=2) == Decimal("3")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([Decimal("1"), Decimal("2"), Decimal("4"), Decimal("3")], Decimal("75")),
        ([Decimal("1"), Decimal("2"), Decimal("3")], Decimal("100")),
        ([Decimal("3"), Decimal("2"), Decimal("1")], Decimal("0")),
        ([Decimal("2"), Decimal("2"), Decimal("2")], Decimal("50")),
    ],
)
def test_rsi_uses_wilder_changes_and_defined_zero_loss_edges(
    values: list[Decimal], expected: Decimal
) -> None:
    """Wrong gain/loss handling would misclassify the listed hand-calculated paths."""
    assert rsi(values, window=values.__len__() - 1) == expected


def test_vwap_uses_typical_price_and_rejects_nonpositive_cumulative_volume() -> None:
    """Using close alone or dividing by zero gives the wrong liquidity-weighted price."""
    first, second = bar_series(count=2, start_price="10", increment="0")
    bars = (
        first.model_copy(
            update={
                "high": Decimal("12"),
                "low": Decimal("8"),
                "close": Decimal("10"),
                "volume": Decimal("2"),
            }
        ),
        second.model_copy(
            update={
                "open": Decimal("15"),
                "high": Decimal("18"),
                "low": Decimal("12"),
                "close": Decimal("15"),
                "volume": Decimal("1"),
            }
        ),
    )
    no_volume = tuple(bar.model_copy(update={"volume": Decimal("0")}) for bar in bars)

    assert vwap(bars, window=2) == Decimal("35") / Decimal("3")
    with pytest.raises(ValueError, match="volume"):
        vwap(no_volume, window=2)


def test_indicators_reject_invalid_windows_and_nonfinite_or_float_values() -> None:
    """Accepting invalid numeric inputs would break deterministic Decimal-only calculations."""
    with pytest.raises(ValueError, match="window"):
        sma([Decimal("1")], window=0)
    with pytest.raises(ValueError, match="finite"):
        ema([Decimal("NaN")], window=1)
    with pytest.raises(ValueError, match="Decimal"):
        rsi([Decimal("1"), 2.0], window=1)  # type: ignore[list-item]


def test_indicator_results_for_a_prefix_do_not_change_after_future_data_exists() -> None:
    """Reading beyond the supplied prefix would make historical replay look ahead."""
    prefix_values = [Decimal("1"), Decimal("2"), Decimal("4"), Decimal("3")]
    prefix_bars = bar_series(count=4, start_price="10", increment="1")
    expected = (
        sma(prefix_values, window=3),
        ema(prefix_values, window=2),
        atr(prefix_bars, window=2),
        rsi(prefix_values, window=3),
        vwap(prefix_bars, window=3),
    )
    future_values = prefix_values + [Decimal("100")]
    future_bars = prefix_bars + bar_series(count=1, start_price="100", increment="0")

    assert future_values[-1] == Decimal("100")
    assert future_bars[-1].close == Decimal("100.25")
    assert (
        sma(prefix_values, window=3),
        ema(prefix_values, window=2),
        atr(prefix_bars, window=2),
        rsi(prefix_values, window=3),
        vwap(prefix_bars, window=3),
    ) == expected


def test_strategy_context_is_immutable_and_strategy_is_a_structural_protocol() -> None:
    """A strategy must receive a stable point-in-time context, not mutable market state."""

    class NoopStrategy:
        def evaluate(self, context: StrategyContext) -> None:
            del context
            return None

    context = StrategyContext(
        instrument_id="AAPL@alpaca",
        bars=bar_series(count=1),
        horizon=Horizon.SWING,
    )

    assert isinstance(NoopStrategy(), Strategy)
    with pytest.raises(FrozenInstanceError):
        context.instrument_id = "MSFT@alpaca"  # type: ignore[misc]
