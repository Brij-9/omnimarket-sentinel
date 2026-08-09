"""Behavioral contracts for the deterministic swing breakout."""

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from market_sentinel.domain.enums import Horizon
from market_sentinel.strategies.base import StrategyContext
from market_sentinel.strategies.indicators import atr
from market_sentinel.strategies.swing import SwingBreakoutStrategy
from tests.factories import trending_bars


def _context(
    *,
    bars: tuple = (),  # type: ignore[type-arg]
    spread_bps: Decimal = Decimal("5"),
    horizon: Horizon = Horizon.SWING,
) -> StrategyContext:
    return StrategyContext(
        instrument_id="AAPL@alpaca",
        bars=bars if bars else trending_bars(80),
        horizon=horizon,
        spread_bps=spread_bps,
    )


def test_swing_breakout_has_atr_invalidation_and_defined_reward() -> None:
    """Wrong ATR multiples would expose a trade to unintended loss or reward."""
    context = _context()
    signal = SwingBreakoutStrategy().evaluate(context)
    average_true_range = atr(context.bars, window=14)

    assert signal is not None
    assert average_true_range is not None
    assert signal.strength > Decimal("0")
    assert signal.entry_price - signal.invalidation_price == 2 * average_true_range
    assert signal.take_profit - signal.entry_price == 3 * average_true_range


def test_swing_metadata_declares_twenty_bar_time_stop() -> None:
    """Omitting lifecycle metadata would let downstream code hold forever."""
    strategy = SwingBreakoutStrategy()

    assert strategy.metadata.max_holding_bars == 20
    assert strategy.metadata.mandatory_preclose_closeout is False
    with pytest.raises(FrozenInstanceError):
        strategy.metadata.max_holding_bars = 21  # type: ignore[misc]


@pytest.mark.parametrize(
    "context",
    [
        _context(horizon=Horizon.INTRADAY),
        _context(bars=trending_bars(69)),
        _context(spread_bps=Decimal("25")),
    ],
)
def test_swing_abstains_when_horizon_history_or_spread_is_ineligible(
    context: StrategyContext,
) -> None:
    """Bypassing any eligibility gate could turn inadequate data into a trade."""
    assert SwingBreakoutStrategy().evaluate(context) is None


def test_swing_requires_prior_high_breakout_ma_alignment_and_momentum() -> None:
    """A missing structural gate would admit a flat or non-breakout market."""
    no_breakout = list(trending_bars(80))
    current = no_breakout[-1]
    no_breakout[-1] = current.model_copy(
        update={
            "open": current.open - Decimal("2"),
            "high": current.high - Decimal("2"),
            "low": current.low - Decimal("2"),
            "close": current.close - Decimal("2"),
        }
    )
    flat = tuple(
        bar.model_copy(
            update={
                "open": Decimal("100"),
                "high": Decimal("100.5"),
                "low": Decimal("99.5"),
                "close": Decimal("100"),
            }
        )
        for bar in trending_bars(80)
    )

    assert SwingBreakoutStrategy().evaluate(_context(bars=tuple(no_breakout))) is None
    assert SwingBreakoutStrategy().evaluate(_context(bars=flat)) is None


def test_swing_requires_recent_liquidity_and_current_volume_confirmation() -> None:
    """Old liquidity must not hide a recent collapse or an unconfirmed breakout."""
    recent_collapse = list(trending_bars(80))
    for index in range(60, 80):
        recent_collapse[index] = recent_collapse[index].model_copy(
            update={"volume": Decimal("0")}
        )
    weak_current = list(trending_bars(80))
    weak_current[-1] = weak_current[-1].model_copy(update={"volume": Decimal("1")})

    strategy = SwingBreakoutStrategy(min_average_volume=Decimal("100"))
    assert strategy.evaluate(_context(bars=tuple(recent_collapse))) is None
    assert strategy.evaluate(_context(bars=tuple(weak_current))) is None


def test_swing_fails_closed_for_malformed_ohlc() -> None:
    """Malformed bars must abstain instead of leaking invalid protective prices."""
    malformed = list(trending_bars(80))
    malformed[-1] = malformed[-1].model_copy(update={"high": Decimal("NaN")})

    assert SwingBreakoutStrategy().evaluate(_context(bars=tuple(malformed))) is None
