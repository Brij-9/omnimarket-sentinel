"""Behavioral contracts for the crypto spot volatility breakout."""

from decimal import Decimal

import pytest

from market_sentinel.domain.enums import Horizon, SignalDirection
from market_sentinel.strategies.base import StrategyContext
from market_sentinel.strategies.crypto import CryptoVolatilityBreakoutStrategy
from market_sentinel.strategies.indicators import atr
from tests.factories import bar_series


def _context(
    *,
    bars: tuple = (),  # type: ignore[type-arg]
    spread_bps: Decimal = Decimal("8"),
    horizon: Horizon = Horizon.SWING,
) -> StrategyContext:
    return StrategyContext(
        instrument_id="BTC/USDT@binance-spot",
        bars=bars
        if bars
        else bar_series(count=40, start_price="100", increment="1", volume="1000"),
        horizon=horizon,
        spread_bps=spread_bps,
    )


def test_crypto_breakout_is_spot_long_with_atr_protection() -> None:
    """The initial crypto strategy must never introduce short or leverage intent."""
    context = _context()
    signal = CryptoVolatilityBreakoutStrategy().evaluate(context)
    average_true_range = atr(context.bars, 14)

    assert signal is not None
    assert average_true_range is not None
    assert signal.direction is SignalDirection.LONG
    assert signal.invalidation_price < signal.entry_price < signal.take_profit
    assert signal.entry_price - signal.invalidation_price == Decimal("2.5") * average_true_range
    assert CryptoVolatilityBreakoutStrategy.metadata.spot_only is True
    assert CryptoVolatilityBreakoutStrategy.metadata.leverage_allowed is False
    assert CryptoVolatilityBreakoutStrategy.metadata.allowed_directions == (
        SignalDirection.LONG,
    )


def test_crypto_requires_trailing_high_breakout() -> None:
    """Including the current bar in the trailing high would suppress every real breakout."""
    bars = list(_context().bars)
    current = bars[-1]
    bars[-1] = current.model_copy(
        update={
            "open": current.open - Decimal("2"),
            "high": current.high - Decimal("2"),
            "low": current.low - Decimal("2"),
            "close": current.close - Decimal("2"),
        }
    )

    assert CryptoVolatilityBreakoutStrategy().evaluate(_context(bars=tuple(bars))) is None


def test_crypto_enforces_tradable_volatility_band() -> None:
    """Near-zero and abnormal volatility must both abstain."""
    quiet = tuple(
        bar.model_copy(
            update={
                "open": Decimal("100") + Decimal(index) / Decimal("100"),
                "high": Decimal("100.006") + Decimal(index) / Decimal("100"),
                "low": Decimal("99.994") + Decimal(index) / Decimal("100"),
                "close": Decimal("100.005") + Decimal(index) / Decimal("100"),
            }
        )
        for index, bar in enumerate(bar_series(count=40, volume="1000"))
    )
    violent = list(_context().bars)
    final = violent[-1]
    violent[-1] = final.model_copy(
        update={"high": Decimal("250"), "low": Decimal("1"), "close": Decimal("150")}
    )

    strategy = CryptoVolatilityBreakoutStrategy()
    assert strategy.evaluate(_context(bars=quiet)) is None
    assert strategy.evaluate(_context(bars=tuple(violent))) is None


def test_crypto_enforces_horizon_history_spread_and_recent_liquidity() -> None:
    """Every market-eligibility gate must reject independently."""
    illiquid = tuple(bar.model_copy(update={"volume": Decimal("0")}) for bar in _context().bars)

    strategy = CryptoVolatilityBreakoutStrategy()
    assert strategy.evaluate(_context(horizon=Horizon.INTRADAY)) is None
    assert strategy.evaluate(_context(bars=_context().bars[:20])) is None
    assert strategy.evaluate(_context(spread_bps=Decimal("30"))) is None
    assert strategy.evaluate(_context(bars=illiquid)) is None


def test_crypto_fails_closed_for_malformed_prices() -> None:
    """Nonfinite market input must never generate a spot signal."""
    malformed = list(_context().bars)
    malformed[-1] = malformed[-1].model_copy(update={"close": Decimal("NaN")})

    assert CryptoVolatilityBreakoutStrategy().evaluate(_context(bars=tuple(malformed))) is None


@pytest.mark.parametrize(
    "update",
    [
        {"close": 139.25},
        {"volume": "1000"},
        {"at": "2026-08-10T10:00:00Z"},
    ],
)
def test_crypto_abstains_for_bypassed_malformed_bar_fields(update: dict[str, object]) -> None:
    """Bypassed model validation must never make malformed fields raise from evaluation."""
    malformed = list(_context().bars)
    malformed[-1] = malformed[-1].model_copy(update=update)

    assert CryptoVolatilityBreakoutStrategy().evaluate(_context(bars=tuple(malformed))) is None
