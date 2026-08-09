"""Fail-closed regime classification contracts."""

from decimal import Decimal

from market_sentinel.strategies.regime import MarketRegime, classify_regime
from tests.factories import bar_series, trending_bars


def test_clean_uptrend_is_classified_as_trending() -> None:
    """Breaking MA separation or slope detection would hide a liquid uptrend."""
    assert classify_regime(trending_bars(count=80), max_spread_bps=30) is MarketRegime.TRENDING


def test_flat_liquid_market_is_range_bound() -> None:
    """A flat series must not be reported as trending without directional structure."""
    bars = bar_series(count=80, start_price="100", increment="0")

    assert classify_regime(bars, max_spread_bps=Decimal("30")) is MarketRegime.RANGE_BOUND


def test_high_volatility_takes_precedence_over_an_existing_trend() -> None:
    """Checking trend first would admit a market that exceeds the volatility cap."""
    bars = list(trending_bars(count=80))
    final = bars[-1]
    bars[-1] = final.model_copy(
        update={
            "high": final.close * Decimal("2"),
            "low": final.close / Decimal("2"),
        }
    )

    assert (
        classify_regime(
            tuple(bars), max_spread_bps=30, max_atr_percentage=Decimal("0.05")
        )
        is MarketRegime.HIGH_VOLATILITY
    )


def test_classifier_fails_closed_for_inadequate_invalid_or_illiquid_bars() -> None:
    """Bad history, malformed OHLC, or no volume must never select a tradable regime."""
    valid = trending_bars(count=80)
    malformed = list(valid)
    malformed[-1] = malformed[-1].model_copy(update={"high": Decimal("0")})
    no_volume = tuple(bar.model_copy(update={"volume": Decimal("0")}) for bar in valid)

    assert classify_regime(valid[:50], max_spread_bps=30) is MarketRegime.UNTRADEABLE
    assert classify_regime(tuple(malformed), max_spread_bps=30) is MarketRegime.UNTRADEABLE
    assert classify_regime(no_volume, max_spread_bps=30) is MarketRegime.UNTRADEABLE


def test_classifier_rejects_spread_at_and_above_the_configured_cap() -> None:
    """A spread over the cost cap makes an otherwise clean trend untradeable."""
    bars = trending_bars(count=80)

    assert (
        classify_regime(bars, max_spread_bps=30, spread_bps=Decimal("30"))
        is MarketRegime.UNTRADEABLE
    )
    assert (
        classify_regime(bars, max_spread_bps=30, spread_bps=Decimal("30.01"))
        is MarketRegime.UNTRADEABLE
    )


def test_classifier_rejects_recently_illiquid_market_despite_liquid_old_history() -> None:
    """A full-history volume average can hide a current liquidity collapse."""
    bars = list(trending_bars(count=80))
    for index in range(60, 80):
        bars[index] = bars[index].model_copy(update={"volume": Decimal("0")})

    assert classify_regime(tuple(bars), max_spread_bps=30) is MarketRegime.UNTRADEABLE
