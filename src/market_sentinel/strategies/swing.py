"""Deterministic, signal-only swing trend breakout."""

from decimal import Decimal, InvalidOperation

from market_sentinel.domain.enums import Horizon, SignalDirection
from market_sentinel.domain.models import Signal
from market_sentinel.strategies.base import StrategyContext, StrategyMetadata
from market_sentinel.strategies.indicators import atr, sma
from market_sentinel.strategies.regime import MarketRegime, classify_regime
from market_sentinel.strategies.validation import bars_are_strictly_valid

_ZERO = Decimal("0")


class SwingBreakoutStrategy:
    """Emit a long swing signal only after every trend and liquidity gate passes."""

    metadata = StrategyMetadata(
        strategy_id="swing-breakout",
        version="1.0.0",
        allowed_horizons=(Horizon.SWING,),
        allowed_directions=(SignalDirection.LONG,),
        max_holding_bars=20,
    )

    def __init__(
        self,
        *,
        max_spread_bps: Decimal = Decimal("25"),
        min_average_volume: Decimal = Decimal("100"),
    ) -> None:
        self.max_spread_bps = _positive_decimal(max_spread_bps)
        self.min_average_volume = _positive_decimal(min_average_volume)

    def evaluate(self, context: StrategyContext) -> Signal | None:
        """Evaluate only the supplied point-in-time history and otherwise abstain."""
        try:
            if context.horizon is not Horizon.SWING or len(context.bars) < 70:
                return None
            if (
                not bars_are_strictly_valid(context.bars)
                or context.spread_bps >= self.max_spread_bps
            ):
                return None
            if (
                classify_regime(
                    context.bars,
                    self.max_spread_bps,
                    spread_bps=context.spread_bps,
                    min_average_volume=self.min_average_volume,
                )
                is not MarketRegime.TRENDING
            ):
                return None

            closes = tuple(bar.close for bar in context.bars)
            fast_average = sma(closes, 20)
            slow_average = sma(closes, 50)
            average_true_range = atr(context.bars, 14)
            current = context.bars[-1]
            prior_high = max(bar.high for bar in context.bars[-21:-1])
            recent_average_volume = sum(
                (bar.volume for bar in context.bars[-20:]), _ZERO
            ) / Decimal("20")
            if (
                fast_average is None
                or slow_average is None
                or average_true_range is None
                or average_true_range <= _ZERO
                or fast_average <= slow_average
                or current.close <= prior_high
                or current.close <= closes[-6]
                or recent_average_volume < self.min_average_volume
                or current.volume < recent_average_volume
            ):
                return None

            return Signal(
                strategy_id=self.metadata.strategy_id,
                strategy_version=self.metadata.version,
                instrument_id=context.instrument_id,
                direction=SignalDirection.LONG,
                strength=Decimal("0.70"),
                horizon=context.horizon,
                entry_price=current.close,
                invalidation_price=current.close - Decimal("2") * average_true_range,
                take_profit=current.close + Decimal("3") * average_true_range,
                research_required=False,
                evidence_uris=(),
            )
        except (AttributeError, InvalidOperation, TypeError, ValueError, ZeroDivisionError):
            return None


def _positive_decimal(value: Decimal) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= _ZERO:
        raise ValueError("strategy thresholds must be finite positive Decimals")
    return value
