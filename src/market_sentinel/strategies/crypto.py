"""Deterministic crypto spot-long volatility breakout."""

from decimal import Decimal, InvalidOperation

from market_sentinel.domain.enums import Horizon, SignalDirection
from market_sentinel.domain.models import Signal
from market_sentinel.strategies.base import StrategyContext, StrategyMetadata
from market_sentinel.strategies.indicators import atr
from market_sentinel.strategies.validation import bars_are_strictly_valid

_ZERO = Decimal("0")


class CryptoVolatilityBreakoutStrategy:
    """Emit long-only spot signals inside a bounded, liquid volatility regime."""

    metadata = StrategyMetadata(
        strategy_id="crypto-volatility-breakout",
        version="1.0.0",
        allowed_horizons=(Horizon.SWING,),
        allowed_directions=(SignalDirection.LONG,),
        spot_only=True,
        leverage_allowed=False,
    )

    def __init__(
        self,
        *,
        min_atr_percentage: Decimal = Decimal("0.003"),
        max_atr_percentage: Decimal = Decimal("0.05"),
        max_spread_bps: Decimal = Decimal("30"),
        min_average_volume: Decimal = Decimal("100"),
    ) -> None:
        self.min_atr_percentage = _positive_decimal(min_atr_percentage)
        self.max_atr_percentage = _positive_decimal(max_atr_percentage)
        self.max_spread_bps = _positive_decimal(max_spread_bps)
        self.min_average_volume = _positive_decimal(min_average_volume)
        if self.min_atr_percentage >= self.max_atr_percentage:
            raise ValueError("minimum ATR percentage must be below maximum")

    def evaluate(self, context: StrategyContext) -> Signal | None:
        """Evaluate only a complete trailing prefix and otherwise abstain."""
        try:
            if context.horizon is not Horizon.SWING or len(context.bars) < 21:
                return None
            if (
                not bars_are_strictly_valid(context.bars)
                or context.spread_bps >= self.max_spread_bps
            ):
                return None
            average_true_range = atr(context.bars, 14)
            current = context.bars[-1]
            prior_high = max(bar.high for bar in context.bars[-21:-1])
            recent = context.bars[-20:]
            recent_average_volume = sum((bar.volume for bar in recent), _ZERO) / Decimal("20")
            if average_true_range is None or current.close <= _ZERO:
                return None
            volatility = average_true_range / current.close
            if (
                volatility < self.min_atr_percentage
                or volatility > self.max_atr_percentage
                or current.close <= prior_high
                or recent_average_volume < self.min_average_volume
                or current.volume < recent_average_volume
            ):
                return None

            invalidation = current.close - Decimal("2.5") * average_true_range
            if invalidation <= _ZERO:
                return None
            return Signal(
                strategy_id=self.metadata.strategy_id,
                strategy_version=self.metadata.version,
                instrument_id=context.instrument_id,
                direction=SignalDirection.LONG,
                strength=Decimal("0.68"),
                horizon=context.horizon,
                entry_price=current.close,
                invalidation_price=invalidation,
                take_profit=current.close + Decimal("5") * average_true_range,
                research_required=False,
                evidence_uris=(),
            )
        except (AttributeError, InvalidOperation, TypeError, ValueError, ZeroDivisionError):
            return None


def _positive_decimal(value: Decimal) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= _ZERO:
        raise ValueError("strategy thresholds must be finite positive Decimals")
    return value
