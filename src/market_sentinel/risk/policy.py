"""Immutable risk-policy values bounded by the project's global safe defaults."""

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from market_sentinel.config import (
    SAFE_MAX_DAILY_LOSS_FRACTION,
    SAFE_MAX_DRAWDOWN_FRACTION,
    SAFE_MAX_GROSS_EXPOSURE_FRACTION,
    SAFE_MAX_POSITION_FRACTION,
    SAFE_MAX_TRADE_RISK_FRACTION,
    RiskSettings,
)


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    """Finite, conservative limits used by the sizing and assessment stages."""

    max_trade_risk_fraction: Decimal
    max_position_fraction: Decimal
    max_gross_exposure_fraction: Decimal
    max_daily_loss_fraction: Decimal
    max_drawdown_fraction: Decimal
    decision_ttl: timedelta = timedelta(seconds=60)

    def __post_init__(self) -> None:
        for value in (
            self.max_trade_risk_fraction,
            self.max_position_fraction,
            self.max_gross_exposure_fraction,
            self.max_daily_loss_fraction,
            self.max_drawdown_fraction,
        ):
            if not value.is_finite() or not Decimal("0") < value <= Decimal("1"):
                raise ValueError("risk-policy fractions must be finite Decimals in (0, 1]")
        if self.decision_ttl <= timedelta(0):
            raise ValueError("decision_ttl must be positive")

    @classmethod
    def safe_defaults(cls) -> "RiskPolicy":
        """Return the immutable global safe limits without consulting targets."""
        return cls(
            max_trade_risk_fraction=SAFE_MAX_TRADE_RISK_FRACTION,
            max_position_fraction=SAFE_MAX_POSITION_FRACTION,
            max_gross_exposure_fraction=SAFE_MAX_GROSS_EXPOSURE_FRACTION,
            max_daily_loss_fraction=SAFE_MAX_DAILY_LOSS_FRACTION,
            max_drawdown_fraction=SAFE_MAX_DRAWDOWN_FRACTION,
        )

    @classmethod
    def from_settings(cls, settings: RiskSettings) -> "RiskPolicy":
        """Clamp supplied settings to the non-relaxable global limits."""
        return cls(
            max_trade_risk_fraction=min(
                settings.max_trade_risk_fraction, SAFE_MAX_TRADE_RISK_FRACTION
            ),
            max_position_fraction=min(settings.max_position_fraction, SAFE_MAX_POSITION_FRACTION),
            max_gross_exposure_fraction=min(
                settings.max_gross_exposure_fraction, SAFE_MAX_GROSS_EXPOSURE_FRACTION
            ),
            max_daily_loss_fraction=min(
                settings.max_daily_loss_fraction, SAFE_MAX_DAILY_LOSS_FRACTION
            ),
            max_drawdown_fraction=min(settings.max_drawdown_fraction, SAFE_MAX_DRAWDOWN_FRACTION),
        )
