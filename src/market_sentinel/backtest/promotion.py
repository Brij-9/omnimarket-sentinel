"""Exact evidence gates for backtest-to-paper and paper-to-live promotion."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from market_sentinel.backtest.metrics import PerformanceMetrics
from market_sentinel.domain.enums import Horizon


class PromotionStatus(StrEnum):
    """The only stable promotion outcomes."""

    ELIGIBLE = "ELIGIBLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PROMOTION_REJECTED = "PROMOTION_REJECTED"


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Machine-readable promotion outcome with stable ordered reasons."""

    status: PromotionStatus
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PaperEvidence:
    """Observed paper evidence for one strategy/venue/horizon combination."""

    horizon: Horizon
    trading_days: int
    calendar_days: int
    completed_trade_count: int
    after_cost_return: Decimal
    risk_breach: bool
    unexplained_reconciliation_event: bool
    realized_slippage_bps: Decimal
    stressed_slippage_bps: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, Horizon):
            raise ValueError("horizon must be a Horizon")
        if min(self.trading_days, self.calendar_days, self.completed_trade_count) < 0:
            raise ValueError("paper durations and trade count must be nonnegative")
        for name, value in (
            ("after_cost_return", self.after_cost_return),
            ("realized_slippage_bps", self.realized_slippage_bps),
            ("stressed_slippage_bps", self.stressed_slippage_bps),
        ):
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"{name} must be a finite Decimal")
        if self.realized_slippage_bps < Decimal("0") or self.stressed_slippage_bps < Decimal("0"):
            raise ValueError("slippage assumptions must be nonnegative")


class PromotionEvaluator:
    """Apply sample sufficiency before exact performance and safety thresholds."""

    def evaluate_backtest(
        self,
        *,
        metrics: PerformanceMetrics,
        stressed_total_return: Decimal,
        horizon: Horizon,
    ) -> PromotionDecision:
        """Evaluate untouched OOS base- and stressed-cost evidence for paper eligibility."""
        if not isinstance(horizon, Horizon):
            raise ValueError("horizon must be a Horizon")
        if not isinstance(stressed_total_return, Decimal) or not stressed_total_return.is_finite():
            raise ValueError("stressed_total_return must be a finite Decimal")
        required_trades = 100 if horizon is Horizon.INTRADAY else 30
        if metrics.completed_trade_count < required_trades:
            return PromotionDecision(
                status=PromotionStatus.INSUFFICIENT_EVIDENCE,
                reason_codes=("INSUFFICIENT_COMPLETED_TRADES",),
            )

        reasons: list[str] = []
        if metrics.total_return <= Decimal("0"):
            reasons.append("NON_POSITIVE_OOS_AFTER_COST_RETURN")
        if metrics.benchmark_excess_return <= Decimal("0"):
            reasons.append("NON_POSITIVE_BENCHMARK_EXCESS")
        if metrics.maximum_drawdown > Decimal("0.10"):
            reasons.append("MAX_DRAWDOWN_EXCEEDED")
        if metrics.profit_factor is None or metrics.profit_factor < Decimal("1.10"):
            reasons.append("PROFIT_FACTOR_BELOW_MINIMUM")
        if stressed_total_return <= Decimal("0"):
            reasons.append("NON_POSITIVE_STRESSED_RETURN")
        return _performance_decision(reasons)

    def evaluate_paper(self, *, evidence: PaperEvidence) -> PromotionDecision:
        """Evaluate observed paper evidence for live eligibility without inferring gaps."""
        insufficient: list[str] = []
        required_duration = 20 if evidence.horizon is Horizon.INTRADAY else 90
        observed_duration = (
            evidence.trading_days
            if evidence.horizon is Horizon.INTRADAY
            else evidence.calendar_days
        )
        required_trades = 100 if evidence.horizon is Horizon.INTRADAY else 30
        if observed_duration < required_duration:
            insufficient.append("INSUFFICIENT_PAPER_DURATION")
        if evidence.completed_trade_count < required_trades:
            insufficient.append("INSUFFICIENT_COMPLETED_TRADES")
        if insufficient:
            return PromotionDecision(
                status=PromotionStatus.INSUFFICIENT_EVIDENCE,
                reason_codes=tuple(insufficient),
            )

        reasons: list[str] = []
        if evidence.after_cost_return <= Decimal("0"):
            reasons.append("NON_POSITIVE_PAPER_AFTER_COST_RETURN")
        if evidence.risk_breach:
            reasons.append("RISK_BREACH")
        if evidence.unexplained_reconciliation_event:
            reasons.append("UNEXPLAINED_RECONCILIATION_EVENT")
        if evidence.realized_slippage_bps > evidence.stressed_slippage_bps:
            reasons.append("REALIZED_SLIPPAGE_EXCEEDS_STRESS")
        return _performance_decision(reasons)


def _performance_decision(reasons: list[str]) -> PromotionDecision:
    if reasons:
        return PromotionDecision(
            status=PromotionStatus.PROMOTION_REJECTED,
            reason_codes=tuple(reasons),
        )
    return PromotionDecision(status=PromotionStatus.ELIGIBLE, reason_codes=())
