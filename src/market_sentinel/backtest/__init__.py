"""Deterministic historical simulation and promotion evidence."""

from market_sentinel.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    CompletedTrade,
    CostModel,
    FillModel,
    RobustnessResult,
)
from market_sentinel.backtest.metrics import PerformanceMetrics, calculate_result_metrics
from market_sentinel.backtest.promotion import (
    PaperEvidence,
    PromotionDecision,
    PromotionEvaluator,
    PromotionStatus,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "CompletedTrade",
    "CostModel",
    "FillModel",
    "PaperEvidence",
    "PerformanceMetrics",
    "PromotionDecision",
    "PromotionEvaluator",
    "PromotionStatus",
    "RobustnessResult",
    "calculate_result_metrics",
]
