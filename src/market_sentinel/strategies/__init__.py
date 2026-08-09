"""Deterministic, point-in-time strategy foundations."""

from market_sentinel.strategies.base import (
    CanonicalParameter,
    Strategy,
    StrategyConfiguration,
    StrategyContext,
    StrategyMetadata,
    canonical_strategy_configuration,
)
from market_sentinel.strategies.crypto import CryptoVolatilityBreakoutStrategy
from market_sentinel.strategies.ensemble import EnsembleWeights, SignalEnsemble
from market_sentinel.strategies.intraday import OpeningRangeVwapStrategy
from market_sentinel.strategies.regime import MarketRegime, classify_regime
from market_sentinel.strategies.swing import SwingBreakoutStrategy

__all__ = [
    "CanonicalParameter",
    "CryptoVolatilityBreakoutStrategy",
    "EnsembleWeights",
    "MarketRegime",
    "OpeningRangeVwapStrategy",
    "SignalEnsemble",
    "Strategy",
    "StrategyConfiguration",
    "StrategyContext",
    "StrategyMetadata",
    "SwingBreakoutStrategy",
    "canonical_strategy_configuration",
    "classify_regime",
]
