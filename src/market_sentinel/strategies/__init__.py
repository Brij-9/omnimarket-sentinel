"""Deterministic, point-in-time strategy foundations."""

from market_sentinel.strategies.base import Strategy, StrategyContext, StrategyMetadata
from market_sentinel.strategies.crypto import CryptoVolatilityBreakoutStrategy
from market_sentinel.strategies.ensemble import EnsembleWeights, SignalEnsemble
from market_sentinel.strategies.intraday import OpeningRangeVwapStrategy
from market_sentinel.strategies.regime import MarketRegime, classify_regime
from market_sentinel.strategies.swing import SwingBreakoutStrategy

__all__ = [
    "CryptoVolatilityBreakoutStrategy",
    "EnsembleWeights",
    "MarketRegime",
    "OpeningRangeVwapStrategy",
    "SignalEnsemble",
    "Strategy",
    "StrategyContext",
    "StrategyMetadata",
    "SwingBreakoutStrategy",
    "classify_regime",
]
