"""Canonical strategy evidence must be derived from the strategy instance."""

from datetime import time, timedelta
from decimal import Decimal

import pytest

from market_sentinel.backtest.engine import BacktestEngine, CostModel
from market_sentinel.strategies import StrategyConfiguration
from market_sentinel.strategies.base import canonical_strategy_configuration
from market_sentinel.strategies.crypto import CryptoVolatilityBreakoutStrategy
from market_sentinel.strategies.intraday import OpeningRangeVwapStrategy
from market_sentinel.strategies.swing import SwingBreakoutStrategy
from tests.backtest.test_engine import _bars
from tests.factories import instrument


def test_official_strategy_configuration_contains_actual_constructor_values() -> None:
    """Defaults or caller-authored metadata must not replace the instance's real thresholds."""
    swing = SwingBreakoutStrategy(
        max_spread_bps=Decimal("12.50"),
        min_average_volume=Decimal("250"),
    )
    intraday = OpeningRangeVwapStrategy(
        session_start=time(9, 15),
        session_end=time(15, 30),
        session_timezone="Asia/Calcutta",
        opening_range_bars=8,
        closeout_buffer=timedelta(minutes=10),
        max_spread_bps=Decimal("9"),
        min_average_volume=Decimal("300"),
    )

    assert isinstance(swing.configuration, StrategyConfiguration)
    assert tuple((item.name, item.kind, item.value) for item in swing.configuration.parameters) == (
        ("max_spread_bps", "decimal", "12.5"),
        ("min_average_volume", "decimal", "250"),
    )
    actual_intraday = tuple(
        (item.name, item.kind, item.value) for item in intraday.configuration.parameters
    )
    assert actual_intraday == (
        ("closeout_buffer", "timedelta", "0:600:0"),
        ("max_spread_bps", "decimal", "9"),
        ("min_average_volume", "decimal", "300"),
        ("opening_range_bars", "integer", "8"),
        ("session_end", "time", "15:30:00"),
        ("session_start", "time", "09:15:00"),
        ("session_timezone", "string", "Asia/Calcutta"),
    )


def test_canonical_configuration_rejects_unsupported_values() -> None:
    """Object reprs can include addresses and therefore cannot be stable evidence."""
    with pytest.raises(ValueError, match="canonical"):
        canonical_strategy_configuration(
            metadata=SwingBreakoutStrategy.metadata,
            parameters={"unsupported": object()},
        )


def test_result_uses_strategy_configuration_and_exact_cost_model() -> None:
    """Equivalent instances must produce the same self-derived evidence artifact."""
    first_strategy = CryptoVolatilityBreakoutStrategy(max_spread_bps=Decimal("20"))
    second_strategy = CryptoVolatilityBreakoutStrategy(max_spread_bps=Decimal("20.0"))
    costs = CostModel(fee_bps=Decimal("1"), latency=timedelta(seconds=5))
    kwargs = {
        "instrument": instrument(minimum_notional="0.01"),
        "bars": _bars("10", "11"),
        "initial_cash": Decimal("10"),
    }

    first = BacktestEngine(costs=costs).run(strategy=first_strategy, **kwargs)  # type: ignore[arg-type]
    second = BacktestEngine(costs=costs).run(strategy=second_strategy, **kwargs)  # type: ignore[arg-type]

    assert first.strategy_configuration == first_strategy.configuration
    assert first.strategy_configuration == second.strategy_configuration
    assert first.costs == costs


def test_caller_cannot_supply_or_misstate_strategy_parameters() -> None:
    """A caller-provided mapping must not be accepted as evidence about actual strategy state."""
    with pytest.raises(TypeError, match="parameters"):
        BacktestEngine().run(
            instrument=instrument(),
            bars=_bars("10"),
            strategy=SwingBreakoutStrategy(),
            initial_cash=Decimal("10"),
            parameters={"max_spread_bps": Decimal("999")},  # type: ignore[call-arg]
        )
