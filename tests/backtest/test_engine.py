"""Behavioral tests for point-in-time event-driven backtesting."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from market_sentinel.backtest.engine import BacktestEngine, CostModel, FillModel
from market_sentinel.domain.enums import Horizon, Side, SignalDirection
from market_sentinel.domain.models import Bar, Signal
from market_sentinel.strategies.base import StrategyContext, StrategyMetadata
from tests.factories import instrument


def _bars(*prices: str) -> tuple[Bar, ...]:
    start = datetime(2026, 8, 3, 14, 30, tzinfo=UTC)
    return tuple(
        Bar(
            at=start + timedelta(minutes=index),
            open=Decimal(price),
            high=Decimal(price),
            low=Decimal(price),
            close=Decimal(price),
            volume=Decimal("1000"),
        )
        for index, price in enumerate(prices)
    )


@dataclass
class _BuyThenFlat:
    metadata: StrategyMetadata = field(
        default_factory=lambda: StrategyMetadata(
            strategy_id="fixture",
            version="1.2.3",
            allowed_horizons=(Horizon.INTRADAY,),
            allowed_directions=(SignalDirection.LONG,),
            max_holding_bars=2,
        )
    )
    seen_prefixes: list[tuple[Decimal, ...]] = field(default_factory=list)

    def evaluate(self, context: StrategyContext) -> Signal | None:
        self.seen_prefixes.append(tuple(bar.close for bar in context.bars))
        if len(context.bars) != 1:
            return None
        return Signal(
            strategy_id=self.metadata.strategy_id,
            strategy_version=self.metadata.version,
            instrument_id=context.instrument_id,
            direction=SignalDirection.LONG,
            strength=Decimal("1"),
            horizon=Horizon.INTRADAY,
            entry_price=context.bars[-1].close,
            invalidation_price=context.bars[-1].close - Decimal("1"),
            take_profit=context.bars[-1].close + Decimal("5"),
            research_required=False,
            evidence_uris=(),
        )


def test_backtest_deducts_spread_slippage_and_fees() -> None:
    """Dropping any cost component would make net equity equal the frictionless result."""
    strategy = _BuyThenFlat()
    result = BacktestEngine(
        costs=CostModel(
            fee_bps=Decimal("10"),
            spread_bps=Decimal("20"),
            slippage_bps=Decimal("10"),
        )
    ).run(
        instrument=instrument(minimum_notional="0.01", price_tick="0.0001"),
        bars=_bars("10", "10", "12", "12"),
        strategy=strategy,
        initial_cash=Decimal("10"),
    )

    assert result.total_fees > Decimal("0")
    assert result.ending_equity < result.gross_ending_equity
    assert strategy.seen_prefixes == [
        (Decimal("10"),),
        (Decimal("10"), Decimal("10")),
        (Decimal("10"), Decimal("10"), Decimal("12")),
        (Decimal("10"), Decimal("10"), Decimal("12"), Decimal("12")),
    ]
    assert result.fills[0].filled_at == _bars("10", "10")[1].at
    assert result.fills[0].price == Decimal("10.0200")
    assert result.fills[0].fee == Decimal("0.000501")
    assert result.ending_equity == Decimal("10.098499")
    assert result.gross_ending_equity == Decimal("10.100000")


def test_fill_model_waits_for_the_first_event_at_or_after_exact_latency() -> None:
    """Filling merely on the next bar would ignore a latency longer than one interval."""
    result = BacktestEngine(
        costs=CostModel(latency=timedelta(seconds=90))
    ).run(
        instrument=instrument(minimum_notional="0.01"),
        bars=_bars("10", "11", "12", "13"),
        strategy=_BuyThenFlat(),
        initial_cash=Decimal("10"),
    )

    assert len(result.fills) == 1
    assert result.fills[0].filled_at == _bars("10", "11", "12")[2].at
    assert result.fills[0].price == Decimal("12")


def test_fill_model_is_stateless_and_reproducible() -> None:
    """Hidden counters or randomness would make paper and historical fills drift."""
    model = FillModel(
        costs=CostModel(
            fee_bps=Decimal("1"),
            spread_bps=Decimal("2"),
            slippage_bps=Decimal("3"),
        )
    )
    venue = instrument(price_tick="0.0001")
    kwargs = {
        "fill_id": "fill-1",
        "order_id": "order-1",
        "instrument": venue,
        "side": Side.BUY,
        "quantity": Decimal("0.25"),
        "reference_price": Decimal("10"),
        "filled_at": _bars("10")[0].at,
    }

    assert model.fill(**kwargs) == model.fill(**kwargs)  # type: ignore[arg-type]


def test_minimum_notional_rejection_proves_entries_use_shared_risk_path() -> None:
    """Directly filling a strategy signal would bypass the venue minimum enforced by sizing."""
    result = BacktestEngine().run(
        instrument=instrument(minimum_notional="2"),
        bars=_bars("10", "11"),
        strategy=_BuyThenFlat(),
        initial_cash=Decimal("10"),
    )

    assert result.fills == ()
    assert result.events[0].kind == "RISK_REJECTED"
    assert result.events[0].reason_codes == ("BELOW_MINIMUM_NOTIONAL",)


@dataclass
class _ProtectiveExitStrategy:
    metadata: StrategyMetadata = field(
        default_factory=lambda: StrategyMetadata(
            strategy_id="protective",
            version="1.0.0",
            allowed_horizons=(Horizon.SWING,),
            allowed_directions=(SignalDirection.LONG,),
            max_holding_bars=10,
        )
    )

    def evaluate(self, context: StrategyContext) -> Signal | None:
        if len(context.bars) != 1:
            return None
        return Signal(
            strategy_id="protective",
            strategy_version="1.0.0",
            instrument_id=context.instrument_id,
            direction=SignalDirection.LONG,
            strength=Decimal("1"),
            horizon=Horizon.SWING,
            entry_price=Decimal("10"),
            invalidation_price=Decimal("9"),
            take_profit=Decimal("11"),
            research_required=False,
            evidence_uris=(),
        )


def test_protective_exit_observed_after_entry_fills_only_on_following_event() -> None:
    """Using a trigger bar's price for its own exit would introduce same-bar look-ahead."""
    bars = list(_bars("10", "10", "10", "11"))
    bars[1] = bars[1].model_copy(update={"high": Decimal("20")})
    bars[2] = bars[2].model_copy(update={"high": Decimal("12")})

    result = BacktestEngine().run(
        instrument=instrument(minimum_notional="0.01"),
        bars=tuple(bars),
        strategy=_ProtectiveExitStrategy(),
        initial_cash=Decimal("10"),
    )

    assert tuple(fill.side for fill in result.fills) == (Side.BUY, Side.SELL)
    assert result.fills[1].filled_at == bars[3].at
    assert result.fills[1].price == Decimal("11")


def test_identical_inputs_produce_identical_immutable_results() -> None:
    """Mutable or nondeterministic artifacts would invalidate reproducible evidence."""
    kwargs = {
        "instrument": instrument(minimum_notional="0.01"),
        "bars": _bars("10", "10", "12"),
        "initial_cash": Decimal("10"),
        "parameters": {"window": 5, "threshold": Decimal("0.2")},
    }
    first = BacktestEngine().run(strategy=_BuyThenFlat(), **kwargs)  # type: ignore[arg-type]
    second = BacktestEngine().run(strategy=_BuyThenFlat(), **kwargs)  # type: ignore[arg-type]

    assert first == second
    assert first.parameters == (("threshold", "Decimal('0.2')"), ("window", "5"))


def test_robustness_rerun_doubles_costs_without_changing_parameters() -> None:
    """Retuning parameters during stress would make the 2x-cost gate incomparable."""
    parameters = {"window": 5, "threshold": Decimal("0.2")}
    result = BacktestEngine(
        costs=CostModel(
            fee_bps=Decimal("5"),
            spread_bps=Decimal("6"),
            slippage_bps=Decimal("7"),
            latency=timedelta(seconds=10),
        )
    ).run_robustness(
        instrument=instrument(minimum_notional="0.01", price_tick="0.0001"),
        bars=_bars("10", "10", "12"),
        strategy=_BuyThenFlat(),
        initial_cash=Decimal("10"),
        parameters=parameters,
    )

    assert result.base.parameters == result.stressed.parameters
    assert result.stressed_costs == CostModel(
        fee_bps=Decimal("10"),
        spread_bps=Decimal("12"),
        slippage_bps=Decimal("14"),
        latency=timedelta(seconds=10),
    )
    assert result.stressed.ending_equity < result.base.ending_equity


def test_completed_trade_records_net_after_cost_pnl_once() -> None:
    """Omitting or double-counting either fill fee would corrupt profit factor."""
    bars = list(_bars("10", "10", "10", "11"))
    bars[2] = bars[2].model_copy(update={"high": Decimal("12")})
    result = BacktestEngine(
        costs=CostModel(fee_bps=Decimal("10"))
    ).run(
        instrument=instrument(minimum_notional="0.01"),
        bars=tuple(bars),
        strategy=_ProtectiveExitStrategy(),
        initial_cash=Decimal("10"),
    )

    trade = result.completed_trades[0]
    assert trade.net_pnl == Decimal("0.04895")
    assert result.traded_notional == Decimal("1.05")
    assert result.exposed_periods == 2


@dataclass
class _SessionCloseStrategy(_ProtectiveExitStrategy):
    metadata: StrategyMetadata = field(
        default_factory=lambda: StrategyMetadata(
            strategy_id="protective",
            version="1.0.0",
            allowed_horizons=(Horizon.INTRADAY,),
            allowed_directions=(SignalDirection.LONG,),
            mandatory_preclose_closeout=True,
            preclose_buffer=timedelta(minutes=5),
        )
    )


def test_session_boundary_submits_deterministic_next_event_closeout() -> None:
    """Retaining an intraday position across a known session boundary violates its metadata."""
    bars = list(_bars("10", "10", "11"))
    bars[2] = bars[2].model_copy(update={"at": bars[2].at + timedelta(days=1)})

    result = BacktestEngine().run(
        instrument=instrument(minimum_notional="0.01"),
        bars=tuple(bars),
        strategy=_SessionCloseStrategy(),
        initial_cash=Decimal("10"),
    )

    assert tuple(fill.side for fill in result.fills) == (Side.BUY, Side.SELL)
    assert any(event.kind == "SESSION_CLOSEOUT" for event in result.events)
