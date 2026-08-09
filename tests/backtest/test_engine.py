"""Behavioral tests for point-in-time event-driven backtesting."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from market_sentinel.backtest.engine import BacktestEngine, CostModel, FillModel
from market_sentinel.domain.enums import Horizon, OrderType, Side, SignalDirection
from market_sentinel.domain.models import Bar, Instrument, MarketSnapshot, OrderIntent, Signal
from market_sentinel.execution.paper import PaperBroker
from market_sentinel.strategies.base import (
    StrategyConfiguration,
    StrategyContext,
    StrategyMetadata,
    canonical_strategy_configuration,
)
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

    @property
    def configuration(self) -> StrategyConfiguration:
        return canonical_strategy_configuration(metadata=self.metadata, parameters={})

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
        bars=_bars("10", "9.9", "12", "12"),
        strategy=strategy,
        initial_cash=Decimal("10"),
    )

    assert result.total_fees > Decimal("0")
    assert result.ending_equity < result.gross_ending_equity
    assert strategy.seen_prefixes == [
        (Decimal("10"),),
        (Decimal("10"), Decimal("9.9")),
        (Decimal("10"), Decimal("9.9"), Decimal("12")),
        (Decimal("10"), Decimal("9.9"), Decimal("12"), Decimal("12")),
    ]
    assert result.fills[0].filled_at == _bars("10", "9.9")[1].at
    assert result.fills[0].price == Decimal("9.9198")
    assert result.fills[0].fee == Decimal("0.000495990")
    assert result.ending_equity == Decimal("10.103514010")
    assert result.gross_ending_equity == Decimal("10.105000000")


def test_fill_model_waits_for_the_first_event_at_or_after_exact_latency() -> None:
    """Filling merely on the next bar would ignore a latency longer than one interval."""
    result = BacktestEngine(
        costs=CostModel(latency=timedelta(seconds=90))
    ).run(
        instrument=instrument(minimum_notional="0.01"),
        bars=_bars("10", "11", "9.5", "10"),
        strategy=_BuyThenFlat(),
        initial_cash=Decimal("10"),
    )

    assert len(result.fills) == 1
    assert result.fills[0].filled_at == _bars("10", "11", "12")[2].at
    assert result.fills[0].price == Decimal("9.5")


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
        "submitted_at": _bars("10")[0].at - timedelta(minutes=1),
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

    @property
    def configuration(self) -> StrategyConfiguration:
        return canonical_strategy_configuration(metadata=self.metadata, parameters={})

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
    }
    first = BacktestEngine().run(strategy=_BuyThenFlat(), **kwargs)  # type: ignore[arg-type]
    second = BacktestEngine().run(strategy=_BuyThenFlat(), **kwargs)  # type: ignore[arg-type]

    assert first == second
    assert first.strategy_configuration == _BuyThenFlat().configuration
    assert first.costs == CostModel()


def test_robustness_rerun_doubles_costs_without_changing_parameters() -> None:
    """Retuning parameters during stress would make the 2x-cost gate incomparable."""
    result = BacktestEngine(
        costs=CostModel(
            fee_bps=Decimal("5"),
            spread_bps=Decimal("6"),
            slippage_bps=Decimal("7"),
            latency=timedelta(seconds=10),
        )
    ).run_robustness(
        instrument=instrument(minimum_notional="0.01", price_tick="0.0001"),
        bars=_bars("10", "9.5", "12"),
        strategy=_BuyThenFlat(),
        initial_cash=Decimal("10"),
    )

    assert result.base.strategy_configuration == result.stressed.strategy_configuration
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


def test_gap_up_never_converts_approved_buy_limit_into_market_fill() -> None:
    """The former loop turned a $0.50 approval into a $50 leveraged fill at the next open."""
    result = BacktestEngine().run(
        instrument=instrument(minimum_notional="0.01"),
        bars=_bars("10", "1000"),
        strategy=_BuyThenFlat(),
        initial_cash=Decimal("10"),
    )

    assert result.fills == ()
    assert result.ending_equity == Decimal("10")


def test_unexecutable_limit_remains_pending_until_later_event_can_fill() -> None:
    """Dropping a skipped pending order would lose a later valid limit execution."""
    bars = _bars("10", "1000", "9.5")
    result = BacktestEngine().run(
        instrument=instrument(minimum_notional="0.01"),
        bars=bars,
        strategy=_BuyThenFlat(),
        initial_cash=Decimal("10"),
    )

    assert len(result.fills) == 1
    assert result.fills[0].filled_at == bars[2].at
    assert result.fills[0].price == Decimal("9.5")


def test_entry_does_not_fill_when_fee_would_exceed_available_cash() -> None:
    """Checking approved notional without all-in fees can still make ledger cash negative."""
    result = BacktestEngine(costs=CostModel(fee_bps=Decimal("200000"))).run(
        instrument=instrument(minimum_notional="0.01"),
        bars=_bars("10", "10"),
        strategy=_BuyThenFlat(),
        initial_cash=Decimal("10"),
    )

    assert result.fills == ()
    assert result.ending_equity == Decimal("10")


def test_entry_does_not_fill_below_its_protective_stop() -> None:
    """A gap below the approved stop would invert protection and increase actual risk."""
    result = BacktestEngine().run(
        instrument=instrument(minimum_notional="0.01"),
        bars=_bars("10", "8"),
        strategy=_ProtectiveExitStrategy(),
        initial_cash=Decimal("10"),
    )

    assert result.fills == ()


def test_fill_model_public_latency_rule_rejects_predeadline_and_accepts_deadline() -> None:
    """Paper callers must use the same latency eligibility rule as the backtest loop."""
    submitted = _bars("10")[0].at
    model = FillModel(costs=CostModel(latency=timedelta(seconds=30)))

    assert (
        model.can_fill(submitted_at=submitted, event_at=submitted + timedelta(seconds=29))
        is False
    )
    assert (
        model.can_fill(submitted_at=submitted, event_at=submitted + timedelta(seconds=30))
        is True
    )
    with pytest.raises(ValueError, match="latency"):
        model.fill(
            fill_id="early",
            order_id="order",
            instrument=instrument(),
            side=Side.BUY,
            quantity=Decimal("0.05"),
            reference_price=Decimal("10"),
            submitted_at=submitted,
            filled_at=submitted + timedelta(seconds=29),
        )


def test_fill_model_allocates_conservative_step_aligned_event_liquidity() -> None:
    """Paper and replay callers must share one conservative partial-quantity rule."""
    model = FillModel()

    assert model.liquidity_budget(
        event_volume=Decimal("1.27"),
        max_participation=Decimal("0.5"),
        quantity_step=Decimal("0.1"),
    ) == Decimal("0.6")
    assert model.allocate_quantity(
        remaining_quantity=Decimal("1"),
        available_liquidity=Decimal("0.35"),
        quantity_step=Decimal("0.1"),
    ) == Decimal("0.3")


@pytest.mark.parametrize(
    ("remaining", "liquidity", "step"),
    [
        (Decimal("NaN"), Decimal("1"), Decimal("0.1")),
        (Decimal("1"), Decimal("-1"), Decimal("0.1")),
        (Decimal("1"), Decimal("1"), Decimal("0")),
    ],
)
def test_fill_model_liquidity_allocation_rejects_invalid_decimals(
    remaining: Decimal,
    liquidity: Decimal,
    step: Decimal,
) -> None:
    with pytest.raises(ValueError):
        FillModel().allocate_quantity(
            remaining_quantity=remaining,
            available_liquidity=liquidity,
            quantity_step=step,
        )


@pytest.mark.parametrize(
    ("quantity", "reference_price", "venue", "side", "filled_at"),
    [
        (Decimal("NaN"), Decimal("10"), instrument(), Side.BUY, _bars("10")[0].at),
        (Decimal("Infinity"), Decimal("10"), instrument(), Side.BUY, _bars("10")[0].at),
        (
            Decimal("0.051"),
            Decimal("10"),
            instrument(quantity_step="0.01"),
            Side.BUY,
            _bars("10")[0].at,
        ),
        (Decimal("0.05"), Decimal("NaN"), instrument(), Side.BUY, _bars("10")[0].at),
        (
            Decimal("0.05"),
            Decimal("10"),
            instrument().model_copy(update={"price_tick": Decimal("0")}),
            Side.BUY,
            _bars("10")[0].at,
        ),
        (
            Decimal("0.05"),
            Decimal("10"),
            instrument(price_tick="20"),
            Side.SELL,
            _bars("10")[0].at,
        ),
        (Decimal("0.05"), Decimal("10"), instrument(), "buy", _bars("10")[0].at),
        (
            Decimal("0.05"),
            Decimal("10"),
            instrument(),
            Side.BUY,
            datetime(2026, 8, 3, 14, 31),
        ),
    ],
)
def test_fill_model_rejects_invalid_boundary_values(
    quantity: Decimal,
    reference_price: Decimal,
    venue: Instrument,
    side: Side | str,
    filled_at: datetime,
) -> None:
    """Noncanonical public fill inputs must fail before producing a domain Fill."""
    submitted = _bars("10")[0].at
    with pytest.raises(ValueError):
        FillModel().fill(
            fill_id="invalid",
            order_id="order",
            instrument=venue,
            side=side,  # type: ignore[arg-type]
            quantity=quantity,
            reference_price=reference_price,
            submitted_at=submitted,
            filled_at=filled_at,
        )


def test_fill_model_rejects_sell_impact_at_or_above_full_price() -> None:
    """A sell multiplier at zero or below cannot produce a valid execution price."""
    with pytest.raises(ValueError, match="impact"):
        FillModel(costs=CostModel(slippage_bps=Decimal("10000"))).fill(
            fill_id="invalid-sell",
            order_id="order",
            instrument=instrument(),
            side=Side.SELL,
            quantity=Decimal("0.05"),
            reference_price=Decimal("10"),
            submitted_at=_bars("10")[0].at,
            filled_at=_bars("10", "10")[1].at,
        )


def test_fill_model_rejects_event_before_submission() -> None:
    """Reverse timestamp ordering is invalid independently of the configured latency."""
    submitted = _bars("10")[0].at
    with pytest.raises(ValueError, match="precede"):
        FillModel().can_fill(
            submitted_at=submitted,
            event_at=submitted - timedelta(microseconds=1),
        )


@pytest.mark.parametrize(
    ("submitted", "thirty_minutes", "sixty_minutes"),
    [
        (
            datetime(2026, 3, 8, 1, 30, tzinfo=ZoneInfo("America/New_York")),
            datetime(2026, 3, 8, 3, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2026, 3, 8, 3, 30, tzinfo=ZoneInfo("America/New_York")),
        ),
        (
            datetime(2026, 11, 1, 1, 30, fold=0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2026, 11, 1, 1, 0, fold=1, tzinfo=ZoneInfo("America/New_York")),
            datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=ZoneInfo("America/New_York")),
        ),
    ],
)
def test_fill_model_latency_uses_absolute_elapsed_time_across_dst_folds(
    submitted: datetime,
    thirty_minutes: datetime,
    sixty_minutes: datetime,
) -> None:
    """Same-zone wall-clock arithmetic is wrong across spring and fall DST transitions."""
    model = FillModel(costs=CostModel(latency=timedelta(minutes=60)))

    assert model.can_fill(submitted_at=submitted, event_at=thirty_minutes) is False
    assert model.can_fill(submitted_at=submitted, event_at=sixty_minutes) is True
    with pytest.raises(ValueError, match="latency"):
        model.fill(
            fill_id="too-early",
            order_id="order",
            instrument=instrument(),
            side=Side.BUY,
            quantity=Decimal("0.05"),
            reference_price=Decimal("10"),
            submitted_at=submitted,
            filled_at=thirty_minutes,
        )
    exact = model.fill(
        fill_id="exact",
        order_id="order",
        instrument=instrument(),
        side=Side.BUY,
        quantity=Decimal("0.05"),
        reference_price=Decimal("10"),
        submitted_at=submitted,
        filled_at=sixty_minutes,
    )
    assert exact.filled_at == sixty_minutes.astimezone(UTC)


def test_sell_fill_applies_adverse_spread_slippage_and_floor_rounding() -> None:
    """Rounding a reducing sell upward would understate adverse execution cost."""
    bars = _bars("10", "10")
    fill = FillModel(
        costs=CostModel(spread_bps=Decimal("20"), slippage_bps=Decimal("10"))
    ).fill(
        fill_id="sell",
        order_id="order",
        instrument=instrument(price_tick="0.01", quantity_step="0.01"),
        side=Side.SELL,
        quantity=Decimal("0.05"),
        reference_price=Decimal("10.007"),
        submitted_at=bars[0].at,
        filled_at=bars[1].at,
    )

    assert fill.price == Decimal("9.98")


@dataclass
class _StopLossStrategy(_ProtectiveExitStrategy):
    pass


def test_stop_loss_fills_on_event_after_trigger() -> None:
    """A stop observed in one completed bar must not fill using that same bar."""
    bars = list(_bars("10", "10", "10", "8.5"))
    bars[2] = bars[2].model_copy(update={"low": Decimal("8")})
    result = BacktestEngine().run(
        instrument=instrument(minimum_notional="0.01"),
        bars=tuple(bars),
        strategy=_StopLossStrategy(),
        initial_cash=Decimal("10"),
    )

    assert result.fills[1].side is Side.SELL
    assert result.fills[1].filled_at == bars[3].at


@dataclass
class _TimedExitStrategy(_ProtectiveExitStrategy):
    metadata: StrategyMetadata = field(
        default_factory=lambda: StrategyMetadata(
            strategy_id="protective",
            version="1.0.0",
            allowed_horizons=(Horizon.SWING,),
            allowed_directions=(SignalDirection.LONG,),
            max_holding_bars=1,
        )
    )


def test_max_holding_exit_fills_on_following_event() -> None:
    """A holding-period boundary must submit first and execute only on a later event."""
    bars = _bars("10", "10", "10", "11")
    result = BacktestEngine().run(
        instrument=instrument(minimum_notional="0.01"),
        bars=bars,
        strategy=_TimedExitStrategy(),
        initial_cash=Decimal("10"),
    )

    assert result.fills[1].side is Side.SELL
    assert result.fills[1].filled_at == bars[3].at


@dataclass
class _NeverTrade:
    metadata: StrategyMetadata = field(
        default_factory=lambda: StrategyMetadata(
            strategy_id="never",
            version="1",
            allowed_horizons=(Horizon.SWING,),
            allowed_directions=(SignalDirection.LONG,),
        )
    )

    @property
    def configuration(self) -> StrategyConfiguration:
        return canonical_strategy_configuration(metadata=self.metadata, parameters={})

    def evaluate(self, context: StrategyContext) -> None:
        del context
        return None


@pytest.mark.parametrize("initial_cash", [Decimal("NaN"), Decimal("Infinity")])
def test_engine_rejects_nonfinite_initial_cash(initial_cash: Decimal) -> None:
    """Nonfinite capital must fail before ledger construction or benchmark arithmetic."""
    with pytest.raises(ValueError, match="initial_cash"):
        BacktestEngine().run(
            instrument=instrument(),
            bars=_bars("10"),
            strategy=_NeverTrade(),
            initial_cash=initial_cash,
        )


@pytest.mark.parametrize(
    "bad_bar",
    [
        _bars("10")[0].model_copy(update={"at": datetime(2026, 8, 3, 14, 30)}),
        _bars("10")[0].model_copy(update={"open": Decimal("NaN")}),
        _bars("10")[0].model_copy(update={"low": Decimal("11")}),
        _bars("10")[0].model_copy(update={"volume": Decimal("-1")}),
    ],
)
def test_engine_rejects_invalid_bar_boundaries(bad_bar: Bar) -> None:
    """Bypassed Pydantic construction must not inject invalid OHLCV into public replay."""
    with pytest.raises(ValueError, match="bars"):
        BacktestEngine().run(
            instrument=instrument(),
            bars=(bad_bar,),
            strategy=_NeverTrade(),
            initial_cash=Decimal("10"),
        )


@dataclass
class _PartialTradeStrategy(_ProtectiveExitStrategy):
    metadata: StrategyMetadata = field(
        default_factory=lambda: StrategyMetadata(
            strategy_id="partial",
            version="1.0.0",
            allowed_horizons=(Horizon.SWING,),
            allowed_directions=(SignalDirection.LONG,),
            max_holding_bars=4,
        )
    )

    def evaluate(self, context: StrategyContext) -> Signal | None:
        value = super().evaluate(context)
        return None if value is None else value.model_copy(
            update={
                "strategy_id": self.metadata.strategy_id,
                "strategy_version": self.metadata.version,
            }
        )


def test_backtest_carries_partial_entry_and_exit_across_event_liquidity() -> None:
    """Dropping a pending residual would underfill entry or overstate a partial exit."""
    bars = list(_bars("10", "10", "10", "10", "10", "10", "10", "10"))
    volumes = ("1000", "0.02", "0.02", "0.01", "0", "0", "0.02", "0.03")
    bars = [
        bar.model_copy(update={"volume": Decimal(volume)})
        for bar, volume in zip(bars, volumes, strict=True)
    ]

    result = BacktestEngine(max_volume_participation=Decimal("1")).run(
        instrument=instrument(minimum_notional="0.01", quantity_step="0.01"),
        bars=tuple(bars),
        strategy=_PartialTradeStrategy(),
        initial_cash=Decimal("10"),
    )

    assert [(fill.side, fill.quantity) for fill in result.fills] == [
        (Side.BUY, Decimal("0.02")),
        (Side.BUY, Decimal("0.02")),
        (Side.BUY, Decimal("0.01")),
        (Side.SELL, Decimal("0.02")),
        (Side.SELL, Decimal("0.03")),
    ]
    assert len(result.completed_trades) == 1
    assert result.completed_trades[0].quantity == Decimal("0.05")
    assert result.completed_trades[0].entry_price == Decimal("10")
    assert result.completed_trades[0].exit_price == Decimal("10")


def test_zero_volume_keeps_pending_entry_for_a_later_liquid_event() -> None:
    """A zero-volume bar must neither fabricate a fill nor discard the remaining order."""
    bars = list(_bars("10", "10", "10"))
    bars[1] = bars[1].model_copy(update={"volume": Decimal("0")})
    bars[2] = bars[2].model_copy(update={"volume": Decimal("0.05")})

    result = BacktestEngine().run(
        instrument=instrument(minimum_notional="0.01", quantity_step="0.01"),
        bars=tuple(bars),
        strategy=_BuyThenFlat(),
        initial_cash=Decimal("10"),
    )

    assert len(result.fills) == 1
    assert result.fills[0].quantity == Decimal("0.05")
    assert result.fills[0].filled_at == bars[2].at


def test_protective_trigger_cancels_unfilled_entry_before_partial_position_exit() -> None:
    """A stop on a partial position must not add more entry exposure on its trigger bar."""
    bars = list(_bars("10", "10", "10", "8.5"))
    bars[1] = bars[1].model_copy(update={"volume": Decimal("0.02")})
    bars[2] = bars[2].model_copy(
        update={"low": Decimal("8"), "volume": Decimal("1")}
    )
    bars[3] = bars[3].model_copy(update={"volume": Decimal("0.02")})

    result = BacktestEngine().run(
        instrument=instrument(minimum_notional="0.01", quantity_step="0.01"),
        bars=tuple(bars),
        strategy=_ProtectiveExitStrategy(),
        initial_cash=Decimal("10"),
    )

    assert [(fill.side, fill.quantity) for fill in result.fills] == [
        (Side.BUY, Decimal("0.02")),
        (Side.SELL, Decimal("0.02")),
    ]
    assert len(result.completed_trades) == 1
    assert result.completed_trades[0].quantity == Decimal("0.02")


def test_paper_and_backtest_share_identical_partial_fill_cost_behavior() -> None:
    """Equal order, event, costs, and participation must produce equal execution facts."""
    bars = list(_bars("10", "9.9"))
    bars[1] = bars[1].model_copy(update={"volume": Decimal("0.04")})
    venue = instrument(
        minimum_notional="0.01",
        quantity_step="0.01",
        price_tick="0.0001",
    )
    costs = CostModel(
        fee_bps=Decimal("10"),
        spread_bps=Decimal("20"),
        slippage_bps=Decimal("10"),
    )
    backtest = BacktestEngine(
        costs=costs,
        max_volume_participation=Decimal("0.5"),
    ).run(
        instrument=venue,
        bars=tuple(bars),
        strategy=_BuyThenFlat(),
        initial_cash=Decimal("10"),
    )
    initial = MarketSnapshot(
        instrument_id="AAPL@alpaca",
        observed_at=bars[0].at,
        source_at=bars[0].at,
        bars=(bars[0],),
        provider="backtest",
        max_age_seconds=0,
    )
    paper = PaperBroker(
        fill_model=FillModel(costs=costs),
        starting_cash=Decimal("10"),
        max_volume_participation=Decimal("0.5"),
    )
    paper.submit(
        OrderIntent(
            intent_id="parity-intent",
            instrument_id="AAPL@alpaca",
            side=Side.BUY,
            quantity=Decimal("0.05"),
            notional=None,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("10"),
            stop_loss=Decimal("9"),
            take_profit=Decimal("15"),
            time_in_force="day",
            product="cash",
            session="regular",
            snapshot_hash="a" * 64,
            created_at=bars[0].at,
            expires_at=bars[0].at + timedelta(minutes=10),
        ),
        initial,
    )
    paper_event = initial.model_copy(
        update={
            "observed_at": bars[1].at,
            "source_at": bars[1].at,
            "bars": tuple(bars),
        }
    )
    [paper_fill] = paper.on_snapshot(paper_event, venue)
    [backtest_fill] = backtest.fills

    assert (paper_fill.quantity, paper_fill.price, paper_fill.fee) == (
        backtest_fill.quantity,
        backtest_fill.price,
        backtest_fill.fee,
    )


def test_robustness_rerun_preserves_volume_participation() -> None:
    """The cost stress must not silently restore unlimited event liquidity."""
    bars = list(_bars("10", "10"))
    bars[1] = bars[1].model_copy(update={"volume": Decimal("0.02")})

    result = BacktestEngine(
        costs=CostModel(fee_bps=Decimal("1")),
        max_volume_participation=Decimal("0.5"),
    ).run_robustness(
        instrument=instrument(minimum_notional="0.01", quantity_step="0.01"),
        bars=tuple(bars),
        strategy=_BuyThenFlat(),
        initial_cash=Decimal("10"),
    )

    assert result.base.fills[0].quantity == Decimal("0.01")
    assert result.stressed.fills[0].quantity == Decimal("0.01")
