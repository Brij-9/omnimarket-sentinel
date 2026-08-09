"""Point-in-time event simulation through the production risk and ledger contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from zoneinfo import ZoneInfo

from market_sentinel.domain.enums import Side
from market_sentinel.domain.models import Bar, Fill, Instrument, MarketSnapshot, OrderIntent
from market_sentinel.portfolio.ledger import PortfolioLedger
from market_sentinel.risk.engine import PositionSizer, RiskEngine, portfolio_hash
from market_sentinel.risk.policy import RiskPolicy
from market_sentinel.strategies.base import (
    Strategy,
    StrategyConfiguration,
    StrategyContext,
    StrategyMetadata,
)

_BPS = Decimal("10000")


def _require_nonnegative_decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value < Decimal("0"):
        raise ValueError(f"{name} must be a finite nonnegative Decimal")
    return value


@dataclass(frozen=True, slots=True)
class CostModel:
    """Provider-neutral deterministic execution-cost assumptions."""

    fee_bps: Decimal = Decimal("0")
    spread_bps: Decimal = Decimal("0")
    slippage_bps: Decimal = Decimal("0")
    latency: timedelta = timedelta(0)

    def __post_init__(self) -> None:
        _require_nonnegative_decimal(self.fee_bps, "fee_bps")
        _require_nonnegative_decimal(self.spread_bps, "spread_bps")
        _require_nonnegative_decimal(self.slippage_bps, "slippage_bps")
        if not isinstance(self.latency, timedelta) or self.latency < timedelta(0):
            raise ValueError("latency must be a nonnegative timedelta")

    def stressed(self, multiplier: Decimal = Decimal("2")) -> CostModel:
        """Scale monetary friction assumptions while retaining identical latency."""
        if not isinstance(multiplier, Decimal) or not multiplier.is_finite() or multiplier <= 0:
            raise ValueError("cost multiplier must be a positive finite Decimal")
        return CostModel(
            fee_bps=self.fee_bps * multiplier,
            spread_bps=self.spread_bps * multiplier,
            slippage_bps=self.slippage_bps * multiplier,
            latency=self.latency,
        )


@dataclass(frozen=True, slots=True)
class BacktestEvent:
    """Immutable audit event emitted by the simulation."""

    at: datetime
    kind: str
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """One after-cost or benchmark valuation."""

    at: datetime
    value: Decimal


@dataclass(frozen=True, slots=True)
class CompletedTrade:
    """One closed long trade with realized after-cost PnL."""

    opened_at: datetime
    closed_at: datetime
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    fees: Decimal
    net_pnl: Decimal


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Immutable, reproducible evidence artifact."""

    strategy_id: str
    strategy_version: str
    strategy_configuration: StrategyConfiguration
    costs: CostModel
    data_cutoff: datetime
    events: tuple[BacktestEvent, ...]
    fills: tuple[Fill, ...]
    equity_curve: tuple[EquityPoint, ...]
    benchmark_curve: tuple[EquityPoint, ...]
    initial_cash: Decimal
    ending_equity: Decimal
    gross_ending_equity: Decimal
    total_fees: Decimal
    completed_trades: tuple[CompletedTrade, ...]
    traded_notional: Decimal
    exposed_periods: int


@dataclass(frozen=True, slots=True)
class RobustnessResult:
    """Comparable base and 2x-cost artifacts produced with identical parameters."""

    base: BacktestResult
    stressed: BacktestResult
    stressed_costs: CostModel


class FillModel:
    """Stateless deterministic fill rules reusable by backtests and paper brokers."""

    def __init__(self, *, costs: CostModel | None = None) -> None:
        self.costs = CostModel() if costs is None else costs

    def fill(
        self,
        *,
        fill_id: str,
        order_id: str,
        instrument: Instrument,
        side: Side,
        quantity: Decimal,
        reference_price: Decimal,
        submitted_at: datetime,
        filled_at: datetime,
    ) -> Fill:
        """Create one full fill with adverse half-spread and slippage exactly once."""
        if not _finite_positive(quantity):
            raise ValueError("quantity must be a positive Decimal")
        if not _finite_positive(reference_price):
            raise ValueError("reference_price must be a positive Decimal")
        if not isinstance(side, Side):
            raise ValueError("side must be a Side")
        if not isinstance(instrument, Instrument):
            raise ValueError("instrument must be an Instrument")
        if not _finite_positive(instrument.price_tick) or not _finite_positive(
            instrument.quantity_step
        ):
            raise ValueError("instrument precision must be finite and positive")
        if not _step_aligned(quantity, instrument.quantity_step):
            raise ValueError("quantity must align with the instrument quantity step")
        if not self.can_fill(submitted_at=submitted_at, event_at=filled_at):
            raise ValueError("fill event occurs before the latency deadline")
        impact_bps = self.costs.spread_bps / Decimal("2") + self.costs.slippage_bps
        if side is Side.SELL and impact_bps >= _BPS:
            raise ValueError("sell execution impact must be below 10000 bps")
        multiplier = (
            Decimal("1") + impact_bps / _BPS
            if side is Side.BUY
            else Decimal("1") - impact_bps / _BPS
        )
        try:
            raw_price = reference_price * multiplier
            rounding = ROUND_CEILING if side is Side.BUY else ROUND_FLOOR
            price = (
                (raw_price / instrument.price_tick).to_integral_value(rounding=rounding)
                * instrument.price_tick
            )
            fee = quantity * price * self.costs.fee_bps / _BPS
        except (InvalidOperation, OverflowError, ZeroDivisionError) as error:
            raise ValueError("fill arithmetic must remain finite") from error
        if not _finite_positive(price):
            raise ValueError("rounded fill price must be finite and positive")
        if not _finite_nonnegative(fee):
            raise ValueError("fill fee must be finite and nonnegative")
        return Fill(
            fill_id=fill_id,
            order_id=order_id,
            instrument_id=f"{instrument.symbol}@{instrument.venue}",
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            filled_at=filled_at,
        )

    def can_fill(self, *, submitted_at: datetime, event_at: datetime) -> bool:
        """Return latency eligibility after validating aware chronological timestamps."""
        submitted_instant = _require_aware_datetime(
            submitted_at, "submitted_at"
        ).astimezone(UTC)
        event_instant = _require_aware_datetime(event_at, "event_at").astimezone(UTC)
        if event_instant < submitted_instant:
            raise ValueError("event_at must not precede submitted_at")
        return event_instant >= submitted_instant + self.costs.latency

    def liquidity_budget(
        self,
        *,
        event_volume: Decimal,
        max_participation: Decimal,
        quantity_step: Decimal,
    ) -> Decimal:
        """Return the conservative step-aligned quantity available to one market event."""
        if not _finite_nonnegative(event_volume):
            raise ValueError("event_volume must be a finite nonnegative Decimal")
        if (
            not _finite_positive(max_participation)
            or max_participation > Decimal("1")
        ):
            raise ValueError("max_participation must be a Decimal in (0, 1]")
        if not _finite_positive(quantity_step):
            raise ValueError("quantity_step must be a finite positive Decimal")
        return _floor_quantity(event_volume * max_participation, quantity_step)

    def allocate_quantity(
        self,
        *,
        remaining_quantity: Decimal,
        available_liquidity: Decimal,
        quantity_step: Decimal,
    ) -> Decimal:
        """Allocate at most one shared event budget without exceeding remaining quantity."""
        if not _finite_nonnegative(remaining_quantity):
            raise ValueError("remaining_quantity must be a finite nonnegative Decimal")
        if not _finite_nonnegative(available_liquidity):
            raise ValueError("available_liquidity must be a finite nonnegative Decimal")
        if not _finite_positive(quantity_step):
            raise ValueError("quantity_step must be a finite positive Decimal")
        return _floor_quantity(
            min(remaining_quantity, available_liquidity),
            quantity_step,
        )


@dataclass(frozen=True, slots=True)
class _PendingOrder:
    order_id: str
    side: Side
    quantity: Decimal
    submitted_index: int
    submitted_at: datetime
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    limit_price: Decimal | None = None
    approved_notional: Decimal | None = None
    approved_stop_risk: Decimal | None = None


@dataclass(frozen=True, slots=True)
class _OpenTrade:
    quantity: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    opened_index: int
    entry_fill: Fill


class BacktestEngine:
    """Run a strategy against bar-close prefixes and next-event fills."""

    def __init__(
        self,
        *,
        costs: CostModel | None = None,
        fill_model: FillModel | None = None,
        risk_policy: RiskPolicy | None = None,
    ) -> None:
        if costs is not None and fill_model is not None:
            raise ValueError("provide costs or fill_model, not both")
        self.fill_model = FillModel(costs=costs) if fill_model is None else fill_model
        self.risk_policy = RiskPolicy.safe_defaults() if risk_policy is None else risk_policy

    def run(
        self,
        *,
        instrument: Instrument,
        bars: tuple[Bar, ...],
        strategy: Strategy,
        initial_cash: Decimal,
    ) -> BacktestResult:
        """Evaluate only prefixes and route every long entry through shared risk approval."""
        _validate_bars(bars)
        if not _finite_positive(initial_cash):
            raise ValueError("initial_cash must be a positive Decimal")
        metadata = getattr(strategy, "metadata", None)
        if not isinstance(metadata, StrategyMetadata):
            raise ValueError("strategy must expose StrategyMetadata")
        configuration = getattr(strategy, "configuration", None)
        if not isinstance(configuration, StrategyConfiguration) or (
            configuration.strategy_id != metadata.strategy_id
            or configuration.strategy_version != metadata.version
        ):
            raise ValueError("strategy must expose matching canonical configuration")

        instrument_id = f"{instrument.symbol}@{instrument.venue}"
        horizon = metadata.allowed_horizons[0]
        ledger = PortfolioLedger(starting_cash=initial_cash, currency=instrument.quote_currency)
        sizer = PositionSizer(policy=self.risk_policy)
        risk = RiskEngine(policy=self.risk_policy)
        pending: _PendingOrder | None = None
        open_trade: _OpenTrade | None = None
        fills: list[Fill] = []
        completed_trades: list[CompletedTrade] = []
        events: list[BacktestEvent] = []
        equity_curve: list[EquityPoint] = []
        execution_impact = Decimal("0")
        exposed_periods = 0

        for index, bar in enumerate(bars):
            if (
                pending is not None
                and index > pending.submitted_index
                and self.fill_model.can_fill(
                    submitted_at=pending.submitted_at,
                    event_at=bar.at,
                )
            ):
                fill = _create_pending_fill(
                    pending=pending,
                    bar=bar,
                    fill_model=self.fill_model,
                    instrument=instrument,
                    fill_id=f"backtest-fill-{len(fills) + 1}",
                    available_cash=ledger.cash,
                )
                if fill is None:
                    events.append(BacktestEvent(at=bar.at, kind="ORDER_NOT_EXECUTABLE"))
                else:
                    ledger.apply_fill(fill)
                    execution_impact += (
                        fill.quantity * (fill.price - bar.open)
                        if fill.side is Side.BUY
                        else fill.quantity * (bar.open - fill.price)
                    )
                    fills.append(fill)
                    events.append(BacktestEvent(at=bar.at, kind="FILL"))
                    if fill.side is Side.BUY:
                        assert pending.stop_loss is not None and pending.take_profit is not None
                        open_trade = _OpenTrade(
                            quantity=fill.quantity,
                            stop_loss=pending.stop_loss,
                            take_profit=pending.take_profit,
                            opened_index=index,
                            entry_fill=fill,
                        )
                    else:
                        assert open_trade is not None
                        fees = open_trade.entry_fill.fee + fill.fee
                        completed_trades.append(
                            CompletedTrade(
                                opened_at=open_trade.entry_fill.filled_at,
                                closed_at=fill.filled_at,
                                quantity=fill.quantity,
                                entry_price=open_trade.entry_fill.price,
                                exit_price=fill.price,
                                fees=fees,
                                net_pnl=(fill.price - open_trade.entry_fill.price)
                                * fill.quantity
                                - fees,
                            )
                        )
                        open_trade = None
                    pending = None

            snapshot = ledger.mark({instrument_id: bar.close}, bar.at)
            if snapshot.positions:
                exposed_periods += 1
            equity_curve.append(EquityPoint(at=bar.at, value=snapshot.equity))
            if open_trade is not None and pending is None and index > open_trade.opened_index:
                exit_reason: str | None = None
                if bar.low <= open_trade.stop_loss:
                    exit_reason = "STOP_LOSS"
                elif bar.high >= open_trade.take_profit:
                    exit_reason = "TAKE_PROFIT"
                elif (
                    metadata.max_holding_bars is not None
                    and index - open_trade.opened_index >= metadata.max_holding_bars
                ):
                    exit_reason = "MAX_HOLDING_BARS"
                if exit_reason is not None:
                    pending = _PendingOrder(
                        order_id=f"exit:{metadata.strategy_id}:{bar.at.isoformat()}",
                        side=Side.SELL,
                        quantity=open_trade.quantity,
                        submitted_index=index,
                        submitted_at=bar.at,
                    )
                    events.append(BacktestEvent(at=bar.at, kind=exit_reason))
            if (
                open_trade is not None
                and pending is None
                and metadata.mandatory_preclose_closeout
                and index + 1 < len(bars)
                and bar.at.astimezone(ZoneInfo(instrument.timezone)).date()
                != bars[index + 1].at.astimezone(ZoneInfo(instrument.timezone)).date()
            ):
                pending = _PendingOrder(
                    order_id=f"session-exit:{metadata.strategy_id}:{bar.at.isoformat()}",
                    side=Side.SELL,
                    quantity=open_trade.quantity,
                    submitted_index=index,
                    submitted_at=bar.at,
                )
                events.append(BacktestEvent(at=bar.at, kind="SESSION_CLOSEOUT"))
            context = StrategyContext(
                instrument_id=instrument_id,
                bars=tuple(bars[: index + 1]),
                horizon=horizon,
                spread_bps=self.fill_model.costs.spread_bps,
            )
            signal = strategy.evaluate(context)
            if signal is None or snapshot.positions or pending is not None:
                continue
            sized = sizer.create_intent(
                signal=signal,
                instrument=instrument,
                portfolio=snapshot,
                snapshot_hash=portfolio_hash(snapshot),
                now=bar.at,
            )
            if not isinstance(sized, OrderIntent):
                events.append(
                    BacktestEvent(at=bar.at, kind="RISK_REJECTED", reason_codes=sized.reason_codes)
                )
                continue
            market = MarketSnapshot(
                instrument_id=instrument_id,
                observed_at=bar.at,
                source_at=bar.at,
                bars=context.bars,
                provider="backtest",
                max_age_seconds=0,
            )
            decision = risk.assess(
                intent=sized,
                instrument=instrument,
                market=market,
                portfolio=snapshot,
                now=bar.at,
            )
            if not decision.approved or decision.approved_quantity is None:
                events.append(
                    BacktestEvent(
                        at=bar.at,
                        kind="RISK_REJECTED",
                        reason_codes=decision.reason_codes,
                    )
                )
                continue
            if (
                decision.approved_notional is None
                or sized.limit_price is None
                or sized.stop_loss is None
                or sized.take_profit is None
            ):
                events.append(
                    BacktestEvent(
                        at=bar.at,
                        kind="RISK_REJECTED",
                        reason_codes=("INVALID_APPROVAL_BOUNDS",),
                    )
                )
                continue
            pending = _PendingOrder(
                order_id=sized.intent_id,
                side=Side.BUY,
                quantity=decision.approved_quantity,
                submitted_index=index,
                submitted_at=bar.at,
                stop_loss=sized.stop_loss,
                take_profit=sized.take_profit,
                limit_price=sized.limit_price,
                approved_notional=decision.approved_notional,
                approved_stop_risk=decision.approved_quantity
                * (sized.limit_price - sized.stop_loss),
            )
            events.append(BacktestEvent(at=bar.at, kind="ORDER_APPROVED"))

        ending = equity_curve[-1].value
        benchmark = tuple(
            EquityPoint(at=bar.at, value=initial_cash * bar.close / bars[0].close)
            for bar in bars
        )
        return BacktestResult(
            strategy_id=metadata.strategy_id,
            strategy_version=metadata.version,
            strategy_configuration=configuration,
            costs=self.fill_model.costs,
            data_cutoff=bars[-1].at,
            events=tuple(events),
            fills=tuple(fills),
            equity_curve=tuple(equity_curve),
            benchmark_curve=benchmark,
            initial_cash=initial_cash,
            ending_equity=ending,
            gross_ending_equity=ending + ledger.fees + execution_impact,
            total_fees=ledger.fees,
            completed_trades=tuple(completed_trades),
            traded_notional=sum(
                (fill.quantity * fill.price for fill in fills), Decimal("0")
            ),
            exposed_periods=exposed_periods,
        )

    def run_robustness(
        self,
        *,
        instrument: Instrument,
        bars: tuple[Bar, ...],
        strategy: Strategy,
        initial_cash: Decimal,
    ) -> RobustnessResult:
        """Run base and 2x-cost simulations with the exact same strategy parameters."""
        base = self.run(
            instrument=instrument,
            bars=bars,
            strategy=strategy,
            initial_cash=initial_cash,
        )
        stressed_costs = self.fill_model.costs.stressed()
        stressed = BacktestEngine(
            costs=stressed_costs,
            risk_policy=self.risk_policy,
        ).run(
            instrument=instrument,
            bars=bars,
            strategy=strategy,
            initial_cash=initial_cash,
        )
        return RobustnessResult(base=base, stressed=stressed, stressed_costs=stressed_costs)


def _create_pending_fill(
    *,
    pending: _PendingOrder,
    bar: Bar,
    fill_model: FillModel,
    instrument: Instrument,
    fill_id: str,
    available_cash: Decimal,
) -> Fill | None:
    if pending.side is Side.SELL:
        return fill_model.fill(
            fill_id=fill_id,
            order_id=pending.order_id,
            instrument=instrument,
            side=Side.SELL,
            quantity=pending.quantity,
            reference_price=bar.open,
            submitted_at=pending.submitted_at,
            filled_at=bar.at,
        )

    assert pending.limit_price is not None
    assert pending.approved_notional is not None
    assert pending.approved_stop_risk is not None
    assert pending.stop_loss is not None
    assert pending.take_profit is not None
    if bar.low > pending.limit_price:
        return None
    candidate = fill_model.fill(
        fill_id=fill_id,
        order_id=pending.order_id,
        instrument=instrument,
        side=Side.BUY,
        quantity=pending.quantity,
        reference_price=min(bar.open, pending.limit_price),
        submitted_at=pending.submitted_at,
        filled_at=bar.at,
    )
    actual_notional = candidate.quantity * candidate.price
    actual_stop_risk = candidate.quantity * (candidate.price - pending.stop_loss)
    if (
        candidate.price > pending.limit_price
        or actual_notional > pending.approved_notional
        or actual_notional < instrument.minimum_notional
        or actual_notional + candidate.fee > available_cash
        or not pending.stop_loss < candidate.price < pending.take_profit
        or actual_stop_risk < Decimal("0")
        or actual_stop_risk > pending.approved_stop_risk
    ):
        return None
    return candidate


def _validate_bars(bars: object) -> None:
    if not isinstance(bars, tuple) or not bars:
        raise ValueError("bars must be a nonempty tuple")
    previous: datetime | None = None
    for bar in bars:
        if not isinstance(bar, Bar):
            raise ValueError("bars must contain Bar instances")
        _require_aware_datetime(bar.at, "bars timestamps")
        prices = (bar.open, bar.high, bar.low, bar.close)
        if not all(_finite_positive(value) for value in prices):
            raise ValueError("bars must contain finite positive Decimal OHLC values")
        if not bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high:
            raise ValueError("bars must have consistent OHLC ordering")
        if not _finite_nonnegative(bar.volume):
            raise ValueError("bars must contain finite nonnegative Decimal volume")
        if previous is not None and previous >= bar.at:
            raise ValueError("bars must be strictly chronological")
        previous = bar.at


def _require_aware_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _finite_positive(value: object) -> bool:
    return isinstance(value, Decimal) and value.is_finite() and value > Decimal("0")


def _finite_nonnegative(value: object) -> bool:
    return isinstance(value, Decimal) and value.is_finite() and value >= Decimal("0")


def _step_aligned(value: Decimal, step: Decimal) -> bool:
    try:
        quotient = value / step
        return quotient.is_finite() and quotient == quotient.to_integral_value()
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return False


def _floor_quantity(value: Decimal, step: Decimal) -> Decimal:
    try:
        return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step
    except (InvalidOperation, OverflowError, ZeroDivisionError) as error:
        raise ValueError("quantity step arithmetic must remain finite") from error
