"""Explicit factories for independent, immutable domain fixtures."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from market_sentinel.domain.enums import AssetClass, Horizon, OrderType, Side, SignalDirection
from market_sentinel.domain.models import (
    Bar,
    Evidence,
    Fill,
    Instrument,
    MarketSnapshot,
    OrderIntent,
    PortfolioSnapshot,
    Position,
    ResearchPacket,
    RiskDecision,
    Signal,
)

DEFAULT_INSTANT = datetime(2026, 8, 9, 10, tzinfo=UTC)


def instrument(
    symbol: str = "AAPL",
    venue: str = "alpaca",
    asset_class: AssetClass = AssetClass.EQUITY,
    quote_currency: str = "USD",
    timezone: str = "America/New_York",
    price_tick: Decimal | str = Decimal("0.01"),
    quantity_step: Decimal | str = Decimal("0.000000001"),
    minimum_notional: Decimal | str = Decimal("1"),
    session_calendar: str | None = "NYSE",
) -> Instrument:
    return Instrument(
        symbol=symbol,
        venue=venue,
        asset_class=asset_class,
        quote_currency=quote_currency,
        timezone=timezone,
        price_tick=Decimal(price_tick),
        quantity_step=Decimal(quantity_step),
        minimum_notional=Decimal(minimum_notional),
        session_calendar=session_calendar,
    )


def bar_series(
    count: int = 60,
    start_at: datetime = DEFAULT_INSTANT - timedelta(minutes=59),
    start_price: Decimal | str = Decimal("100"),
    increment: Decimal | str = Decimal("1"),
    volume: Decimal | str = Decimal("1000"),
) -> tuple[Bar, ...]:
    initial_price = Decimal(start_price)
    price_increment = Decimal(increment)
    bar_volume = Decimal(volume)
    return tuple(
        Bar(
            at=start_at + timedelta(minutes=index),
            open=initial_price + price_increment * Decimal(index),
            high=initial_price + price_increment * Decimal(index) + Decimal("0.5"),
            low=initial_price + price_increment * Decimal(index) - Decimal("0.5"),
            close=initial_price + price_increment * Decimal(index) + Decimal("0.25"),
            volume=bar_volume,
        )
        for index in range(count)
    )


def trending_bars(count: int = 80) -> tuple[Bar, ...]:
    """Return a liquid, steady uptrend suitable for deterministic strategy fixtures."""
    return bar_series(count=count, start_price="100", increment="1", volume="1000")


def snapshot(
    instrument_id: str = "AAPL@alpaca",
    observed_at: datetime = DEFAULT_INSTANT,
    source_at: datetime | None = None,
    bars: tuple[Bar, ...] | None = None,
    provider: str = "fixture",
    max_age_seconds: int = 60,
) -> MarketSnapshot:
    snapshot_bars = bar_series() if bars is None else bars
    effective_source_at = observed_at if source_at is None else source_at
    return MarketSnapshot(
        instrument_id=instrument_id,
        observed_at=observed_at,
        source_at=effective_source_at,
        bars=snapshot_bars,
        provider=provider,
        max_age_seconds=max_age_seconds,
    )


def research_packet(
    instrument_id: str = "AAPL@alpaca",
    as_of: datetime = DEFAULT_INSTANT,
    thesis: str = "Demand is improving.",
    bear_case: str = "Demand may reverse.",
    catalysts: tuple[str, ...] = ("earnings",),
    risks: tuple[str, ...] = ("volatility",),
    evidence: tuple[Evidence, ...] = (),
    confidence: Decimal | str = Decimal("0.5"),
    model_id: str = "fixture-model",
    prompt_version: str = "v1",
    configuration_hash: str = "configuration-hash",
) -> ResearchPacket:
    return ResearchPacket(
        instrument_id=instrument_id,
        as_of=as_of,
        thesis=thesis,
        bear_case=bear_case,
        catalysts=catalysts,
        risks=risks,
        evidence=evidence,
        confidence=Decimal(confidence),
        model_id=model_id,
        prompt_version=prompt_version,
        configuration_hash=configuration_hash,
    )


def signal(
    strategy_id: str = "fixture-strategy",
    strategy_version: str = "v1",
    instrument_id: str = "AAPL@alpaca",
    direction: SignalDirection = SignalDirection.LONG,
    strength: Decimal | str = Decimal("0.5"),
    horizon: Horizon = Horizon.SWING,
    entry_price: Decimal | str = Decimal("100"),
    invalidation_price: Decimal | str = Decimal("95"),
    take_profit: Decimal | str = Decimal("110"),
    research_required: bool = False,
    evidence_uris: tuple[str, ...] = (),
) -> Signal:
    return Signal(
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        instrument_id=instrument_id,
        direction=direction,
        strength=Decimal(strength),
        horizon=horizon,
        entry_price=Decimal(entry_price),
        invalidation_price=Decimal(invalidation_price),
        take_profit=Decimal(take_profit),
        research_required=research_required,
        evidence_uris=evidence_uris,
    )


def intent(
    intent_id: str = "intent-1",
    instrument_id: str = "AAPL@alpaca",
    side: Side = Side.BUY,
    quantity: Decimal | str | None = None,
    notional: Decimal | str | None = Decimal("10"),
    order_type: OrderType | None = None,
    limit_price: Decimal | str | None = None,
    trigger_price: Decimal | str | None = None,
    stop_loss: Decimal | str | None = Decimal("95"),
    take_profit: Decimal | str | None = Decimal("110"),
    time_in_force: str = "day",
    product: str = "cash",
    session: str = "regular",
    snapshot_hash: str = "snapshot-hash",
    created_at: datetime = DEFAULT_INSTANT,
    expires_at: datetime = DEFAULT_INSTANT + timedelta(minutes=1),
) -> OrderIntent:
    return OrderIntent(
        intent_id=intent_id,
        instrument_id=instrument_id,
        side=side,
        quantity=None if quantity is None else Decimal(quantity),
        notional=None if notional is None else Decimal(notional),
        order_type=(
            OrderType.LIMIT if order_type is None and limit_price is not None else order_type
        )
        or OrderType.MARKET,
        limit_price=None if limit_price is None else Decimal(limit_price),
        trigger_price=None if trigger_price is None else Decimal(trigger_price),
        stop_loss=None if stop_loss is None else Decimal(stop_loss),
        take_profit=None if take_profit is None else Decimal(take_profit),
        time_in_force=time_in_force,
        product=product,
        session=session,
        snapshot_hash=snapshot_hash,
        created_at=created_at,
        expires_at=expires_at,
    )


def risk_decision(
    approved: bool = False,
    reason_codes: tuple[str, ...] = ("REJECTED",),
    approved_quantity: Decimal | str | None = None,
    approved_notional: Decimal | str | None = None,
    portfolio_hash: str = "portfolio-hash",
    decided_at: datetime = DEFAULT_INSTANT,
    expires_at: datetime = DEFAULT_INSTANT + timedelta(minutes=1),
) -> RiskDecision:
    return RiskDecision(
        approved=approved,
        reason_codes=reason_codes,
        approved_quantity=None if approved_quantity is None else Decimal(approved_quantity),
        approved_notional=None if approved_notional is None else Decimal(approved_notional),
        portfolio_hash=portfolio_hash,
        decided_at=decided_at,
        expires_at=expires_at,
    )


def fill(
    fill_id: str = "fill-1",
    order_id: str = "order-1",
    instrument_id: str = "AAPL@alpaca",
    side: Side = Side.BUY,
    quantity: Decimal | str = Decimal("1"),
    price: Decimal | str = Decimal("100"),
    fee: Decimal | str = Decimal("0"),
    filled_at: datetime = DEFAULT_INSTANT,
) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id=order_id,
        instrument_id=instrument_id,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        filled_at=filled_at,
    )


def portfolio(
    currency: str = "USD",
    cash: Decimal | str = Decimal("10"),
    equity: Decimal | str = Decimal("10"),
    peak_equity: Decimal | str = Decimal("10"),
    gross_exposure: Decimal | str = Decimal("0"),
    daily_pnl: Decimal | str = Decimal("0"),
    realized_pnl: Decimal | str = Decimal("0"),
    positions: tuple[Position, ...] = (),
    observed_at: datetime = DEFAULT_INSTANT,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        currency=currency,
        cash=Decimal(cash),
        equity=Decimal(equity),
        peak_equity=Decimal(peak_equity),
        gross_exposure=Decimal(gross_exposure),
        daily_pnl=Decimal(daily_pnl),
        realized_pnl=Decimal(realized_pnl),
        positions=positions,
        observed_at=observed_at,
    )


def position(
    instrument_id: str = "AAPL@alpaca",
    quantity: Decimal | str = Decimal("1"),
    average_price: Decimal | str = Decimal("1"),
    market_price: Decimal | str = Decimal("1"),
    unrealized_pnl: Decimal | str = Decimal("0"),
) -> Position:
    return Position(
        instrument_id=instrument_id,
        quantity=Decimal(quantity),
        average_price=Decimal(average_price),
        market_price=Decimal(market_price),
        unrealized_pnl=Decimal(unrealized_pnl),
    )
