"""Validated immutable records shared by market, research, risk, and broker code."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from market_sentinel.domain.enums import (
    AssetClass,
    Horizon,
    OrderStatus,
    OrderType,
    Side,
    SignalDirection,
)


class FrozenModel(BaseModel):
    """Base model that rejects mutation and naive timestamps."""

    model_config = ConfigDict(frozen=True)

    @field_validator("*", mode="after")
    @classmethod
    def require_aware_datetimes(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("datetime fields must be timezone-aware")
            return value.astimezone(UTC)
        return value


class GateResult(FrozenModel):
    name: str
    passed: bool
    reason_code: str


class Instrument(FrozenModel):
    symbol: str
    venue: str
    asset_class: AssetClass
    quote_currency: str
    timezone: str
    price_tick: Decimal
    quantity_step: Decimal
    minimum_notional: Decimal
    session_calendar: str | None = None

    @field_validator("price_tick", "quantity_step", "minimum_notional")
    @classmethod
    def require_positive_precision(cls, value: Decimal) -> Decimal:
        if value <= Decimal("0"):
            raise ValueError("precision and minimum notional must be positive")
        return value


class Bar(FrozenModel):
    at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    @field_validator("volume")
    @classmethod
    def require_nonnegative_volume(cls, value: Decimal) -> Decimal:
        if value < Decimal("0"):
            raise ValueError("volume must be nonnegative")
        return value


class MarketSnapshot(FrozenModel):
    instrument_id: str
    observed_at: datetime
    source_at: datetime
    bars: tuple[Bar, ...]
    provider: str
    max_age_seconds: int

    @field_validator("max_age_seconds")
    @classmethod
    def require_nonnegative_max_age(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_age_seconds must be nonnegative")
        return value

    def is_stale(self, now: datetime) -> bool:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        return now - self.source_at > timedelta(seconds=self.max_age_seconds)


class Evidence(FrozenModel):
    uri: str
    title: str
    published_at: datetime


class ResearchPacket(FrozenModel):
    instrument_id: str
    as_of: datetime
    thesis: str
    bear_case: str
    catalysts: tuple[str, ...]
    risks: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    confidence: Decimal
    model_id: str
    prompt_version: str
    configuration_hash: str

    @field_validator("confidence")
    @classmethod
    def require_confidence_in_unit_interval(cls, value: Decimal) -> Decimal:
        if not Decimal("0") <= value <= Decimal("1"):
            raise ValueError("confidence must be between 0 and 1")
        return value


class Signal(FrozenModel):
    strategy_id: str
    strategy_version: str
    instrument_id: str
    direction: SignalDirection
    strength: Decimal
    horizon: Horizon
    entry_price: Decimal
    invalidation_price: Decimal
    take_profit: Decimal
    research_required: bool
    evidence_uris: tuple[str, ...]

    @field_validator("strength")
    @classmethod
    def require_strength_in_unit_interval(cls, value: Decimal) -> Decimal:
        if not Decimal("-1") <= value <= Decimal("1"):
            raise ValueError("strength must be between -1 and 1")
        return value

    @model_validator(mode="after")
    def require_consistent_protective_prices(self) -> Self:
        if self.direction is SignalDirection.LONG and not (
            self.invalidation_price < self.entry_price < self.take_profit
        ):
            raise ValueError("long signals require invalidation < entry < take_profit")
        if self.direction is SignalDirection.SHORT and not (
            self.invalidation_price > self.entry_price > self.take_profit
        ):
            raise ValueError("short signals require invalidation > entry > take_profit")
        return self


class OrderIntent(FrozenModel):
    intent_id: str
    instrument_id: str
    side: Side
    quantity: Decimal | None
    notional: Decimal | None
    order_type: OrderType
    limit_price: Decimal | None
    stop_loss: Decimal | None
    take_profit: Decimal | None
    time_in_force: str
    product: str
    session: str
    snapshot_hash: str
    created_at: datetime
    expires_at: datetime
    trigger_price: Decimal | None = None

    @field_validator("limit_price", "trigger_price", "stop_loss", "take_profit")
    @classmethod
    def require_positive_order_prices(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and (not value.is_finite() or value <= Decimal("0")):
            raise ValueError("order prices must be finite and positive")
        return value

    @model_validator(mode="after")
    def require_single_size_and_consistent_protective_prices(self) -> Self:
        if (self.quantity is None) == (self.notional is None):
            raise ValueError("exactly one of quantity or notional must be populated")
        expected_fields = {
            OrderType.MARKET: (False, False),
            OrderType.LIMIT: (True, False),
            OrderType.STOP: (False, True),
            OrderType.STOP_LIMIT: (True, True),
        }
        expected_limit, expected_trigger = expected_fields[self.order_type]
        if (self.limit_price is not None) is not expected_limit or (
            (self.trigger_price is not None) is not expected_trigger
        ):
            raise ValueError("order type requires its exact limit/trigger price fields")
        if self.order_type is OrderType.STOP_LIMIT:
            assert self.limit_price is not None and self.trigger_price is not None
            if self.side is Side.BUY and self.limit_price < self.trigger_price:
                raise ValueError("buy stop-limit requires limit at or above trigger")
            if self.side is Side.SELL and self.limit_price > self.trigger_price:
                raise ValueError("sell stop-limit requires limit at or below trigger")
        if self.stop_loss is not None and self.take_profit is not None:
            if self.side is Side.BUY and self.stop_loss >= self.take_profit:
                raise ValueError("buy intent requires stop_loss below take_profit")
            if self.side is Side.SELL and self.stop_loss <= self.take_profit:
                raise ValueError("sell intent requires stop_loss above take_profit")
        return self


class Position(FrozenModel):
    instrument_id: str
    quantity: Decimal
    average_price: Decimal
    market_price: Decimal
    unrealized_pnl: Decimal


class PortfolioSnapshot(FrozenModel):
    currency: str
    cash: Decimal
    equity: Decimal
    peak_equity: Decimal
    gross_exposure: Decimal
    daily_pnl: Decimal
    realized_pnl: Decimal
    positions: tuple[Position, ...]
    observed_at: datetime


class RiskDecision(FrozenModel):
    approved: bool
    reason_codes: tuple[str, ...]
    approved_quantity: Decimal | None
    approved_notional: Decimal | None
    portfolio_hash: str
    decided_at: datetime
    expires_at: datetime


class BrokerOrder(FrozenModel):
    order_id: str
    client_order_id: str
    broker: str
    instrument_id: str
    status: OrderStatus
    requested_quantity: Decimal | None
    requested_notional: Decimal | None = None
    filled_quantity: Decimal
    average_fill_price: Decimal | None
    submitted_at: datetime
    updated_at: datetime


class Fill(FrozenModel):
    fill_id: str
    order_id: str
    instrument_id: str
    side: Side
    quantity: Decimal
    price: Decimal
    fee: Decimal
    filled_at: datetime
