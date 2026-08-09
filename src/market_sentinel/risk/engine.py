"""Deterministic sizing and ordered, fail-closed order risk assessment."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal, InvalidOperation

from market_sentinel.domain.clock import Clock, SystemClock
from market_sentinel.domain.enums import AssetClass, OrderType, Side, SignalDirection
from market_sentinel.domain.models import (
    Instrument,
    MarketSnapshot,
    OrderIntent,
    PortfolioSnapshot,
    RiskDecision,
    Signal,
)
from market_sentinel.risk.policy import RiskPolicy

_REASON_ORDER = (
    "KILL_SWITCH_ACTIVE",
    "EXPIRED_INTENT",
    "STALE_DATA",
    "PORTFOLIO_HASH_MISMATCH",
    "MISSING_PROTECTIVE_EXIT",
    "DRAWDOWN_LIMIT",
    "DAILY_LOSS_LIMIT",
    "LEVERAGE_FORBIDDEN",
    "SHORT_FORBIDDEN",
    "DERIVATIVE_FORBIDDEN",
    "POSITION_LIMIT",
    "GROSS_EXPOSURE_LIMIT",
    "BELOW_MINIMUM_NOTIONAL",
    "INVALID_PRECISION",
)


class PositionSizer:
    """Create only conservative, venue-precise cash long limit intents."""

    def __init__(self, *, policy: RiskPolicy) -> None:
        self._policy = policy

    def create_intent(
        self,
        *,
        signal: Signal,
        instrument: Instrument,
        portfolio: PortfolioSnapshot,
        snapshot_hash: str,
        now: datetime,
        time_in_force: str = "day",
        product: str = "cash",
        session: str = "regular",
    ) -> OrderIntent | RiskDecision:
        """Size to risk, position, and gross headroom, rounding strictly down."""
        instant = _normalize_now(now)
        actual_hash = portfolio_hash(portfolio)
        if (
            not _valid_portfolio(portfolio)
            or not snapshot_hash
            or snapshot_hash != actual_hash
            or not _valid_instrument(instrument)
            or signal.instrument_id != _canonical_instrument_id(instrument)
        ):
            return _rejection("INVALID_PRECISION", actual_hash, instant)
        if signal.direction is SignalDirection.SHORT:
            return _rejection("SHORT_FORBIDDEN", actual_hash, instant)
        if signal.direction is not SignalDirection.LONG:
            return _rejection("INVALID_PRECISION", actual_hash, instant)
        if product.strip().lower() != "cash":
            return _rejection("LEVERAGE_FORBIDDEN", actual_hash, instant)
        if not _valid_signal_prices(
            signal.entry_price,
            signal.invalidation_price,
            signal.take_profit,
            Side.BUY,
            instrument.price_tick,
        ):
            reason = (
                "MISSING_PROTECTIVE_EXIT"
                if _finite_positive(signal.invalidation_price)
                and _finite_positive(signal.take_profit)
                else "INVALID_PRECISION"
            )
            return _rejection(reason, actual_hash, instant)

        entry = signal.entry_price
        stop_distance = entry - signal.invalidation_price
        risk_quantity = portfolio.equity * self._policy.max_trade_risk_fraction / stop_distance
        held_notional = _held_notional(portfolio, signal.instrument_id)
        position_headroom = portfolio.equity * self._policy.max_position_fraction - held_notional
        gross_headroom = (
            portfolio.equity * self._policy.max_gross_exposure_fraction - portfolio.gross_exposure
        )
        quantity = _round_down(
            min(risk_quantity, position_headroom / entry, gross_headroom / entry),
            instrument.quantity_step,
        )
        if quantity <= Decimal("0"):
            reason = (
                "POSITION_LIMIT" if position_headroom <= Decimal("0") else "GROSS_EXPOSURE_LIMIT"
            )
            return _rejection(reason, actual_hash, instant)
        notional = quantity * entry
        if notional < instrument.minimum_notional:
            return _rejection("BELOW_MINIMUM_NOTIONAL", actual_hash, instant)
        if portfolio.cash < notional:
            return _rejection("LEVERAGE_FORBIDDEN", actual_hash, instant)
        return OrderIntent(
            intent_id=f"{signal.strategy_id}:{signal.instrument_id}:{instant.isoformat()}",
            instrument_id=signal.instrument_id,
            side=Side.BUY,
            quantity=quantity,
            notional=None,
            order_type=OrderType.LIMIT,
            limit_price=entry,
            stop_loss=signal.invalidation_price,
            take_profit=signal.take_profit,
            time_in_force=time_in_force,
            product="cash",
            session=session,
            snapshot_hash=actual_hash,
            created_at=instant,
            expires_at=instant + self._policy.decision_ttl,
        )


class RiskEngine:
    """Accumulate every applicable safety rejection in stable documented order."""

    def __init__(
        self,
        *,
        policy: RiskPolicy,
        kill_switch: bool = False,
        expected_portfolio_hash: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._policy = policy
        self._kill_switch = kill_switch
        self._expected_portfolio_hash = expected_portfolio_hash
        self._clock = SystemClock() if clock is None else clock

    @classmethod
    def safe_defaults(
        cls,
        *,
        kill_switch: bool = False,
        expected_portfolio_hash: str | None = None,
        clock: Clock | None = None,
    ) -> RiskEngine:
        return cls(
            policy=RiskPolicy.safe_defaults(),
            kill_switch=kill_switch,
            expected_portfolio_hash=expected_portfolio_hash,
            clock=clock,
        )

    def assess(
        self,
        *,
        intent: OrderIntent,
        instrument: Instrument,
        market: MarketSnapshot,
        portfolio: PortfolioSnapshot,
        now: datetime | None = None,
    ) -> RiskDecision:
        """Assess every gate; any rejection makes executable fields unavailable."""
        instant = _normalize_now(self._clock.now() if now is None else now)
        actual_hash = portfolio_hash(portfolio)
        reasons: set[str] = set()
        portfolio_valid = _valid_portfolio(portfolio)

        if self._kill_switch:
            reasons.add("KILL_SWITCH_ACTIVE")
        if not _valid_intent_time(intent, instant, self._policy.decision_ttl):
            reasons.add("EXPIRED_INTENT")
        if not _valid_market_time(market, instant):
            reasons.add("STALE_DATA")
        if not _valid_portfolio_time(portfolio, instant, self._policy.decision_ttl):
            reasons.add("STALE_DATA")
        if (
            not intent.snapshot_hash
            or intent.snapshot_hash != actual_hash
            or (
                self._expected_portfolio_hash is not None
                and self._expected_portfolio_hash != actual_hash
            )
        ):
            reasons.add("PORTFOLIO_HASH_MISMATCH")
        if not _valid_portfolio(portfolio) or not _valid_instrument(instrument):
            reasons.add("INVALID_PRECISION")
        if (
            not _canonical_instrument_id(instrument)
            or intent.instrument_id != _canonical_instrument_id(instrument)
            or market.instrument_id != _canonical_instrument_id(instrument)
        ):
            reasons.add("INVALID_PRECISION")

        quantity, notional, price, precision_valid = _intent_size(intent, instrument, market)
        if not _valid_exits(intent, price, instrument.price_tick):
            if intent.stop_loss is None or intent.take_profit is None:
                reasons.add("MISSING_PROTECTIVE_EXIT")
            elif not _exits_are_numeric(intent) or not _exits_are_step_aligned(
                intent, instrument.price_tick
            ):
                reasons.add("INVALID_PRECISION")
            else:
                reasons.add("MISSING_PROTECTIVE_EXIT")
        if not precision_valid:
            reasons.add("INVALID_PRECISION")

        if portfolio_valid:
            drawdown = (portfolio.peak_equity - portfolio.equity) / portfolio.peak_equity
            if drawdown >= self._policy.max_drawdown_fraction:
                reasons.add("DRAWDOWN_LIMIT")
            if portfolio.daily_pnl <= -(portfolio.equity * self._policy.max_daily_loss_fraction):
                reasons.add("DAILY_LOSS_LIMIT")
        if intent.product.strip().lower() != "cash":
            reasons.add("LEVERAGE_FORBIDDEN")
        if intent.side is Side.SELL:
            reasons.add("SHORT_FORBIDDEN")
        if instrument.asset_class not in {AssetClass.EQUITY, AssetClass.CRYPTO_SPOT}:
            reasons.add("DERIVATIVE_FORBIDDEN")

        if portfolio_valid and precision_valid and quantity is not None and notional is not None:
            held_notional = _held_notional(portfolio, intent.instrument_id)
            if held_notional + notional > portfolio.equity * self._policy.max_position_fraction:
                reasons.add("POSITION_LIMIT")
            if portfolio.gross_exposure + notional > (
                portfolio.equity * self._policy.max_gross_exposure_fraction
            ):
                reasons.add("GROSS_EXPOSURE_LIMIT")
            if intent.stop_loss is not None and price is not None:
                stop_risk = quantity * abs(price - intent.stop_loss)
                if stop_risk > portfolio.equity * self._policy.max_trade_risk_fraction:
                    reasons.add("POSITION_LIMIT")
            if intent.side is Side.BUY and portfolio.cash < notional:
                reasons.add("LEVERAGE_FORBIDDEN")
            if notional < instrument.minimum_notional:
                reasons.add("BELOW_MINIMUM_NOTIONAL")

        ordered = tuple(code for code in _REASON_ORDER if code in reasons)
        if ordered:
            return RiskDecision(
                approved=False,
                reason_codes=ordered,
                approved_quantity=None,
                approved_notional=None,
                portfolio_hash=actual_hash,
                decided_at=instant,
                expires_at=instant,
            )
        assert quantity is not None and notional is not None
        return RiskDecision(
            approved=True,
            reason_codes=(),
            approved_quantity=quantity,
            approved_notional=notional,
            portfolio_hash=actual_hash,
            decided_at=instant,
            expires_at=instant + self._policy.decision_ttl,
        )


def portfolio_hash(portfolio: PortfolioSnapshot) -> str:
    """Hash every risk-relevant portfolio field using canonical UTC and Decimal encodings."""
    payload = {
        "currency": portfolio.currency,
        "cash": _decimal_text(portfolio.cash),
        "equity": _decimal_text(portfolio.equity),
        "peak_equity": _decimal_text(portfolio.peak_equity),
        "gross_exposure": _decimal_text(portfolio.gross_exposure),
        "daily_pnl": _decimal_text(portfolio.daily_pnl),
        "realized_pnl": _decimal_text(portfolio.realized_pnl),
        "observed_at": _datetime_text(portfolio.observed_at),
        "positions": [
            [
                position.instrument_id,
                _decimal_text(position.quantity),
                _decimal_text(position.average_price),
                _decimal_text(position.market_price),
                _decimal_text(position.unrealized_pnl),
            ]
            for position in sorted(
                portfolio.positions,
                key=lambda item: (
                    item.instrument_id,
                    _decimal_text(item.quantity),
                    _decimal_text(item.average_price),
                    _decimal_text(item.market_price),
                    _decimal_text(item.unrealized_pnl),
                ),
            )
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _intent_size(
    intent: OrderIntent, instrument: Instrument, market: MarketSnapshot
) -> tuple[Decimal | None, Decimal | None, Decimal | None, bool]:
    price = (
        intent.limit_price
        if intent.limit_price is not None
        else intent.trigger_price
        if intent.trigger_price is not None
        else _market_price(market)
    )
    if price is None or not _finite_positive(price) or not _valid_instrument(instrument):
        return None, None, price, False
    if (
        intent.limit_price is not None or intent.trigger_price is not None
    ) and not _is_step_aligned(price, instrument.price_tick):
        return None, None, price, False
    if intent.quantity is not None:
        if not _finite_positive(intent.quantity) or not _is_step_aligned(
            intent.quantity, instrument.quantity_step
        ):
            return None, None, price, False
        return intent.quantity, intent.quantity * price, price, True
    if intent.notional is None or not _finite_positive(intent.notional):
        return None, None, price, False
    quantity = _round_down(intent.notional / price, instrument.quantity_step)
    return quantity, quantity * price, price, quantity > Decimal("0")


def _valid_portfolio(portfolio: PortfolioSnapshot) -> bool:
    if not portfolio.currency.strip() or not _finite_nonnegative(portfolio.cash):
        return False
    if not _finite_positive(portfolio.equity) or not _finite_positive(portfolio.peak_equity):
        return False
    if portfolio.peak_equity < portfolio.equity or not _finite_nonnegative(
        portfolio.gross_exposure
    ):
        return False
    if not _is_finite_decimal(portfolio.daily_pnl) or not _is_finite_decimal(
        portfolio.realized_pnl
    ):
        return False
    ids: set[str] = set()
    market_value = Decimal("0")
    for position in portfolio.positions:
        if (
            not position.instrument_id.strip()
            or position.instrument_id in ids
            or not _finite_positive(position.quantity)
            or not _finite_positive(position.average_price)
            or not _finite_positive(position.market_price)
            or not _is_finite_decimal(position.unrealized_pnl)
        ):
            return False
        ids.add(position.instrument_id)
        market_value += position.quantity * position.market_price
    return (
        portfolio.gross_exposure == market_value
        and portfolio.cash + market_value == portfolio.equity
    )


def _valid_instrument(instrument: Instrument) -> bool:
    return (
        bool(instrument.symbol.strip())
        and bool(instrument.venue.strip())
        and _finite_positive(instrument.price_tick)
        and _finite_positive(instrument.quantity_step)
        and _finite_positive(instrument.minimum_notional)
    )


def _valid_intent_time(intent: OrderIntent, now: datetime, ttl: timedelta) -> bool:
    return (
        intent.created_at <= now < intent.expires_at
        and timedelta(0) < intent.expires_at - intent.created_at <= ttl
    )


def _valid_market_time(market: MarketSnapshot, now: datetime) -> bool:
    return (
        market.source_at <= market.observed_at <= now
        and market.source_at <= now
        and not market.is_stale(now)
    )


def _valid_portfolio_time(portfolio: PortfolioSnapshot, now: datetime, ttl: timedelta) -> bool:
    return portfolio.observed_at <= now and now - portfolio.observed_at <= ttl


def _valid_exits(intent: OrderIntent, price: Decimal | None, price_tick: Decimal) -> bool:
    if price is None or intent.stop_loss is None or intent.take_profit is None:
        return False
    if not all(_finite_positive(value) for value in (price, intent.stop_loss, intent.take_profit)):
        return False
    if not _is_step_aligned(intent.stop_loss, price_tick) or not _is_step_aligned(
        intent.take_profit, price_tick
    ):
        return False
    if intent.side is Side.BUY:
        return intent.stop_loss < price < intent.take_profit
    return intent.stop_loss > price > intent.take_profit


def _valid_signal_prices(
    entry: Decimal,
    stop_loss: Decimal,
    take_profit: Decimal,
    side: Side,
    price_tick: Decimal,
) -> bool:
    if not all(_finite_positive(value) for value in (entry, stop_loss, take_profit)):
        return False
    if not all(_is_step_aligned(value, price_tick) for value in (entry, stop_loss, take_profit)):
        return False
    return stop_loss < entry < take_profit if side is Side.BUY else stop_loss > entry > take_profit


def _exits_are_numeric(intent: OrderIntent) -> bool:
    return (
        intent.stop_loss is not None
        and intent.take_profit is not None
        and all(_finite_positive(value) for value in (intent.stop_loss, intent.take_profit))
    )


def _exits_are_step_aligned(intent: OrderIntent, price_tick: Decimal) -> bool:
    return (
        intent.stop_loss is not None
        and intent.take_profit is not None
        and _is_step_aligned(intent.stop_loss, price_tick)
        and _is_step_aligned(intent.take_profit, price_tick)
    )


def _canonical_instrument_id(instrument: Instrument) -> str:
    return f"{instrument.symbol}@{instrument.venue}" if _valid_instrument(instrument) else ""


def _market_price(market: MarketSnapshot) -> Decimal | None:
    return market.bars[-1].close if market.bars else None


def _held_notional(portfolio: PortfolioSnapshot, instrument_id: str) -> Decimal:
    return sum(
        (
            position.quantity * position.market_price
            for position in portfolio.positions
            if position.instrument_id == instrument_id
        ),
        Decimal("0"),
    )


def _round_down(value: Decimal, step: Decimal) -> Decimal:
    try:
        return (value / step).to_integral_value(rounding=ROUND_DOWN) * step
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _is_step_aligned(value: Decimal, step: Decimal) -> bool:
    try:
        quotient = value / step
        return quotient.to_integral_value() == quotient
    except (InvalidOperation, ValueError):
        return False


def _is_finite_decimal(value: object) -> bool:
    return isinstance(value, Decimal) and value.is_finite()


def _finite_positive(value: object) -> bool:
    return isinstance(value, Decimal) and value.is_finite() and value > Decimal("0")


def _finite_nonnegative(value: object) -> bool:
    return isinstance(value, Decimal) and value.is_finite() and value >= Decimal("0")


def _normalize_now(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("risk timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _rejection(reason: str, portfolio_hash_value: str, now: datetime) -> RiskDecision:
    return RiskDecision(
        approved=False,
        reason_codes=(reason,),
        approved_quantity=None,
        approved_notional=None,
        portfolio_hash=portfolio_hash_value,
        decided_at=now,
        expires_at=now,
    )


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f") if value.is_finite() else str(value)


def _datetime_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()
