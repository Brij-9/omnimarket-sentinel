"""Deterministic sizing and ordered, fail-closed order risk assessment."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal, InvalidOperation

from market_sentinel.domain.clock import Clock, SystemClock
from market_sentinel.domain.enums import AssetClass, OrderType, Side
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
    """Create a conservative limit intent from one validated signal."""

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
        """Size to stop risk, position, and gross caps; venue-round strictly down."""
        instant = _normalize_now(now)
        values = (
            signal.entry_price,
            signal.invalidation_price,
            signal.take_profit,
            portfolio.equity,
            portfolio.gross_exposure,
            instrument.quantity_step,
            instrument.minimum_notional,
        )
        if not all(_is_finite_decimal(value) for value in values):
            return _rejection("INVALID_PRECISION", snapshot_hash, instant)

        entry = signal.entry_price
        stop_distance = abs(entry - signal.invalidation_price)
        if entry <= Decimal("0") or stop_distance <= Decimal("0"):
            return _rejection("INVALID_PRECISION", snapshot_hash, instant)

        risk_quantity = portfolio.equity * self._policy.max_trade_risk_fraction / stop_distance
        held_notional = _held_notional(portfolio, signal.instrument_id)
        position_headroom = portfolio.equity * self._policy.max_position_fraction - held_notional
        gross_headroom = (
            portfolio.equity * self._policy.max_gross_exposure_fraction - portfolio.gross_exposure
        )
        caps = (
            risk_quantity,
            position_headroom / entry,
            gross_headroom / entry,
        )
        quantity = _round_down(min(caps), instrument.quantity_step)
        if quantity <= Decimal("0"):
            reason = (
                "POSITION_LIMIT" if position_headroom <= Decimal("0") else "GROSS_EXPOSURE_LIMIT"
            )
            return _rejection(reason, snapshot_hash, instant)

        notional = quantity * entry
        if notional < instrument.minimum_notional:
            return _rejection("BELOW_MINIMUM_NOTIONAL", snapshot_hash, instant)

        side = Side.BUY if signal.direction.value == "long" else Side.SELL
        return OrderIntent(
            intent_id=f"{signal.strategy_id}:{signal.instrument_id}:{instant.isoformat()}",
            instrument_id=signal.instrument_id,
            side=side,
            quantity=quantity,
            notional=None,
            order_type=OrderType.LIMIT,
            limit_price=entry,
            stop_loss=signal.invalidation_price,
            take_profit=signal.take_profit,
            time_in_force=time_in_force,
            product=product,
            session=session,
            snapshot_hash=snapshot_hash,
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
        """Build an engine with the fixed conservative global policy."""
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
        """Assess all gates, never approving a partial or rejected order."""
        instant = _normalize_now(self._clock.now() if now is None else now)
        actual_portfolio_hash = _portfolio_hash(portfolio)
        reasons: set[str] = set()

        if self._kill_switch:
            reasons.add("KILL_SWITCH_ACTIVE")
        if instant >= intent.expires_at:
            reasons.add("EXPIRED_INTENT")
        if market.is_stale(instant):
            reasons.add("STALE_DATA")
        if (
            self._expected_portfolio_hash is not None
            and intent.snapshot_hash != self._expected_portfolio_hash
        ):
            reasons.add("PORTFOLIO_HASH_MISMATCH")
        if intent.stop_loss is None or intent.take_profit is None:
            reasons.add("MISSING_PROTECTIVE_EXIT")

        equity_is_usable = _is_finite_decimal(portfolio.equity) and portfolio.equity > Decimal("0")
        peak_is_usable = _is_finite_decimal(
            portfolio.peak_equity
        ) and portfolio.peak_equity > Decimal("0")
        if not equity_is_usable or not peak_is_usable:
            reasons.add("INVALID_PRECISION")
        else:
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

        quantity, notional, precision_valid = _intent_size(intent, instrument, market)
        if not precision_valid:
            reasons.add("INVALID_PRECISION")
        elif equity_is_usable and quantity is not None and notional is not None:
            held_notional = _held_notional(portfolio, intent.instrument_id)
            if held_notional + notional > portfolio.equity * self._policy.max_position_fraction:
                reasons.add("POSITION_LIMIT")
            if portfolio.gross_exposure + notional > (
                portfolio.equity * self._policy.max_gross_exposure_fraction
            ):
                reasons.add("GROSS_EXPOSURE_LIMIT")
            if notional < instrument.minimum_notional:
                reasons.add("BELOW_MINIMUM_NOTIONAL")

        ordered_reasons = tuple(code for code in _REASON_ORDER if code in reasons)
        if ordered_reasons:
            return RiskDecision(
                approved=False,
                reason_codes=ordered_reasons,
                approved_quantity=None,
                approved_notional=None,
                portfolio_hash=actual_portfolio_hash,
                decided_at=instant,
                expires_at=instant,
            )
        assert quantity is not None and notional is not None
        return RiskDecision(
            approved=True,
            reason_codes=(),
            approved_quantity=quantity,
            approved_notional=notional,
            portfolio_hash=actual_portfolio_hash,
            decided_at=instant,
            expires_at=instant + self._policy.decision_ttl,
        )


def _intent_size(
    intent: OrderIntent, instrument: Instrument, market: MarketSnapshot
) -> tuple[Decimal | None, Decimal | None, bool]:
    price = intent.limit_price if intent.limit_price is not None else _market_price(market)
    if price is None or not _is_finite_decimal(price) or price <= Decimal("0"):
        return None, None, False
    if not _is_finite_decimal(instrument.quantity_step) or instrument.quantity_step <= Decimal("0"):
        return None, None, False
    if not _is_finite_decimal(instrument.price_tick) or instrument.price_tick <= Decimal("0"):
        return None, None, False
    if intent.limit_price is not None and not _is_step_aligned(
        intent.limit_price, instrument.price_tick
    ):
        return None, None, False
    if intent.quantity is not None:
        requested_quantity = intent.quantity
        if not _is_finite_decimal(requested_quantity) or requested_quantity <= Decimal("0"):
            return None, None, False
        if not _is_step_aligned(requested_quantity, instrument.quantity_step):
            return None, None, False
        assert price is not None
        return requested_quantity, requested_quantity * price, True
    requested_notional = intent.notional
    if (
        requested_notional is None
        or not _is_finite_decimal(requested_notional)
        or requested_notional <= Decimal("0")
    ):
        return None, None, False
    assert price is not None
    quantity = _round_down(requested_notional / price, instrument.quantity_step)
    if quantity <= Decimal("0"):
        return quantity, Decimal("0"), True
    return quantity, quantity * price, True


def _market_price(market: MarketSnapshot) -> Decimal | None:
    if not market.bars:
        return None
    return market.bars[-1].close


def _held_notional(portfolio: PortfolioSnapshot, instrument_id: str) -> Decimal:
    return sum(
        (
            abs(position.quantity) * position.market_price
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
        return (value / step).to_integral_value() == value / step
    except (InvalidOperation, ValueError):
        return False


def _is_finite_decimal(value: object) -> bool:
    return isinstance(value, Decimal) and value.is_finite()


def _normalize_now(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("risk timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _rejection(reason: str, portfolio_hash: str, now: datetime) -> RiskDecision:
    return RiskDecision(
        approved=False,
        reason_codes=(reason,),
        approved_quantity=None,
        approved_notional=None,
        portfolio_hash=portfolio_hash,
        decided_at=now,
        expires_at=now,
    )


def _portfolio_hash(portfolio: PortfolioSnapshot) -> str:
    payload = {
        "cash": _decimal_text(portfolio.cash),
        "equity": _decimal_text(portfolio.equity),
        "positions": [
            [
                position.instrument_id,
                _decimal_text(position.quantity),
                _decimal_text(position.average_price),
                _decimal_text(position.market_price),
            ]
            for position in sorted(portfolio.positions, key=lambda item: item.instrument_id)
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f") if value.is_finite() else str(value)
