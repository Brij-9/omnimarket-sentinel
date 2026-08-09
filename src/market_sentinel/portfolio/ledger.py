"""Deterministic, long-only portfolio fill accounting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from market_sentinel.domain.enums import Side
from market_sentinel.domain.models import Fill, PortfolioSnapshot, Position


class DuplicateFillError(ValueError):
    """Raised when a fill ID has already been applied to this ledger."""


class InsufficientPositionError(ValueError):
    """Raised when a long-only sell requests more than the held quantity."""


@dataclass(slots=True)
class _PositionState:
    quantity: Decimal
    average_price: Decimal


class PortfolioLedger:
    """Apply fills once, value open long positions, and produce reconciliation snapshots."""

    def __init__(self, *, starting_cash: Decimal, currency: str) -> None:
        self._starting_cash = starting_cash
        self._currency = currency
        self._cash = starting_cash
        self._positions: dict[str, _PositionState] = {}
        self._market_prices: dict[str, Decimal] = {}
        self._fill_ids: set[str] = set()
        self._gross_realized_pnl = Decimal("0")
        self._fees = Decimal("0")
        self._equity = starting_cash
        self._peak_equity = starting_cash
        self._drawdown = Decimal("0")

    @property
    def cash(self) -> Decimal:
        """Return cash after all applied fill consideration and fees."""
        return self._cash

    @property
    def fees(self) -> Decimal:
        """Return all fees expensed through the applied fill sequence."""
        return self._fees

    @property
    def drawdown(self) -> Decimal:
        """Return the current fractional decline from the ledger's peak equity."""
        return self._drawdown

    def apply_fill(self, fill: Fill) -> None:
        """Apply one fill atomically using average-cost long-only accounting."""
        quantity = _require_positive_decimal(fill.quantity, field_name="quantity")
        price = _require_positive_decimal(fill.price, field_name="price")
        fee = _require_nonnegative_decimal(fill.fee, field_name="fee")
        if fill.fill_id in self._fill_ids:
            raise DuplicateFillError(f"fill ID already applied: {fill.fill_id}")

        position = self._positions.get(fill.instrument_id)
        if fill.side is Side.SELL and (
            position is None or quantity > position.quantity
        ):
            held = Decimal("0") if position is None else position.quantity
            raise InsufficientPositionError(
                f"cannot sell {quantity} of {fill.instrument_id}; held {held}"
            )

        notional = quantity * price
        if fill.side is Side.BUY:
            self._apply_buy(fill.instrument_id, quantity, notional, fee)
        else:
            assert position is not None
            self._apply_sell(
                fill.instrument_id,
                position,
                quantity,
                price,
                notional,
                fee,
            )

        self._fill_ids.add(fill.fill_id)
        self._market_prices[fill.instrument_id] = price
        self._revalue()

    def mark(self, prices: Mapping[str, Decimal], at: datetime) -> PortfolioSnapshot:
        """Value every open position at supplied prices and return a snapshot."""
        observed_at = _normalize_aware_datetime(at)
        validated_prices = {
            instrument_id: _require_positive_decimal(price, field_name="price")
            for instrument_id, price in prices.items()
        }
        for instrument_id in self._positions:
            if instrument_id not in validated_prices:
                raise ValueError(f"missing price for open position: {instrument_id}")

        candidate_prices = self._market_prices | validated_prices
        equity, peak_equity, drawdown = self._valuation(candidate_prices)
        self._market_prices = candidate_prices
        self._equity = equity
        self._peak_equity = peak_equity
        self._drawdown = drawdown
        return self._snapshot(observed_at)

    def snapshot(self, at: datetime) -> PortfolioSnapshot:
        """Return a snapshot from the most recently applied fills or marks."""
        observed_at = _normalize_aware_datetime(at)
        for instrument_id in self._positions:
            if instrument_id not in self._market_prices:
                raise ValueError(f"missing price for open position: {instrument_id}")
        return self._snapshot(observed_at)

    def position_hash(self) -> str:
        """Hash a canonical representation of the current position reconciliation state."""
        payload = {
            "cash": _canonical_decimal(self._cash),
            "equity": _canonical_decimal(self._equity),
            "positions": [
                [
                    instrument_id,
                    _canonical_decimal(position.quantity),
                    _canonical_decimal(position.average_price),
                ]
                for instrument_id, position in sorted(self._positions.items())
            ],
        }
        encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _apply_buy(
        self, instrument_id: str, quantity: Decimal, notional: Decimal, fee: Decimal
    ) -> None:
        position = self._positions.get(instrument_id)
        if position is None:
            self._positions[instrument_id] = _PositionState(
                quantity=quantity,
                average_price=notional / quantity,
            )
        else:
            total_quantity = position.quantity + quantity
            position.average_price = (
                position.quantity * position.average_price + notional
            ) / total_quantity
            position.quantity = total_quantity
        self._cash -= notional + fee
        self._fees += fee

    def _apply_sell(
        self,
        instrument_id: str,
        position: _PositionState,
        quantity: Decimal,
        price: Decimal,
        notional: Decimal,
        fee: Decimal,
    ) -> None:
        self._cash += notional - fee
        self._gross_realized_pnl += (price - position.average_price) * quantity
        self._fees += fee
        remaining_quantity = position.quantity - quantity
        if remaining_quantity == Decimal("0"):
            del self._positions[instrument_id]
        else:
            position.quantity = remaining_quantity

    def _revalue(self) -> None:
        self._equity, self._peak_equity, self._drawdown = self._valuation(self._market_prices)

    def _valuation(self, prices: Mapping[str, Decimal]) -> tuple[Decimal, Decimal, Decimal]:
        market_value = sum(
            (
                position.quantity * prices[instrument_id]
                for instrument_id, position in self._positions.items()
            ),
            Decimal("0"),
        )
        equity = self._cash + market_value
        peak_equity = max(self._peak_equity, equity)
        drawdown = (
            (peak_equity - equity) / peak_equity
            if peak_equity > Decimal("0")
            else Decimal("0")
        )
        return equity, peak_equity, drawdown

    def _snapshot(self, observed_at: datetime) -> PortfolioSnapshot:
        positions = tuple(
            Position(
                instrument_id=instrument_id,
                quantity=position.quantity,
                average_price=position.average_price,
                market_price=self._market_prices[instrument_id],
                unrealized_pnl=(self._market_prices[instrument_id] - position.average_price)
                * position.quantity,
            )
            for instrument_id, position in sorted(self._positions.items())
        )
        gross_exposure = sum(
            (position.quantity * position.market_price for position in positions),
            Decimal("0"),
        )
        return PortfolioSnapshot(
            currency=self._currency,
            cash=self._cash,
            equity=self._equity,
            peak_equity=self._peak_equity,
            gross_exposure=gross_exposure,
            daily_pnl=self._equity - self._starting_cash,
            realized_pnl=self._gross_realized_pnl - self._fees,
            positions=positions,
            observed_at=observed_at,
        )


def _normalize_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("snapshot timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _canonical_decimal(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    return format(value.normalize(), "f")


def _require_positive_decimal(value: object, *, field_name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= Decimal("0"):
        raise ValueError(f"{field_name} must be a finite Decimal greater than zero")
    return value


def _require_nonnegative_decimal(value: object, *, field_name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value < Decimal("0"):
        raise ValueError(f"{field_name} must be a finite Decimal greater than or equal to zero")
    return value
