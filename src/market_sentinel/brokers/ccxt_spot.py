"""Offline-testable, spot-only CCXT adapter with sandbox-first preflight."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from market_sentinel.brokers._records import (
    broker_order,
    decimal,
    mapping,
    position,
    symbol_from_instrument,
    value,
)
from market_sentinel.brokers.preflight import PreflightReport, gate
from market_sentinel.config import Settings
from market_sentinel.domain import (
    AssetClass,
    BrokerOrder,
    MarketSnapshot,
    OrderIntent,
    OrderType,
    Position,
)
from market_sentinel.domain.enums import OperatingMode
from market_sentinel.execution.base import BrokerCapabilities


class CcxtExchange(Protocol):
    has: Mapping[str, object]

    def set_sandbox_mode(self, enabled: bool) -> None: ...
    def load_markets(self) -> Mapping[str, object]: ...
    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: str,
        price: str | None,
        params: Mapping[str, Any],
    ) -> object: ...
    def fetch_order(self, order_id: str) -> object: ...
    def fetch_positions(self) -> Sequence[object]: ...


CcxtExchangeFactory = Callable[[str, Mapping[str, Any]], CcxtExchange]


class CcxtSpotBroker:
    broker_name = "ccxt-spot"

    def __init__(
        self,
        settings: Settings,
        *,
        exchange: CcxtExchange | None = None,
        exchange_factory: CcxtExchangeFactory | None = None,
    ) -> None:
        self._settings = settings
        self._exchange = exchange
        self._exchange_factory = exchange_factory
        self._markets: Mapping[str, object] = {}

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        exchange: CcxtExchange | None = None,
        exchange_factory: CcxtExchangeFactory | None = None,
    ) -> CcxtSpotBroker:
        return cls(settings, exchange=exchange, exchange_factory=exchange_factory)

    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            broker=self.broker_name,
            supported_asset_classes=frozenset({AssetClass.CRYPTO_SPOT}),
            supported_order_types=frozenset({OrderType.MARKET, OrderType.LIMIT}),
            supports_fractional_quantity=True,
            supports_notional_orders=False,
            supports_partial_fills=True,
            supports_shorting=False,
            supports_leverage=False,
            supports_derivatives=False,
            supports_cancel=False,
            is_paper=self._settings.ccxt_sandbox,
        )

    def preflight(self) -> PreflightReport:
        settings = self._settings
        gates = [
            gate("MARKET_SENTINEL_MODE", settings.mode is OperatingMode.LIVE_SMALL),
            gate("CCXT_LIVE_TRADING_ENABLED", settings.ccxt_live_trading_enabled),
            gate("CCXT_REAL_API_ENABLED", settings.ccxt_real_api_enabled),
            gate("CCXT_EXCHANGE_ID_CONFIGURED", bool(settings.ccxt_exchange_id)),
            gate("CCXT_SPOT_ONLY", True),
            gate(
                "CCXT_LOCAL_CREDENTIALS_PRESENT",
                bool(settings.ccxt_api_key and settings.ccxt_secret),
            ),
            gate(
                "CCXT_WITHDRAWALS_DISABLED_CONFIRMED", settings.ccxt_withdrawals_disabled_confirmed
            ),
            gate("CCXT_IP_RESTRICTED_CONFIRMED", settings.ccxt_ip_restricted_confirmed),
            gate(
                "CCXT_NO_SANDBOX_ACKNOWLEDGED",
                settings.ccxt_sandbox or settings.ccxt_no_sandbox_acknowledged,
            ),
        ]
        exchange = self._get_exchange()
        if all(item.passed for item in gates) and exchange is not None:
            try:
                if settings.ccxt_sandbox:
                    exchange.set_sandbox_mode(True)
                self._markets = exchange.load_markets()
                gates.append(gate("CCXT_SPOT_MARKETS_AVAILABLE", _valid_market_set(self._markets)))
                gates.append(
                    gate(
                        "CCXT_CREATE_ORDER_SUPPORTED",
                        bool(exchange.has.get("createOrder", False)),
                    )
                )
            except Exception:
                gates.append(gate("CCXT_MARKETS_ACCESSIBLE", False, "MARKETS_UNAVAILABLE"))
        else:
            gates.append(gate("CCXT_MARKETS_ACCESSIBLE", False, "MARKETS_UNAVAILABLE"))
        return PreflightReport(broker=self.broker_name, gates=tuple(gates))

    def submit(self, intent: OrderIntent, snapshot: MarketSnapshot) -> BrokerOrder:
        del snapshot
        self._require_ready()
        if intent.product not in {"spot", "cash"}:
            raise ValueError("CCXT adapter is spot-only; leverage and derivatives are rejected")
        if intent.notional is not None:
            raise ValueError("CCXT adapter requires an explicit base quantity")
        if intent.order_type not in {OrderType.MARKET, OrderType.LIMIT}:
            raise ValueError("CCXT market-order emulation and unsupported order types are rejected")
        symbol = symbol_from_instrument(intent.instrument_id)
        market = self._markets.get(symbol)
        if market is None or not _valid_market(market):
            raise ValueError("spot market minimum or precision is unavailable")
        assert intent.quantity is not None
        limits = mapping(value(market, "limits", {}))
        amount_limits = mapping(value(limits, "amount", {}))
        if intent.quantity < decimal(amount_limits.get("min", 0)):
            raise ValueError("quantity is below the configured spot market minimum")
        exchange = self._require_exchange()
        price = None if intent.limit_price is None else str(intent.limit_price)
        result = exchange.create_order(
            symbol,
            intent.order_type.value,
            intent.side.value,
            str(intent.quantity),
            price,
            {"clientOrderId": intent.intent_id},
        )
        return broker_order(
            result, broker=self.broker_name, default_client_order_id=intent.intent_id
        )

    def get_order(self, order_id: str) -> BrokerOrder:
        return broker_order(self._require_exchange().fetch_order(order_id), broker=self.broker_name)

    def get_order_by_client_id(self, client_intent_id: str) -> BrokerOrder:
        return self.get_order(client_intent_id)

    def list_orders(self) -> tuple[BrokerOrder, ...]:
        return ()

    def open_orders(self) -> tuple[BrokerOrder, ...]:
        return ()

    def reconcile_unknown_fills(
        self,
        authoritative_order: BrokerOrder,
        new_fills: tuple[object, ...],
        *,
        instrument: object,
    ) -> BrokerOrder:
        del new_fills, instrument
        return authoritative_order

    def cancel(self, order_id: str, *, at: object) -> BrokerOrder:
        del order_id, at
        raise NotImplementedError("cancellation requires an explicit injected exchange capability")

    def positions(self) -> tuple[Position, ...]:
        return tuple(
            position(item, broker=self.broker_name)
            for item in self._require_exchange().fetch_positions()
        )

    def _get_exchange(self) -> CcxtExchange | None:
        if self._exchange is None and self._exchange_factory is not None:
            self._exchange = self._exchange_factory(
                self._settings.ccxt_exchange_id,
                {"enableRateLimit": True},
            )
        return self._exchange

    def _require_ready(self) -> None:
        if not self.preflight().ready:
            raise PermissionError("broker preflight is not ready")

    def _require_exchange(self) -> CcxtExchange:
        exchange = self._get_exchange()
        if exchange is None:
            raise PermissionError("injected CCXT exchange is required")
        return exchange


def _valid_market_set(markets: Mapping[str, object]) -> bool:
    return any(_valid_market(market) for market in markets.values())


def _valid_market(market: object) -> bool:
    precision = mapping(value(market, "precision", {}))
    limits = mapping(value(market, "limits", {}))
    amount_limits = mapping(value(limits, "amount", {}))
    cost_limits = mapping(value(limits, "cost", {}))
    return (
        bool(value(market, "spot", False))
        and bool(value(market, "active", False))
        and decimal(value(precision, "amount", 0)) > 0
        and decimal(value(precision, "price", 0)) > 0
        and decimal(value(amount_limits, "min", 0)) > 0
        and decimal(value(cost_limits, "min", 0)) > 0
    )
