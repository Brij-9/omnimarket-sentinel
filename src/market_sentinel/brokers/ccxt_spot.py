"""Fail-closed native-spot CCXT adapter using only injected boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol, TypeVar

from market_sentinel.brokers._records import (
    broker_order,
    decimal,
    mapping,
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

_T = TypeVar("_T")
_MAX_VALUATION_AGE_SECONDS = 300


@dataclass(frozen=True, slots=True, repr=False)
class CcxtConnectionConfig:
    exchange_id: str
    enable_rate_limit: bool
    api_key: str = ""
    secret: str = ""

    def __repr__(self) -> str:
        present = bool(self.api_key and self.secret)
        return (
            "CcxtConnectionConfig("
            f"exchange_id={self.exchange_id!r}, credentials_present={present!r})"
        )


class CcxtExchange(Protocol):
    id: str
    enableRateLimit: bool
    options: Mapping[str, object]
    has: Mapping[str, object]

    def set_sandbox_mode(self, enabled: bool) -> None: ...
    def load_markets(self) -> Mapping[str, object]: ...
    def amount_to_precision(self, symbol: str, amount: str) -> str: ...
    def price_to_precision(self, symbol: str, price: str) -> str: ...
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
    def fetch_order_by_client_id(self, client_order_id: str) -> object: ...
    def fetch_balance(self) -> Mapping[str, object]: ...


class SpotValuationProvider(Protocol):
    def value_spot(
        self, exchange_id: str, currency: str, quote_currency: str, quantity: Decimal
    ) -> object: ...


CcxtExchangeFactory = Callable[[CcxtConnectionConfig], CcxtExchange]


class CcxtSpotBroker:
    broker_name = "ccxt-spot"

    def __init__(
        self,
        settings: Settings,
        *,
        exchange: CcxtExchange | None = None,
        exchange_factory: CcxtExchangeFactory | None = None,
        valuation_provider: SpotValuationProvider | None = None,
        quote_currency: str = "USDT",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings, self._exchange, self._exchange_factory = (
            settings,
            exchange,
            exchange_factory,
        )
        self._valuation, self._quote, self._clock = valuation_provider, quote_currency, clock
        self._markets: Mapping[str, object] = {}
        self._prepared = False

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: object) -> CcxtSpotBroker:
        return cls(settings, **kwargs)  # type: ignore[arg-type]

    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            self.broker_name,
            frozenset({AssetClass.CRYPTO_SPOT}),
            frozenset({OrderType.MARKET, OrderType.LIMIT}),
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            self._settings.ccxt_sandbox,
        )

    def preflight(self) -> PreflightReport:
        s = self._settings
        gates = [
            gate("MARKET_SENTINEL_MODE", s.mode is OperatingMode.LIVE_SMALL),
            gate("CCXT_LIVE_TRADING_ENABLED", s.ccxt_live_trading_enabled),
            gate("CCXT_REAL_API_ENABLED", s.ccxt_real_api_enabled),
            gate("CCXT_EXCHANGE_ID_CONFIGURED", bool(s.ccxt_exchange_id)),
            gate("CCXT_SPOT_ONLY", True),
            gate("CCXT_LOCAL_CREDENTIALS_PRESENT", bool(s.ccxt_api_key and s.ccxt_secret)),
            gate("CCXT_WITHDRAWALS_DISABLED_CONFIRMED", s.ccxt_withdrawals_disabled_confirmed),
            gate("CCXT_IP_RESTRICTED_CONFIRMED", s.ccxt_ip_restricted_confirmed),
            gate("CCXT_NO_SANDBOX_ACKNOWLEDGED", s.ccxt_sandbox or s.ccxt_no_sandbox_acknowledged),
        ]
        if not all(g.passed for g in gates):
            gates.extend(
                [
                    gate("CCXT_EXCHANGE_CONFIGURED", False, "EXCHANGE_INVALID"),
                    gate("CCXT_SPOT_MARKETS_AVAILABLE", False, "MARKETS_UNAVAILABLE"),
                    gate("CCXT_CREATE_ORDER_SUPPORTED", False, "ORDERS_UNSUPPORTED"),
                ]
            )
            return PreflightReport(self.broker_name, tuple(gates))
        try:
            exchange = self._prepare_exchange()
            gates.extend(
                [
                    gate("CCXT_EXCHANGE_CONFIGURED", True),
                    gate("CCXT_SPOT_MARKETS_AVAILABLE", _valid_market_set(self._markets)),
                    gate("CCXT_CREATE_ORDER_SUPPORTED", exchange.has.get("createOrder") is True),
                ]
            )
        except RuntimeError:
            gates.extend(
                [
                    gate("CCXT_EXCHANGE_CONFIGURED", False, "EXCHANGE_INVALID"),
                    gate("CCXT_SPOT_MARKETS_AVAILABLE", False, "MARKETS_UNAVAILABLE"),
                    gate("CCXT_CREATE_ORDER_SUPPORTED", False, "ORDERS_UNSUPPORTED"),
                ]
            )
        return PreflightReport(self.broker_name, tuple(gates))

    def submit(self, intent: OrderIntent, snapshot: MarketSnapshot) -> BrokerOrder:
        self._require_ready()
        exchange = self._prepare_exchange()
        if intent.product not in {"spot", "cash"}:
            raise ValueError("CCXT adapter is spot-only; leverage and derivatives are rejected")
        if intent.notional is not None or intent.quantity is None:
            raise ValueError("CCXT adapter requires an explicit base quantity")
        if intent.order_type not in {OrderType.MARKET, OrderType.LIMIT}:
            raise ValueError("CCXT market-order emulation and unsupported order types are rejected")
        symbol = symbol_from_instrument(intent.instrument_id)
        market = self._markets.get(symbol)
        if market is None or not _valid_market(market):
            raise ValueError("spot market minimum or precision is unavailable")
        if (
            intent.order_type is OrderType.MARKET
            and exchange.has.get("createMarketOrder") is not True
        ):
            raise ValueError("CCXT requires exact native market-order capability")
        amount = decimal(intent.quantity, positive=True)
        if (
            decimal(
                self._call(lambda: exchange.amount_to_precision(symbol, str(amount))), positive=True
            )
            != amount
        ):
            raise ValueError("amount precision normalization changes the order")
        price = intent.limit_price
        if (
            price is not None
            and decimal(
                self._call(lambda: exchange.price_to_precision(symbol, str(price))), positive=True
            )
            != price
        ):
            raise ValueError("price precision normalization changes the order")
        limits = mapping(value(market, "limits"))
        amount_limits = mapping(value(limits, "amount"))
        cost_limits = mapping(value(limits, "cost"))
        if amount < decimal(value(amount_limits, "min"), positive=True):
            raise ValueError("quantity is below the configured spot market minimum")
        reference = (
            price
            if price is not None
            else _snapshot_reference(snapshot, intent.instrument_id, self._now())
        )
        if amount * reference < decimal(value(cost_limits, "min"), positive=True):
            raise ValueError("order cost is below the configured spot market minimum")
        return self._order(
            lambda: exchange.create_order(
                symbol,
                intent.order_type.value,
                intent.side.value,
                str(amount),
                None if price is None else str(price),
                {"clientOrderId": intent.intent_id},
            ),
            intent.intent_id,
        )

    def get_order(self, order_id: str) -> BrokerOrder:
        return self._order(lambda: self._prepare_exchange().fetch_order(order_id))

    def get_order_by_client_id(self, client_intent_id: str) -> BrokerOrder:
        return self._order(
            lambda: self._prepare_exchange().fetch_order_by_client_id(client_intent_id)
        )

    def list_orders(self) -> tuple[BrokerOrder, ...]:
        return ()

    def open_orders(self) -> tuple[BrokerOrder, ...]:
        return ()

    def reconcile_unknown_fills(
        self, authoritative_order: BrokerOrder, new_fills: tuple[object, ...], *, instrument: object
    ) -> BrokerOrder:
        del new_fills, instrument
        return authoritative_order

    def cancel(self, order_id: str, *, at: object) -> BrokerOrder:
        del order_id, at
        raise NotImplementedError("CCXT spot cancellation is not advertised")

    def positions(self) -> tuple[Position, ...]:
        try:
            now = self._now()
            balance = self._call(self._prepare_exchange().fetch_balance)
            total = _standard_total(balance)
            positions: list[Position] = []
            for currency, raw_quantity in total.items():
                quantity = decimal(raw_quantity, nonnegative=True)
                if quantity == 0 or currency == self._quote:
                    continue
                positions.append(self._valued_position(currency, quantity, now))
            return tuple(positions)
        except Exception:
            pass
        raise RuntimeError("ccxt client operation failed")

    def _valued_position(self, currency: str, quantity: Decimal, now: datetime) -> Position:
        if self._valuation is None:
            raise RuntimeError("ccxt client operation failed")
        valuation = self._valuation
        assert valuation is not None
        record = mapping(
            self._call(
                lambda: valuation.value_spot(
                    self._settings.ccxt_exchange_id, currency, self._quote, quantity
                )
            )
        )
        if record.get("instrument_id") != f"{currency}/{self._quote}@{self.broker_name}":
            raise RuntimeError("ccxt client operation failed")
        observed_at = record.get("observed_at")
        max_age = record.get("max_age_seconds")
        if (
            type(observed_at) is not datetime
            or observed_at.tzinfo is None
            or observed_at.utcoffset() is None
            or type(max_age) is not int
            or not 0 <= max_age <= _MAX_VALUATION_AGE_SECONDS
        ):
            raise RuntimeError("ccxt client operation failed")
        observed_at = observed_at.astimezone(UTC)
        if observed_at > now or now - observed_at > timedelta(seconds=max_age):
            raise RuntimeError("ccxt client operation failed")
        average, market = (
            decimal(record.get("average_price"), positive=True),
            decimal(record.get("market_price"), positive=True),
        )
        return Position(
            instrument_id=str(record["instrument_id"]),
            quantity=quantity,
            average_price=average,
            market_price=market,
            unrealized_pnl=(market - average) * quantity,
        )

    def _prepare_exchange(self) -> CcxtExchange:
        exchange = self._get_exchange()
        if self._prepared:
            return exchange
        if (
            exchange.id != self._settings.ccxt_exchange_id
            or exchange.enableRateLimit is not True
            or exchange.options.get("defaultType") != "spot"
        ):
            raise RuntimeError("ccxt client operation failed")
        if self._settings.ccxt_sandbox:
            self._call(lambda: exchange.set_sandbox_mode(True))
        self._markets = self._call(exchange.load_markets)
        self._prepared = True
        return exchange

    def _order(self, operation: Callable[[], object], client_id: str = "") -> BrokerOrder:
        return self._call(
            lambda: broker_order(
                operation(), broker=self.broker_name, default_client_order_id=client_id
            )
        )

    def _call(self, operation: Callable[[], _T]) -> _T:
        try:
            return operation()
        except Exception:
            pass
        raise RuntimeError("ccxt client operation failed")

    def _get_exchange(self) -> CcxtExchange:
        if self._exchange is None:
            if self._exchange_factory is None:
                raise RuntimeError("ccxt client operation failed")
            config = CcxtConnectionConfig(
                self._settings.ccxt_exchange_id,
                True,
                self._settings.ccxt_api_key,
                self._settings.ccxt_secret,
            )
            factory = self._exchange_factory
            assert factory is not None
            self._exchange = self._call(lambda: factory(config))
        return self._exchange

    def _now(self) -> datetime:
        if self._clock is None:
            raise ValueError("snapshot reference is unavailable")
        now = self._call(self._clock)
        if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("snapshot reference is unavailable")
        return now.astimezone(UTC)

    def _require_ready(self) -> None:
        if not self.preflight().ready:
            raise PermissionError("broker preflight is not ready")


def _standard_total(balance: object) -> Mapping[str, object]:
    data = mapping(balance)
    total = mapping(data.get("total"))
    for currency, amount in total.items():
        if not isinstance(currency, str) or not currency or currency != currency.strip():
            raise RuntimeError("ccxt client operation failed")
        decimal(amount, nonnegative=True)
    return total


def _snapshot_reference(snapshot: MarketSnapshot, instrument_id: str, now: datetime) -> Decimal:
    if (
        snapshot.instrument_id != instrument_id
        or snapshot.observed_at > now
        or snapshot.source_at > now
        or snapshot.is_stale(now)
        or not snapshot.bars
    ):
        raise ValueError("snapshot reference is unavailable")
    return decimal(snapshot.bars[-1].close, positive=True)


def _valid_market_set(markets: Mapping[str, object]) -> bool:
    return any(_valid_market(m) for m in markets.values())


def _valid_market(market: object) -> bool:
    try:
        precision, limits = mapping(value(market, "precision")), mapping(value(market, "limits"))
        amount, cost = mapping(value(limits, "amount")), mapping(value(limits, "cost"))
        return (
            value(market, "spot") is True
            and value(market, "active") is True
            and decimal(value(precision, "amount"), positive=True) > 0
            and decimal(value(precision, "price"), positive=True) > 0
            and decimal(value(amount, "min"), positive=True) > 0
            and decimal(value(cost, "min"), positive=True) > 0
        )
    except ValueError:
        return False
