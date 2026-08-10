"""Fail-closed Alpaca adapter with only injected client operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from typing import Any, Protocol, TypeVar

from market_sentinel.brokers._records import (
    broker_order,
    decimal,
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

_PAPER_ENDPOINT = "https://paper-api.alpaca.markets"
_LIVE_ENDPOINT = "https://api.alpaca.markets"
_T = TypeVar("_T")


class AlpacaClient(Protocol):
    def get_account(self) -> object: ...
    def get_asset(self, symbol: str) -> object: ...
    def submit_order(self, payload: Mapping[str, Any]) -> object: ...
    def get_order(self, order_id: str) -> object: ...
    def get_order_by_client_id(self, client_order_id: str) -> object: ...
    def cancel_order(self, order_id: str) -> object: ...
    def get_positions(self) -> Sequence[object]: ...


class AlpacaBroker:
    broker_name = "alpaca"

    def __init__(self, settings: Settings, *, client: AlpacaClient | None = None) -> None:
        self._settings = settings
        self._client = client

    @classmethod
    def from_settings(
        cls, settings: Settings, *, client: AlpacaClient | None = None
    ) -> AlpacaBroker:
        return cls(settings, client=client)

    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            self.broker_name,
            frozenset({AssetClass.EQUITY}),
            frozenset(OrderType),
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            self._settings.alpaca_trading_endpoint == _PAPER_ENDPOINT,
        )

    def preflight(self) -> PreflightReport:
        settings = self._settings
        gates = [
            gate("MARKET_SENTINEL_MODE", settings.mode is OperatingMode.LIVE_SMALL),
            gate("ALPACA_LIVE_TRADING_ENABLED", settings.alpaca_live_trading_enabled),
            gate("ALPACA_REAL_API_ENABLED", settings.alpaca_real_api_enabled),
            gate("ALPACA_LIVE_ENDPOINT", settings.alpaca_trading_endpoint == _LIVE_ENDPOINT),
            gate("ALPACA_ACCOUNT_ID_PRESENT", bool(settings.alpaca_account_id)),
            gate(
                "ALPACA_LOCAL_CREDENTIALS_PRESENT",
                bool(settings.alpaca_key_id and settings.alpaca_secret_key),
            ),
        ]
        if not all(item.passed for item in gates) or self._client is None:
            gates.extend(
                [
                    gate("ALPACA_ACCOUNT_ID_MATCHED", False, "ACCOUNT_UNAVAILABLE"),
                    gate("ALPACA_ACCOUNT_ACTIVE", False, "ACCOUNT_UNAVAILABLE"),
                    gate("ALPACA_ACCOUNT_UNBLOCKED", False, "ACCOUNT_UNAVAILABLE"),
                    gate("ALPACA_SUFFICIENT_BUYING_POWER", False, "ACCOUNT_UNAVAILABLE"),
                ]
            )
            return PreflightReport(self.broker_name, tuple(gates))
        try:
            account = self._call(self._client.get_account)
            gates.extend(
                [
                    gate(
                        "ALPACA_ACCOUNT_ID_MATCHED",
                        value(account, "id") == settings.alpaca_account_id,
                    ),
                    gate("ALPACA_ACCOUNT_ACTIVE", value(account, "status") == "ACTIVE"),
                    gate("ALPACA_ACCOUNT_UNBLOCKED", value(account, "account_blocked") is False),
                    gate(
                        "ALPACA_SUFFICIENT_BUYING_POWER",
                        _positive_money(value(account, "buying_power")),
                    ),
                ]
            )
        except RuntimeError:
            gates.extend(
                [
                    gate("ALPACA_ACCOUNT_ID_MATCHED", False, "ACCOUNT_UNAVAILABLE"),
                    gate("ALPACA_ACCOUNT_ACTIVE", False, "ACCOUNT_UNAVAILABLE"),
                    gate("ALPACA_ACCOUNT_UNBLOCKED", False, "ACCOUNT_UNAVAILABLE"),
                    gate("ALPACA_SUFFICIENT_BUYING_POWER", False, "ACCOUNT_UNAVAILABLE"),
                ]
            )
        return PreflightReport(self.broker_name, tuple(gates))

    def submit(self, intent: OrderIntent, snapshot: MarketSnapshot) -> BrokerOrder:
        del snapshot
        self._require_ready()
        if intent.time_in_force not in {"day", "gtc"} or intent.session != "regular":
            raise ValueError("unsupported Alpaca order session")
        client = self._require_client()
        asset = self._call(lambda: client.get_asset(symbol_from_instrument(intent.instrument_id)))
        if value(asset, "tradable") is not True:
            raise ValueError("asset is not tradable")
        if intent.notional is not None and value(asset, "fractionable") is not True:
            raise ValueError("notional order requires a fractionable asset")
        payload: dict[str, Any] = {
            "symbol": symbol_from_instrument(intent.instrument_id),
            "side": intent.side.value,
            "type": intent.order_type.value,
            "time_in_force": intent.time_in_force,
            "client_order_id": intent.intent_id,
        }
        if intent.quantity is not None:
            payload["qty"] = str(intent.quantity)
        else:
            payload["notional"] = str(intent.notional)
        if intent.limit_price is not None:
            payload["limit_price"] = str(intent.limit_price)
        if intent.trigger_price is not None:
            payload["stop_price"] = str(intent.trigger_price)
        return self._order(lambda: client.submit_order(payload), intent.intent_id)

    def get_order(self, order_id: str) -> BrokerOrder:
        return self._order(lambda: self._require_client().get_order(order_id))

    def get_order_by_client_id(self, client_intent_id: str) -> BrokerOrder:
        return self._order(lambda: self._require_client().get_order_by_client_id(client_intent_id))

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
        del at
        return self._order(lambda: self._require_client().cancel_order(order_id))

    def positions(self) -> tuple[Position, ...]:
        return self._call(
            lambda: tuple(
                position(item, broker=self.broker_name)
                for item in self._require_client().get_positions()
            )
        )

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
        raise RuntimeError("alpaca client operation failed")

    def _require_ready(self) -> None:
        if not self.preflight().ready:
            raise PermissionError("broker preflight is not ready")

    def _require_client(self) -> AlpacaClient:
        if self._client is None:
            raise PermissionError("injected Alpaca client is required")
        return self._client


def _positive_money(value_: object) -> bool:
    try:
        return decimal(value_, positive=True) > Decimal("0")
    except ValueError:
        return False
