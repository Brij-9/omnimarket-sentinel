"""Offline-testable, fail-closed Alpaca adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from market_sentinel.brokers._records import (
    broker_order,
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


class AlpacaClient(Protocol):
    def get_account(self) -> object: ...
    def get_asset(self, symbol: str) -> object: ...
    def submit_order(self, payload: Mapping[str, Any]) -> object: ...
    def get_order(self, order_id: str) -> object: ...
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
            broker=self.broker_name,
            supported_asset_classes=frozenset({AssetClass.EQUITY}),
            supported_order_types=frozenset(OrderType),
            supports_fractional_quantity=True,
            supports_notional_orders=True,
            supports_partial_fills=True,
            supports_shorting=True,
            supports_leverage=False,
            supports_derivatives=False,
            supports_cancel=True,
            is_paper=self._settings.alpaca_trading_endpoint == _PAPER_ENDPOINT,
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
        if all(item.passed for item in gates) and self._client is not None:
            try:
                account = self._client.get_account()
                active = str(value(account, "status", "")).upper() == "ACTIVE"
                unblocked = not bool(value(account, "account_blocked", True))
                buying_power = _positive_decimal(value(account, "buying_power", 0))
                gates.extend(
                    [
                        gate("ALPACA_ACCOUNT_ACTIVE", active),
                        gate("ALPACA_ACCOUNT_UNBLOCKED", unblocked),
                        gate("ALPACA_SUFFICIENT_BUYING_POWER", buying_power),
                    ]
                )
            except Exception:
                gates.append(gate("ALPACA_ACCOUNT_ACCESSIBLE", False, "ACCOUNT_UNAVAILABLE"))
        else:
            gates.append(gate("ALPACA_ACCOUNT_ACCESSIBLE", False, "ACCOUNT_UNAVAILABLE"))
        return PreflightReport(broker=self.broker_name, gates=tuple(gates))

    def submit(self, intent: OrderIntent, snapshot: MarketSnapshot) -> BrokerOrder:
        del snapshot
        self._require_ready()
        if self._client is None:
            raise PermissionError("broker preflight is not ready")
        if intent.time_in_force not in {"day", "gtc"} or intent.session != "regular":
            raise ValueError("unsupported Alpaca order session")
        asset = self._client.get_asset(symbol_from_instrument(intent.instrument_id))
        if not bool(value(asset, "tradable", False)):
            raise ValueError("asset is not tradable")
        if intent.notional is not None and not bool(value(asset, "fractionable", False)):
            raise ValueError("notional order requires a fractionable asset")
        payload: dict[str, Any] = {
            "symbol": symbol_from_instrument(intent.instrument_id),
            "side": intent.side.value,
            "type": intent.order_type.value.replace("_", "_"),
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
        return broker_order(
            self._client.submit_order(payload),
            broker=self.broker_name,
            default_client_order_id=intent.intent_id,
        )

    def get_order(self, order_id: str) -> BrokerOrder:
        return broker_order(self._require_client().get_order(order_id), broker=self.broker_name)

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
        del at
        raise NotImplementedError("cancellation requires an explicit injected client capability")

    def positions(self) -> tuple[Position, ...]:
        return tuple(
            position(item, broker=self.broker_name)
            for item in self._require_client().get_positions()
        )

    def _require_ready(self) -> None:
        if not self.preflight().ready:
            raise PermissionError("broker preflight is not ready")

    def _require_client(self) -> AlpacaClient:
        if self._client is None:
            raise PermissionError("injected Alpaca client is required")
        return self._client


def _positive_decimal(value_: object) -> bool:
    try:
        return Decimal(str(value_)) > 0
    except (InvalidOperation, ValueError):
        return False
