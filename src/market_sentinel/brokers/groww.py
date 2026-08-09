"""Offline-testable, fail-closed Groww adapter."""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping, Sequence
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


class GrowwClient(Protocol):
    def profile(self) -> object: ...
    def capabilities(self) -> object: ...
    def submit_order(self, payload: Mapping[str, Any]) -> object: ...
    def get_order(self, order_id: str) -> object: ...
    def positions(self) -> Sequence[object]: ...


class GrowwAuthProvider(Protocol):
    """Injected local auth provider; it must never persist or log credentials."""

    def authenticated_client(self, settings: Settings) -> GrowwClient: ...


class GrowwBroker:
    broker_name = "groww"

    def __init__(
        self,
        settings: Settings,
        *,
        client: GrowwClient | None = None,
        auth_provider: GrowwAuthProvider | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._auth_provider = auth_provider

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        client: GrowwClient | None = None,
        auth_provider: GrowwAuthProvider | None = None,
    ) -> GrowwBroker:
        return cls(settings, client=client, auth_provider=auth_provider)

    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            broker=self.broker_name,
            supported_asset_classes=frozenset({AssetClass.EQUITY}),
            supported_order_types=frozenset(
                {OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT}
            ),
            supports_fractional_quantity=False,
            supports_notional_orders=False,
            supports_partial_fills=True,
            supports_shorting=False,
            supports_leverage=False,
            supports_derivatives=False,
            supports_cancel=True,
            is_paper=False,
        )

    def preflight(self) -> PreflightReport:
        settings = self._settings
        local_credentials = bool(settings.groww_access_token) or bool(
            settings.groww_api_key and settings.groww_secret_key
        )
        gates = [
            gate("GROWW_PRIMARY_BROKER", settings.primary_broker == self.broker_name),
            gate("MARKET_SENTINEL_MODE", settings.mode is OperatingMode.LIVE_SMALL),
            gate("INDIA_LIVE_TRADING_ENABLED", settings.india_live_trading_enabled),
            gate("INDIA_ALGO_COMPLIANCE_VERIFIED", settings.india_algo_compliance_verified),
            gate("GROWW_REAL_API_ENABLED", settings.groww_real_api_enabled),
            gate("GROWW_API_SUBSCRIPTION_ACTIVE", settings.groww_api_subscription_active),
            gate("GROWW_PROTECTED_ORDER_CLIENT", settings.groww_protected_order_client),
            gate("GROWW_STATIC_OUTBOUND_IPV4", _is_public_ipv4(settings.groww_static_outbound_ip)),
            gate("GROWW_STATIC_IP_ALLOWLISTED", settings.groww_static_ip_allowlisted),
            gate("GROWW_BROKER_APPROVED_ALGO_ID", bool(settings.groww_algo_id)),
            gate("GROWW_LOCAL_CREDENTIALS_PRESENT", local_credentials),
        ]
        client = self._client_or_auth_client()
        if all(item.passed for item in gates) and client is not None:
            try:
                profile = client.profile()
                capabilities = client.capabilities()
                gates.extend(
                    [
                        gate("GROWW_PROFILE_ACTIVE", bool(value(profile, "active", False))),
                        gate(
                            "GROWW_REGULAR_SESSION_SUPPORTED",
                            bool(value(capabilities, "regular_session", False)),
                        ),
                        gate(
                            "GROWW_PROTECTED_ORDERS_SUPPORTED",
                            bool(value(capabilities, "protected_orders", False)),
                        ),
                    ]
                )
            except Exception:
                gates.append(gate("GROWW_READ_ONLY_PROFILE_ACCESS", False, "PROFILE_UNAVAILABLE"))
        else:
            gates.append(gate("GROWW_READ_ONLY_PROFILE_ACCESS", False, "PROFILE_UNAVAILABLE"))
        return PreflightReport(broker=self.broker_name, gates=tuple(gates))

    def submit(self, intent: OrderIntent, snapshot: MarketSnapshot) -> BrokerOrder:
        del snapshot
        self._require_ready()
        if intent.product != "cash" or intent.session != "regular":
            raise ValueError("Groww adapter supports protected regular-session cash orders only")
        if intent.notional is not None:
            raise ValueError("Groww adapter requires an explicit quantity")
        client = self._require_client()
        payload: dict[str, Any] = {
            "symbol": symbol_from_instrument(intent.instrument_id),
            "side": intent.side.value,
            "order_type": intent.order_type.value,
            "quantity": str(intent.quantity),
            "time_in_force": intent.time_in_force,
            "order_reference_id": intent.intent_id,
            "protected": True,
        }
        if intent.limit_price is not None:
            payload["limit_price"] = str(intent.limit_price)
        if intent.trigger_price is not None:
            payload["trigger_price"] = str(intent.trigger_price)
        return broker_order(
            client.submit_order(payload),
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
        del order_id, at
        raise NotImplementedError("cancellation requires an explicit injected client capability")

    def positions(self) -> tuple[Position, ...]:
        return tuple(
            position(item, broker=self.broker_name) for item in self._require_client().positions()
        )

    def _client_or_auth_client(self) -> GrowwClient | None:
        if self._client is None and self._auth_provider is not None:
            self._client = self._auth_provider.authenticated_client(self._settings)
        return self._client

    def _require_ready(self) -> None:
        if not self.preflight().ready:
            raise PermissionError("broker preflight is not ready")

    def _require_client(self) -> GrowwClient:
        client = self._client_or_auth_client()
        if client is None:
            raise PermissionError("injected Groww client is required")
        return client


def _is_public_ipv4(value_: str) -> bool:
    try:
        address = ipaddress.ip_address(value_)
    except ValueError:
        return False
    return isinstance(address, ipaddress.IPv4Address) and not address.is_unspecified
