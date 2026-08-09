"""Fail-closed Groww adapter with local-gates-first authentication."""

from __future__ import annotations

import ipaddress
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol, TypeVar

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

_T = TypeVar("_T")


class GrowwClient(Protocol):
    def profile(self) -> object: ...
    def capabilities(self) -> object: ...
    def instrument_capabilities(self, symbol: str) -> object: ...
    def submit_order(self, payload: Mapping[str, Any]) -> object: ...
    def get_order(self, order_id: str) -> object: ...
    def get_order_by_client_id(self, client_order_id: str) -> object: ...
    def cancel_order(self, order_id: str) -> object: ...
    def positions(self) -> Sequence[object]: ...


@dataclass(frozen=True, slots=True)
class GrowwSession:
    client: GrowwClient
    expires_at: datetime


class GrowwAuthProvider(Protocol):
    def authenticated_session(self, settings: Settings, now: datetime) -> GrowwSession: ...


class GrowwBroker:
    broker_name = "groww"

    def __init__(
        self,
        settings: Settings,
        *,
        client: GrowwClient | None = None,
        auth_provider: GrowwAuthProvider | None = None,
        access_token_expires_at: datetime | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings, self._client, self._auth_provider = settings, client, auth_provider
        self._access_token_expires_at, self._clock = access_token_expires_at, clock

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        client: GrowwClient | None = None,
        auth_provider: GrowwAuthProvider | None = None,
        access_token_expires_at: datetime | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> GrowwBroker:
        return cls(
            settings,
            client=client,
            auth_provider=auth_provider,
            access_token_expires_at=access_token_expires_at,
            clock=clock,
        )

    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            self.broker_name,
            frozenset({AssetClass.EQUITY}),
            frozenset(OrderType),
            False,
            False,
            True,
            False,
            False,
            False,
            True,
            False,
        )

    def preflight(self) -> PreflightReport:
        s = self._settings
        local = [
            gate("GROWW_PRIMARY_BROKER", s.primary_broker == self.broker_name),
            gate("MARKET_SENTINEL_MODE", s.mode is OperatingMode.LIVE_SMALL),
            gate("INDIA_LIVE_TRADING_ENABLED", s.india_live_trading_enabled),
            gate("INDIA_ALGO_COMPLIANCE_VERIFIED", s.india_algo_compliance_verified),
            gate("GROWW_REAL_API_ENABLED", s.groww_real_api_enabled),
            gate("GROWW_API_SUBSCRIPTION_ACTIVE", s.groww_api_subscription_active),
            gate("GROWW_PROTECTED_ORDER_CLIENT", s.groww_protected_order_client),
            gate("GROWW_STATIC_OUTBOUND_IPV4", _is_public_ipv4(s.groww_static_outbound_ip)),
            gate("GROWW_STATIC_IP_ALLOWLISTED", s.groww_static_ip_allowlisted),
            gate("GROWW_BROKER_APPROVED_ALGO_ID", bool(s.groww_algo_id)),
            gate(
                "GROWW_LOCAL_CREDENTIALS_PRESENT",
                bool(s.groww_access_token) or bool(s.groww_api_key and s.groww_secret_key),
            ),
        ]
        if not all(item.passed for item in local):
            local.extend(
                [
                    gate("GROWW_AUTH_SESSION_FRESH", False, "AUTH_NOT_ATTEMPTED"),
                    gate("GROWW_READ_ONLY_PROFILE_ACCESS", False, "PROFILE_UNAVAILABLE"),
                ]
            )
            return PreflightReport(self.broker_name, tuple(local))
        try:
            now = self._now()
            client, expiry = self._authenticated_client(now)
            fresh = _is_fresh(expiry, now)
            local.append(gate("GROWW_AUTH_SESSION_FRESH", fresh))
            if not fresh:
                return PreflightReport(self.broker_name, tuple(local))
            profile = self._call(client.profile)
            capabilities = self._call(client.capabilities)
            local.extend(
                [
                    gate("GROWW_PROFILE_ACTIVE", value(profile, "active") is True),
                    gate(
                        "GROWW_REGULAR_SESSION_SUPPORTED",
                        value(capabilities, "regular_session") is True,
                    ),
                    gate(
                        "GROWW_PROTECTED_ORDERS_SUPPORTED",
                        value(capabilities, "protected_orders") is True,
                    ),
                ]
            )
        except RuntimeError:
            local.extend(
                [
                    gate("GROWW_AUTH_SESSION_FRESH", False, "AUTH_UNAVAILABLE"),
                    gate("GROWW_READ_ONLY_PROFILE_ACCESS", False, "PROFILE_UNAVAILABLE"),
                ]
            )
        return PreflightReport(self.broker_name, tuple(local))

    def submit(self, intent: OrderIntent, snapshot: MarketSnapshot) -> BrokerOrder:
        del snapshot
        self._require_ready()
        if intent.product != "cash" or intent.session != "regular":
            raise ValueError("Groww adapter supports protected regular-session cash orders only")
        if (
            intent.notional is not None
            or intent.quantity is None
            or intent.quantity != intent.quantity.to_integral_value()
        ):
            raise ValueError("Groww quantity must be an integer lot")
        client = self._require_client()
        symbol = symbol_from_instrument(intent.instrument_id)
        capability = self._call(lambda: client.instrument_capabilities(symbol))
        if not _supports_intent(capability, intent, symbol):
            raise ValueError("Groww instrument or protected-order capability is unsupported")
        payload: dict[str, Any] = {
            "symbol": symbol,
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
                for item in self._require_client().positions()
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
        raise RuntimeError("groww client operation failed")

    def _now(self) -> datetime:
        if self._clock is None:
            raise RuntimeError("groww client operation failed")
        now = self._call(self._clock)
        if now.tzinfo is None or now.utcoffset() is None:
            raise RuntimeError("groww client operation failed")
        return now.astimezone(UTC)

    def _authenticated_client(self, now: datetime) -> tuple[GrowwClient, datetime]:
        if self._client is not None:
            if self._access_token_expires_at is None:
                raise RuntimeError("groww client operation failed")
            return self._client, self._access_token_expires_at
        if self._auth_provider is None:
            raise RuntimeError("groww client operation failed")
        provider = self._auth_provider
        assert provider is not None
        session = self._call(lambda: provider.authenticated_session(self._settings, now))
        if not isinstance(session, GrowwSession):
            raise RuntimeError("groww client operation failed")
        self._client = session.client
        return session.client, session.expires_at

    def _require_ready(self) -> None:
        if not self.preflight().ready:
            raise PermissionError("broker preflight is not ready")

    def _require_client(self) -> GrowwClient:
        if self._client is None:
            raise PermissionError("injected Groww client is required")
        return self._client


def _is_public_ipv4(value_: str) -> bool:
    try:
        address = ipaddress.ip_address(value_)
    except ValueError:
        return False
    return (
        isinstance(address, ipaddress.IPv4Address)
        and address.is_global is True
        and not address.is_multicast
        and not address.is_reserved
    )


def _is_fresh(expiry: object, now: datetime) -> bool:
    return (
        isinstance(expiry, datetime)
        and expiry.tzinfo is not None
        and expiry.utcoffset() is not None
        and expiry.astimezone(UTC) > now
    )


def _supports_intent(capability: object, intent: OrderIntent, symbol: str) -> bool:
    try:
        data = mapping(capability)
        products = data["products"]
        sessions = data["sessions"]
        order_types = data["order_types"]
        lot_size = decimal(data["lot_size"], positive=True)
        return (
            data.get("symbol") == symbol
            and data.get("tradable") is True
            and data.get("protected_orders") is True
            and isinstance(products, tuple)
            and intent.product in products
            and isinstance(sessions, tuple)
            and intent.session in sessions
            and isinstance(order_types, tuple)
            and intent.order_type.value in order_types
            and lot_size == Decimal("1")
        )
    except (KeyError, ValueError):
        return False
