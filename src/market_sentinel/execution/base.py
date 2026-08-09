"""Broker-neutral execution capabilities and adapter protocol."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from market_sentinel.domain.enums import AssetClass, OrderType
from market_sentinel.domain.models import BrokerOrder, MarketSnapshot, OrderIntent, Position
from market_sentinel.operations.audit import AuditEvent


@dataclass(frozen=True, slots=True)
class BrokerCapabilities:
    """Immutable, explicit features discovered or guaranteed by an adapter."""

    broker: str
    supported_asset_classes: frozenset[AssetClass]
    supported_order_types: frozenset[OrderType]
    supports_fractional_quantity: bool
    supports_notional_orders: bool
    supports_partial_fills: bool
    supports_shorting: bool
    supports_leverage: bool
    supports_derivatives: bool
    supports_cancel: bool
    is_paper: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.broker, str)
            or not self.broker
            or self.broker != self.broker.strip()
        ):
            raise ValueError("broker capability identity must be nonempty and trimmed")
        if not isinstance(self.supported_asset_classes, frozenset) or not all(
            isinstance(item, AssetClass) for item in self.supported_asset_classes
        ):
            raise ValueError("supported_asset_classes must be a frozenset of AssetClass")
        if not isinstance(self.supported_order_types, frozenset) or not all(
            isinstance(item, OrderType) for item in self.supported_order_types
        ):
            raise ValueError("supported_order_types must be a frozenset of OrderType")
        flag_names = (
            "supports_fractional_quantity",
            "supports_notional_orders",
            "supports_partial_fills",
            "supports_shorting",
            "supports_leverage",
            "supports_derivatives",
            "supports_cancel",
            "is_paper",
        )
        if not all(type(getattr(self, name)) is bool for name in flag_names):
            raise ValueError("broker capability flags must be exact bool values")


@runtime_checkable
class BrokerAdapter(Protocol):
    """Minimum common execution surface shared by paper and future live adapters."""

    @property
    def broker_name(self) -> str: ...

    def capabilities(self) -> BrokerCapabilities: ...

    def submit(self, intent: OrderIntent, snapshot: MarketSnapshot) -> BrokerOrder: ...

    def get_order(self, order_id: str) -> BrokerOrder: ...

    def get_order_by_client_id(self, client_intent_id: str) -> BrokerOrder: ...

    def list_orders(self) -> tuple[BrokerOrder, ...]: ...

    def open_orders(self) -> tuple[BrokerOrder, ...]: ...

    def cancel(self, order_id: str, *, at: datetime) -> BrokerOrder: ...

    def positions(self) -> tuple[Position, ...]: ...


class AuditRecorder(Protocol):
    """Structural subset of the application audit facade used by execution."""

    def record_many(self, batch: tuple[AuditEvent, ...]) -> None: ...
