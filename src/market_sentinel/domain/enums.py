"""String enums used by immutable domain records."""

from enum import StrEnum


class AssetClass(StrEnum):
    EQUITY = "equity"
    CRYPTO_SPOT = "crypto_spot"
    FUTURE = "future"
    OPTION = "option"
    FOREX = "forex"
    COMMODITY = "commodity"


class Horizon(StrEnum):
    INTRADAY = "intraday"
    SWING = "swing"


class OperatingMode(StrEnum):
    RESEARCH = "research"
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE_SMALL = "live-small"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class SignalDirection(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(StrEnum):
    PROPOSED = "proposed"
    RISK_APPROVED = "risk_approved"
    CONFIRMED = "confirmed"
    SUBMITTING = "submitting"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    UNKNOWN = "unknown"
