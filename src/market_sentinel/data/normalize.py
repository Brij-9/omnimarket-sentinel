"""Strict conversion of provider OHLCV rows into immutable domain bars."""

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from market_sentinel.domain import Bar


def normalize_ohlcv(
    rows: Iterable[Mapping[str, object]], *, cutoff: datetime
) -> tuple[Bar, ...]:
    """Validate provider rows and return UTC, ascending, exact-price bars.

    The cutoff represents the latest information usable for the current analysis.
    """
    normalized_cutoff = _normalize_aware_datetime(cutoff, field_name="cutoff")
    bars: list[Bar] = []
    previous_at: datetime | None = None
    for row in rows:
        at = _parse_timestamp(_required(row, "at"))
        if at > normalized_cutoff:
            raise ValueError("future bar is after cutoff")
        if previous_at is not None and at <= previous_at:
            raise ValueError("bar timestamps must be strictly ascending")

        open_price = _parse_decimal(_required(row, "open"), field_name="open")
        high = _parse_decimal(_required(row, "high"), field_name="high")
        low = _parse_decimal(_required(row, "low"), field_name="low")
        close = _parse_decimal(_required(row, "close"), field_name="close")
        volume = _parse_decimal(_required(row, "volume"), field_name="volume")
        if high < max(open_price, close):
            raise ValueError("high must be at least open and close")
        if low > min(open_price, close):
            raise ValueError("low must be at most open and close")
        if volume < Decimal("0"):
            raise ValueError("volume must be nonnegative")

        bars.append(
            Bar(
                at=at,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
        previous_at = at
    return tuple(bars)


def _required(row: Mapping[str, object], field_name: str) -> object:
    try:
        return row[field_name]
    except KeyError as error:
        raise ValueError(f"missing required OHLCV field: {field_name}") from error


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("bar timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("bar timestamp must be ISO-8601") from error
    return _normalize_aware_datetime(parsed, field_name="bar timestamp")


def _normalize_aware_datetime(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_decimal(value: object, *, field_name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, str)):
        raise ValueError(f"{field_name} must be a Decimal-compatible string or integer")
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{field_name} must be a valid Decimal") from error
    if not decimal_value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return decimal_value
