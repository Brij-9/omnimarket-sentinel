"""Tests for provider OHLCV normalization at the market-data boundary."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from market_sentinel.data.normalize import normalize_ohlcv


def test_normalizer_rejects_future_bars_before_other_row_errors() -> None:
    """Removing the cutoff comparison would accept a bar unavailable at the analysis time."""
    cutoff = datetime(2026, 8, 9, 10, tzinfo=UTC)
    rows = [
        {
            "at": "2026-08-09T10:01:00Z",
            "open": "10",
            "high": "11",
            "low": "9",
            "close": "10",
            "volume": "2",
        },
        {
            "at": "2026-08-09T09:59:00Z",
            "open": "10",
            "high": "11",
            "low": "9",
            "close": "10",
            "volume": "2",
        },
    ]

    with pytest.raises(ValueError, match="future bar"):
        normalize_ohlcv(rows, cutoff=cutoff)


def test_normalizer_preserves_decimal_prices_and_returns_an_immutable_tuple() -> None:
    """Replacing Decimal parsing or the tuple result would lose exact financial values."""
    cutoff = datetime(2026, 8, 9, 10, tzinfo=UTC)

    bars = normalize_ohlcv(
        [
            {
                "at": "2026-08-09T09:59:00Z",
                "open": "0.1",
                "high": "0.2",
                "low": "0.1",
                "close": "0.2",
                "volume": "2",
            }
        ],
        cutoff=cutoff,
    )

    assert isinstance(bars, tuple)
    assert bars[0].close == Decimal("0.2")
    assert type(bars[0].close) is Decimal


def test_normalizer_converts_offset_timestamp_to_utc() -> None:
    """Removing UTC conversion would retain provider-local time at the domain boundary."""
    cutoff = datetime(2026, 8, 9, 10, tzinfo=UTC)

    bars = normalize_ohlcv(
        [
            {
                "at": "2026-08-09T15:29:00+05:30",
                "open": "10",
                "high": "11",
                "low": "9",
                "close": "10",
                "volume": "2",
            }
        ],
        cutoff=cutoff,
    )

    assert bars[0].at == datetime(2026, 8, 9, 9, 59, tzinfo=UTC)
    assert bars[0].at.tzinfo is UTC


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {
                    "at": "2026-08-09T09:59:00",
                    "open": "10",
                    "high": "11",
                    "low": "9",
                    "close": "10",
                    "volume": "2",
                }
            ],
            "timezone-aware",
        ),
        (
            [
                {
                    "at": "2026-08-09T09:59:00Z",
                    "open": "10",
                    "high": "9",
                    "low": "8",
                    "close": "10",
                    "volume": "2",
                }
            ],
            "high",
        ),
        (
            [
                {
                    "at": "2026-08-09T09:59:00Z",
                    "open": "10",
                    "high": "11",
                    "low": "10.5",
                    "close": "10",
                    "volume": "2",
                }
            ],
            "low",
        ),
    ],
)
def test_normalizer_rejects_invalid_timestamp_and_price_ranges(
    rows: list[dict[str, str]], message: str
) -> None:
    """Removing boundary validation would permit malformed provider data into strategies."""
    cutoff = datetime(2026, 8, 9, 10, tzinfo=UTC)

    with pytest.raises(ValueError, match=message):
        normalize_ohlcv(rows, cutoff=cutoff)


def test_normalizer_rejects_nonascending_timestamps() -> None:
    """Replacing the strict timestamp comparison would allow duplicate or reversed bars."""
    cutoff = datetime(2026, 8, 9, 10, tzinfo=UTC)
    rows = [
        {
            "at": "2026-08-09T09:59:00Z",
            "open": "10",
            "high": "11",
            "low": "9",
            "close": "10",
            "volume": "2",
        },
        {
            "at": "2026-08-09T09:58:00Z",
            "open": "10",
            "high": "11",
            "low": "9",
            "close": "10",
            "volume": "2",
        },
    ]

    with pytest.raises(ValueError, match="ascending"):
        normalize_ohlcv(rows, cutoff=cutoff)


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=-10_000, max_value=10_000),
            st.integers(min_value=0, max_value=1_000),
            st.integers(min_value=0, max_value=1_000),
            st.integers(min_value=0, max_value=1_000),
        ),
        min_size=1,
        max_size=20,
    )
)
def test_normalizer_retains_valid_ohlc_order_and_ascending_timestamps(
    values: list[tuple[int, int, int, int]],
) -> None:
    """Broken Decimal conversion or ordering would violate valid provider-row invariants."""
    cutoff = datetime(2026, 8, 9, 10, tzinfo=UTC)
    start = cutoff - timedelta(minutes=len(values))
    rows = []
    for index, (base, open_offset, close_offset, range_size) in enumerate(values):
        open_price = Decimal(base) + Decimal(open_offset) / Decimal("100")
        close_price = Decimal(base) + Decimal(close_offset) / Decimal("100")
        low = min(open_price, close_price) - Decimal(range_size) / Decimal("100")
        high = max(open_price, close_price) + Decimal(range_size) / Decimal("100")
        rows.append(
            {
                "at": (start + timedelta(minutes=index)).isoformat(),
                "open": str(open_price),
                "high": str(high),
                "low": str(low),
                "close": str(close_price),
                "volume": str(index),
            }
        )

    bars = normalize_ohlcv(rows, cutoff=cutoff)

    assert tuple(bar.at for bar in bars) == tuple(sorted(bar.at for bar in bars))
    assert all(bar.low <= bar.open <= bar.high for bar in bars)
    assert all(bar.low <= bar.close <= bar.high for bar in bars)
