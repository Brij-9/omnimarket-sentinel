from datetime import UTC, datetime, timedelta

import pytest

from market_sentinel.domain.clock import FrozenClock


def test_frozen_clock_is_deterministic() -> None:
    instant = datetime(2026, 8, 9, 9, tzinfo=UTC)
    assert FrozenClock(instant).now() == instant


def test_frozen_clock_rejects_backward_movement() -> None:
    clock = FrozenClock(datetime(2026, 8, 9, 9, tzinfo=UTC))

    with pytest.raises(ValueError, match="cannot move backward"):
        clock.advance(timedelta(seconds=-1))
