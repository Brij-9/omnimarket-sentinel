from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from market_sentinel.domain.clock import FrozenClock
from market_sentinel.domain.enums import OrderStatus, Side
from market_sentinel.execution.reconcile import (
    KILL_SWITCH_ACKNOWLEDGEMENT,
    BrokerOpenOrderRecord,
    BrokerPositionRecord,
    BrokerReconciliationSnapshot,
    KillSwitchError,
    Reconciler,
)
from market_sentinel.operations.audit import AuditLog
from market_sentinel.portfolio.ledger import PortfolioLedger
from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventStore
from tests.factories import DEFAULT_INSTANT, fill


def _services(path: Path | None = None) -> tuple[FrozenClock, EventStore, Reconciler]:
    url = "sqlite+pysqlite:///:memory:" if path is None else f"sqlite+pysqlite:///{path}"
    clock = FrozenClock(DEFAULT_INSTANT)
    store = EventStore(create_engine_and_schema(url))
    return clock, store, Reconciler(audit_log=AuditLog(store, clock), clock=clock)


def _ledger() -> PortfolioLedger:
    ledger = PortfolioLedger(starting_cash=Decimal("100"), currency="USD")
    ledger.apply_fill(fill(quantity="1", price="10"))
    return ledger


def _snapshot(**changes: object) -> BrokerReconciliationSnapshot:
    values: dict[str, object] = {
        "broker": "alpaca",
        "currency": "USD",
        "cash": Decimal("90"),
        "positions": (BrokerPositionRecord("AAPL@alpaca", Side.BUY, Decimal("1")),),
        "open_orders": (),
        "observed_at": DEFAULT_INSTANT,
    }
    values.update(changes)
    return BrokerReconciliationSnapshot(**values)  # type: ignore[arg-type]


def test_compare_is_order_independent_and_persists_exact_hashes() -> None:
    """Provider ordering must not affect the report or its durable canonical evidence."""
    _clock, store, reconciler = _services()
    ledger = _ledger()
    expected = (
        BrokerOpenOrderRecord(
            "intent-a",
            "order-a",
            "AAPL@alpaca",
            Side.BUY,
            Decimal("1"),
            Decimal("0"),
            OrderStatus.ACKNOWLEDGED,
        ),
        BrokerOpenOrderRecord(
            "intent-b",
            "order-b",
            "MSFT@alpaca",
            Side.SELL,
            Decimal("2"),
            Decimal("0.5"),
            OrderStatus.PARTIALLY_FILLED,
        ),
    )
    report = reconciler.compare(_snapshot(open_orders=tuple(reversed(expected))), ledger, expected)

    assert report.healthy is True
    assert report.reason_codes == ()
    [event] = tuple(store.stream("live-reconciliation"))
    assert event.payload["broker_hash"] == report.broker_hash
    assert event.payload["ledger_hash"] == report.ledger_hash


@pytest.mark.parametrize(
    ("changes", "expected_code"),
    [
        ({"cash": Decimal("89.99")}, "CASH_MISMATCH"),
        ({"currency": "EUR"}, "CURRENCY_MISMATCH"),
        ({"positions": ()}, "POSITION_MISSING"),
        (
            {"positions": (BrokerPositionRecord("MSFT@alpaca", Side.BUY, Decimal("1")),)},
            "POSITION_UNKNOWN",
        ),
        (
            {"positions": (BrokerPositionRecord("AAPL@alpaca", Side.SELL, Decimal("1")),)},
            "POSITION_SIDE_MISMATCH",
        ),
        (
            {"positions": (BrokerPositionRecord("AAPL@alpaca", Side.BUY, Decimal("2")),)},
            "POSITION_QUANTITY_MISMATCH",
        ),
    ],
)
def test_position_and_cash_mismatches_activate_persistent_kill_switch(
    changes: dict[str, object], expected_code: str, tmp_path: Path
) -> None:
    """Any exact account-state mismatch must survive a service restart as a kill switch."""
    db = tmp_path / "reconciliation.db"
    clock, _store, reconciler = _services(db)
    report = reconciler.compare(_snapshot(**changes), _ledger(), ())

    assert report.healthy is False
    assert expected_code in report.reason_codes
    restarted_store = EventStore(create_engine_and_schema(f"sqlite+pysqlite:///{db}"))
    restarted = Reconciler(audit_log=AuditLog(restarted_store, clock), clock=clock)
    assert restarted.kill_switch_active() is True


@pytest.mark.parametrize(
    ("actual", "expected_code"),
    [
        (
            BrokerOpenOrderRecord(
                "unknown",
                "order",
                "AAPL@alpaca",
                Side.BUY,
                Decimal("1"),
                Decimal("0"),
                OrderStatus.ACKNOWLEDGED,
            ),
            "ORDER_UNKNOWN",
        ),
        (
            BrokerOpenOrderRecord(
                "intent",
                "order",
                "MSFT@alpaca",
                Side.BUY,
                Decimal("1"),
                Decimal("0"),
                OrderStatus.ACKNOWLEDGED,
            ),
            "ORDER_INSTRUMENT_MISMATCH",
        ),
        (
            BrokerOpenOrderRecord(
                "intent",
                "order",
                "AAPL@alpaca",
                Side.SELL,
                Decimal("1"),
                Decimal("0"),
                OrderStatus.ACKNOWLEDGED,
            ),
            "ORDER_SIDE_MISMATCH",
        ),
        (
            BrokerOpenOrderRecord(
                "intent",
                "order",
                "AAPL@alpaca",
                Side.BUY,
                Decimal("2"),
                Decimal("0"),
                OrderStatus.ACKNOWLEDGED,
            ),
            "ORDER_QUANTITY_MISMATCH",
        ),
        (
            BrokerOpenOrderRecord(
                "intent",
                "order",
                "AAPL@alpaca",
                Side.BUY,
                Decimal("1"),
                Decimal("0.5"),
                OrderStatus.ACKNOWLEDGED,
            ),
            "ORDER_FILL_MISMATCH",
        ),
        (
            BrokerOpenOrderRecord(
                "intent",
                "order",
                "AAPL@alpaca",
                Side.BUY,
                Decimal("1"),
                Decimal("0"),
                OrderStatus.PARTIALLY_FILLED,
            ),
            "ORDER_STATUS_MISMATCH",
        ),
    ],
)
def test_each_open_order_mismatch_has_a_stable_reason_code(
    actual: BrokerOpenOrderRecord, expected_code: str
) -> None:
    """Weakening any order comparison must make its specific mismatch observable."""
    _clock, _store, reconciler = _services()
    expected = BrokerOpenOrderRecord(
        "intent",
        "order",
        "AAPL@alpaca",
        Side.BUY,
        Decimal("1"),
        Decimal("0"),
        OrderStatus.ACKNOWLEDGED,
    )
    report = reconciler.compare(_snapshot(open_orders=(actual,)), _ledger(), (expected,))
    assert expected_code in report.reason_codes
    assert reconciler.kill_switch_active() is True


def test_provider_uncertainty_is_unhealthy_and_fail_closed() -> None:
    """A read exception must become a safe stable reason without exposing provider text."""
    _clock, _store, reconciler = _services()

    def unavailable() -> BrokerReconciliationSnapshot:
        raise RuntimeError("api_key=do-not-leak")

    report = reconciler.read_and_compare(unavailable, _ledger(), ())
    assert report.reason_codes == ("PROVIDER_UNAVAILABLE",)
    assert "api_key" not in repr(report)
    assert reconciler.kill_switch_active() is True


def test_clear_requires_new_healthy_reconciliation_and_exact_acknowledgement() -> None:
    """Submission cannot clear state, and an old healthy report cannot clear a later mismatch."""
    _clock, _store, reconciler = _services()
    reconciler.compare(_snapshot(cash=Decimal("89")), _ledger(), ())
    with pytest.raises(KillSwitchError, match="acknowledgement"):
        reconciler.clear_kill_switch(f"{KILL_SWITCH_ACKNOWLEDGEMENT} ")
    with pytest.raises(KillSwitchError, match="healthy"):
        reconciler.clear_kill_switch(KILL_SWITCH_ACKNOWLEDGEMENT)

    reconciler.compare(_snapshot(), _ledger(), ())
    reconciler.clear_kill_switch(KILL_SWITCH_ACKNOWLEDGEMENT)
    assert reconciler.kill_switch_active() is False


def test_clear_rejects_a_healthy_reconciliation_that_has_become_stale() -> None:
    """A once-healthy report cannot authorize clearing account drift minutes later."""
    clock, _store, reconciler = _services()
    reconciler.compare(_snapshot(cash=Decimal("89")), _ledger(), ())
    reconciler.compare(_snapshot(), _ledger(), ())
    clock.advance(timedelta(seconds=61))

    with pytest.raises(KillSwitchError, match="healthy"):
        reconciler.clear_kill_switch(KILL_SWITCH_ACKNOWLEDGEMENT)
    assert reconciler.kill_switch_active() is True


def test_racing_unhealthy_event_cannot_be_erased_by_clear(tmp_path: Path) -> None:
    """A clear acknowledging an older generation must not cover a concurrent later mismatch."""
    db = tmp_path / "race.db"
    _clock, _store, first = _services(db)
    clock2 = FrozenClock(DEFAULT_INSTANT + timedelta(seconds=1))
    store2 = EventStore(create_engine_and_schema(f"sqlite+pysqlite:///{db}"))
    second = Reconciler(audit_log=AuditLog(store2, clock2), clock=clock2)
    first.compare(_snapshot(cash=Decimal("89")), _ledger(), ())
    first.compare(_snapshot(), _ledger(), ())

    with ThreadPoolExecutor(max_workers=2) as executor:
        unhealthy = executor.submit(
            second.compare, _snapshot(cash=Decimal("88"), observed_at=clock2.now()), _ledger(), ()
        )
        clear = executor.submit(first.clear_kill_switch, KILL_SWITCH_ACKNOWLEDGEMENT)
        unhealthy.result()
        with suppress(KillSwitchError):
            clear.result()
    assert first.kill_switch_active() is True
