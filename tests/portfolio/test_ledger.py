"""Behavioral tests for the long-only portfolio ledger."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from market_sentinel.domain.enums import Side
from market_sentinel.domain.models import Fill
from market_sentinel.portfolio.ledger import (
    DuplicateFillError,
    InsufficientPositionError,
    PortfolioLedger,
    PortfolioLedgerState,
)


def _fill(
    fill_id: str,
    side: Side,
    quantity: str,
    price: str,
    fee: str = "0",
    instrument_id: str = "AAPL@alpaca",
) -> Fill:
    """Build a fill with exact values for independent accounting assertions."""
    return Fill(
        fill_id=fill_id,
        order_id=f"order-{fill_id}",
        instrument_id=instrument_id,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        filled_at=datetime(2026, 8, 9, 10, tzinfo=UTC),
    )


def _unvalidated_fill(
    fill_id: str, quantity: str, price: str, fee: str
) -> Fill:
    """Construct an invalid domain record to exercise the ledger's defensive boundary."""
    return Fill.model_construct(
        fill_id=fill_id,
        order_id=f"order-{fill_id}",
        instrument_id="AAPL@alpaca",
        side=Side.BUY,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        filled_at=datetime(2026, 8, 9, 10, tzinfo=UTC),
    )


def test_buy_then_partial_sell_updates_cash_position_and_realized_pnl() -> None:
    """Charging a fee twice or folding it into cost basis changes these hand-calculated totals."""
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    ledger = PortfolioLedger(starting_cash=Decimal("10"), currency="USD")
    ledger.apply_fill(_fill("f1", Side.BUY, "0.5", "10", "0.01"))
    ledger.apply_fill(_fill("f2", Side.SELL, "0.2", "12", "0.01"))

    snapshot = ledger.mark({"AAPL@alpaca": Decimal("12")}, at)

    assert snapshot.cash == Decimal("7.38")
    assert snapshot.positions[0].quantity == Decimal("0.3")
    assert snapshot.realized_pnl == Decimal("0.38")
    assert snapshot.equity == Decimal("10.98")


def test_duplicate_fill_is_rejected_without_changing_portfolio_state() -> None:
    """Removing fill-ID protection would debit cash and increase the position twice."""
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    ledger = PortfolioLedger(starting_cash=Decimal("10"), currency="USD")
    fill = _fill("f1", Side.BUY, "0.5", "10", "0.01")
    ledger.apply_fill(fill)

    with pytest.raises(DuplicateFillError, match="f1"):
        ledger.apply_fill(fill)

    snapshot = ledger.mark({"AAPL@alpaca": Decimal("10")}, at)
    assert snapshot.cash == Decimal("4.99")
    assert snapshot.positions[0].quantity == Decimal("0.5")
    assert snapshot.realized_pnl == Decimal("-0.01")


def test_multiple_buys_use_average_cost_for_later_realized_pnl() -> None:
    """Using FIFO or the last purchase price would not realize the hand-calculated five dollars."""
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    ledger = PortfolioLedger(starting_cash=Decimal("100"), currency="USD")
    ledger.apply_fill(_fill("f1", Side.BUY, "1", "10"))
    ledger.apply_fill(_fill("f2", Side.BUY, "1", "20"))
    ledger.apply_fill(_fill("f3", Side.SELL, "1", "20"))

    snapshot = ledger.mark({"AAPL@alpaca": Decimal("20")}, at)

    assert snapshot.positions[0].quantity == Decimal("1")
    assert snapshot.positions[0].average_price == Decimal("15")
    assert snapshot.realized_pnl == Decimal("5")


def test_oversell_is_rejected_without_reserving_its_fill_id_or_mutating_state() -> None:
    """Mutating before the long-only check would leave a rejected sell in the ledger."""
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    ledger = PortfolioLedger(starting_cash=Decimal("10"), currency="USD")
    ledger.apply_fill(_fill("buy", Side.BUY, "0.5", "10"))

    with pytest.raises(InsufficientPositionError, match="held"):
        ledger.apply_fill(_fill("sell", Side.SELL, "0.6", "12"))

    ledger.apply_fill(_fill("sell", Side.SELL, "0.5", "12"))
    snapshot = ledger.mark({}, at)
    assert snapshot.cash == Decimal("11")
    assert snapshot.positions == ()
    assert snapshot.realized_pnl == Decimal("1")


def test_mark_calculates_peak_equity_and_fractional_drawdown() -> None:
    """Failing to retain the prior peak would report no drawdown after a price decline."""
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    ledger = PortfolioLedger(starting_cash=Decimal("10"), currency="USD")
    ledger.apply_fill(_fill("f1", Side.BUY, "0.5", "10"))

    ledger.mark({"AAPL@alpaca": Decimal("12")}, at)
    snapshot = ledger.mark({"AAPL@alpaca": Decimal("8")}, at)

    assert snapshot.equity == Decimal("9")
    assert snapshot.peak_equity == Decimal("11")
    assert ledger.drawdown == Decimal("0.1818181818181818181818181818")


def test_mark_requires_a_price_for_every_open_position() -> None:
    """Silently valuing an unpriced holding at zero would conceal a stale market-data failure."""
    ledger = PortfolioLedger(starting_cash=Decimal("10"), currency="USD")
    ledger.apply_fill(_fill("f1", Side.BUY, "0.5", "10"))

    with pytest.raises(ValueError, match="missing price.*AAPL@alpaca"):
        ledger.mark({}, datetime(2026, 8, 9, 10, tzinfo=UTC))


def test_mark_does_not_change_fill_accounting_or_average_cost() -> None:
    """Using a mark price as fill state would change accounting between snapshots."""
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    ledger = PortfolioLedger(starting_cash=Decimal("20"), currency="USD")
    ledger.apply_fill(_fill("f1", Side.BUY, "1", "10", "0.25"))

    ledger.mark({"AAPL@alpaca": Decimal("12")}, at)
    snapshot = ledger.mark({"AAPL@alpaca": Decimal("8")}, at)

    assert snapshot.positions[0].average_price == Decimal("10")
    assert snapshot.realized_pnl == Decimal("-0.25")
    assert snapshot.positions[0].unrealized_pnl == Decimal("-2")


def test_snapshot_uses_the_latest_mark_and_normalizes_aware_timestamp_to_utc() -> None:
    """Discarding a mark or retaining its local timestamp breaks reconciliation snapshots."""
    ledger = PortfolioLedger(starting_cash=Decimal("10"), currency="USD")
    ledger.apply_fill(_fill("f1", Side.BUY, "0.5", "10"))
    ledger.mark({"AAPL@alpaca": Decimal("12")}, datetime(2026, 8, 9, 10, tzinfo=UTC))

    india = timezone(timedelta(hours=5, minutes=30))
    snapshot = ledger.snapshot(datetime(2026, 8, 9, 16, tzinfo=india))

    assert snapshot.positions[0].market_price == Decimal("12")
    assert snapshot.observed_at == datetime(2026, 8, 9, 10, 30, tzinfo=UTC)


def test_position_hash_is_stable_for_equal_economic_state() -> None:
    """Depending on insertion order would give the same ledger two reconciliation hashes."""
    first = PortfolioLedger(starting_cash=Decimal("100"), currency="USD")
    second = PortfolioLedger(starting_cash=Decimal("100"), currency="USD")

    first.apply_fill(_fill("a1", Side.BUY, "1", "10", instrument_id="AAPL@alpaca"))
    first.apply_fill(_fill("m1", Side.BUY, "2", "20", instrument_id="MSFT@alpaca"))
    second.apply_fill(_fill("m2", Side.BUY, "2", "20", instrument_id="MSFT@alpaca"))
    second.apply_fill(_fill("a2", Side.BUY, "1", "10", instrument_id="AAPL@alpaca"))

    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    first.mark({"AAPL@alpaca": Decimal("11"), "MSFT@alpaca": Decimal("19")}, at)
    second.mark({"MSFT@alpaca": Decimal("19"), "AAPL@alpaca": Decimal("11")}, at)

    assert first.position_hash() == second.position_hash()
    assert len(first.position_hash()) == 64


@pytest.mark.parametrize(
    ("quantity", "price", "fee", "field_name"),
    [
        ("0", "10", "0", "quantity"),
        ("-1", "10", "0", "quantity"),
        ("NaN", "10", "0", "quantity"),
        ("Infinity", "10", "0", "quantity"),
        ("1", "0", "0", "price"),
        ("1", "-10", "0", "price"),
        ("1", "NaN", "0", "price"),
        ("1", "Infinity", "0", "price"),
        ("1", "10", "-0.01", "fee"),
        ("1", "10", "NaN", "fee"),
        ("1", "10", "Infinity", "fee"),
    ],
)
def test_apply_fill_rejects_invalid_numbers_without_changing_or_reserving_state(
    quantity: str, price: str, fee: str, field_name: str
) -> None:
    """Accepting any invalid numeric fill would corrupt cash, positions, or duplicate protection."""
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    ledger = PortfolioLedger(starting_cash=Decimal("10"), currency="USD")
    ledger.apply_fill(_fill("seed", Side.BUY, "0.5", "10"))
    before = ledger.mark({"AAPL@alpaca": Decimal("12")}, at)
    before_hash = ledger.position_hash()
    before_drawdown = ledger.drawdown

    with pytest.raises(ValueError, match=field_name):
        ledger.apply_fill(_unvalidated_fill("rejected", quantity, price, fee))

    after = ledger.snapshot(at)
    assert after.cash == before.cash
    assert after.positions == before.positions
    assert after.peak_equity == before.peak_equity
    assert ledger.position_hash() == before_hash
    assert ledger.drawdown == before_drawdown

    ledger.apply_fill(_fill("rejected", Side.BUY, "0.1", "10"))
    assert ledger.snapshot(at).positions[0].quantity == Decimal("0.6")


@pytest.mark.parametrize(
    "prices",
    [
        {"AAPL@alpaca": Decimal("NaN")},
        {"AAPL@alpaca": Decimal("Infinity")},
        {"AAPL@alpaca": Decimal("0")},
        {"AAPL@alpaca": Decimal("-1")},
        {"AAPL@alpaca": "12"},
        {"AAPL@alpaca": Decimal("12"), "MSFT@alpaca": Decimal("NaN")},
    ],
)
def test_mark_rejects_each_unusable_supplied_price_without_changing_valuation(
    prices: dict[str, object],
) -> None:
    """Committing an invalid mark before valuation would corrupt later snapshots and hashes."""
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    ledger = PortfolioLedger(starting_cash=Decimal("10"), currency="USD")
    ledger.apply_fill(_fill("seed", Side.BUY, "0.5", "10"))
    ledger.mark({"AAPL@alpaca": Decimal("12")}, at)
    before = ledger.mark({"AAPL@alpaca": Decimal("8")}, at)
    before_hash = ledger.position_hash()
    before_drawdown = ledger.drawdown

    with pytest.raises(ValueError, match="price"):
        ledger.mark(prices, at)  # type: ignore[arg-type]

    after = ledger.snapshot(at)
    assert after == before
    assert ledger.position_hash() == before_hash
    assert ledger.drawdown == before_drawdown


def test_validated_state_round_trip_preserves_every_accounting_accumulator() -> None:
    """Replaying only fills and the latest mark loses the historical peak and drawdown."""
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    ledger = PortfolioLedger(starting_cash=Decimal("1000"), currency="USD")
    ledger.apply_fill(_fill("entry", Side.BUY, "1", "100"))
    ledger.mark({"AAPL@alpaca": Decimal("150")}, at)
    before = ledger.mark({"AAPL@alpaca": Decimal("80")}, at + timedelta(minutes=1))

    state = ledger.export_state()
    restored = PortfolioLedger.from_state(state)

    assert isinstance(state, PortfolioLedgerState)
    assert restored.export_state() == state
    assert restored.snapshot(at + timedelta(minutes=2)).model_copy(
        update={"observed_at": before.observed_at}
    ) == before
    with pytest.raises(DuplicateFillError):
        restored.apply_fill(_fill("entry", Side.BUY, "1", "100"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: replace(state, cash=Decimal("NaN")),
        lambda state: replace(state, equity=state.equity + Decimal("1")),
        lambda state: replace(state, peak_equity=state.equity - Decimal("1")),
        lambda state: replace(state, drawdown=Decimal("-1")),
        lambda state: replace(state, fill_ids=("entry", "entry")),
    ],
)
def test_ledger_state_import_rejects_malformed_or_inconsistent_values(
    mutation: object,
) -> None:
    """A forged replay state must not bypass ledger accounting invariants."""
    ledger = PortfolioLedger(starting_cash=Decimal("1000"), currency="USD")
    ledger.apply_fill(_fill("entry", Side.BUY, "1", "100"))
    ledger.mark(
        {"AAPL@alpaca": Decimal("110")},
        datetime(2026, 8, 9, 10, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="ledger state"):
        PortfolioLedger.from_state(mutation(ledger.export_state()))  # type: ignore[operator]
