"""Deterministic, sanitized, no-network end-to-end fixture runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal, DecimalException
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from market_sentinel.backtest.engine import BacktestEngine, CostModel, FillModel
from market_sentinel.backtest.metrics import PerformanceMetrics, calculate_result_metrics
from market_sentinel.backtest.promotion import PromotionEvaluator
from market_sentinel.brokers.preflight import PreflightReport, required_gate_names
from market_sentinel.data.normalize import normalize_ohlcv
from market_sentinel.domain.clock import FrozenClock
from market_sentinel.domain.enums import AssetClass, OrderStatus, OrderType, Side
from market_sentinel.domain.models import (
    Bar,
    BrokerOrder,
    Evidence,
    Fill,
    GateResult,
    Instrument,
    MarketSnapshot,
    OrderIntent,
    PortfolioSnapshot,
    ResearchPacket,
    RiskDecision,
    Signal,
)
from market_sentinel.execution.approval import ApprovalService, OrderConfirmation
from market_sentinel.execution.base import BrokerCapabilities
from market_sentinel.execution.live import LiveOrderError, LiveOrderService
from market_sentinel.execution.paper import PaperBroker
from market_sentinel.execution.reconcile import (
    BrokerOpenOrderRecord,
    BrokerPositionRecord,
    BrokerReconciliationSnapshot,
    Reconciler,
    ReconciliationReport,
)
from market_sentinel.execution.safety import create_safety_capabilities
from market_sentinel.operations.audit import AuditLog
from market_sentinel.operations.dashboard import (
    DashboardAspiration,
    DashboardBroker,
    DashboardOrder,
    DashboardPortfolio,
    DashboardPromotion,
    DashboardResearch,
    DashboardRisk,
    DashboardSafetyState,
    DashboardStatus,
    DashboardStrategy,
    export_dashboard,
)
from market_sentinel.portfolio.ledger import PortfolioLedger, PortfolioLedgerState
from market_sentinel.risk.engine import PositionSizer, RiskEngine, portfolio_hash
from market_sentinel.risk.policy import RiskPolicy
from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventStore
from market_sentinel.strategies.base import Strategy, StrategyContext
from market_sentinel.strategies.crypto import CryptoVolatilityBreakoutStrategy
from market_sentinel.strategies.intraday import OpeningRangeVwapStrategy

_FIXTURE_PATH = Path(__file__).resolve().with_name("fixtures") / "e2e_markets.json"
_TARGET = Decimal("1000000")
_REQUEST_MAX_AGE = timedelta(minutes=5)
_CAPABILITY_KEYS = frozenset(
    {
        "broker",
        "supported_asset_classes",
        "supported_order_types",
        "supports_fractional_quantity",
        "supports_notional_orders",
        "supports_partial_fills",
        "supports_shorting",
        "supports_leverage",
        "supports_derivatives",
        "supports_cancel",
        "is_paper",
    }
)
_ACCOUNT_KEYS = frozenset(
    {
        "currency",
        "cash",
        "equity",
        "peak_equity",
        "gross_exposure",
        "daily_pnl",
        "realized_pnl",
        "positions",
    }
)
_COSTS = CostModel(
    fee_bps=Decimal("1"),
    spread_bps=Decimal("2"),
    slippage_bps=Decimal("1"),
)


class FixtureRequestError(ValueError):
    """One stable fail-closed fixture request rejection."""


@dataclass(frozen=True, slots=True)
class BacktestEvidence:
    """After-cost, benchmark, robustness, and sufficiency evidence."""

    metrics: PerformanceMetrics
    after_cost: bool
    costs: CostModel
    total_fees: Decimal
    robustness_stressed_return: Decimal
    evidence_sufficiency_reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LiveRejection:
    """Secret-free result of an intentionally locked live attempt."""

    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FixturePipelineResult:
    """Typed evidence from every local pipeline stage."""

    market: str
    instrument_id: str
    instrument: Instrument
    account: PortfolioSnapshot
    broker_capabilities: BrokerCapabilities
    cutoff: datetime
    maximum_request_age: timedelta
    raw_bar_count: int
    normalized_bars: tuple[Bar, ...]
    research_packet: ResearchPacket
    signal: Signal
    intent: OrderIntent
    risk_decision: RiskDecision
    backtest: BacktestEvidence
    paper_order: BrokerOrder
    paper_fill: Fill
    ledger_state: PortfolioLedgerState
    broker_snapshot: BrokerReconciliationSnapshot
    expected_open_orders: tuple[BrokerOpenOrderRecord, ...]
    reconciliation: ReconciliationReport
    kill_switch_active: bool
    dashboard: DashboardStatus
    dashboard_payload: dict[str, object]
    audit_kinds: tuple[str, ...]
    strategy_spot_only: bool
    strategy_leverage_allowed: bool
    live_order: BrokerOrder | None
    live_rejection: LiveRejection | None
    live_submit_calls: int
    live_query_calls: int
    live_flags_enabled: bool


class _NoCredentialLiveBroker:
    """A local-only adapter boundary whose preflight is deliberately not ready."""

    broker_name = "alpaca"

    def __init__(self) -> None:
        self.submit_calls = 0
        self.query_calls = 0

    def preflight(self) -> PreflightReport:
        return PreflightReport(
            self.broker_name,
            tuple(
                GateResult(name=name, passed=False, reason_code="NOT_READY")
                for name in sorted(required_gate_names(self.broker_name))
            ),
        )

    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            broker=self.broker_name,
            supported_asset_classes=frozenset({AssetClass.EQUITY}),
            supported_order_types=frozenset(OrderType),
            supports_fractional_quantity=True,
            supports_notional_orders=True,
            supports_partial_fills=True,
            supports_shorting=False,
            supports_leverage=False,
            supports_derivatives=False,
            supports_cancel=True,
            is_paper=False,
        )

    def submit(self, intent: OrderIntent, snapshot: MarketSnapshot) -> BrokerOrder:
        del intent, snapshot
        self.submit_calls += 1
        raise AssertionError("locked live adapter submit was called")

    def get_order_by_client_id(self, client_intent_id: str) -> BrokerOrder:
        del client_intent_id
        self.query_calls += 1
        raise AssertionError("locked live adapter query was called")


class FixturePipelineRunner:
    """Run each request once against immutable local JSON evidence."""

    def __init__(self) -> None:
        self._request_ids: set[str] = set()

    def run(
        self,
        market: str,
        *,
        request_id: str | None = None,
        requested_as_of: datetime | None = None,
        expected_instrument_id: str | None = None,
        request_live: bool = False,
        simulate_reconciliation_mismatch: bool = False,
        simulate_broker_open_order_mismatch: bool = False,
    ) -> FixturePipelineResult:
        fixture = _load_market_fixture(market)
        instrument = _instrument(fixture)
        broker_capabilities = _broker_capabilities(fixture, market, instrument)
        instrument_id = f"{instrument.symbol}@{instrument.venue}"
        raw_bars = _exact_list(fixture, "bars")
        cutoff = _fixture_cutoff(raw_bars)
        effective_request_id = request_id or f"{market}:{cutoff.isoformat()}"
        if effective_request_id in self._request_ids:
            raise FixtureRequestError("DUPLICATE_FIXTURE_REQUEST")
        if requested_as_of is not None:
            requested = _aware_utc(requested_as_of)
            if requested < cutoff:
                raise FixtureRequestError("FIXTURE_LOOKAHEAD_REJECTED")
            if requested >= cutoff + _REQUEST_MAX_AGE:
                raise FixtureRequestError("STALE_FIXTURE_REQUEST")
        if expected_instrument_id is not None and expected_instrument_id != instrument_id:
            raise FixtureRequestError("FIXTURE_INSTRUMENT_MISMATCH")
        self._request_ids.add(effective_request_id)

        bars = normalize_ohlcv(cast(list[dict[str, object]], raw_bars), cutoff=cutoff)
        analysis_bars = bars[:-1]
        analysis_at = analysis_bars[-1].at
        strategy, spread_bps = _strategy(fixture, instrument)
        research = _research_packet(fixture, instrument_id, analysis_at)
        if (
            not research.evidence
            or research.as_of > analysis_at
            or any(item.published_at > research.as_of for item in research.evidence)
        ):
            raise FixtureRequestError("FIXTURE_RESEARCH_INVALID")
        signal = strategy.evaluate(
            StrategyContext(
                instrument_id=instrument_id,
                bars=analysis_bars,
                horizon=strategy.metadata.allowed_horizons[0],  # type: ignore[attr-defined]
                spread_bps=spread_bps,
            )
        )
        if signal is None:
            raise FixtureRequestError("FIXTURE_SIGNAL_ABSENT")

        portfolio = _initial_portfolio(fixture, instrument, analysis_at)
        policy = RiskPolicy.safe_defaults()
        sized = PositionSizer(policy=policy).create_intent(
            signal=signal,
            instrument=instrument,
            portfolio=portfolio,
            snapshot_hash=portfolio_hash(portfolio),
            now=analysis_at,
        )
        if type(sized) is not OrderIntent:
            raise FixtureRequestError("FIXTURE_SIZING_REJECTED")
        sized = sized.model_copy(
            update={
                "intent_id": (
                    f"fixture-{market}-"
                    f"{hashlib.sha256(sized.intent_id.encode('utf-8')).hexdigest()[:24]}"
                )
            }
        )
        if (
            sized.order_type not in broker_capabilities.supported_order_types
            or (
                sized.quantity is not None
                and sized.quantity != sized.quantity.to_integral_value()
                and not broker_capabilities.supports_fractional_quantity
            )
            or (
                sized.notional is not None
                and not broker_capabilities.supports_notional_orders
            )
        ):
            raise FixtureRequestError("FIXTURE_CAPABILITIES_MISMATCH")
        market_snapshot = MarketSnapshot(
            instrument_id=instrument_id,
            observed_at=analysis_at,
            source_at=analysis_at,
            bars=analysis_bars,
            provider="fixture",
            max_age_seconds=60,
        )
        risk = RiskEngine(policy=policy).assess(
            intent=sized,
            instrument=instrument,
            market=market_snapshot,
            portfolio=portfolio,
            now=analysis_at,
        )
        if not risk.approved:
            raise FixtureRequestError("FIXTURE_RISK_REJECTED")

        backtest = _backtest_evidence(
            instrument,
            bars,
            strategy,
            signal,
            initial_cash=portfolio.cash,
        )
        paper = PaperBroker(
            fill_model=FillModel(costs=_COSTS),
            starting_cash=portfolio.cash,
            currency=instrument.quote_currency,
            session_id=f"fixture-{market}",
        )
        submitted = paper.submit(sized, market_snapshot)
        execution_snapshot = MarketSnapshot(
            instrument_id=instrument_id,
            observed_at=bars[-1].at,
            source_at=bars[-1].at,
            bars=bars,
            provider="fixture",
            max_age_seconds=60,
        )
        fills = paper.on_snapshot(execution_snapshot, instrument)
        if len(fills) != 1:
            raise FixtureRequestError("FIXTURE_PAPER_FILL_MISSING")
        paper_fill = fills[0]
        paper_order = paper.get_order(submitted.order_id)

        ledger = PortfolioLedger(
            starting_cash=portfolio.cash,
            currency=instrument.quote_currency,
        )
        ledger.apply_fill(paper_fill)
        marked = ledger.mark({instrument_id: bars[-1].close}, bars[-1].at)
        ledger_state = ledger.export_state()
        clock = FrozenClock(bars[-1].at)
        store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
        safety_key = hashlib.sha256(b"offline fixture safety authority").digest()
        approval_safety, reconciliation_safety, live_safety = create_safety_capabilities(
            audit_log=AuditLog(store, clock),
            key=safety_key,
            nonce_source=lambda: b"f" * 32,
        )
        reconciler = Reconciler(safety_capability=reconciliation_safety, clock=clock)
        paper_account = paper.portfolio_snapshot()
        broker_positions = tuple(
            BrokerPositionRecord(item.instrument_id, Side.BUY, item.quantity)
            for item in paper_account.positions
        )
        if simulate_reconciliation_mismatch:
            if not broker_positions:
                raise FixtureRequestError("FIXTURE_BROKER_POSITION_MISSING")
            first_position = broker_positions[0]
            broker_positions = (
                BrokerPositionRecord(
                    first_position.instrument_id,
                    first_position.side,
                    first_position.quantity + instrument.quantity_step,
                ),
                *broker_positions[1:],
            )
        broker_open_orders = tuple(
            BrokerOpenOrderRecord(
                order.client_order_id,
                order.order_id,
                order.instrument_id,
                sized.side,
                cast(Decimal, order.requested_quantity),
                order.filled_quantity,
                order.status,
            )
            for order in paper.open_orders()
        )
        if paper_order.status is not OrderStatus.FILLED:
            raise FixtureRequestError("FIXTURE_PAPER_ORDER_NOT_TERMINAL")
        expected_open_orders: tuple[BrokerOpenOrderRecord, ...] = ()
        if simulate_broker_open_order_mismatch:
            if sized.quantity is None:
                raise FixtureRequestError("FIXTURE_ORDER_QUANTITY_MISSING")
            broker_open_orders = (
                BrokerOpenOrderRecord(
                    sized.intent_id,
                    submitted.order_id,
                    sized.instrument_id,
                    sized.side,
                    sized.quantity,
                    Decimal("0"),
                    OrderStatus.ACKNOWLEDGED,
                ),
            )
        broker_snapshot = BrokerReconciliationSnapshot(
            broker=_broker_identity(market),
            currency=paper_account.currency,
            cash=paper_account.cash,
            positions=broker_positions,
            open_orders=broker_open_orders,
            observed_at=bars[-1].at,
        )
        reconciliation = reconciler.compare(
            broker_snapshot,
            ledger,
            expected_open_orders,
        )
        kill_switch_active = reconciler.kill_switch_active()
        reconciliation_reason = (
            reconciliation.reason_codes[0] if reconciliation.reason_codes else "OK"
        )
        dashboard = _dashboard(
            market=market,
            instrument=instrument,
            strategy=strategy,
            research=research,
            paper_order=paper_order,
            equity=marked.equity,
            at=bars[-1].at,
            starting_capital=portfolio.cash,
            kill_switch_active=kill_switch_active,
            kill_switch_reason=reconciliation_reason,
        )
        with TemporaryDirectory(prefix="market-sentinel-e2e-") as temporary:
            dashboard_path = Path(temporary) / "status.json"
            export_dashboard(dashboard, dashboard_path)
            dashboard_payload = cast(
                dict[str, object], json.loads(dashboard_path.read_text(encoding="utf-8"))
            )

        live_order: BrokerOrder | None = None
        live_rejection: LiveRejection | None = None
        live_broker = _NoCredentialLiveBroker()
        if request_live:
            approval = ApprovalService(clock=clock, safety_capability=approval_safety)
            service = LiveOrderService(
                broker=live_broker,
                approval_service=approval,
                reconciler=reconciler,
                safety_capability=live_safety,
                clock=clock,
                ledger=ledger,
            )
            locked_preflight = live_broker.preflight()
            try:
                live_order = service.submit_confirmed(
                    intent=sized,
                    risk_decision=risk,
                    snapshot=execution_snapshot,
                    confirmation=cast(OrderConfirmation, object()),
                    preflight=locked_preflight,
                    reconciliation=reconciliation,
                )
            except LiveOrderError as error:
                live_rejection = LiveRejection((str(error),))

        audit_kinds = tuple(event.kind for event in paper.audit_events) + tuple(
            event.kind for event in store.stream("live-reconciliation")
        )
        return FixturePipelineResult(
            market=market,
            instrument_id=instrument_id,
            instrument=instrument,
            account=portfolio,
            broker_capabilities=broker_capabilities,
            cutoff=cutoff,
            maximum_request_age=_REQUEST_MAX_AGE,
            raw_bar_count=len(raw_bars),
            normalized_bars=bars,
            research_packet=research,
            signal=signal,
            intent=sized,
            risk_decision=risk,
            backtest=backtest,
            paper_order=paper_order,
            paper_fill=paper_fill,
            ledger_state=ledger_state,
            broker_snapshot=broker_snapshot,
            expected_open_orders=expected_open_orders,
            reconciliation=reconciliation,
            kill_switch_active=kill_switch_active,
            dashboard=dashboard,
            dashboard_payload=dashboard_payload,
            audit_kinds=audit_kinds,
            strategy_spot_only=bool(getattr(strategy.metadata, "spot_only", False)),  # type: ignore[attr-defined]
            strategy_leverage_allowed=bool(
                getattr(strategy.metadata, "leverage_allowed", False)  # type: ignore[attr-defined]
            ),
            live_order=live_order,
            live_rejection=live_rejection,
            live_submit_calls=live_broker.submit_calls,
            live_query_calls=live_broker.query_calls,
            live_flags_enabled=False,
        )


def run_fixture_pipeline(
    market: str,
    *,
    request_live: bool = False,
    simulate_reconciliation_mismatch: bool = False,
    simulate_broker_open_order_mismatch: bool = False,
) -> FixturePipelineResult:
    """Run one isolated market fixture through the complete local platform."""
    return FixturePipelineRunner().run(
        market,
        request_live=request_live,
        simulate_reconciliation_mismatch=simulate_reconciliation_mismatch,
        simulate_broker_open_order_mismatch=simulate_broker_open_order_mismatch,
    )


def _load_market_fixture(market: str) -> dict[str, object]:
    if market not in {"india", "us", "crypto"}:
        raise FixtureRequestError("UNKNOWN_FIXTURE_MARKET")
    loaded = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if type(loaded) is not dict or type(loaded.get(market)) is not dict:
        raise FixtureRequestError("FIXTURE_FORMAT_INVALID")
    return cast(dict[str, object], loaded[market])


def _exact_mapping(value: dict[str, object], key: str) -> dict[str, object]:
    found = value.get(key)
    if type(found) is not dict:
        raise FixtureRequestError("FIXTURE_FORMAT_INVALID")
    return cast(dict[str, object], found)


def _exact_list(value: dict[str, object], key: str) -> list[object]:
    found = value.get(key)
    if type(found) is not list:
        raise FixtureRequestError("FIXTURE_FORMAT_INVALID")
    return cast(list[object], found)


def _instrument(fixture: dict[str, object]) -> Instrument:
    value = _exact_mapping(fixture, "instrument")
    return Instrument(
        symbol=str(value["symbol"]),
        venue=str(value["venue"]),
        asset_class=AssetClass(str(value["asset_class"])),
        quote_currency=str(value["quote_currency"]),
        timezone=str(value["timezone"]),
        price_tick=Decimal(str(value["price_tick"])),
        quantity_step=Decimal(str(value["quantity_step"])),
        minimum_notional=Decimal(str(value["minimum_notional"])),
        session_calendar=(
            None if value["session_calendar"] is None else str(value["session_calendar"])
        ),
    )


def _broker_capabilities(
    fixture: dict[str, object],
    market: str,
    instrument: Instrument,
) -> BrokerCapabilities:
    value = _exact_mapping(fixture, "broker_capabilities")
    if frozenset(value) != _CAPABILITY_KEYS:
        raise FixtureRequestError("FIXTURE_CAPABILITIES_INVALID")
    broker = value.get("broker")
    if type(broker) is not str:
        raise FixtureRequestError("FIXTURE_CAPABILITIES_INVALID")
    try:
        capabilities = BrokerCapabilities(
            broker=broker,
            supported_asset_classes=frozenset(
                AssetClass(item)
                for item in _exact_string_list(
                    value,
                    "supported_asset_classes",
                    "FIXTURE_CAPABILITIES_INVALID",
                )
            ),
            supported_order_types=frozenset(
                OrderType(item)
                for item in _exact_string_list(
                    value,
                    "supported_order_types",
                    "FIXTURE_CAPABILITIES_INVALID",
                )
            ),
            supports_fractional_quantity=_exact_bool(
                value,
                "supports_fractional_quantity",
                "FIXTURE_CAPABILITIES_INVALID",
            ),
            supports_notional_orders=_exact_bool(
                value,
                "supports_notional_orders",
                "FIXTURE_CAPABILITIES_INVALID",
            ),
            supports_partial_fills=_exact_bool(
                value,
                "supports_partial_fills",
                "FIXTURE_CAPABILITIES_INVALID",
            ),
            supports_shorting=_exact_bool(
                value,
                "supports_shorting",
                "FIXTURE_CAPABILITIES_INVALID",
            ),
            supports_leverage=_exact_bool(
                value,
                "supports_leverage",
                "FIXTURE_CAPABILITIES_INVALID",
            ),
            supports_derivatives=_exact_bool(
                value,
                "supports_derivatives",
                "FIXTURE_CAPABILITIES_INVALID",
            ),
            supports_cancel=_exact_bool(
                value,
                "supports_cancel",
                "FIXTURE_CAPABILITIES_INVALID",
            ),
            is_paper=_exact_bool(
                value,
                "is_paper",
                "FIXTURE_CAPABILITIES_INVALID",
            ),
        )
    except (KeyError, TypeError, ValueError):
        raise FixtureRequestError("FIXTURE_CAPABILITIES_INVALID") from None
    if (
        capabilities.broker != _broker_identity(market)
        or instrument.asset_class not in capabilities.supported_asset_classes
    ):
        raise FixtureRequestError("FIXTURE_CAPABILITIES_MISMATCH")
    return capabilities


def _exact_bool(value: dict[str, object], key: str, reason: str) -> bool:
    found = value.get(key)
    if type(found) is not bool:
        raise FixtureRequestError(reason)
    return found


def _exact_string_list(
    value: dict[str, object],
    key: str,
    reason: str,
) -> tuple[str, ...]:
    found = value.get(key)
    if type(found) is not list or not found or not all(type(item) is str for item in found):
        raise FixtureRequestError(reason)
    return tuple(cast(list[str], found))


def _fixture_cutoff(rows: list[object]) -> datetime:
    if not rows or type(rows[-1]) is not dict:
        raise FixtureRequestError("FIXTURE_FORMAT_INVALID")
    value = cast(dict[str, object], rows[-1]).get("at")
    if type(value) is not str:
        raise FixtureRequestError("FIXTURE_FORMAT_INVALID")
    return _aware_utc(datetime.fromisoformat(value))


def _research_packet(
    fixture: dict[str, object], instrument_id: str, as_of: datetime
) -> ResearchPacket:
    value = _exact_mapping(fixture, "research")
    configuration = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return ResearchPacket(
        instrument_id=instrument_id,
        as_of=as_of,
        thesis=str(value["thesis"]),
        bear_case=str(value["bear_case"]),
        catalysts=tuple(str(item) for item in cast(list[object], value["catalysts"])),
        risks=tuple(str(item) for item in cast(list[object], value["risks"])),
        evidence=(
            Evidence(
                uri=str(value["evidence_uri"]),
                title=str(value["evidence_title"]),
                published_at=as_of - timedelta(minutes=1),
            ),
        ),
        confidence=Decimal(str(value["confidence"])),
        model_id="tauric-fixture-no-graph",
        prompt_version="fixture-v1",
        configuration_hash=hashlib.sha256(configuration.encode("utf-8")).hexdigest(),
    )


def _strategy(fixture: dict[str, object], instrument: Instrument) -> tuple[Strategy, Decimal]:
    value = _exact_mapping(fixture, "strategy")
    spread = Decimal(str(value["spread_bps"]))
    if value["kind"] == "opening_range_vwap":
        return (
            OpeningRangeVwapStrategy(
                session_start=time.fromisoformat(str(value["session_start"])),
                session_end=time.fromisoformat(str(value["session_end"])),
                session_timezone=instrument.timezone,
                opening_range_bars=int(str(value["opening_range_bars"])),
            ),
            spread,
        )
    if value["kind"] == "crypto_volatility_breakout":
        return CryptoVolatilityBreakoutStrategy(), spread
    raise FixtureRequestError("FIXTURE_STRATEGY_INVALID")


def _initial_portfolio(
    fixture: dict[str, object],
    instrument: Instrument,
    at: datetime,
) -> PortfolioSnapshot:
    value = _exact_mapping(fixture, "account")
    if frozenset(value) != _ACCOUNT_KEYS:
        raise FixtureRequestError("FIXTURE_ACCOUNT_INVALID")
    try:
        positions = _exact_list(value, "positions")
        if positions:
            raise FixtureRequestError("FIXTURE_ACCOUNT_INVALID")
        currency = value["currency"]
        decimal_values = tuple(
            value[key]
            for key in (
                "cash",
                "equity",
                "peak_equity",
                "gross_exposure",
                "daily_pnl",
                "realized_pnl",
            )
        )
        if type(currency) is not str or not all(
            type(item) is str for item in decimal_values
        ):
            raise FixtureRequestError("FIXTURE_ACCOUNT_INVALID")
        snapshot = PortfolioSnapshot(
            currency=currency,
            cash=Decimal(cast(str, value["cash"])),
            equity=Decimal(cast(str, value["equity"])),
            peak_equity=Decimal(cast(str, value["peak_equity"])),
            gross_exposure=Decimal(cast(str, value["gross_exposure"])),
            daily_pnl=Decimal(cast(str, value["daily_pnl"])),
            realized_pnl=Decimal(cast(str, value["realized_pnl"])),
            positions=(),
            observed_at=at,
        )
    except (DecimalException, KeyError, TypeError, ValueError):
        raise FixtureRequestError("FIXTURE_ACCOUNT_INVALID") from None
    if snapshot.currency != instrument.quote_currency or snapshot.cash <= Decimal("0"):
        raise FixtureRequestError("FIXTURE_ACCOUNT_MISMATCH")
    return snapshot


def _backtest_evidence(
    instrument: Instrument,
    bars: tuple[Bar, ...],
    strategy: Strategy,
    signal: Signal,
    *,
    initial_cash: Decimal,
) -> BacktestEvidence:
    robustness = BacktestEngine(costs=_COSTS).run_robustness(
        instrument=instrument,
        bars=bars,
        strategy=strategy,
        initial_cash=initial_cash,
    )
    metrics = calculate_result_metrics(robustness.base, periods_per_year=252)
    stressed_return = (
        robustness.stressed.ending_equity / robustness.stressed.initial_cash - Decimal("1")
    )
    promotion = PromotionEvaluator().evaluate_backtest(
        metrics=metrics,
        stressed_total_return=stressed_return,
        horizon=signal.horizon,
    )
    return BacktestEvidence(
        metrics=metrics,
        after_cost=robustness.base.costs == _COSTS,
        costs=robustness.base.costs,
        total_fees=robustness.base.total_fees,
        robustness_stressed_return=stressed_return,
        evidence_sufficiency_reason_codes=promotion.reason_codes,
    )


def _dashboard(
    *,
    market: str,
    instrument: Instrument,
    strategy: Strategy,
    research: ResearchPacket,
    paper_order: BrokerOrder,
    equity: Decimal,
    at: datetime,
    starting_capital: Decimal,
    kill_switch_active: bool,
    kill_switch_reason: str,
) -> DashboardStatus:
    broker_key = "ccxt-spot" if market == "crypto" else _broker_identity(market)
    dashboard_broker = "ccxt" if market == "crypto" else broker_key
    policy = RiskPolicy.safe_defaults()
    return DashboardStatus(
        generated_at=at,
        data_as_of=research.as_of,
        research=DashboardResearch(research.prompt_version, True),
        strategies=(
            DashboardStrategy(
                strategy.metadata.strategy_id,  # type: ignore[attr-defined]
                strategy.metadata.version,  # type: ignore[attr-defined]
            ),
        ),
        promotion=DashboardPromotion("backtest"),
        portfolio=DashboardPortfolio(instrument.quote_currency, equity),
        risk=DashboardRisk(
            policy.max_trade_risk_fraction,
            policy.max_position_fraction,
            policy.max_gross_exposure_fraction,
            policy.max_daily_loss_fraction,
            policy.max_drawdown_fraction,
        ),
        brokers=(
            DashboardBroker(
                dashboard_broker,
                tuple(
                    GateResult(name=name, passed=False, reason_code="NOT_READY")
                    for name in sorted(required_gate_names(broker_key))
                ),
            ),
        ),
        orders=(DashboardOrder(paper_order.order_id, paper_order.status),),
        kill_switches=(DashboardSafetyState(kill_switch_active, kill_switch_reason),),
        interlocks=(DashboardSafetyState(False, "OK"),),
        aspirational_target=DashboardAspiration(
            starting_capital,
            equity,
            _TARGET,
            _TARGET / starting_capital,
            equity / starting_capital,
            _TARGET - equity,
            True,
        ),
    )


def _broker_identity(market: str) -> str:
    return {"india": "groww", "us": "alpaca", "crypto": "ccxt-spot"}[market]


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise FixtureRequestError("FIXTURE_TIME_INVALID")
    return value.astimezone(UTC)
