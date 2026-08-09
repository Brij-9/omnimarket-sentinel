# OmniMarket Sentinel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a locally operable trading-agent platform for India, US equities, and crypto with Tauric multi-agent research, deterministic strategies and risk, backtesting, paper execution, and preflight-gated live adapters.

**Architecture:** A typed Python control plane normalizes market inputs, converts Tauric TradingAgents output into evidence-bearing research packets, runs deterministic regime/strategy logic, and passes order intents through a fail-closed portfolio risk gate. Replay, paper, and live modes share one durable order pipeline; only the terminal broker adapter changes, and live submission additionally requires current broker preflight plus an exact expiring confirmation.

**Tech Stack:** Python 3.12, Pydantic 2, Typer/Rich, pandas/NumPy, SQLAlchemy/SQLite, HTTPX, APScheduler, TradingAgents pinned at `a33fd4c0f134485a43553a2c23a63cb14adbd88f`, Alpaca-py, CCXT, pytest/Hypothesis/respx, Ruff, mypy, pip-audit, GitHub Actions.

## Global Constraints

- Package name is `market_sentinel`; the authoritative CLI is `python -m market_sentinel.cli`.
- Python support is `>=3.12,<3.14`; production and test code use timezone-aware UTC datetimes and `Decimal` for money, price, and quantity.
- Starting capital defaults to USD 10 and the aspirational dashboard target defaults to USD 1,000,000; the target is reporting metadata only and never affects sizing, promotion, or live approval.
- Live-small defaults: no leverage, shorting, or derivatives; per-trade risk `0.005`, per-position exposure `0.10`, gross exposure `0.50`, daily-loss limit `0.02`, and drawdown limit `0.10`.
- LLM research can shift a normalized deterministic signal by at most `0.10`; it cannot choose size, create a trade by itself, reverse direction, or override risk rejection.
- Any stale input, unsupported broker capability, unresolved order, position mismatch, missing protective exit, invalid confirmation, or failed live gate produces a rejection or kill switch.
- Broker secrets come only from local environment or a future secret-store implementation; no command-line secret arguments, logs, fixtures, screenshots, or committed files contain credentials.
- No test, CI workflow, or development command submits a real-money order. External integration tests use Alpaca paper, exchange sandbox, or sanitized HTTP fixtures.
- Tauric TradingAgents is consumed as an isolated dependency under Apache-2.0 at the pinned commit; its internal simulated execution is not reused.
- Every production behavior follows red-green-refactor and every task ends with a focused commit after fresh tests pass.

## File Structure

```text
pyproject.toml                         packaging, dependencies, tools, CLI entry point
.env.example                          non-secret configuration names and safe defaults
LICENSE                               Apache-2.0 project license
README.md                             setup, safety boundary, paper/live workflows
.github/workflows/ci.yml              lint, type, unit, fixture-integration, audit, build
src/market_sentinel/__init__.py       package version
src/market_sentinel/cli.py            Typer control surface
src/market_sentinel/domain/enums.py   canonical enums
src/market_sentinel/domain/models.py  immutable typed contracts
src/market_sentinel/domain/clock.py   real and frozen clocks
src/market_sentinel/config.py         validated environment settings
src/market_sentinel/security.py       structured secret redaction
src/market_sentinel/storage/db.py     SQLAlchemy engine and schema lifecycle
src/market_sentinel/storage/events.py append-only audit/event repository
src/market_sentinel/data/base.py      market-data provider protocol
src/market_sentinel/data/normalize.py provider payload normalization
src/market_sentinel/data/freshness.py freshness and capability checks
src/market_sentinel/portfolio/ledger.py positions, fills, equity, reconciliation snapshots
src/market_sentinel/risk/policy.py     immutable risk configuration
src/market_sentinel/risk/engine.py     sizing and fail-closed order assessment
src/market_sentinel/strategies/base.py common strategy context/protocol
src/market_sentinel/strategies/indicators.py deterministic indicators
src/market_sentinel/strategies/regime.py market-regime classification
src/market_sentinel/strategies/swing.py swing trend/breakout strategy
src/market_sentinel/strategies/intraday.py ORB/VWAP intraday strategy
src/market_sentinel/strategies/crypto.py crypto spot volatility-breakout strategy
src/market_sentinel/strategies/ensemble.py conflict and research-adjustment rules
src/market_sentinel/backtest/engine.py event-driven simulation
src/market_sentinel/backtest/metrics.py after-cost metrics and benchmarks
src/market_sentinel/backtest/promotion.py backtest/paper promotion gates
src/market_sentinel/research/base.py research provider protocol
src/market_sentinel/research/tauric.py pinned TradingAgents adapter
src/market_sentinel/execution/base.py broker protocol and capability model
src/market_sentinel/execution/state_machine.py valid durable order transitions
src/market_sentinel/execution/paper.py deterministic paper broker
src/market_sentinel/execution/approval.py expiring exact live confirmations
src/market_sentinel/execution/reconcile.py broker-ledger reconciliation
src/market_sentinel/brokers/preflight.py common gate report
src/market_sentinel/brokers/alpaca.py Alpaca paper/live adapter and gates
src/market_sentinel/brokers/groww.py Groww adapter and India compliance gates
src/market_sentinel/brokers/ccxt_spot.py one selected CCXT spot exchange adapter
src/market_sentinel/operations/audit.py immutable decision audit facade
src/market_sentinel/operations/scheduler.py intraday/daily jobs and health checks
src/market_sentinel/operations/dashboard.py redacted JSON status export
scripts/check-readiness.ps1            local read-only preflight wrapper
scripts/export-dashboard-status.ps1    local status export wrapper
tests/                                 unit, property, contract, replay, and end-to-end tests
tests/fixtures/                         sanitized point-in-time market and broker payloads
tests/factories.py                     reusable immutable domain constructors
tests/fakes.py                         injected provider and broker fakes
tests/settings.py                      safe broker-settings constructors
```

---

### Task 1: Reproducible package, CLI smoke test, and dependency contract

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `LICENSE`
- Create: `src/market_sentinel/__init__.py`
- Create: `src/market_sentinel/cli.py`
- Create: `tests/test_package.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: approved package name `market_sentinel` and Python version constraint.
- Produces: `market_sentinel.__version__: str`, `market_sentinel.cli.app: typer.Typer`, editable-install and test commands.

- [ ] **Step 1: Write the failing package and CLI smoke test**

```python
from typer.testing import CliRunner

from market_sentinel import __version__
from market_sentinel.cli import app


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_status_command_is_safe_by_default() -> None:
    result = CliRunner().invoke(app, ["status"])
    assert result.exit_code == 0
    assert "mode=research" in result.stdout
    assert "live_ready=false" in result.stdout
```

- [ ] **Step 2: Run the test and verify the missing package failure**

Run: `python -m pytest tests/test_package.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'market_sentinel'`.

- [ ] **Step 3: Add packaging, safe CLI, and dependency groups**

Create `pyproject.toml` with this project contract:

```toml
[build-system]
requires = ["setuptools>=80.9,<81"]
build-backend = "setuptools.build_meta"

[project]
name = "omnimarket-sentinel"
version = "0.1.0"
requires-python = ">=3.12,<3.14"
dependencies = [
  "apscheduler>=3.11,<4",
  "httpx>=0.28,<1",
  "numpy>=2.2,<3",
  "pandas>=2.3,<3",
  "pydantic>=2.11,<3",
  "pydantic-settings>=2.8,<3",
  "rich>=14,<15",
  "sqlalchemy>=2.0,<3",
  "tenacity>=9,<10",
  "typer>=0.21,<1",
]

[project.optional-dependencies]
research = ["tradingagents @ git+https://github.com/TauricResearch/TradingAgents.git@a33fd4c0f134485a43553a2c23a63cb14adbd88f"]
brokers = ["alpaca-py>=0.42,<1", "ccxt>=4.4,<5"]
dev = [
  "build>=1.2,<2",
  "hypothesis>=6.135,<7",
  "mypy>=1.15,<2",
  "pip-audit>=2.9,<3",
  "pip-tools>=7.5,<8",
  "pyyaml>=6,<7",
  "pytest>=8,<10",
  "pytest-cov>=6,<8",
  "respx>=0.22,<1",
  "ruff>=0.15,<1",
]

[project.scripts]
market-sentinel = "market_sentinel.cli:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = ["integration: uses sandbox or sanitized external fixtures", "e2e: full local pipeline"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP", "C4", "SIM"]

[tool.mypy]
python_version = "3.12"
strict = true
packages = ["market_sentinel"]
```

Create the package with:

```python
# src/market_sentinel/__init__.py
__version__ = "0.1.0"
```

```python
# src/market_sentinel/cli.py
import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def status() -> None:
    typer.echo("mode=research live_ready=false")


if __name__ == "__main__":
    app()
```

The environment example contains only names and safe values: `MARKET_SENTINEL_MODE=research`, SQLite URL, starting/target capital, provider selection, and every broker gate set to `false`; credential values remain empty. Copy the Apache-2.0 text into `LICENSE`. Add `requirements*.txt`, `.coverage`, `dist/`, and `*.egg-info/` to `.gitignore`.

- [ ] **Step 4: Install development dependencies and verify smoke tests**

Run: `python -m pip install -e ".[dev]"`

Run: `python -m pytest tests/test_package.py -v`

Expected: two tests pass and the CLI reports research mode with live readiness false.

- [ ] **Step 5: Generate and validate an exact dependency lock**

Run: `python -m piptools compile --extra dev --extra brokers --strip-extras --generate-hashes --output-file requirements.lock pyproject.toml`

Run: `python -m pip install --dry-run --require-hashes -r requirements.lock`

Expected: dependency resolution succeeds without changing the environment.

- [ ] **Step 6: Commit the package foundation**

```powershell
git add pyproject.toml requirements.lock .env.example LICENSE .gitignore src/market_sentinel tests/test_package.py
git commit -m "build: create omnimarket sentinel package"
```

### Task 2: Immutable domain records, enums, and frozen clock

**Files:**
- Create: `src/market_sentinel/domain/__init__.py`
- Create: `src/market_sentinel/domain/enums.py`
- Create: `src/market_sentinel/domain/models.py`
- Create: `src/market_sentinel/domain/clock.py`
- Create: `tests/domain/test_models.py`
- Create: `tests/domain/test_clock.py`
- Create: `tests/factories.py`

**Interfaces:**
- Consumes: Pydantic and Python `Decimal`/timezone-aware datetime.
- Produces: `AssetClass`, `Horizon`, `OperatingMode`, `OrderStatus`, `OrderType`, `Side`, `SignalDirection`; `GateResult`, `Instrument`, `Bar`, `MarketSnapshot`, `Evidence`, `ResearchPacket`, `Signal`, `OrderIntent`, `Position`, `PortfolioSnapshot`, `RiskDecision`, `BrokerOrder`, `Fill`; `Clock`, `SystemClock`, `FrozenClock`.

- [ ] **Step 1: Write failing validation and immutability tests**

```python
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from market_sentinel.domain.enums import AssetClass
from market_sentinel.domain.models import Bar, Instrument, MarketSnapshot


def test_instrument_rejects_nonpositive_precision() -> None:
    with pytest.raises(ValidationError):
        Instrument(
            symbol="AAPL",
            venue="alpaca",
            asset_class=AssetClass.EQUITY,
            quote_currency="USD",
            timezone="America/New_York",
            price_tick=Decimal("0"),
            quantity_step=Decimal("0.000000001"),
            minimum_notional=Decimal("1"),
        )


def test_snapshot_staleness_uses_source_timestamp() -> None:
    now = datetime(2026, 8, 9, 12, tzinfo=UTC)
    bar = Bar(at=now - timedelta(minutes=2), open=Decimal("10"), high=Decimal("11"), low=Decimal("9"), close=Decimal("10.5"), volume=Decimal("1000"))
    snapshot = MarketSnapshot(instrument_id="AAPL@alpaca", observed_at=now, source_at=bar.at, bars=(bar,), provider="fixture", max_age_seconds=60)
    assert snapshot.is_stale(now) is True
```

```python
from datetime import UTC, datetime

from market_sentinel.domain.clock import FrozenClock


def test_frozen_clock_is_deterministic() -> None:
    instant = datetime(2026, 8, 9, 9, tzinfo=UTC)
    assert FrozenClock(instant).now() == instant
```

- [ ] **Step 2: Run the domain tests and verify missing-module failures**

Run: `python -m pytest tests/domain -v`

Expected: collection fails because `market_sentinel.domain` does not exist.

- [ ] **Step 3: Implement the canonical enums and models**

Use string enums with these exact members:

```python
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
```

Add `Side(BUY, SELL)`, `SignalDirection(LONG, SHORT, FLAT)`, `OrderType(MARKET, LIMIT, STOP, STOP_LIMIT)`, and `OrderStatus(PROPOSED, RISK_APPROVED, CONFIRMED, SUBMITTING, ACKNOWLEDGED, PARTIALLY_FILLED, FILLED, REJECTED, CANCELLED, EXPIRED, UNKNOWN)`. Implement frozen Pydantic models using `ConfigDict(frozen=True)` and field validators for aware datetimes, nonnegative volume, positive price/quantity precision, confidence in `[0,1]`, signal strength in `[-1,1]`, and protective-price consistency. `MarketSnapshot.is_stale(now)` returns `now - source_at > timedelta(seconds=max_age_seconds)`.

Use these exact record shapes across later tasks:

```python
class GateResult(FrozenModel):
    name: str
    passed: bool
    reason_code: str


class Position(FrozenModel):
    instrument_id: str
    quantity: Decimal
    average_price: Decimal
    market_price: Decimal
    unrealized_pnl: Decimal


class PortfolioSnapshot(FrozenModel):
    currency: str
    cash: Decimal
    equity: Decimal
    peak_equity: Decimal
    gross_exposure: Decimal
    daily_pnl: Decimal
    realized_pnl: Decimal
    positions: tuple[Position, ...]
    observed_at: datetime


class Evidence(FrozenModel):
    uri: str
    title: str
    published_at: datetime


class ResearchPacket(FrozenModel):
    instrument_id: str
    as_of: datetime
    thesis: str
    bear_case: str
    catalysts: tuple[str, ...]
    risks: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    confidence: Decimal
    model_id: str
    prompt_version: str
    configuration_hash: str


class Signal(FrozenModel):
    strategy_id: str
    strategy_version: str
    instrument_id: str
    direction: SignalDirection
    strength: Decimal
    horizon: Horizon
    entry_price: Decimal
    invalidation_price: Decimal
    take_profit: Decimal
    research_required: bool
    evidence_uris: tuple[str, ...]


class BrokerOrder(FrozenModel):
    order_id: str
    client_order_id: str
    broker: str
    instrument_id: str
    status: OrderStatus
    requested_quantity: Decimal | None
    filled_quantity: Decimal
    average_fill_price: Decimal | None
    submitted_at: datetime
    updated_at: datetime


class Fill(FrozenModel):
    fill_id: str
    order_id: str
    instrument_id: str
    side: Side
    quantity: Decimal
    price: Decimal
    fee: Decimal
    filled_at: datetime
```

`OrderIntent` contains `intent_id`, `instrument_id`, `side`, `quantity`, `notional`, `order_type`, `limit_price`, `stop_loss`, `take_profit`, `time_in_force`, `product`, `session`, `snapshot_hash`, `created_at`, and `expires_at`; exactly one of quantity/notional is populated. `RiskDecision` contains `approved`, `reason_codes`, `approved_quantity`, `approved_notional`, `portfolio_hash`, `decided_at`, and `expires_at`.

- [ ] **Step 4: Implement real and frozen clocks**

```python
class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    def __init__(self, instant: datetime) -> None:
        if instant.tzinfo is None:
            raise ValueError("FrozenClock requires an aware datetime")
        self._instant = instant.astimezone(UTC)

    def now(self) -> datetime:
        return self._instant

    def advance(self, delta: timedelta) -> None:
        if delta.total_seconds() < 0:
            raise ValueError("FrozenClock cannot move backward")
        self._instant += delta
```

Add `tests/factories.py` constructors for a valid default instrument, bar series, snapshot, research packet, signal, order intent, risk decision, fill, and portfolio. Every override is explicit and factories return new immutable records.

- [ ] **Step 5: Verify domain behavior and type checking**

Run: `python -m pytest tests/domain -v`

Run: `python -m mypy src/market_sentinel/domain`

Expected: all domain tests pass and mypy reports no issues.

- [ ] **Step 6: Commit domain contracts**

```powershell
git add src/market_sentinel/domain tests/domain
git commit -m "feat: add immutable trading domain contracts"
```

### Task 3: Validated settings, capital-target reporting, and secret redaction

**Files:**
- Create: `src/market_sentinel/config.py`
- Create: `src/market_sentinel/security.py`
- Create: `tests/test_config.py`
- Create: `tests/test_security.py`

**Interfaces:**
- Consumes: `OperatingMode`, Pydantic Settings.
- Produces: `Settings`, `RiskSettings`, `TargetProgress`, `redact_mapping(value: Mapping[str, object]) -> dict[str, object]`.

- [ ] **Step 1: Write failing settings and redaction tests**

```python
from decimal import Decimal

from market_sentinel.config import Settings


def test_target_is_reporting_only() -> None:
    settings = Settings(_env_file=None)
    assert settings.starting_capital == Decimal("10")
    assert settings.aspirational_target == Decimal("1000000")
    assert settings.required_multiple == Decimal("100000")
    assert settings.risk.max_position_fraction == Decimal("0.10")


def test_live_mode_cannot_relax_risk_defaults(monkeypatch) -> None:
    monkeypatch.setenv("MARKET_SENTINEL_MODE", "live-small")
    monkeypatch.setenv("MARKET_SENTINEL_MAX_DRAWDOWN", "0.20")
    settings = Settings(_env_file=None)
    assert settings.risk.max_drawdown_fraction == Decimal("0.10")
```

```python
from market_sentinel.security import redact_mapping


def test_nested_secrets_are_redacted() -> None:
    value = {"Authorization": "Bearer abc", "broker": {"api_key": "secret", "account_id": "A1"}}
    assert redact_mapping(value) == {"Authorization": "[REDACTED]", "broker": {"api_key": "[REDACTED]", "account_id": "A1"}}
```

- [ ] **Step 2: Verify the tests fail because settings and redaction are absent**

Run: `python -m pytest tests/test_config.py tests/test_security.py -v`

Expected: import errors for `market_sentinel.config` and `market_sentinel.security`.

- [ ] **Step 3: Implement fail-safe settings**

`Settings` uses `env_prefix="MARKET_SENTINEL_"`, ignores unknown environment keys, and defaults to research mode. Broker fields use explicit Pydantic `validation_alias` values for the exact unprefixed gate names such as `ALPACA_LIVE_TRADING_ENABLED`, `GROWW_STATIC_IP_ALLOWLISTED`, and `CCXT_WITHDRAWALS_DISABLED_CONFIRMED`; this avoids silently looking for a double-prefixed variable. `RiskSettings` clamps live-small values to the approved defaults by taking the minimum of the configured value and safe default. The exact fields are `max_trade_risk_fraction`, `max_position_fraction`, `max_gross_exposure_fraction`, `max_daily_loss_fraction`, and `max_drawdown_fraction`. `required_multiple` divides the aspiration by starting capital; it is not referenced by `RiskSettings` or strategy modules. `target_progress(current_equity)` returns `TargetProgress(starting_capital, current_equity, aspirational_target, required_multiple, achieved_multiple, remaining_gap)` for reporting only.

- [ ] **Step 4: Implement recursive redaction**

Redact mapping keys matching this case-insensitive set: `authorization`, `api_key`, `secret_key`, `access_token`, `password`, `totp`, `private_key`, and any suffix `_secret` or `_token`. Preserve keys and non-secret values; recurse through dictionaries, lists, and tuples without mutating the input.

- [ ] **Step 5: Verify safe configuration and redaction**

Run: `python -m pytest tests/test_config.py tests/test_security.py -v`

Expected: all tests pass; no secret literal is emitted in assertion diffs.

- [ ] **Step 6: Commit configuration security**

```powershell
git add src/market_sentinel/config.py src/market_sentinel/security.py tests/test_config.py tests/test_security.py
git commit -m "feat: validate risk settings and redact secrets"
```

### Task 4: Append-only SQLite event store and audit facade

**Files:**
- Create: `src/market_sentinel/storage/__init__.py`
- Create: `src/market_sentinel/storage/db.py`
- Create: `src/market_sentinel/storage/events.py`
- Create: `src/market_sentinel/operations/__init__.py`
- Create: `src/market_sentinel/operations/audit.py`
- Create: `tests/storage/test_events.py`

**Interfaces:**
- Consumes: domain models serialized through Pydantic JSON mode and `redact_mapping`.
- Produces: `create_engine_and_schema(url: str) -> Engine`, `EventRecord`, `EventStore.append(...)`, `EventStore.stream(...)`, `AuditLog.record(...)`.

- [ ] **Step 1: Write the failing immutable event-store test**

```python
from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError

from market_sentinel.storage.db import create_engine_and_schema
from market_sentinel.storage.events import EventStore


def test_event_store_is_ordered_and_event_ids_are_immutable() -> None:
    store = EventStore(create_engine_and_schema("sqlite+pysqlite:///:memory:"))
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    store.append("evt-1", "risk.rejected", "intent-1", {"reason": "STALE_DATA"}, at)
    with pytest.raises(IntegrityError):
        store.append("evt-1", "risk.approved", "intent-1", {}, at)
    events = list(store.stream("intent-1"))
    assert [event.kind for event in events] == ["risk.rejected"]
```

- [ ] **Step 2: Verify the storage test fails before schema implementation**

Run: `python -m pytest tests/storage/test_events.py -v`

Expected: import errors for the storage modules.

- [ ] **Step 3: Implement the append-only schema and repository**

Create a SQLAlchemy table `events` with primary-key `event_id`, indexed `aggregate_id`, `kind`, UTC `occurred_at`, canonical JSON `payload_json`, and monotonic integer `sequence`. `append` uses one transaction and never exposes update/delete methods. `stream(aggregate_id)` orders by `(occurred_at, sequence)` and returns frozen `EventRecord` values.

- [ ] **Step 4: Implement the redacted audit facade**

```python
class AuditLog:
    def __init__(self, store: EventStore, clock: Clock) -> None:
        self._store = store
        self._clock = clock

    def record(self, event_id: str, kind: str, aggregate_id: str, payload: Mapping[str, object]) -> None:
        self._store.append(event_id, kind, aggregate_id, redact_mapping(payload), self._clock.now())
```

- [ ] **Step 5: Verify persistence, ordering, duplicate rejection, and redaction**

Run: `python -m pytest tests/storage tests/test_security.py -v`

Expected: all tests pass, and duplicate event IDs raise `IntegrityError`.

- [ ] **Step 6: Commit storage and audit**

```powershell
git add src/market_sentinel/storage src/market_sentinel/operations tests/storage
git commit -m "feat: add immutable audit event store"
```

### Task 5: Market-data protocol, normalization, and freshness gates

**Files:**
- Create: `src/market_sentinel/data/__init__.py`
- Create: `src/market_sentinel/data/base.py`
- Create: `src/market_sentinel/data/normalize.py`
- Create: `src/market_sentinel/data/freshness.py`
- Create: `tests/data/test_normalize.py`
- Create: `tests/data/test_freshness.py`

**Interfaces:**
- Consumes: `Instrument`, `Bar`, `MarketSnapshot`, `Clock`.
- Produces: `MarketDataProvider.fetch_snapshot(instrument, horizon, as_of)`, `normalize_ohlcv(...)`, `FreshnessGate.check(snapshot, now) -> GateResult`, `ProviderCapabilities`.

- [ ] **Step 1: Write failing normalization and freshness tests**

```python
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from market_sentinel.data.normalize import normalize_ohlcv


def test_normalizer_rejects_future_and_out_of_order_bars() -> None:
    cutoff = datetime(2026, 8, 9, 10, tzinfo=UTC)
    rows = [
        {"at": "2026-08-09T10:01:00Z", "open": "10", "high": "11", "low": "9", "close": "10", "volume": "2"},
        {"at": "2026-08-09T09:59:00Z", "open": "10", "high": "11", "low": "9", "close": "10", "volume": "2"},
    ]
    with pytest.raises(ValueError, match="future bar"):
        normalize_ohlcv(rows, cutoff=cutoff)


def test_normalizer_preserves_decimal_prices() -> None:
    cutoff = datetime(2026, 8, 9, 10, tzinfo=UTC)
    bars = normalize_ohlcv([{"at": "2026-08-09T09:59:00Z", "open": "0.1", "high": "0.2", "low": "0.1", "close": "0.2", "volume": "2"}], cutoff=cutoff)
    assert bars[0].close == Decimal("0.2")
```

- [ ] **Step 2: Run the data tests and verify missing implementation failures**

Run: `python -m pytest tests/data -v`

Expected: import errors for `market_sentinel.data`.

- [ ] **Step 3: Implement provider and capability protocols**

```python
@dataclass(frozen=True)
class ProviderCapabilities:
    asset_classes: frozenset[AssetClass]
    horizons: frozenset[Horizon]
    supports_historical: bool
    supports_live_quotes: bool
    max_requests_per_second: Decimal


class MarketDataProvider(Protocol):
    @property
    def capabilities(self) -> ProviderCapabilities: ...

    def fetch_snapshot(self, instrument: Instrument, horizon: Horizon, as_of: datetime) -> MarketSnapshot: ...
```

- [ ] **Step 4: Implement strict normalization and freshness checks**

`normalize_ohlcv` parses ISO-8601 timestamps to UTC, rejects naive/future/nonascending rows, rejects `high < max(open, close)` or `low > min(open, close)`, and returns an immutable tuple of `Bar`. `FreshnessGate.check` returns `GateResult(name="market_data_fresh", passed=False, reason_code="STALE_DATA")` when the snapshot is stale and never includes raw provider credentials or headers.

- [ ] **Step 5: Verify normalization and property invariants**

Add a Hypothesis test generating valid decimal OHLC rows and asserting that normalized bars retain ascending timestamps and `low <= open/close <= high`.

Run: `python -m pytest tests/data -v`

Expected: all deterministic and generated cases pass.

- [ ] **Step 6: Commit market-data contracts**

```powershell
git add src/market_sentinel/data tests/data
git commit -m "feat: normalize and gate market data"
```

### Task 6: Portfolio ledger, fill accounting, and reconciliation snapshot

**Files:**
- Create: `src/market_sentinel/portfolio/__init__.py`
- Create: `src/market_sentinel/portfolio/ledger.py`
- Create: `tests/portfolio/test_ledger.py`

**Interfaces:**
- Consumes: `Fill`, `Position`, `PortfolioSnapshot`, `Side`.
- Produces: `PortfolioLedger.apply_fill(fill)`, `PortfolioLedger.mark(prices, at)`, `PortfolioLedger.snapshot(at)`, `PortfolioLedger.position_hash()`.

- [ ] **Step 1: Write failing fill-accounting tests**

```python
from datetime import UTC, datetime
from decimal import Decimal

from market_sentinel.domain.enums import Side
from market_sentinel.domain.models import Fill
from market_sentinel.portfolio.ledger import PortfolioLedger


def test_buy_then_partial_sell_updates_cash_position_and_realized_pnl() -> None:
    at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    ledger = PortfolioLedger(starting_cash=Decimal("10"), currency="USD")
    ledger.apply_fill(Fill(fill_id="f1", order_id="o1", instrument_id="AAPL@alpaca", side=Side.BUY, quantity=Decimal("0.5"), price=Decimal("10"), fee=Decimal("0.01"), filled_at=at))
    ledger.apply_fill(Fill(fill_id="f2", order_id="o2", instrument_id="AAPL@alpaca", side=Side.SELL, quantity=Decimal("0.2"), price=Decimal("12"), fee=Decimal("0.01"), filled_at=at))
    snapshot = ledger.mark({"AAPL@alpaca": Decimal("12")}, at)
    assert snapshot.cash == Decimal("7.38")
    assert snapshot.positions[0].quantity == Decimal("0.3")
    assert snapshot.realized_pnl == Decimal("0.38")
    assert snapshot.equity == Decimal("10.98")
```

- [ ] **Step 2: Verify the ledger test fails before implementation**

Run: `python -m pytest tests/portfolio/test_ledger.py -v`

Expected: import error for `market_sentinel.portfolio.ledger`.

- [ ] **Step 3: Implement average-cost accounting with duplicate-fill protection**

`PortfolioLedger` maintains cash, positions keyed by instrument, realized P&L, fees, peak equity, and a set of fill IDs. A duplicate fill ID raises `DuplicateFillError`. Long-only live-small accounting rejects a sell greater than held quantity. `mark` calculates market value, gross exposure, unrealized P&L, equity, peak equity, and drawdown from supplied prices without mutating historical fill records. Fees are expensed immediately and realized P&L is gross realized price P&L minus all fees charged through that fill sequence, matching the hand-calculated test.

- [ ] **Step 4: Add deterministic position hashing**

Serialize sorted `(instrument_id, quantity, average_price)` tuples plus cash and equity using canonical JSON and SHA-256. Two ledgers with the same economic state produce the same hash regardless of fill insertion dictionary ordering.

- [ ] **Step 5: Verify accounting, duplicate handling, and hash stability**

Run: `python -m pytest tests/portfolio -v`

Expected: all tests pass, including a duplicate-fill rejection and stable-hash test.

- [ ] **Step 6: Commit portfolio accounting**

```powershell
git add src/market_sentinel/portfolio tests/portfolio
git commit -m "feat: add portfolio ledger and fill accounting"
```

### Task 7: Position sizing and fail-closed risk engine

**Files:**
- Create: `src/market_sentinel/risk/__init__.py`
- Create: `src/market_sentinel/risk/policy.py`
- Create: `src/market_sentinel/risk/engine.py`
- Create: `tests/risk/test_engine.py`
- Create: `tests/risk/test_properties.py`

**Interfaces:**
- Consumes: `RiskSettings`, `Instrument`, `MarketSnapshot`, `Signal`, `OrderIntent`, `PortfolioSnapshot`, `Clock`.
- Produces: `RiskPolicy`, `PositionSizer.create_intent(...) -> OrderIntent | RiskDecision`, `RiskEngine.assess(...) -> RiskDecision`.

- [ ] **Step 1: Write failing rejection and sizing tests**

```python
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from market_sentinel.risk.engine import RiskEngine
from tests.factories import instrument, intent, portfolio, snapshot


def test_stale_snapshot_is_rejected_even_when_order_is_small() -> None:
    now = datetime(2026, 8, 9, 10, tzinfo=UTC)
    decision = RiskEngine.safe_defaults().assess(
        intent=intent(notional="1", stop_loss="9"),
        instrument=instrument(minimum_notional="1"),
        market=snapshot(source_at=now - timedelta(minutes=5), max_age_seconds=60),
        portfolio=portfolio(equity="10"),
        now=now,
    )
    assert decision.approved is False
    assert "STALE_DATA" in decision.reason_codes


def test_below_minimum_notional_is_not_scaled_up() -> None:
    decision = RiskEngine.safe_defaults().assess(
        intent=intent(notional="0.50", stop_loss="9"),
        instrument=instrument(minimum_notional="1"),
        market=snapshot(),
        portfolio=portfolio(equity="10"),
        now=snapshot().observed_at,
    )
    assert decision.approved is False
    assert decision.approved_notional is None
    assert "BELOW_MINIMUM_NOTIONAL" in decision.reason_codes
```

- [ ] **Step 2: Run risk tests and verify the missing-engine failure**

Run: `python -m pytest tests/risk/test_engine.py -v`

Expected: import error for `market_sentinel.risk.engine`.

- [ ] **Step 3: Implement immutable policy and deterministic sizing**

`RiskPolicy.safe_defaults()` uses the exact global constraint values. `PositionSizer` computes risk budget `equity * 0.005`, stop distance `abs(entry-stop)`, risk-sized quantity `risk_budget / stop_distance`, then caps it by position and gross-exposure budgets and rounds down to `quantity_step`. It returns a rejection rather than increasing size when rounded notional is below `minimum_notional`.

- [ ] **Step 4: Implement ordered fail-closed gates**

`RiskEngine.assess` accumulates stable reason codes in this order: `KILL_SWITCH_ACTIVE`, `EXPIRED_INTENT`, `STALE_DATA`, `PORTFOLIO_HASH_MISMATCH`, `MISSING_PROTECTIVE_EXIT`, `DRAWDOWN_LIMIT`, `DAILY_LOSS_LIMIT`, `LEVERAGE_FORBIDDEN`, `SHORT_FORBIDDEN`, `DERIVATIVE_FORBIDDEN`, `POSITION_LIMIT`, `GROSS_EXPOSURE_LIMIT`, `BELOW_MINIMUM_NOTIONAL`, `INVALID_PRECISION`. Any code yields `approved=False`; only a clean result carries approved quantity/notional and a short expiry.

- [ ] **Step 5: Add property tests for unbypassable limits**

Use Hypothesis to generate positive equity, prices, stop distances, and venue precision. Assert approved notional never exceeds `equity * 0.10`, approved risk never exceeds `equity * 0.005`, and adding a rejection condition cannot change a rejected decision to approved.

Run: `python -m pytest tests/risk -v`

Expected: unit and property tests pass.

- [ ] **Step 6: Commit deterministic risk controls**

```powershell
git add src/market_sentinel/risk tests/risk tests/factories.py
git commit -m "feat: enforce deterministic trading risk limits"
```

### Task 8: Indicators and regime classifier

**Files:**
- Create: `src/market_sentinel/strategies/__init__.py`
- Create: `src/market_sentinel/strategies/base.py`
- Create: `src/market_sentinel/strategies/indicators.py`
- Create: `src/market_sentinel/strategies/regime.py`
- Create: `tests/strategies/test_indicators.py`
- Create: `tests/strategies/test_regime.py`

**Interfaces:**
- Consumes: immutable `Bar` sequences and `Horizon`.
- Produces: `StrategyContext`, `Strategy.evaluate(context) -> Signal | None`, `sma`, `ema`, `atr`, `rsi`, `vwap`, `classify_regime(...) -> MarketRegime`.

- [ ] **Step 1: Write failing indicator and regime tests with hand-calculated fixtures**

```python
from decimal import Decimal

from market_sentinel.strategies.indicators import sma


def test_sma_uses_only_trailing_values() -> None:
    values = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("100")]
    assert sma(values[:3], window=3) == Decimal("2")
    assert sma(values, window=3) == Decimal("35")
```

```python
from market_sentinel.strategies.regime import MarketRegime, classify_regime
from tests.factories import trending_bars


def test_clean_uptrend_is_classified_as_trending() -> None:
    assert classify_regime(trending_bars(count=80), max_spread_bps=30) is MarketRegime.TRENDING
```

- [ ] **Step 2: Verify strategy tests fail before modules exist**

Run: `python -m pytest tests/strategies/test_indicators.py tests/strategies/test_regime.py -v`

Expected: import errors for strategy modules.

- [ ] **Step 3: Implement deterministic indicators without forward access**

Each indicator accepts only the passed sequence, returns `None` when history is shorter than its window, and performs calculations with `Decimal`. ATR uses true range against the previous close; RSI uses Wilder smoothing; VWAP rejects nonpositive cumulative volume.

- [ ] **Step 4: Implement regime classification**

`MarketRegime` has `TRENDING`, `RANGE_BOUND`, `HIGH_VOLATILITY`, and `UNTRADEABLE`. The classifier uses trailing-only ATR percentage, 20/50 moving-average separation, slope, spread basis points, and average volume. Invalid bars, excessive spread, or inadequate history produce `UNTRADEABLE`; volatility above the configured cap takes precedence over trend.

- [ ] **Step 5: Verify indicators never observe future bars**

Add a prefix-invariance test: appending future bars cannot change any indicator value computed on the original prefix.

Run: `python -m pytest tests/strategies/test_indicators.py tests/strategies/test_regime.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit indicator and regime foundations**

```powershell
git add src/market_sentinel/strategies tests/strategies tests/factories.py
git commit -m "feat: add deterministic indicators and regimes"
```

### Task 9: Swing, intraday, crypto, and ensemble strategies

**Files:**
- Create: `src/market_sentinel/strategies/swing.py`
- Create: `src/market_sentinel/strategies/intraday.py`
- Create: `src/market_sentinel/strategies/crypto.py`
- Create: `src/market_sentinel/strategies/ensemble.py`
- Create: `tests/strategies/test_swing.py`
- Create: `tests/strategies/test_intraday.py`
- Create: `tests/strategies/test_crypto.py`
- Create: `tests/strategies/test_ensemble.py`

**Interfaces:**
- Consumes: `StrategyContext`, `MarketRegime`, optional `ResearchPacket`.
- Produces: `SwingBreakoutStrategy`, `OpeningRangeVwapStrategy`, `CryptoVolatilityBreakoutStrategy`, `SignalEnsemble.combine(signals, research) -> Signal | None`.

- [ ] **Step 1: Write failing strategy-behavior tests**

```python
from decimal import Decimal

from market_sentinel.strategies.swing import SwingBreakoutStrategy
from tests.factories import swing_breakout_context


def test_swing_breakout_has_atr_invalidation_and_defined_reward() -> None:
    signal = SwingBreakoutStrategy().evaluate(swing_breakout_context())
    assert signal is not None
    assert signal.strength > Decimal("0")
    assert signal.invalidation_price < signal.entry_price < signal.take_profit
```

```python
from market_sentinel.strategies.ensemble import SignalEnsemble
from tests.factories import long_signal, short_signal


def test_conflicting_high_strength_signals_produce_no_trade() -> None:
    assert SignalEnsemble().combine([long_signal("0.8"), short_signal("-0.8")], research=None) is None
```

- [ ] **Step 2: Verify all four strategy modules are initially absent**

Run: `python -m pytest tests/strategies/test_swing.py tests/strategies/test_intraday.py tests/strategies/test_crypto.py tests/strategies/test_ensemble.py -v`

Expected: import errors for the new strategy modules.

- [ ] **Step 3: Implement explicit entry and exit rules**

`SwingBreakoutStrategy` requires `TRENDING`, close above the prior 20-bar high, 20-SMA above 50-SMA, positive momentum, adequate volume, stop at `entry - 2*ATR`, take profit at `entry + 3*ATR`, and a 20-bar time stop. `OpeningRangeVwapStrategy` uses the first configured session range, requires an in-session breakout with close above VWAP, spread/liquidity gates, stop below the range, take profit at twice risk, and mandatory closeout before session end. `CryptoVolatilityBreakoutStrategy` runs spot-long only, requires a trailing high breakout in a tradable volatility band, stop at `2.5*ATR`, and never emits leverage or short intent.

- [ ] **Step 4: Implement deterministic ensemble and bounded research adjustment**

Normalize strategy strengths to `[-1,1]`, apply versioned weights, reject high-strength directional conflicts, and return no trade below the after-cost threshold. A valid `ResearchPacket` can add or subtract at most `0.10` without changing direction. Missing research yields no trade only when the selected strategy has `research_required=True`.

- [ ] **Step 5: Verify session, conflict, and research-cap edge cases**

Run: `python -m pytest tests/strategies -v`

Expected: every strategy and ensemble test passes, including a test that research cannot reverse direction.

- [ ] **Step 6: Commit the initial strategy ensemble**

```powershell
git add src/market_sentinel/strategies tests/strategies
git commit -m "feat: add multi-horizon strategy ensemble"
```

### Task 10: Event-driven backtester, metrics, and promotion gates

**Files:**
- Create: `src/market_sentinel/backtest/__init__.py`
- Create: `src/market_sentinel/backtest/engine.py`
- Create: `src/market_sentinel/backtest/metrics.py`
- Create: `src/market_sentinel/backtest/promotion.py`
- Create: `tests/backtest/test_engine.py`
- Create: `tests/backtest/test_metrics.py`
- Create: `tests/backtest/test_promotion.py`

**Interfaces:**
- Consumes: strategies, `RiskEngine`, `PortfolioLedger`, market events.
- Produces: `CostModel`, `BacktestEngine.run(...) -> BacktestResult`, `PerformanceMetrics`, `PromotionEvaluator.evaluate_backtest(...)`, `evaluate_paper(...)`.

- [ ] **Step 1: Write a failing after-cost backtest test**

```python
from decimal import Decimal

from market_sentinel.backtest.engine import BacktestEngine, CostModel
from tests.factories import buy_then_exit_strategy, flat_bars_with_jump, instrument


def test_backtest_deducts_spread_slippage_and_fees() -> None:
    result = BacktestEngine(costs=CostModel(fee_bps=Decimal("10"), spread_bps=Decimal("20"), slippage_bps=Decimal("10"))).run(
        instrument=instrument(),
        bars=flat_bars_with_jump(),
        strategy=buy_then_exit_strategy(),
        initial_cash=Decimal("10"),
    )
    assert result.total_fees > Decimal("0")
    assert result.ending_equity < result.gross_ending_equity
```

- [ ] **Step 2: Verify backtest tests fail before engine creation**

Run: `python -m pytest tests/backtest -v`

Expected: import errors for `market_sentinel.backtest`.

- [ ] **Step 3: Implement event simulation through the shared pipeline**

At each bar, pass only the prefix ending at that bar to the strategy, size through `PositionSizer`, assess through `RiskEngine`, simulate next-event fills through a deterministic `FillModel`, and apply fills to `PortfolioLedger`. `CostModel` deterministically applies fee, half-spread, slippage, and optional fixed latency. The result stores strategy version, parameters, data cutoff, events, fills, equity curve, and benchmark curve. Task 12's `PaperBroker` delegates to this same `FillModel`, preventing cost/fill-rule drift between historical and current-data paper execution.

- [ ] **Step 4: Implement after-cost metrics and robustness runs**

Calculate total return, benchmark excess return, maximum drawdown, annualized Sharpe/Sortino with explicit period frequency, profit factor, turnover, exposure, hit rate, and completed-trade count. A robustness helper reruns with `2x` costs without tuning strategy parameters.

- [ ] **Step 5: Implement exact promotion decisions**

Backtest paper promotion requires positive after-cost return, benchmark outperformance, drawdown at most `0.10`, profit factor at least `1.10`, positive 2x-cost result, and at least 100 intraday or 30 swing trades. Live promotion requires the specified paper duration/trade count, positive observed after-cost result, no risk breach, no unexplained reconciliation event, and realized slippage at or below stressed assumptions. Return `ELIGIBLE`, `INSUFFICIENT_EVIDENCE`, or `PROMOTION_REJECTED` with stable reason codes.

- [ ] **Step 6: Verify no-look-ahead, costs, metrics, and promotion thresholds**

Run: `python -m pytest tests/backtest -v`

Expected: all tests pass, including prefix invariance and 2x-cost rejection cases.

- [ ] **Step 7: Commit backtesting and promotion evidence**

```powershell
git add src/market_sentinel/backtest tests/backtest
git commit -m "feat: add after-cost backtesting and promotion gates"
```

### Task 11: Tauric TradingAgents research adapter

**Files:**
- Create: `src/market_sentinel/research/__init__.py`
- Create: `src/market_sentinel/research/base.py`
- Create: `src/market_sentinel/research/tauric.py`
- Create: `tests/research/test_tauric.py`
- Create: `tests/fixtures/tauric_decision.json`
- Create: `tests/fakes.py`

**Interfaces:**
- Consumes: pinned optional TradingAgents dependency, `Instrument`, `Evidence`, `ResearchPacket`, `Clock`.
- Produces: `ResearchProvider.analyze(instrument, as_of) -> ResearchPacket`, `TauricRunner.propagate(symbol, date)`, `TauricResearchProvider`.

- [ ] **Step 1: Write failing adapter tests using an injected runner**

```python
from datetime import UTC, datetime

import pytest

from market_sentinel.research.tauric import TauricResearchProvider
from tests.fakes import FakeTauricRunner
from tests.factories import instrument


def test_tauric_decision_becomes_timestamped_research_packet() -> None:
    as_of = datetime(2026, 8, 8, 20, tzinfo=UTC)
    provider = TauricResearchProvider(runner=FakeTauricRunner.from_fixture("tauric_decision.json"), prompt_version="tauric-v1")
    packet = provider.analyze(instrument(symbol="AAPL"), as_of)
    assert packet.model_id == "fixture-model"
    assert packet.prompt_version == "tauric-v1"
    assert packet.evidence
    assert all(item.published_at <= as_of for item in packet.evidence)


def test_future_dated_evidence_is_rejected() -> None:
    provider = TauricResearchProvider(runner=FakeTauricRunner.with_future_evidence(), prompt_version="tauric-v1")
    with pytest.raises(ValueError, match="look-ahead evidence"):
        provider.analyze(instrument(symbol="AAPL"), datetime(2026, 8, 8, 20, tzinfo=UTC))
```

- [ ] **Step 2: Verify research tests fail before adapter creation**

Run: `python -m pytest tests/research/test_tauric.py -v`

Expected: import error for `market_sentinel.research`.

- [ ] **Step 3: Implement research protocol and strict output conversion**

The provider accepts a `TauricRunner` protocol so unit tests never import or call external LLMs. It converts the upstream state and decision into thesis, bear case, catalysts, risks, normalized confidence, model ID, prompt/config hash, and evidence. It rejects evidence after `as_of`, missing source timestamps when research is required, malformed confidence, or a decision lacking both thesis and bear case.

- [ ] **Step 4: Implement the lazy pinned upstream runner**

Only when research mode is configured, lazy-import `TradingAgentsGraph` and `DEFAULT_CONFIG`, copy the config, set provider/model/checkpoint values from validated settings, call `propagate(symbol, as_of.date().isoformat())`, and return a sanitized mapping. Import failure raises `ResearchUnavailable` with the exact installation command `python -m pip install -e ".[research]"`; it never falls back to a different unselected provider.

- [ ] **Step 5: Verify fixture conversion and optional-dependency behavior**

Run: `python -m pytest tests/research -v`

Run: `python -m pip install -e ".[research]"`

Run: `python -c "from tradingagents.graph.trading_graph import TradingAgentsGraph; print('tauric-import-ok')"`

Expected: unit tests pass and the pinned upstream import prints `tauric-import-ok`; no live LLM request runs.

- [ ] **Step 6: Record upstream license and commit the adapter**

Add a `THIRD_PARTY_NOTICES.md` entry naming TradingAgents, commit SHA, Apache-2.0 license, and repository URL.

```powershell
git add src/market_sentinel/research tests/research tests/fixtures/tauric_decision.json tests/fakes.py THIRD_PARTY_NOTICES.md pyproject.toml requirements.lock
git commit -m "feat: integrate pinned Tauric research agents"
```

### Task 12: Durable order state machine and deterministic paper broker

**Files:**
- Create: `src/market_sentinel/execution/__init__.py`
- Create: `src/market_sentinel/execution/base.py`
- Create: `src/market_sentinel/execution/state_machine.py`
- Create: `src/market_sentinel/execution/paper.py`
- Create: `tests/execution/test_state_machine.py`
- Create: `tests/execution/test_paper.py`

**Interfaces:**
- Consumes: `OrderIntent`, `BrokerOrder`, `Fill`, market snapshots, audit log.
- Produces: `BrokerCapabilities`, `BrokerAdapter`, `OrderStateMachine.transition(...)`, `PaperBroker.submit(...)`, `PaperBroker.on_snapshot(...)`.

- [ ] **Step 1: Write failing transition and duplicate-order tests**

```python
import pytest

from market_sentinel.domain.enums import OrderStatus
from market_sentinel.execution.state_machine import InvalidOrderTransition, OrderStateMachine


def test_filled_order_cannot_return_to_submitting() -> None:
    with pytest.raises(InvalidOrderTransition):
        OrderStateMachine.transition(OrderStatus.FILLED, OrderStatus.SUBMITTING)
```

```python
from market_sentinel.execution.paper import PaperBroker
from tests.factories import intent, snapshot


def test_same_client_intent_is_idempotent() -> None:
    broker = PaperBroker()
    first = broker.submit(intent(intent_id="intent-1"), snapshot())
    second = broker.submit(intent(intent_id="intent-1"), snapshot())
    assert second.order_id == first.order_id
    assert broker.order_count == 1
```

- [ ] **Step 2: Verify execution tests fail before modules exist**

Run: `python -m pytest tests/execution/test_state_machine.py tests/execution/test_paper.py -v`

Expected: import errors for execution modules.

- [ ] **Step 3: Implement explicit transition map**

Allow only the approved forward states plus cancel/reject/expire/unknown branches. Terminal states are immutable. Persist each transition as an audit event with prior/new state, client intent ID, broker order ID, and timestamp.

- [ ] **Step 4: Implement deterministic paper execution**

Market orders fill at next quote plus configured half-spread/slippage; limit orders fill only when the subsequent bar crosses the limit; stop and stop-limit rules use subsequent market events. Delegate price, fee, latency, and partial-fill calculations to Task 10's tested `FillModel`, adding only durable current-session order state. Paper execution implements the same `BrokerAdapter` protocol as live adapters and exposes no credential settings.

- [ ] **Step 5: Verify fills, partial fills, idempotency, and terminal states**

Run: `python -m pytest tests/execution -v`

Expected: all execution tests pass, including duplicate submit, partial-fill quantities, and invalid backward transition.

- [ ] **Step 6: Commit the shared paper/live pipeline foundation**

```powershell
git add src/market_sentinel/execution tests/execution
git commit -m "feat: add durable orders and paper execution"
```

### Task 13: Common preflight plus Alpaca, Groww, and CCXT adapters

**Files:**
- Create: `src/market_sentinel/brokers/__init__.py`
- Create: `src/market_sentinel/brokers/preflight.py`
- Create: `src/market_sentinel/brokers/alpaca.py`
- Create: `src/market_sentinel/brokers/groww.py`
- Create: `src/market_sentinel/brokers/ccxt_spot.py`
- Create: `tests/brokers/test_preflight.py`
- Create: `tests/brokers/test_alpaca.py`
- Create: `tests/brokers/test_groww.py`
- Create: `tests/brokers/test_ccxt_spot.py`
- Create: `tests/fixtures/alpaca_account.json`
- Create: `tests/fixtures/groww_order.json`
- Create: `tests/fixtures/ccxt_market.json`
- Create: `tests/settings.py`

**Interfaces:**
- Consumes: local environment settings, `BrokerAdapter`, `BrokerCapabilities`, HTTPX/Alpaca-py/CCXT clients injected behind protocols.
- Produces: `PreflightReport`, `AlpacaBroker`, `GrowwBroker`, `CcxtSpotBroker`; all expose `preflight()`, `capabilities()`, `submit()`, `get_order()`, and `positions()`.

- [ ] **Step 1: Write failing gate-name and secret-safety tests**

```python
from market_sentinel.brokers.alpaca import AlpacaBroker
from market_sentinel.config import Settings


def test_alpaca_preflight_lists_missing_gates_without_values() -> None:
    report = AlpacaBroker.from_settings(Settings(_env_file=None)).preflight()
    assert report.ready is False
    assert "ALPACA_LIVE_TRADING_ENABLED" in report.missing_gate_names
    rendered = report.safe_summary()
    assert "secret" not in rendered.lower()
    assert "api_key" not in rendered.lower()
```

```python
from market_sentinel.brokers.groww import GrowwBroker
from tests.settings import groww_settings


def test_groww_static_ip_gate_cannot_be_bypassed() -> None:
    report = GrowwBroker.from_settings(groww_settings(static_ip_allowlisted=False)).preflight()
    assert report.ready is False
    assert "GROWW_STATIC_IP_ALLOWLISTED" in report.missing_gate_names
```

- [ ] **Step 2: Verify all broker tests fail before adapter modules exist**

Run: `python -m pytest tests/brokers -v`

Expected: import errors for broker modules.

- [ ] **Step 3: Implement common preflight data model**

Reuse the domain `GateResult(name, passed, reason_code)`. `PreflightReport.ready` is derived as `all(gate.passed for gate in gates)` and has no writable ready field. `safe_summary` prints broker and missing gate names only.

- [ ] **Step 4: Implement Alpaca adapter contract**

Paper uses `https://paper-api.alpaca.markets`; live accepts only `https://api.alpaca.markets`. Live gates require `MARKET_SENTINEL_MODE=live-small`, `ALPACA_LIVE_TRADING_ENABLED=true`, `ALPACA_REAL_API_ENABLED=true`, live endpoint, account ID, key ID, secret key, active/unblocked account, sufficient buying power, fractionable asset for notional orders, and supported order/session. Convert SDK objects into domain records and use `intent_id` as client order ID.

- [ ] **Step 5: Implement Groww adapter contract**

Use HTTPX with either a fresh bearer access token or an API-key/secret-key flow injected through an auth provider. Gates match the broker-live-setup contract: primary broker, live-small mode, India live flag, algo-compliance verified, real API flag, active subscription, protected-order client, static public IPv4, allow-listed confirmation, broker-approved algo ID, and local credential presence. Preflight makes a read-only authenticated profile/account request, queries instrument/session/order capabilities before submission, maps `order_reference_id=intent_id`, and never logs the authorization header.

- [ ] **Step 6: Implement CCXT spot adapter contract**

Instantiate only the configured exchange ID, enable built-in rate limiting, load markets, call `set_sandbox_mode(True)` before any other request when sandbox is configured, and inspect exchange feature/capability metadata. Live gates require spot-only mode, local API key and secret, withdrawal-disabled confirmation, IP restriction confirmation when configured, market minimum/precision, and either a successful sandbox path or an explicit venue `NO_SANDBOX_AVAILABLE` acknowledgement. Reject margin, futures, options, leverage parameters, and market-order emulation.

- [ ] **Step 7: Verify sanitized contract fixtures and no-network unit suite**

Run: `python -m pytest tests/brokers -v`

Expected: all broker contract tests pass using injected clients and sanitized fixtures; no external request appears in test logs.

- [ ] **Step 8: Commit broker adapters and readiness gates**

```powershell
git add src/market_sentinel/brokers tests/brokers tests/fixtures tests/settings.py pyproject.toml requirements.lock
git commit -m "feat: add multi-market broker preflights"
```

### Task 14: Exact live confirmation, submission, and reconciliation kill switch

**Files:**
- Create: `src/market_sentinel/execution/approval.py`
- Create: `src/market_sentinel/execution/reconcile.py`
- Create: `src/market_sentinel/execution/live.py`
- Create: `tests/execution/test_approval.py`
- Create: `tests/execution/test_reconcile.py`
- Create: `tests/execution/test_live.py`

**Interfaces:**
- Consumes: passing `PreflightReport`, `OrderIntent`, `RiskDecision`, `BrokerAdapter`, `PortfolioLedger`, `Clock`, `AuditLog`.
- Produces: `OrderConfirmation`, `ApprovalService.create/verify`, `Reconciler.compare(...)`, `LiveOrderService.submit_confirmed(...)`.

- [ ] **Step 1: Write failing confirmation-expiry and mismatch tests**

```python
from datetime import timedelta
from decimal import Decimal

import pytest

from market_sentinel.execution.approval import ApprovalError, ApprovalService
from tests.factories import approved_risk, frozen_clock, intent


def test_confirmation_expires_and_is_bound_to_every_order_parameter() -> None:
    clock = frozen_clock()
    service = ApprovalService(clock=clock)
    original = intent(symbol="AAPL", quantity="0.1", limit_price="100", stop_loss="98", take_profit="104")
    confirmation = service.create(original, approved_risk(), phrase="I_CONFIRM_REAL_MONEY_ORDER")
    changed = original.model_copy(update={"quantity": Decimal("0.2")})
    with pytest.raises(ApprovalError, match="fingerprint"):
        service.verify(changed, approved_risk(), confirmation)
    clock.advance(timedelta(minutes=6))
    with pytest.raises(ApprovalError, match="expired"):
        service.verify(original, approved_risk(), confirmation)
```

- [ ] **Step 2: Verify live-service tests fail before modules are implemented**

Run: `python -m pytest tests/execution/test_approval.py tests/execution/test_reconcile.py tests/execution/test_live.py -v`

Expected: import errors for approval, reconcile, and live modules.

- [ ] **Step 3: Implement exact expiring confirmation**

Create a SHA-256 fingerprint from broker, symbol, side, quantity/notional, order type, limit, stop, target, product, session, snapshot hash, risk-decision hash, and expiry. Require the exact phrase and a default five-minute lifetime. A confirmation is single-use; success records an audit event before submission.

- [ ] **Step 4: Implement reconciliation and persistent kill switch**

Compare sorted broker positions/open orders with the ledger. Quantity, side, unknown order, or cash mismatch produces a `ReconciliationReport(healthy=False, reason_codes=...)` and persists a kill-switch event. Clearing requires a new healthy reconciliation plus an explicit local operator acknowledgement; code cannot clear it as a side effect of order submission.

- [ ] **Step 5: Implement the live submission sequence**

`LiveOrderService.submit_confirmed` performs, in order: preflight ready, fresh risk approval, healthy reconciliation, inactive kill switch, exact unused confirmation, audit `live.submission_started`, broker submit once, audit acknowledgement. Timeout after submit becomes `UNKNOWN`; it reconciles by client order ID and never blindly retries.

- [ ] **Step 6: Verify all bypass and unknown-order paths fail closed**

Run: `python -m pytest tests/execution -v`

Expected: live submission is rejected for every missing gate, mismatch, stale decision, reused confirmation, and unknown unreconciled order.

- [ ] **Step 7: Commit supervised live execution**

```powershell
git add src/market_sentinel/execution tests/execution
git commit -m "feat: gate and reconcile live orders"
```

### Task 15: CLI workflows, scheduler, dashboard export, and PowerShell helpers

**Files:**
- Modify: `src/market_sentinel/cli.py`
- Create: `src/market_sentinel/operations/scheduler.py`
- Create: `src/market_sentinel/operations/dashboard.py`
- Create: `scripts/check-readiness.ps1`
- Create: `scripts/export-dashboard-status.ps1`
- Create: `tests/operations/test_dashboard.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_scripts.py`

**Interfaces:**
- Consumes: all completed services through a `ServiceContainer` factory.
- Produces: commands `research`, `backtest`, `paper-run`, `live-preflight`, `propose-order`, `submit-confirmed-order`, `export-dashboard`; `Scheduler`; redacted dashboard JSON schema v1.

- [ ] **Step 1: Write failing CLI safety and dashboard tests**

```python
from typer.testing import CliRunner

from market_sentinel.cli import app


def test_live_submit_requires_exact_confirmation_and_never_accepts_secrets() -> None:
    result = CliRunner().invoke(app, ["submit-confirmed-order", "--broker", "alpaca", "--symbol", "AAPL"])
    assert result.exit_code != 0
    assert "I_CONFIRM_REAL_MONEY_ORDER" in result.stdout
    assert "--secret" not in result.stdout
```

```python
import json

from market_sentinel.operations.dashboard import export_dashboard
from tests.factories import system_status


def test_dashboard_reports_target_gap_without_using_it_as_risk_budget(tmp_path) -> None:
    path = tmp_path / "status.json"
    export_dashboard(system_status(equity="10"), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["aspirational_target"]["required_multiple"] == "100000"
    assert "api_key" not in path.read_text(encoding="utf-8").lower()
    assert data["risk"]["max_position_fraction"] == "0.10"
```

- [ ] **Step 2: Verify workflow tests fail before command implementation**

Run: `python -m pytest tests/test_cli.py tests/operations tests/test_scripts.py -v`

Expected: command and operations imports/behavior fail.

- [ ] **Step 3: Implement dependency-injected commands**

Each command builds services from validated settings and emits structured Rich output. Research/backtest/paper accept canonical instrument and dates. Live preflight is read-only. Propose-order prints all exact approval parameters. Submit-confirmed-order requires every parameter plus `--confirm-real-money I_CONFIRM_REAL_MONEY_ORDER`; broker credentials are not CLI options. Exceptions render stable reason codes and exit nonzero without tracebacks or secrets.

- [ ] **Step 4: Implement scheduler and redacted dashboard**

Scheduler jobs acquire a per-strategy/instrument lock, use exchange-aware calendars, store start/end/health audit events, and skip missed stale runs rather than catch up with old data. Dashboard export writes atomically through a temporary sibling file then replace, includes schema version, freshness, research/strategy versions, promotion status, portfolio/risk, broker gates, orders, kill switches, and aspiration progress, all passed through redaction.

- [ ] **Step 5: Implement local PowerShell wrappers**

`check-readiness.ps1` accepts `-Broker` limited to `groww`, `alpaca`, or `ccxt` and invokes `python -m market_sentinel.cli live-preflight --broker <value>`. `export-dashboard-status.ps1` accepts broker and path, resolves the path under the repository unless an explicit absolute destination is supplied, and invokes the export command. Neither wrapper accepts, prints, or reads secret command-line parameters.

- [ ] **Step 6: Verify CLI help, failure codes, scheduling, dashboard, and scripts**

Run: `python -m pytest tests/test_cli.py tests/operations tests/test_scripts.py -v`

Run: `python -m market_sentinel.cli --help`

Run: `python -m market_sentinel.cli live-preflight --broker alpaca`

Expected: tests pass; help lists all seven commands; preflight exits nonzero and lists missing gate names only.

- [ ] **Step 7: Commit operations and control surface**

```powershell
git add src/market_sentinel/cli.py src/market_sentinel/operations scripts tests/test_cli.py tests/operations tests/test_scripts.py
git commit -m "feat: add safe trading control center"
```

### Task 16: End-to-end fixtures, documentation, GitHub CI, and completion audit

**Files:**
- Create: `tests/e2e/test_india_pipeline.py`
- Create: `tests/e2e/test_us_pipeline.py`
- Create: `tests/e2e/test_crypto_pipeline.py`
- Create: `tests/e2e/test_live_lock.py`
- Create: `tests/e2e/runner.py`
- Create: `tests/test_project_config.py`
- Create: `.github/workflows/ci.yml`
- Create: `README.md`
- Create: `SECURITY.md`
- Create: `docs/operations/live-readiness.md`
- Create: `docs/operations/strategy-validation.md`
- Modify: `docs/superpowers/specs/2026-08-09-omnimarket-sentinel-design.md`

**Interfaces:**
- Consumes: the complete local platform and sanitized India/US/crypto fixtures.
- Produces: reproducible end-to-end evidence, operator documentation, and GitHub quality gates.

- [ ] **Step 1: Write failing end-to-end and project-config tests**

```python
import pytest

from tests.e2e.runner import run_fixture_pipeline


@pytest.mark.e2e
@pytest.mark.parametrize("market", ["india", "us", "crypto"])
def test_fixture_pipeline_researches_signals_risks_papers_and_reconciles(market: str) -> None:
    result = run_fixture_pipeline(market)
    assert result.research_packet.evidence
    assert result.backtest.metrics.after_cost_return is not None
    assert result.paper_order.status.value == "filled"
    assert result.reconciliation.healthy is True
    assert result.audit_kinds[-1] == "reconciliation.healthy"


def test_live_submission_is_locked_without_real_local_gates() -> None:
    result = run_fixture_pipeline("us", request_live=True)
    assert result.live_order is None
    assert "PREFLIGHT_NOT_READY" in result.live_rejection.reason_codes
```

```python
from pathlib import Path

import yaml


def test_ci_contains_required_quality_jobs() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert {"lint-type", "unit", "fixture-integration", "security", "package"} <= set(workflow["jobs"])
```

- [ ] **Step 2: Verify end-to-end/config tests fail before fixtures and workflow exist**

Run: `python -m pytest tests/e2e tests/test_project_config.py -v`

Expected: missing e2e runner/workflow failures.

- [ ] **Step 3: Implement three sanitized end-to-end fixture runners**

Each market fixture supplies point-in-time bars, a Tauric-style evidence packet, broker capabilities, account state, and deterministic paper fills. The runner executes normalize -> research -> strategy -> risk -> backtest report -> paper submit -> ledger -> reconciliation -> dashboard. The live-lock test uses no credentials and asserts no adapter submit method is called.

- [ ] **Step 4: Write operator and strategy documentation**

README contains local setup, exact installation extras, research/backtest/paper examples, all live gates, the confirmation phrase, supported market matrix, USD 10 venue constraints, aspiration-versus-guarantee explanation, and links to official broker/regulator sources. `SECURITY.md` documents private vulnerability reporting, secret rotation after exposure, and non-withdrawal crypto keys. Operations docs provide read-only readiness and promotion evidence procedures.

- [ ] **Step 5: Implement GitHub Actions quality gates**

Use Python 3.12 and lock-file installation. `lint-type` runs Ruff and mypy; `unit` runs tests excluding integration/e2e with coverage; `fixture-integration` runs sanitized integration and e2e tests with all real-trading environment flags forced false; `security` runs pip-audit and a repository secret-pattern scan; `package` builds wheel/sdist and installs the wheel into a clean environment for import/CLI smoke. No job defines broker or LLM credentials.

- [ ] **Step 6: Run the fresh full verification suite**

Run: `python -m ruff check .`

Run: `python -m mypy src/market_sentinel`

Run: `python -m pytest -v --cov=market_sentinel --cov-report=term-missing`

Run: `python -m pip_audit -r requirements.lock`

Run: `python -m build`

Run: `python -m pip install --force-reinstall dist/omnimarket_sentinel-0.1.0-py3-none-any.whl`

Run: `python -m market_sentinel.cli status`

Expected: every command exits zero; all tests pass; package build/install works; status prints research mode and live readiness false.

- [ ] **Step 7: Audit every acceptance criterion against evidence**

Create `docs/verification/2026-08-09-completion-audit.md` with one row per design acceptance criterion, the exact test/command/file proving it, and its observed result. Any missing or indirect evidence keeps that row incomplete and returns execution to the owning task.

- [ ] **Step 8: Commit end-to-end verification and documentation**

```powershell
git add .github README.md SECURITY.md docs src tests pyproject.toml requirements.lock
git commit -m "test: verify end-to-end trading agent platform"
```

## Execution Checkpoints

- After Task 5: package, contracts, configuration, storage, and market-data boundary are reviewable.
- After Task 10: deterministic ledger, risk, strategies, backtesting, and promotion evidence are reviewable.
- After Task 14: Tauric research, paper execution, all three broker adapters, and live safety boundary are reviewable.
- After Task 16: CLI, operations, fixtures, documentation, CI, packaging, and the requirement-by-requirement audit are reviewable.

No checkpoint is permission to submit a real-money order. Live-readiness verification remains read-only until the user owns eligible accounts, configures credentials locally, passes every current broker gate, and confirms an exact order in the current conversation.

## Spec Coverage Matrix

| Approved design requirement | Implementing tasks | Verification evidence |
|---|---|---|
| Immutable contracts, auditability, local storage | 2, 4, 6 | Domain, storage, and ledger suites |
| India, US, crypto data and capability boundaries | 5, 13, 16 | Provider/broker contract tests and three e2e fixtures |
| Intraday, swing, and crypto deterministic strategies | 8, 9 | Indicator, regime, session, and ensemble tests |
| After-cost replay, robustness, benchmarks, promotion | 10 | Backtest, metrics, prefix-invariance, and promotion tests |
| Tauric multi-agent evidence without execution authority | 11 | Fixture adapter tests, pinned import, research-cap tests |
| Shared paper/live order pipeline and durable states | 10, 12, 14 | Fill-model, state-machine, paper, live, and reconciliation tests |
| Risk caps, capital constraints, and kill switches | 3, 7, 14 | Unit/property tests and live bypass suite |
| Groww, Alpaca, and CCXT readiness without secret exposure | 3, 13, 15 | Missing-gate, request-shape, redaction, and CLI tests |
| Exact supervised live confirmation | 14, 15, 16 | Fingerprint/expiry/single-use tests and no-credentials live lock |
| Scheduler, dashboard, aspiration reporting, alerts | 3, 15 | Scheduler and redacted dashboard tests |
| Open-source notices, documentation, GitHub CI, package | 1, 11, 16 | License/notices, config test, Actions jobs, build/install smoke |
| Requirement-by-requirement completion proof | 16 | `docs/verification/2026-08-09-completion-audit.md` backed by fresh commands |
