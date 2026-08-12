# OmniMarket Sentinel

OmniMarket Sentinel is a safety-first Python 3.12 trading research platform. It
normalizes point-in-time data, validates typed research evidence, evaluates
deterministic strategies, applies non-relaxable risk limits, simulates after-cost
backtests, executes paper orders, reconciles a ledger, and exports a redacted local
dashboard. The checked-in end-to-end fixtures represent India, US, and crypto spot
paths without contacting any provider or running a Tauric graph.

It is not an autonomous profit machine, investment advice, a universal market
adapter, or authorization to trade real money. Live adapters are present but locked
behind user-owned accounts, broker readiness, exact local authority, fresh risk and
reconciliation evidence, an inactive kill switch, and a one-use confirmation.

## Financial reality and USD 10 constraints

Turning USD 10 into USD 1,000,000 in one month is a 100,000x aspiration. It is
extraordinarily improbable, is not a guarantee or acceptance criterion, and USD 10
can be lost in full. The target appears only as reporting-only dashboard context and
never changes position sizing or risk limits. No profit is guaranteed.

USD 10 may be below a venue's minimum order or notional, and fees, spread, taxes,
currency conversion, or an instrument's quantity step can make an order impossible.
Fractional quantity and notional support are venue-, account-, instrument-, and
session-specific. The system rejects rather than scaling a too-small order upward.

## Install

Use Python 3.12 in a dedicated environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Extras are explicit:

- `dev`: tests, coverage, Ruff, mypy, build, YAML, and dependency audit tools.
- `brokers`: Alpaca and CCXT client libraries. Installing a client does not enable
  live trading.
- `research`: Tauric Research TradingAgents at the exact reviewed commit.

Examples: `python -m pip install -e ".[dev,brokers]"` or, for a separately reviewed
research environment, `python -m pip install -e ".[dev,research]"`.

### Lock-file boundary

The checked-in lock was generated with Python 3.13.
Python 3.12 lock installation was verified by successful
[GitHub Actions run 31631595574](https://github.com/Brij-9/omnimarket-sentinel/actions/runs/31631595574)
at commit `ea1765be95fdd9a7254e9c6c98f7f4893b5c2d35`: all hash-verifiable
entries installed with `--require-hashes`, and the separately reviewed Tauric VCS
dependency was verified at its exact commit boundary.

`requirements.lock` hash-pins index artifacts and contains this exact 40-character
Tauric VCS pin:

```text
tradingagents @ git+https://github.com/TauricResearch/TradingAgents.git@a33fd4c0f134485a43553a2c23a63cb14adbd88f
```

pip cannot use `--require-hashes` for that VCS requirement. No hash is invented.
CI asserts that the line appears exactly once in both `pyproject.toml` and the lock,
installs all remaining lock entries with `--require-hashes`, then installs that exact
commit as a separately visible step. `pip-audit` audits the index-backed filtered
lock; the VCS commit requires commit-level source/dependency review and is documented
as an audit limitation.

## Deterministic verification and safe CLI

Run the complete sanitized evidence chain:

```powershell
python -m pytest tests/e2e -v
python -m pytest tests/e2e/test_india_pipeline.py -v
python -m pytest tests/e2e/test_us_pipeline.py -v
python -m pytest tests/e2e/test_crypto_pipeline.py -v
```

These runs use frozen JSON bars and a typed Tauric-style packet; they never execute a
graph or call a broker, exchange, data source, or model. Backtest results include
after-cost return, benchmark excess, drawdown, robustness under 2x costs, and
evidence-sufficiency reasons. Paper fills feed the real ledger, reconciliation, and
dashboard contracts.
Each market fixture also supplies a typed account snapshot and explicit
representative broker capabilities.

The seven visible CLI workflows are:

```text
research
backtest
paper-run
live-preflight
propose-order
submit-confirmed-order
export-dashboard
```

Inspect syntax with `python -m market_sentinel.cli --help`. The installed default
container runs only the three packaged, sanitized, point-in-time fixtures; it has no
provider credentials, network client, model call, or live-order authority. Exact
offline examples are:

```powershell
python -m market_sentinel.cli research --instrument RELIANCE@groww --as-of 2026-08-10T03:48:30+00:00
python -m market_sentinel.cli backtest --instrument AAPL@alpaca --start 2026-08-10T13:30:00+00:00 --end 2026-08-10T13:33:30+00:00
python -m market_sentinel.cli paper-run --instrument BTC-USDT@ccxt-spot --as-of 2026-08-10T00:20:30+00:00
```

Any other instrument or time window fails closed. `status` is a hidden compatibility
probe and reports `mode=research live_ready=false`.

Typical read-only checks:

```powershell
python -m market_sentinel.cli live-preflight --broker alpaca
python -m market_sentinel.cli live-preflight --broker groww
python -m market_sentinel.cli live-preflight --broker ccxt
pwsh -File scripts/check-readiness.ps1 -Broker alpaca
```

## Implemented market matrix

| Market | Fixture path | Implemented strategy/execution boundary | Live adapter boundary |
|---|---|---|---|
| India | `RELIANCE@groww` | India-session opening-range/VWAP, INR paper ledger | Groww readiness and protected-order capability checks |
| US | `AAPL@alpaca` | US-session opening-range/VWAP, USD fractional paper ledger | Alpaca account/endpoint/capability checks |
| Crypto | `BTC-USDT@ccxt-spot` | spot-long, unlevered volatility breakout, USDT paper ledger | one configurable CCXT spot exchange; withdrawals remain disabled |

This is representative support, not universal coverage. Other markets require
explicit data, calendar, instrument-normalization, cost, and broker adapters and
their own point-in-time validation, venue rules, paper evidence, and regulatory
review. Provider symbols may require a safe internal identifier mapping.

## Live gates

No command stores credentials. Live configuration is local-only, and every real
trading flag defaults to false. The exact preflight manifests are:

- Alpaca: `MARKET_SENTINEL_MODE`, `ALPACA_LIVE_TRADING_ENABLED`,
  `ALPACA_REAL_API_ENABLED`, `ALPACA_LIVE_ENDPOINT`,
  `ALPACA_ACCOUNT_ID_PRESENT`, `ALPACA_LOCAL_CREDENTIALS_PRESENT`,
  `ALPACA_ACCOUNT_ID_MATCHED`, `ALPACA_ACCOUNT_ACTIVE`,
  `ALPACA_ACCOUNT_UNBLOCKED`, and `ALPACA_SUFFICIENT_BUYING_POWER`.
- Groww: `GROWW_PRIMARY_BROKER`, `MARKET_SENTINEL_MODE`,
  `INDIA_LIVE_TRADING_ENABLED`, `INDIA_ALGO_COMPLIANCE_VERIFIED`,
  `GROWW_REAL_API_ENABLED`, `GROWW_API_SUBSCRIPTION_ACTIVE`,
  `GROWW_PROTECTED_ORDER_CLIENT`, `GROWW_STATIC_OUTBOUND_IPV4`,
  `GROWW_STATIC_IP_ALLOWLISTED`, `GROWW_BROKER_APPROVED_ALGO_ID`,
  `GROWW_LOCAL_CREDENTIALS_PRESENT`, `GROWW_AUTH_SESSION_FRESH`,
  `GROWW_READ_ONLY_PROFILE_ACCESS`, `GROWW_PROFILE_ACTIVE`,
  `GROWW_REGULAR_SESSION_SUPPORTED`, and
  `GROWW_PROTECTED_ORDERS_SUPPORTED`.
- CCXT spot: `MARKET_SENTINEL_MODE`, `CCXT_LIVE_TRADING_ENABLED`,
  `CCXT_REAL_API_ENABLED`, `CCXT_EXCHANGE_ID_CONFIGURED`, `CCXT_SPOT_ONLY`,
  `CCXT_LOCAL_CREDENTIALS_PRESENT`, `CCXT_WITHDRAWALS_DISABLED_CONFIRMED`,
  `CCXT_IP_RESTRICTED_CONFIRMED`, `CCXT_NO_SANDBOX_ACKNOWLEDGED`,
  `CCXT_EXCHANGE_CONFIGURED`, `CCXT_SPOT_MARKETS_AVAILABLE`, and
  `CCXT_CREATE_ORDER_SUPPORTED`.

Passing preflight is necessary but insufficient. The exact proposal must retain its
fresh, approved risk decision and portfolio hash. A healthy reconciliation report
must match the same broker and ledger, the kill switch and submission interlock must
be inactive, and the local confirmation must be current-session, unexpired, unused,
and exactly `I_CONFIRM_REAL_MONEY_ORDER`. Submission starts only through the Task 14
opaque safety authority. Any ambiguous response becomes unknown and activates the
fail-closed reconciliation/interlock process.

Read [live readiness](docs/operations/live-readiness.md), [strategy validation](docs/operations/strategy-validation.md),
and the [completion audit](docs/verification/2026-08-09-completion-audit.md) before
considering any promotion.

## Authoritative references

- [Tauric Research TradingAgents](https://github.com/TauricResearch/TradingAgents)
  and its [license](https://github.com/TauricResearch/TradingAgents/blob/main/LICENSE)
- [Groww Trading API](https://groww.in/trade-api/docs),
  [order API](https://groww.in/trade-api/docs/curl/orders), and
  [static-IP guide](https://groww.in/blog/static-ip-api-trading-setup)
- [Alpaca account and market coverage](https://docs.alpaca.markets/us/docs/account-plans)
  and [fractional trading](https://docs.alpaca.markets/us/v1.1/docs/fractional-trading)
- [CCXT manual](https://github.com/ccxt/ccxt/wiki/manual)
- [SEBI retail algorithmic trading circular](https://www.sebi.gov.in/legal/circulars/feb-2025/safer-participation-of-retail-investors-in-algorithmic-trading_91614.html)
- [SEC investor warning](https://www.investor.gov/protect-your-investments/fraud/protect-your-money)

See [SECURITY.md](SECURITY.md) for private reporting and credential handling.
