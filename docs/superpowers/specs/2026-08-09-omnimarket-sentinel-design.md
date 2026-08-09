# OmniMarket Sentinel: End-to-End Trading Agent Design

Status: approved architectural direction, pending review of this written specification

Date: 2026-08-09

Primary package: `market_sentinel`

## 1. Objective

Build a locally operable, open-source trading platform that:

- uses Tauric Research's TradingAgents framework for evidence-backed multi-agent research;
- supports intraday and swing workflows;
- covers Indian securities, US securities, and crypto in the first implementation;
- can add forex, commodities, futures, options, and other global venues through stable adapter contracts;
- supports historical backtesting, deterministic paper trading, and tightly gated live trading;
- remains useful with either a configured cloud LLM or a local Ollama-compatible model;
- records every input, decision, risk check, order, and fill for audit and replay.

The stated aspiration of growing USD 10 to USD 1,000,000 in one month is retained as an experiment target displayed in reporting. It requires a 100,000x multiple and is not a promise, an acceptance criterion, or an input to position sizing. No honest engineering or investment process can guarantee that outcome. The platform instead measures whether a strategy has a repeatable after-cost edge while enforcing survival-oriented risk limits.

## 2. Non-goals

- Claiming guaranteed returns or marketing an expected return.
- Allowing an LLM to directly call a broker or bypass deterministic validation.
- Enabling leverage, naked derivatives, martingale sizing, unlimited averaging down, or unbounded grid strategies by default.
- Treating a backtest as proof of future profitability.
- Pretending that one broker or market-data provider covers every financial market.
- Placing any real-money order during development or setup.

## 3. Design principles

1. **Research proposes; deterministic controls decide.** LLM output becomes a typed `ResearchPacket`. Only deterministic strategy and risk code may create an executable `OrderIntent`.
2. **Fail closed.** Stale data, conflicting positions, partial provider failure, invalid capabilities, missing stops, or an unhealthy broker connection results in `NO_TRADE`.
3. **Paper and live share the same order pipeline.** Only the terminal broker adapter differs, reducing simulation-to-live drift.
4. **Evidence is immutable.** Market snapshots, source timestamps, prompts, model identifiers, strategy versions, risk decisions, and fills are stored for replay.
5. **Capabilities are discovered.** The platform checks venue features, precision, minimum notional, sessions, rate limits, and supported order types instead of assuming they exist.
6. **Secrets stay local.** Credentials are loaded from local environment or secret-store integrations, never committed, logged, copied from a browser, or displayed by diagnostics.

## 4. System architecture

```text
Market/broker data + news/fundamentals/macro
                  |
                  v
        Normalized, timestamped snapshots
                  |
                  v
  Tauric analysts -> bull/bear research -> portfolio research
                  |
            ResearchPacket
                  |
                  v
   Regime classifier + deterministic strategy ensemble
                  |
              SignalSet
                  |
                  v
  Portfolio/risk gate -> OrderIntent or RiskRejection
                  |
                  v
 Backtest | PaperBroker | Confirmed Live Broker Adapter
                  |
                  v
      Orders/fills -> ledger -> dashboard/alerts/audit
```

The core package is split into small, independently testable modules:

- `domain`: immutable models, enums, money/quantity rules, and clock abstraction;
- `data`: provider contracts, normalization, freshness, caching, and corporate-action handling;
- `research`: Tauric wrapper, source collection, structured output validation, and local/cloud LLM configuration;
- `strategies`: indicators, regime classification, signal generation, and ensemble scoring;
- `backtest`: event-driven simulation using the same strategy and risk interfaces as live mode;
- `portfolio`: ledger, valuation, exposure, P&L, and reconciliation;
- `risk`: order validation, position sizing, loss limits, concentration, liquidity, and kill switches;
- `execution`: paper broker, broker adapters, order state machine, idempotency, and fill reconciliation;
- `operations`: scheduler, preflight, health checks, alerts, dashboard export, and audit log;
- `cli`: explicit commands for research, backtesting, paper sessions, live preflight, and confirmed order submission.

## 5. Core contracts

The following typed records form the internal boundary:

- `Instrument`: canonical symbol, venue, asset class, quote currency, timezone, session calendar, precision, and minimum notional.
- `MarketSnapshot`: timestamped bars, quote/order-book data, provenance, freshness, and adjustment metadata.
- `ResearchPacket`: thesis, bear case, catalysts, risks, evidence links, evidence timestamps, confidence, model, and prompt version.
- `Signal`: strategy identifier/version, direction, strength, horizon, entry model, invalidation, and evidence references.
- `OrderIntent`: broker-neutral symbol, side, quantity/notional, order type, limit, stop loss, take profit, time in force, session/product, and correlation ID.
- `RiskDecision`: approved/rejected status, sizing calculation, breached gates, portfolio snapshot hash, and expiry.
- `BrokerOrder` and `Fill`: broker identifiers, state transitions, quantities, prices, fees, taxes, and timestamps.

An `OrderIntent` expires when its market snapshot or risk decision becomes stale. Re-evaluation is mandatory; it is never silently refreshed.

## 6. Market and broker coverage

| Scope | Research/data | Paper | Live adapter | Initial restrictions |
|---|---|---|---|---|
| India | Groww historical/live data plus exchange-suffixed research tickers | Deterministic paper broker | Groww | Static-IP, subscription, algo-compliance, protected-order, instrument, and session gates |
| US equities | Alpaca data; secondary research snapshots | Alpaca paper or deterministic paper broker | Alpaca | Account eligibility, asset fractionability, session, shorting, and buying-power checks |
| Crypto spot | CCXT unified public data; exchange-specific metadata | Exchange sandbox when supported, otherwise deterministic paper broker | One explicitly selected CCXT exchange | Spot only initially; sandbox capability, minimum notional, precision, jurisdiction, and withdrawal-disabled API key checks |
| Global/other | Provider plug-in contract | Deterministic paper broker | Future IBKR/OANDA/venue adapters | No claimed live support until its adapter and compliance profile pass tests |

The Groww adapter must query current capabilities because public documentation can change and currently contains evolving coverage. Derivatives and commodities may be researched and simulated, but USD 10 is insufficient for typical contract sizes; live derivatives remain rejected until instrument margin and configured capital gates pass.

## 7. Tauric Research integration

Use `TauricResearch/TradingAgents` as a pinned dependency or isolated service, initially targeting the reviewed 0.3.1 interface and Apache-2.0 license obligations. Do not fork its execution path.

The integration will enable selected roles:

- technical analyst;
- fundamentals analyst when the asset has meaningful fundamentals;
- news and macro analyst;
- sentiment analyst only when timestamped source data is available;
- bull and bear researchers;
- trader and portfolio research roles.

Their output is parsed into `ResearchPacket`. Invalid structure, unsupported claims, missing timestamps, evidence from after the analysis cutoff, or a provider timeout lowers confidence or yields `NO_RESEARCH`. Research may shift a deterministic composite signal by at most 0.10 on its normalized -1.0-to-1.0 scale. It cannot change the direction of an otherwise valid signal, make an ineligible strategy eligible, affect position sizing, increase hard risk limits, or turn a risk rejection into approval. Each strategy declares whether missing research permits deterministic-only operation or requires `NO_TRADE`; that declaration is versioned with the strategy.

## 8. Initial deterministic strategy ensemble

The platform starts with explicit, testable strategies rather than a single opaque “AI strategy.”

### 8.1 Regime classifier

Classify each instrument/timeframe as trending, range-bound, high-volatility, or untradeable using volatility, directional movement, moving-average structure, spread, volume, and liquidity. The classifier selects eligible strategies; it does not forecast price.

### 8.2 Swing strategy

A volatility-scaled trend/breakout model combines multi-timeframe trend alignment, momentum, a rolling breakout, volume/liquidity confirmation, and an ATR-based invalidation level. It exits on invalidation, trailing protection, time stop, or regime change.

### 8.3 Intraday strategy

An opening-range breakout/VWAP-pullback model runs only during configured liquid sessions. It requires spread, volume, volatility, and news-risk filters. It has a fixed session closeout policy unless the product explicitly permits overnight holding.

### 8.4 Crypto spot strategy

A 24/7 volatility-breakout/trend model uses exchange-specific minimum notional and precision, avoids illiquid pairs, and disables leverage. Funding, open-interest, or derivatives features are research-only until a separate derivatives risk specification is approved.

### 8.5 Ensemble rules

Signals are combined by versioned, deterministic weights. Conflicting high-strength signals, abnormal volatility, insufficient data, or a weak after-cost expectancy produce no trade. There is no martingale recovery logic. Strategy parameters are fixed for each validation run and saved with its result.

## 9. Backtesting and validation

Backtesting is event-driven and uses point-in-time data. It models session calendars, corporate actions, fees, taxes, spreads, slippage, latency, partial fills, borrow/short constraints, minimum notional, and quantity precision where applicable.

Validation includes:

- chronological train/validation/test separation;
- walk-forward evaluation without look-ahead;
- multiple market regimes and benchmark comparisons;
- sensitivity tests around parameters rather than selecting one lucky optimum;
- base-cost and 2x-cost stress runs;
- data-gap, stale-data, and delayed-fill fault tests;
- separate reporting for every asset class, venue, strategy, and horizon.

Promotion evidence includes after-cost return versus an appropriate benchmark, maximum drawdown, Sharpe and Sortino ratios, profit factor, turnover, exposure, tail loss, number of independent trades, parameter stability, and confidence intervals. A strategy/venue/horizon combination is backtest-eligible for paper promotion only when its untouched out-of-sample result is positive after costs, exceeds its declared benchmark, stays within the 10% drawdown cap, has profit factor at least 1.10, remains positive under the 2x-cost stress run, and contains at least 100 completed intraday trades or 30 completed swing trades. These thresholds are necessary gates, not evidence of future profitability.

Live promotion is assessed independently for every strategy/venue/horizon combination. Intraday requires at least 20 trading days and 100 completed paper trades; swing requires at least 90 calendar days and 30 completed paper trades. The paper window must be positive after observed costs, remain within the configured loss limits, have no unexplained reconciliation events, and show realized slippage no worse than the stressed backtest assumption. Otherwise the status is `INSUFFICIENT_EVIDENCE` or `PROMOTION_REJECTED`; no metric is inferred from a smaller sample.

## 10. Risk policy and capital constraints

Safe defaults, configurable only downward during a live session:

- no leverage, shorts, or derivatives in `live-small` mode;
- maximum 0.5% of equity at risk per trade;
- maximum 10% of equity in one position, subject to broker minimum notional;
- maximum 50% gross exposure across open positions;
- maximum 2% daily loss;
- maximum 10% peak-to-trough drawdown before the global kill switch locks new orders;
- mandatory protective exit or explicit risk-defined alternative;
- no new order when price, portfolio, broker, or exchange clock is stale;
- no position increase after a strategy’s invalidation level is crossed;
- configurable correlation and asset-class concentration caps.

With USD 10, these percentages may produce an order smaller than the venue permits. The correct result is `BELOW_MINIMUM_NOTIONAL`, not leverage or risk-limit inflation. Alpaca fractional assets can support small notionals, while other venues may not.

## 11. Operating modes and promotion gates

1. `research`: no simulated or real orders.
2. `backtest`: historical event simulation only.
3. `paper`: current data with simulated or broker-paper orders.
4. `live-small`: real-money adapter available, risk limits fixed, every order supervised.

Live readiness requires a passing broker-specific preflight. For Groww this includes the selected broker, `live-small` mode, local credentials, active subscription, static outbound IP and allow-list confirmation, algo-compliance and algo-ID gates, and supported protected-order capability. For Alpaca this includes the live endpoint, active account, locally configured credentials, and live-trading flags. Crypto requires an explicitly selected exchange, a non-withdrawal API key, IP restrictions where supported, capability discovery, and a successful sandbox/dry-run path when the venue provides one.

No code or operator may set `ready_to_trade` manually. Preflight derives it from current local state. Diagnostics display missing gate names, never secret values.

Every real-money order requires a current approval containing broker, symbol, side, quantity/notional, limit price, stop loss, take profit, product/session, and the confirmation phrase `I_CONFIRM_REAL_MONEY_ORDER`. Approval expires if any parameter, price/risk snapshot, or session changes. Development and automated tests never invoke live endpoints.

## 12. Execution reliability

Orders use a durable state machine: `PROPOSED -> RISK_APPROVED -> CONFIRMED -> SUBMITTING -> ACKNOWLEDGED -> PARTIALLY_FILLED -> FILLED`, with terminal rejected/cancelled/expired/unknown states.

- A stable client order ID prevents duplicate submission across retries.
- Network failure after submission produces `UNKNOWN`, followed by broker reconciliation before any retry.
- Partial fills update exposure and protective-order quantities atomically.
- Broker positions and the internal ledger reconcile at startup, periodically, and before live submission.
- Rate limits use bounded exponential backoff with jitter; order submission is never blindly retried.
- Any position mismatch, clock drift, unsupported capability, or unreconciled order activates the broker kill switch.

## 13. Storage, observability, and interface

SQLite is sufficient for the first local deployment, with migrations and repository interfaces that allow PostgreSQL later. Stored entities include instruments, snapshots, research packets, signals, validation runs, portfolios, risk decisions, orders, fills, health events, and audit records.

The Typer CLI is the authoritative control surface:

```text
python -m market_sentinel.cli research
python -m market_sentinel.cli backtest
python -m market_sentinel.cli paper-run
python -m market_sentinel.cli live-preflight
python -m market_sentinel.cli propose-order
python -m market_sentinel.cli submit-confirmed-order
python -m market_sentinel.cli export-dashboard
```

A read-mostly local dashboard shows provider health, market freshness, research evidence, strategy/version, validation metrics, positions, risk budget, promotion status, order approvals, and the aspirational target gap. Alerts are emitted for risk breaches, stale inputs, provider degradation, reconciliation failure, and kill-switch activation.

## 14. Security and compliance

- `.env` and local state are excluded from Git.
- Logs use structured redaction for key-like fields and authorization headers.
- Broker credentials are never accepted as CLI arguments because process lists and shell history can expose them.
- Crypto keys should have trading permission only; withdrawals remain disabled.
- Dependency versions and the upstream Tauric revision are pinned and recorded.
- The project documents that it is software, not investment advice, and that users remain responsible for broker eligibility, taxes, exchange rules, and jurisdiction-specific compliance.
- India live execution remains locked unless the current retail-algo framework and broker requirements are satisfied.

## 15. Testing strategy

- Unit tests for every domain rule, strategy calculation, risk gate, and state transition.
- Property tests for invariants: risk rejection cannot become approval, live size never exceeds caps, and duplicate client IDs never create duplicate logical orders.
- Deterministic replay tests using frozen clocks and recorded market events.
- Contract tests for provider and broker adapters with sanitized fixtures.
- Sandbox integration tests for Alpaca paper and supported crypto testnets.
- Groww request-shape tests plus read-only readiness checks; no CI job has real credentials.
- Fault-injection tests for timeouts, stale data, rate limits, disconnects, partial fills, clock drift, and reconciliation mismatch.
- Secret-scanning, linting, type checking, dependency review, and full test suite in GitHub Actions.

Implementation follows red-green-refactor: each behavior begins with a failing test, then the smallest implementation, then refactoring while green.

## 16. Delivery sequence

1. Repository, domain models, configuration, storage, audit log, and CI.
2. Normalized market data, frozen fixtures, portfolio ledger, risk engine, and paper broker.
3. Strategy ensemble, event-driven backtester, walk-forward reports, and benchmark comparison.
4. Tauric Research adapter with structured evidence and local/cloud model support.
5. Alpaca paper/live adapter, Groww adapter and preflight, and one configurable CCXT spot adapter.
6. Scheduler, reconciliation, kill switches, confirmation workflow, dashboard export, and fault testing.
7. End-to-end paper runs; live adapters remain locked until user-owned accounts and local preflight gates pass.

## 17. Acceptance criteria

- A user can research and backtest representative India, US, and crypto instruments from one CLI.
- The same strategy/risk pipeline drives replay, paper, and live modes.
- Reports include after-cost performance, benchmark comparison, drawdown, robustness, and evidence sufficiency.
- Tauric output is typed, sourced, timestamped, recorded with its model/prompt/configuration for replay, and unable to bypass the risk gate; byte-identical LLM output is not claimed.
- Groww, Alpaca, and CCXT adapters expose capability/readiness status without exposing secrets.
- Paper orders execute end to end and reconcile with the internal ledger.
- Live submission is impossible without a passing preflight and exact, unexpired confirmation.
- Stale data, provider loss, duplicate requests, partial fills, and position mismatch fail closed in automated tests.
- GitHub Actions runs lint, type, unit, integration-fixture, security, and packaging checks.
- Documentation explains setup, paper operation, live gates, limitations, and why no profit is guaranteed.

## 18. Authoritative references

- Tauric Research TradingAgents: https://github.com/TauricResearch/TradingAgents
- TradingAgents license: https://github.com/TauricResearch/TradingAgents/blob/main/LICENSE
- Groww Trading API: https://groww.in/trade-api/docs
- Groww order API: https://groww.in/trade-api/docs/curl/orders
- Groww static-IP guide: https://groww.in/blog/static-ip-api-trading-setup
- Alpaca account and market coverage: https://docs.alpaca.markets/us/docs/account-plans
- Alpaca fractional trading: https://docs.alpaca.markets/us/v1.1/docs/fractional-trading
- CCXT manual: https://github.com/ccxt/ccxt/wiki/manual
- SEBI retail algorithmic trading circular: https://www.sebi.gov.in/legal/circulars/feb-2025/safer-participation-of-retail-investors-in-algorithmic-trading_91614.html
- SEC investor warning on guaranteed returns: https://www.investor.gov/protect-your-investments/fraud/protect-your-money
