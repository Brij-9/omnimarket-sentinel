# Task 10 report: event-driven backtester, metrics, and promotion gates

## Scope completed

- Added a point-in-time bar event loop that evaluates strategy prefixes ending at the current
  bar, sizes entries through `PositionSizer`, obtains entry approval through `RiskEngine`, fills
  on a later eligible event through the public provider-neutral `FillModel`, and applies fills and
  marks through `PortfolioLedger`.
- Added immutable result, event, equity-point, completed-trade, and robustness artifacts. Results
  retain strategy/version, serialized parameter evidence, cutoff, events, fills, after-cost equity,
  frictionless benchmark curve, completed trades, fees, turnover inputs, and exposure periods.
- Added Decimal-only fee, half-spread, slippage, adverse tick rounding, and fixed `timedelta`
  latency. The first eligible fill is the first event strictly after submission whose timestamp is
  at or after the latency deadline.
- Added deterministic stop-loss, take-profit, maximum-holding-period, and session-boundary exits.
  Price-triggered exits never inspect the entry bar's high/low and all exits fill on a later event.
- Added after-cost return, benchmark excess, maximum drawdown, explicit-frequency Sharpe/Sortino,
  profit factor, turnover, exposure, hit rate, and completed-trade count with zero/no-trade cases.
- Added chronological and expanding walk-forward split helpers with disjoint untouched test
  windows.
- Added base versus 2x-cost reruns. Monetary friction assumptions are doubled while latency,
  strategy instance, input bars, and serialized parameters are unchanged; no tuning hook exists in
  the rerun path.
- Added exact backtest-to-paper and paper-to-live decisions with stable reason ordering and
  separate `INSUFFICIENT_EVIDENCE` versus `PROMOTION_REJECTED` outcomes.

## TDD evidence

- Initial engine RED: `python -m pytest tests/backtest/test_engine.py -v` failed during collection
  with `ModuleNotFoundError: No module named 'market_sentinel.backtest'`.
- Metrics RED: `python -m pytest tests/backtest/test_metrics.py -v` failed during collection with
  `ModuleNotFoundError: No module named 'market_sentinel.backtest.metrics'`.
- Promotion RED: `python -m pytest tests/backtest/test_promotion.py -v` failed during collection
  with `ModuleNotFoundError: No module named 'market_sentinel.backtest.promotion'`.
- Behavioral REDs also captured missing next-event protective and session exits. The exact turnover
  test exposed Decimal rounding from dividing through a repeating average; the implementation was
  rearranged algebraically to preserve the exact hand-calculated `2.0625`.
- Focused final: `python -m pytest tests/backtest -v` -> `27 passed`.

## Final verification evidence

- `ruff check .` -> `All checks passed!`
- `mypy --strict src/market_sentinel` -> `Success: no issues found in 35 source files`
- `python -m pytest` -> `198 passed in 10.45s`
- `git diff --check` -> exit 0, no whitespace errors.

## Exact promotion semantics

- Backtest sample sufficiency: 100 completed intraday trades or 30 completed swing trades.
- Eligible evidence then requires OOS after-cost return `> 0`, benchmark excess `> 0`, maximum
  drawdown `<= 0.10`, profit factor `>= 1.10`, and 2x-cost return `> 0`.
- Live sample sufficiency: intraday uses at least 20 trading days and 100 completed trades; swing
  uses at least 90 calendar days and 30 completed trades.
- Eligible paper evidence then requires observed after-cost return `> 0`, no risk breach, no
  unexplained reconciliation event, and realized slippage `<=` the stressed assumption.
- Sample deficits are returned before performance evaluation so smaller samples are never treated
  as negative evidence.

## Assumptions and known design limitations

- `Strategy.evaluate` is treated as the existing pure deterministic strategy contract. A robustness
  rerun reuses the same strategy object and parameters; stateful third-party strategies would violate
  that contract.
- The reusable `FillModel` intentionally models deterministic full fills. Provider liquidity,
  partial fills, exchange taxes, corporate actions, and borrow mechanics need additional event data
  and remain outside Task 10. Long-entry risk already rejects shorts, leverage, derivatives, minimum
  notional failures, and invalid venue precision through the shared risk pipeline.
- Fixed latency is exact as a timestamp deadline but fills only when a supplied market event exists;
  it does not synthesize intrabar events.
- Session closeout uses adjacent event dates in the instrument timezone as the deterministic session
  boundary signal. An exchange-calendar service and explicit session-close events are not yet
  present; therefore sparse data may defer the reducing fill to the next supplied event.
- Benchmark evidence is a frictionless buy-and-hold curve from the first close. Callers remain
  responsible for supplying an appropriate instrument/venue/horizon benchmark dataset when a
  different declared benchmark is required.
- These gates generate evidence status only. They do not claim or guarantee performance or
  profitability and do not submit broker or live orders.
