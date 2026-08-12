# OmniMarket Sentinel completion audit

This audit maps the ten approved design acceptance criteria to direct executable
evidence. `Complete` means the cited behavior has direct observed evidence, not only
an implementation or an unexecuted workflow definition. It does not mean profitable,
suitable for an account, or ready for live money.

| # | Acceptance criterion | Direct evidence | Observed result | Status |
|---:|---|---|---|---|
| 1 | Research and backtest representative India, US, and crypto instruments from one CLI | `tests/e2e/test_cli_fixture_workflows.py`; `tests/e2e/test_india_pipeline.py`; `tests/e2e/test_us_pipeline.py`; `tests/e2e/test_crypto_pipeline.py` | The installed default CLI runs all three fixed sanitized research/backtest/paper workflows with no network connection and rejects any non-allowlisted instrument or time | Complete |
| 2 | The same strategy/risk pipeline drives replay, paper, and live modes | `src/market_sentinel/operations/fixture_pipeline.py`; `tests/e2e/test_cli_fixture_workflows.py`; `tests/e2e/test_live_lock.py` | CLI E2E uses production strategy, PositionSizer/RiskEngine, BacktestEngine, PaperBroker, and a gated LiveOrderService | Complete |
| 3 | Reports include after-cost performance, benchmark, drawdown, robustness, and evidence sufficiency | `tests/e2e/test_india_pipeline.py::test_india_fixture_researches_risks_papers_and_reconciles`; `tests/backtest/test_metrics.py`; `tests/backtest/test_promotion.py` | Typed evidence exposes all required fields and the insufficient-sample reason | Complete |
| 4 | Tauric output is typed, sourced, timestamped, replayable, and cannot bypass risk | `tests/research/test_tauric.py`; `tests/e2e/test_us_pipeline.py::test_us_fixture_is_deterministic_and_uses_typed_point_in_time_evidence` | Typed packet has evidence, model/prompt/config identity, deterministic replay, and then passes through risk | Complete |
| 5 | Groww, Alpaca, and CCXT expose secret-free capability/readiness state | `tests/brokers/test_preflight.py`; `tests/brokers/test_groww.py`; `tests/brokers/test_alpaca.py`; `tests/brokers/test_ccxt_spot.py`; `tests/test_security.py` | Exact manifests/capabilities are covered and output redaction is enforced | Complete |
| 6 | Paper orders execute end to end and reconcile with the internal ledger | `tests/e2e/test_india_pipeline.py`; `tests/e2e/test_us_pipeline.py::test_us_fixture_reconciliation_detects_independent_broker_position_mismatch`; `tests/e2e/test_crypto_pipeline.py` | Each deterministic order reaches `filled`; reconciliation compares independent PaperBroker account state with the ledger and detects injected position drift | Complete |
| 7 | Live submission is impossible without passing preflight and exact unexpired confirmation | `tests/e2e/test_live_lock.py`; `tests/execution/test_live.py`; `tests/execution/test_approval.py`; `tests/execution/test_safety_authority.py` | No-credential E2E rejects `PREFLIGHT_NOT_READY` with zero submit/query calls; exact authority gates have unit coverage | Complete |
| 8 | Stale data, provider loss, duplicate requests, partial fills, and position mismatch fail closed | `tests/e2e/test_india_pipeline.py::test_fixture_runner_enforces_no_lookahead_freshness_and_request_identity`; `tests/e2e/test_us_pipeline.py::test_us_fixture_reconciliation_detects_independent_broker_position_mismatch`; `tests/data/test_freshness.py`; `tests/execution/test_paper.py`; `tests/execution/test_reconcile.py` | Lookahead, naive timestamps, stale and duplicate requests reject; broker position drift produces unhealthy reconciliation and activates the kill switch | Complete |
| 9 | GitHub Actions runs lint, type, unit, fixture integration, security, and package gates | `tests/test_project_config.py`; `.github/workflows/ci.yml` | Required jobs and safety settings are statically parsed, but no post-upload GitHub Actions run has executed them | Incomplete |
| 10 | Documentation explains setup, paper operation, live gates, limitations, and no profit assurance | `tests/test_project_config.py::test_required_operator_documents_exist_and_reject_profit_promises`; `README.md`; `SECURITY.md`; `docs/operations/` | Required operator boundaries and the 100,000x warning are present | Complete |

## Verification commands

The implementing task records fresh command exits and counts in its ignored Task 16
report. The controller must rerun them before release; this file is not evidence for
a command that has not run. Required commands are:

- `python -m pytest tests/e2e tests/test_project_config.py -v`
- `python -m ruff check .`
- `python -m mypy --strict src/market_sentinel`
- `python -m pytest -v --cov=market_sentinel --cov-report=term-missing`
- `python -m pip_audit -r requirements.lock` (expected to report the exact VCS
  limitation; the workflow is configured to audit the hash-verifiable filtered lock
  and separately verify the exact commit, but that workflow has not run)
- `python -m build`
- clean-environment wheel import and hidden `status` smoke
- `git diff --check`

## Important limitations and expected skips

- India, US, and crypto fixtures are representative and sanitized. Other markets
  require explicit data, calendar, broker, cost, instrument, and validation adapters.
- Criterion 9 remains incomplete until the post-upload GitHub Actions jobs run
  successfully.
- The checked-in lock was generated with Python 3.13. Python 3.12 lock installation
  remains unverified; the Python 3.12 workflow is a pending validation gate.
- The Tauric-style e2e packet never runs a graph. Byte-identical LLM output is not
  claimed. The exact VCS pin is not compatible with pip `--require-hashes`.
- No fixture predicts future performance. Turning USD 10 into USD 1,000,000 in one
  month is a 100,000x aspiration, is extraordinarily improbable, is not a guarantee
  or acceptance criterion, and USD 10 can be lost in full.
- All live flags are false. The only e2e live attempt proves rejection and zero
  adapter submit/query calls; no successful live order is fabricated.
- Dashboard filesystem tests may skip Windows directory-symlink cases when the host
  cannot create them, and POSIX anonymous `O_TMPFILE` cases on unsupported hosts.
  Windows exact handle-bound replacement and POSIX anonymous create-once semantics
  remain fail-closed contracts.
