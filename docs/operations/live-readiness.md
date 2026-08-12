# Live readiness: read-only procedure

This procedure observes readiness; it does not authorize or submit an order. Keep all
real-trading flags false while preparing an account. Never paste credentials into a
command, issue, log, or dashboard.

## Sequence

1. Complete provider onboarding and applicable legal/regulatory review outside this
   repository. For India, verify the current SEBI/broker retail-algorithm rules. For
   crypto, use spot-only permissions with withdrawals disabled.
2. Configure least-privilege values locally, restrict outbound IPs where supported,
   and keep paper/live accounts distinct.
3. Run `python -m market_sentinel.cli live-preflight --broker alpaca`, `groww`, or
   `ccxt`. `scripts/check-readiness.ps1` is the equivalent PowerShell wrapper.
4. Read only the sorted missing gate names. A nonzero exit or any absent, duplicate,
   unknown, stale, malformed, or unavailable gate means not ready. Do not override it.
5. Validate a strategy using the paper-first procedure in `strategy-validation.md`.
6. Require a current healthy reconciliation report for the exact broker and ledger.
   Investigate every cash, currency, position, open-order, partial-fill, provider, or
   freshness mismatch. A mismatch activates the durable kill switch.
7. Verify no unresolved submission interlock exists and the current-session safety
   authority shares the exact EventStore with approval and reconciliation.
8. Construct and review one exact proposal. Live submission additionally requires a
   fresh risk decision, unchanged portfolio/snapshot hashes, supported capabilities,
   an inactive kill switch, and an exact, unexpired, unused local confirmation. The
   phrase is `I_CONFIRM_REAL_MONEY_ORDER`; this document intentionally provides no
   command that supplies it.

`PREFLIGHT_NOT_READY` means a fresh exact manifest did not pass. `ORDER_UNSUPPORTED`
means adapter capabilities do not match the intent. `RISK_NOT_APPROVED` or
`RISK_STALE` requires a new proposal, not mutation. `RECONCILIATION_NOT_CURRENT`,
`KILL_SWITCH_ACTIVE`, or `SAFETY_STATE_CHANGED` requires investigation and a new
healthy generation. `CONFIRMATION_INVALID` or `CONFIRMATION_USED` requires a new
operator review. `SUBMISSION_UNKNOWN` forbids retrying blindly; reconcile by the
client intent ID first.

## Dashboard portability contract

Dashboard export is local and create-once/fail-closed. On Windows the implementation
uses exact handle-bound replacement and rejects reparse/network-path uncertainty. On
POSIX it uses an anonymous `O_TMPFILE` create-once path and fails closed when the
filesystem cannot provide the required semantics; expected platform skips are
recorded in the completion audit. Never weaken the path checks for convenience. The
JSON is a redacted status view, not a credential store or an authorization token.
