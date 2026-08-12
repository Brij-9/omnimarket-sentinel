# Strategy validation and promotion evidence

Promotion is evidence-based and paper-first. A profit target does not change a gate.
Turning USD 10 into USD 1,000,000 in one month is a 100,000x aspiration, is
extraordinarily improbable, is not a guarantee or acceptance criterion, and USD 10
can be lost in full. Other markets require explicit data, calendar, broker, cost, and
validation adapters before they enter this process.

## Reproducible local evidence

1. Run `python -m pytest tests/e2e -v`. The India, US, and crypto fixtures are
   sanitized, point-in-time, deterministic, and offline.
2. Confirm research evidence is nonempty, sourced, timestamped, no later than the
   analysis cutoff, and records model, prompt, and configuration identity. The
   fixture packet is Tauric-style typed evidence; it does not run a graph.
3. Confirm the strategy sees only its supplied prefix, produces deterministic
   protective exits, and abstains on malformed, stale, ineligible, or illiquid data.
4. Confirm the shared risk engine approves the exact intent or reports stable
   rejection reasons. Below-minimum orders remain rejected and are never scaled up.
5. Review after-cost return, benchmark excess, maximum drawdown, trade count,
   turnover, exposure, and the identical-parameter 2x-cost robustness run. An
   insufficient sample remains `INSUFFICIENT_EVIDENCE`; it is not inferred as good.
6. Require the documented out-of-sample trade count and duration gates before paper
   or live promotion. Do not reuse validation/test observations across folds.
7. Run paper execution through deterministic fills, fees, ledger, partial-fill and
   duplicate handling, reconciliation, and dashboard export. Any unexplained
   reconciliation event or risk breach rejects promotion.
8. Repeat against the intended provider's sanitized sandbox/fixture evidence and
   exact venue capabilities. Keep every live flag false.

Expected failures are actionable evidence: stale/provider loss rejects before risk;
duplicate or changed client intent IDs fail closed; minimum-notional/precision
failures require a different instrument or more capital; partial fills remain open
until reconciled; cash/position/order mismatch activates the kill switch. Never edit
an evidence artifact to make a gate pass.

The supported fixture matrix is representative India, US, and crypto spot. It does
not validate every instrument, venue, market session, tax treatment, corporate
action, outage, or liquidity regime.
