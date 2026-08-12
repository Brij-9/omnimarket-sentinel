# Security policy

## Supported version

Security fixes are applied to the current `0.1.x` development line only. This is a
local research and supervised-execution project, not a hosted custody service.

## Private reporting

Do not open a public issue containing an exploit, account detail, provider response,
dashboard artifact, database, log, or credential. After the repository owner enables
GitHub private vulnerability reporting, open the repository's **Security tab** and
select **Report a vulnerability**. If that button is unavailable, use a private
contact method published on the repository owner's GitHub profile at
https://github.com/Brij-9. If no private contact method is listed, open a public issue
containing only a request for a private channel and no vulnerability details. Give
the affected version/commit, a minimal sanitized reproduction, impact, and suggested
containment. Do not test against accounts or infrastructure you do not own.

If any credential may have appeared in a terminal, file, log, screenshot, issue,
artifact, commit, or model prompt, revoke and rotate it immediately; deleting the
visible copy is not sufficient. Review account activity, open orders, withdrawal
settings, IP allowlists, and the repository history.

## Credential and account controls

- Keep credentials local in an ignored `.env` or an operating-system secret store.
  Never place values in source, fixtures, CI, prompts, dashboard exports, or logs.
- Grant least privilege. Separate paper and live accounts where the provider permits.
- Use crypto keys with withdrawals disabled. Do not reuse custody keys.
- Restrict keys to known outbound IPs where supported and rotate on personnel,
  workstation, or network changes.
- Treat account IDs and provider responses as sensitive even when they cannot submit
  orders.
- Keep every real-trading flag false until the operator completes the read-only
  readiness procedure. A package install never enables live mode.

Audit and dashboard payloads use bounded secret detection and redaction, but that is
defense in depth, not permission to log raw provider objects. Exceptions must remain
secret-free and provider callbacks or opaque client objects must never enter durable
events.

## Dependencies and updates

Index packages are hash-locked. The GitHub workflow is configured to audit them and
uses immutable action commit SHAs, but a post-upload Actions run has not yet verified
the Python 3.12 lock installation or audit. The current lock was generated with
Python 3.13. The Tauric TradingAgents dependency is pinned to an exact reviewed
commit; pip cannot hash-enforce a VCS requirement, so the workflow verifies and
installs it separately and the commit requires manual source/dependency review.
Update dependencies in a focused change, regenerate the lock in a clean Python 3.12
environment, review licenses and transitive changes, run `pip-audit`, then run the
full offline suite and package smoke.

Known or suspected compromise activates the kill switch, stops new submissions,
preserves sanitized audit evidence, and requires healthy reconciliation before any
clear or resumed operation.
