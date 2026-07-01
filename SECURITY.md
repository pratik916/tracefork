# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x (latest `main`) | :white_check_mark: |

tracefork is pre-1.0; only the latest `0.1.x` release / `main` branch receives
security fixes.

## Reporting a Vulnerability

Please **do not open a public GitHub issue** for security vulnerabilities.

Instead, report privately using one of:

- Email **godofcode.pratik@gmail.com**, or
- [GitHub private vulnerability reporting](https://github.com/pratik916/tracefork/security/advisories/new)
  for this repository.

Please include what you found, how to reproduce it, and its potential impact. I'll
acknowledge reports and follow up as soon as possible.

## Security Posture

For context when assessing impact, here's how tracefork is actually built:

- **Tapes are JSON + base64, never pickle.** `Tape.to_bytes()` / `Tape.from_bytes()`
  (`src/tracefork/tape.py`) serialize to plain JSON with base64-encoded blobs. Loading a
  tape — including one you didn't create yourself — cannot execute arbitrary code.
- **`tracefork serve` binds `127.0.0.1`, same-origin, no CORS.** The live web UI
  (`src/tracefork/server.py`) is a local-only FastAPI app; it is not intended to be
  exposed on a network interface or behind a reverse proxy without additional
  hardening.
- **The HTML report escapes `</script>`.** `report.py` injects tape JSON into the
  single-file report (`web/report.html`) with escaping against `</script>` breakout, so
  a tape containing that sequence in recorded content can't terminate the inline
  `<script>` block early.
- **The only networked code path is `blame` against a real run**, which calls the live
  Anthropic API to re-run counterfactual tails. It is budget-capped: `BudgetGovernor`
  estimates cost from `constants.PRICING_TABLE` before any spend and `BlameEngine.rank()`
  raises `BudgetExceededError` if the estimate exceeds the caller's `budget_usd`. Every
  other command — `replay`, `verify`, `fork`, `report`, `serve`, `validate` — is offline
  and makes no network calls.

If you find a case where any of the above doesn't hold, that's a security bug — please
report it as above.
