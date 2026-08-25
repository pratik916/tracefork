# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x: |

Only the latest `1.0.x` release / `main` branch receives security fixes. Pre-1.0 releases
(`0.x`) are not supported.

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

- **Tapes are a versioned zstd + content-addressed binary envelope with a JSON header — never
  pickle.** `Tape.to_bytes()` (`src/tracefork/tape.py:157-163`) serializes to a magic marker, a
  `uint16` format version, a plain-JSON header (agent name, draws, exchange hash pairs,
  blob-hash order), then each unique request/response blob zstd-compressed and
  content-addressed by sha256 (`_encode_v6`, `src/tracefork/tape.py:291-329`). There is no
  base64 in the current format (earlier releases
  used a JSON + base64 encoding; `from_bytes` still reads it for backward compatibility, but
  nothing produced today writes it). Because none of this is pickle or any other
  code-executing deserializer, loading a tape you didn't create — including through
  `Tape.load()`'s SQLite path — cannot execute arbitrary code.
- **`tracefork serve` binds `127.0.0.1`, same-origin, no CORS.** The live web UI
  (`src/tracefork/server.py`, `src/tracefork/cli.py:1187`) is a local-only FastAPI app; it is
  not intended to be exposed on a network interface or behind a reverse proxy without
  additional hardening.
- **The checkpoint-tail endpoint is confined to an explicit allowlist.** `GET
  /api/checkpoint/tail` opens whatever SQLite file its `path` query parameter names, and
  opening it writes to it (`PRAGMA journal_mode=WAL` plus `CREATE TABLE IF NOT EXISTS`), so an
  unauthenticated, CSRF-reachable `GET` must never be able to point that at an arbitrary
  third-party file. The default allowlist is empty, which 403s every request
  (`resolve_confined_checkpoint_path`, `src/tracefork/checkpoint.py:96-111`); an operator opts
  in per directory via `serve --allow-checkpoint-dir` (`init_checkpoint_dirs`,
  `src/tracefork/server.py:75-77`), and both the requested path and every allowed directory are
  resolved through `os.path.realpath` before comparison so a symlink can't be used to escape
  confinement.
- **The HTML report escapes `</script>`.** `report.py`'s `_safe_json`
  (`src/tracefork/report.py:171-173`) escapes the tape JSON it injects into the single-file
  report (`web/report.html`) so a tape containing that sequence in recorded content can't
  terminate the inline `<script>` block early. `web/report.html` separately HTML-escapes
  recorded values (its `escape()`/`escapeAttr()` helpers) before inserting them into the DOM at
  render time.
- **The networked code paths are: `blame` against a real run, `tracefork proxy record`, and
  any `Recorder`/`AsyncRecorder` wrapping a live client.** `blame` calls the live Anthropic API
  to re-run counterfactual tails and is budget-capped: `BudgetGovernor` estimates cost from
  `constants.PRICING_TABLE` before any spend and `BlameEngine.rank()` raises
  `BudgetExceededError` if the estimate exceeds the caller's `budget_usd`.
  `tracefork proxy record` (`src/tracefork/proxy.py`) forwards every request to the real
  upstream over TLS by design — it's a record-mode proxy, not an offline command. Wrapping a
  real `anthropic.Anthropic`/`AsyncAnthropic` client with `Recorder`/`AsyncRecorder` is the
  product's primary use case and is exactly as networked as calling that client directly.
  Every other command — `replay`, `verify`, `fork`, `report`, `serve`, `validate`, `bench` — is
  offline and makes no network calls.

## What is on a tape

By default, request/response bodies and nondeterminism draws (clock, ids, random) are stored
**unredacted** — a tape is a faithful recording of what the agent actually sent and received,
and redaction is opt-in (`src/tracefork/redact.py`'s module docstring): nothing in `redact.py`
runs unless a `Redactor` is built and passed to `Recorder`/`AsyncRecorder`
(`redactor=None` is the default, unchanged behavior).

Auth headers (`Authorization`, `x-api-key`, etc.) are **not** stored on the default
identity-matcher path: `IdentityMatcher.stored_request` persists `request.content` — the
request body only, not `request.headers` (`src/tracefork/matcher.py:87-88`). A recorded tape
therefore does not contain your API key by default. This is a property of the default matcher,
not a redaction feature — a custom `RequestMatcher` that chose to persist headers could change
it.

To scrub what a tape *does* record — headers plus recognized secret-env-var values, always, once
a `Redactor` is in use; message content, opt-in on top of that — pass
`redact.safe_defaults()` (metadata-only) or `redact.with_content_redaction()` (also redacts
message content, and marks `Tape.content_redacted = True` since that changes what a replayed
agent reads back — see `redact.py`'s module docstring and the README's Redaction section) as
the `redactor=` argument to `Recorder`/`AsyncRecorder`.

## Trusting a tape or bundle you did not create

Opening a tape, a bundle, or an HTML report someone else generated means trusting their file to
the extent described here:

- **No code execution.** Tapes and bundles are never pickle (see Security Posture above);
  `import_bundle` (`src/tracefork/bundle.py:88`) writes through the same CAS-guarded
  `save_tape`/`save_branch` path a normal recording uses, never a raw `INSERT`, so a colliding
  `run_id`/`branch_id` with genuinely different content raises `TapeConflictError` instead of
  silently clobbering your data. `ingest_otel_trace`/`ingest_openinference_dataset`
  (`src/tracefork/interop.py`) read a plain JSON `Mapping` — no code execution surface there
  either.
- **No guaranteed bit-exact replay from an untrusted tape.** A tape someone else recorded
  reflects whatever the recorder chose not to redact; nothing about loading it is unsafe, but
  don't assume its content is what you'd have recorded yourself.
- **Decompression is not size-capped.** `Tape.from_bytes`/`Tape.load` decompress each blob via
  `zstandard.ZstdDecompressor().decompress(...)` (`src/tracefork/tape.py`) with no
  `max_output_size` argument. A maliciously crafted small tape or bundle could decompress to a
  very large size and exhaust memory (a decompression-bomb DoS) before any other check runs.
  Treat an untrusted tape or bundle file the way you'd treat any other untrusted compressed
  archive — don't open one from a source you don't trust on a machine where a memory-exhaustion
  crash matters.
- **The HTML report's escaping is a mitigation, not an audited guarantee.** See the `</script>`
  and DOM-escaping bullet above; it covers the sites that exist today, not a formal claim that
  every future rendering path stays escaped.

If you find a case where any of the above doesn't hold, that's a security bug — please
report it as above.
