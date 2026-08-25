# Stability policy

This document defines what tracefork's SemVer promise ("this project adheres to Semantic
Versioning", `CHANGELOG.md`) actually covers, starting at `1.0.0`. If a change would break
something not listed here as covered, it is not a SemVer-breaking change by this project's
definition — but it's still worth a `CHANGELOG.md` note if it's user-visible.

## 1. The Python API

**The public API is exactly the names in `tracefork.__all__`** (`src/tracefork/__init__.py`).
That list is long on purpose — the record/replay/fork/blame product surface (`Recorder`,
`Tape`, `ReplayVerifier`, `ForkEngine`, `BlameEngine`, `TapeStore`, `generate_report`, …), the
plugin/extension mechanism (`Registry` and the five group constants — see
[`docs/plugin-api.md`](plugin-api.md)), the framework adapters, and the redaction/OTel/MCP
helpers — but it is the *complete* list. If a symbol isn't in `__all__`, importing it works
today (Python doesn't enforce `__all__`) but is not covered:

- **`from tracefork import <Name>` is the only supported import path** for a SemVer-covered
  symbol. `docs/plugin-api.md` states this explicitly for the plugin surface (`Registry`, the
  five group constants, the five protocols) — the same rule applies to every other name in
  `__all__`.
- **A direct `tracefork.<submodule>` import is always internal**, even when the submodule
  happens to define a symbol that's *also* re-exported at the top level. `from tracefork import
  Tape` is covered; `from tracefork.tape import Tape` reaches the same object today but is not
  a promise — a future release could restructure `tape.py` without that counting as a breaking
  change, as long as `tracefork.Tape` keeps working.
- **A breaking change to a symbol in `__all__`** — removing it, changing its call signature
  incompatibly, changing its return type's shape — is a major-version bump, preceded by one
  minor release carrying a `DeprecationWarning` (see §5) and a `CHANGELOG.md` note under a
  `Deprecated` heading.

### Explicitly internal: the test-scaffolding modules

These modules exist to build offline, deterministic test fixtures and self-validation runs.
They live in `src/tracefork/` (not `tests/`) so production code never imports from `tests/`
(see `CLAUDE.md`), but they are not part of the product and carry no stability promise at all —
including within a patch release:

`synthetic.py`, `fixtures.py`, `wire.py`, `faults.py`, `archetypes.py`,
`competing_faults.py`, `concurrent_validate.py`, `ci_calibration.py`, `selfaudit.py`,
`eventstream.py`, `repl.py`.

None of these are reachable from `tracefork.__all__`. If your code imports from one of them
directly, you are depending on test scaffolding, not the library — see
[`examples/record_your_agent.py`](../examples/record_your_agent.py) for the shape a real
integration should take instead.

## 2. The CLI

Covered, under the same "one minor release deprecates, next major removes" discipline as §1:

- **Command names** (`replay`, `verify`, `fork`, `blame`, `report`, `serve`, `validate`,
  `bench`, `coalition-fork`, `tournament`, `diff`, `receipt`, `coverage`, `prune`,
  `bundle-export`/`bundle-import`, `export`/`ingest`, `proxy`, `session`, and the rest listed by
  `tracefork --help`).
- **Flag names** for each command (`--agent`, `--step`, `--budget`, `--k`, `--check`,
  `--corpus`, `--store`, etc.) and their types/defaults.
- **Exit codes** — `0` for success, non-zero on failure/drift (e.g. `verify`'s CI-gate exit
  code on divergence).
- **`--json`/`--output` schemas** — the field names and shapes of any machine-readable output a
  command opts into (structured JSON exports, the trust receipt's JSON document, etc.).

**Not covered:** human-readable stdout formatting — column widths, exact wording, emoji/box
characters, the layout of `tracefork validate`'s printed report. Script against `--json`/
`--output` or a documented exit code, not against parsing plain stdout.

## 3. The tape on-disk format

**Any `1.x` reader opens tapes written by every `0.x` and every `1.x` release.** This is
already how the format works, not a new promise: `Tape.from_bytes`/`Tape.load`
(`src/tracefork/tape.py`) dispatch on an embedded format version and run a read-time upcaster
chain up to the current version — including the original pre-envelope JSON+base64 format
(detected by the absence of the magic marker) forward through every versioned envelope since.
A format bump is always additive: a new version adds fields/encodes more efficiently, but
`from_bytes` keeps reading every version that came before it. This policy commits to keeping
that upcaster chain intact for the life of the `1.x` line — a tape you recorded on `0.3.0` will
still load on whatever `1.x` release you're running.

This is a read compatibility promise, not a write one: `to_bytes()` always writes the current
version's envelope; there's no option to write an older format.

## 4. Python support window

tracefork requires **Python >= 3.12** (`pyproject.toml`'s `requires-python`) and is tested in CI
against 3.12, 3.13, and 3.14 (`.github/workflows/ci.yml`'s matrix). A Python version is dropped
no sooner than one minor tracefork release after CI stops testing it, and the drop is called out
in `CHANGELOG.md`.

## 5. Deprecation procedure

There is one procedure for deprecating anything covered by §1–§3:

1. The replacement (if any) ships first, so there's something to migrate to.
2. The deprecated name/flag/format keeps working for at least one minor release, emitting
   `DeprecationWarning` (Python API) or a printed deprecation notice (CLI) pointing at the
   replacement.
3. `CHANGELOG.md` lists it under a `### Deprecated` heading in the release that starts the
   warning, and again under `### Removed` in the major release that drops it.
4. Removal happens only in a major-version release.

As of `1.0.0` there is no deprecation machinery in place yet (no code path emits
`DeprecationWarning`) because nothing has been deprecated yet — this section states the
procedure future removals will follow, not a claim that it's already been exercised.

## Where this is referenced

Linked from [`README.md`](../README.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md), which
gates any PR that adds, removes, or changes the signature of a name in `tracefork.__all__`, a
CLI command/flag, or the tape format on reading this document first. `docs/plugin-api.md`'s own
Stability policy section covers the plugin mechanism specifically and points back here for the
project-wide policy it used to (incorrectly) assume existed.
