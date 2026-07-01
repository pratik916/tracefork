---
name: Bug report
about: Report a problem
labels: bug
---

## Description

A clear, concise description of the bug.

## Reproduction

- Command(s) run (e.g. `tracefork replay ...`, `tracefork fork ...`, `tracefork blame ...`):
- Tape involved (attach or describe how it was recorded, if applicable):
- Minimal steps to reproduce:

## Expected vs actual

**Expected:**

**Actual:**

## Environment

- `tracefork --version`:
- Python version:
- OS:

## Determinism-boundary note

Did the agent under trace read all time/id nondeterminism only via `NondetSource`
(no direct `datetime.now()`, `uuid`, or `random` calls)? If unsure, say so — this
affects whether replay/fork/blame can be expected to be bit-exact.
