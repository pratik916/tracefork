---
name: Feature request
about: Suggest an idea
labels: enhancement
---

## Problem

What problem are you trying to solve? What's missing or painful today?

## Proposed solution

Describe the feature or change you'd like to see.

## Alternatives

What alternatives or workarounds have you considered?

## Scope/invariants impact

Does this affect any of tracefork's core invariants? Please note if it touches:

- Offline / $0 / no-key operation for tests, spike, `validate`, or the demo
- The determinism boundary (agent reads time/ids only via `NondetSource`)
- The verifier's hash-check proof (vs. assertion) or the drift negative control
- Packaging/PyPI metadata or the public CLI surface
