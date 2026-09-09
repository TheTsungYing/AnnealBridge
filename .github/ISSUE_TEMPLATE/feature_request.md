---
name: Feature request
about: Propose a new capability, backend, or contract change
title: ""
labels: enhancement
assignees: ""
---

## Problem

What can't you do today, or what is awkward? If it is a class of optimization
problem, include a small example JSON.

## Proposed change

What should AnnealBridge do instead. If it touches the problem JSON contract,
say whether it is backwards compatible with schema versions `1.0` / `1.1`.

## Design principles check

The project keeps a few hard lines (see `docs/architecture.md`). Please note
whether the proposal:

- lets the agent supply anything mathematical (QUBO, penalty weights, slack,
  encodings) — this is out of scope by design;
- judges feasibility or ranks by solver energy instead of re-validating
  against the original problem;
- silently falls back to another backend or clamps a user parameter;
- requires the core (`models`, `validation`, `compiler`, `penalty`,
  `orchestration`) to know a specific backend by name.

## Alternatives considered

Other ways to get the same outcome, and why they are worse.
