---
name: gate
description: How the camas gate works — when it runs, how to read a needs_reasoning residual, and the no-masking rule.
---

The camas gate keeps the workspace green as you edit, in two layers:

- On every file change, a `FileChanged` hook runs camas's deterministic fixers (`camas fix`)
  scoped to that file — zero model tokens. Define a `fix` task (your mutating,
  behavior-preserving leaves) for it to run.
- After each batch of edits, a `PostToolBatch` hook calls the `camas_gate` tool: it applies
  the deterministic fixes for free, runs the project's checks over the fixed workspace, and
  classifies the residual — `autofixed` (nothing left) or `needs_reasoning` (a check still fails).

When the gate surfaces a `needs_reasoning` residual, fix the underlying problem — never mask
it: do not suppress, disable, or loosen a check to make the gate pass. A green gate must mean
the work is actually correct.

The gate runs the project's default camas task; scope or time-box it by editing the hook's
`input` in `hooks/hooks.json`.
