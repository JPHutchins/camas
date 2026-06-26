---
name: gate
description: How the camas gate works — the two deterministic layers, how a needs_reasoning residual is delegated to the cheap fixer, and the no-masking rule.
---

The camas gate keeps the workspace green as you edit, in two deterministic layers — both
driven by what the project declares in `Config.agent`:

- **Fix (FileChanged, scoped).** On every file change, a `FileChanged` hook runs
  `camas fix --paths <file>` — the project's declared `Config.agent.fix` node (its mutating,
  behavior-preserving auto-fixers: formatters, `--fix` linters), scoped to that file, at zero
  model tokens. It never asks you anything.
- **Check (PostToolBatch, scoped).** After each batch of edits, a `PostToolBatch` hook runs
  `camas mcp gate` over the just-edited files. The gate is read-only — it does NOT fix — it
  runs the project's check node (`Config.agent.check`, else the default task) scoped to those
  files and returns a binary verdict: `green` (the checks pass) or `needs_reasoning` (a check
  still fails), with the failing diagnostics.

When the gate returns `needs_reasoning`, delegate it to the `camas-fixer` subagent before
reasoning about the failure yourself. It runs on a cheap model, off your context, and loops
fix → re-gate within a bounded turn budget, so a fix-and-recheck loop never costs your tokens.
It hands back only what it could not settle: if its final message says the gate is green, the
work is done; if it quotes remaining diagnostics, those are the ones that actually need your
reasoning — and the gate prints the exact `camas mcp gate …` command to re-run that scope.

Never mask a residual — yours or the fixer's: do not suppress, disable, or loosen a check to
make the gate pass. A green gate must mean the work is actually correct.
