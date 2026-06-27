---
name: gate
description: How the camas gate works — the two deterministic layers, how a needs_reasoning residual is delegated to the cheap fixer, and the no-masking rule.
---

The camas gate keeps the workspace green as you edit, in two deterministic layers — both
driven by what the project declares in `Config.agent`:

- **Fix (FileChanged, scoped).** On every file change, a `FileChanged` hook runs
  `camas mcp fix --paths <file>` — which runs the node the project *registered* as
  `Config.agent.fix` (its mutating, behavior-preserving auto-fixers: formatters, `--fix`
  linters, named whatever the author chose), scoped to that file, at zero model tokens. It
  never asks you anything; with no fix registered it is a no-op.
- **Check (PostToolBatch, scoped).** After each batch of edits, a `PostToolBatch` hook runs
  `camas mcp gate` over the just-edited files. The gate is read-only — it does NOT fix — it
  runs the project's check node (`Config.agent.check`, else the default task) scoped to those
  files and returns a binary verdict: `green` (the checks pass) or `needs_reasoning` (a check
  still fails), with the failing diagnostics.

When the gate returns `needs_reasoning`, delegate it to the `camas-fixer` subagent before
reasoning about the failure yourself — and hand it the gate's **`rerun`** handle (its
`{task, paths, under}`, also printed as the `camas mcp gate …` re-run command). That tells the
fixer exactly which scope to re-gate: it loops fix → `camas mcp gate --paths <those paths>` on
a cheap model, off your context, within a bounded turn budget, so the loop never costs your
tokens. It hands back only what it could not settle: if its final message says the gate is
green, the work is done; if it quotes remaining diagnostics, those actually need your reasoning.

Never mask a residual — yours or the fixer's: do not suppress, disable, or loosen a check to
make the gate pass. A green gate must mean the work is actually correct.
