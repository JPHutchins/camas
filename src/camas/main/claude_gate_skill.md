---
name: gate
description: Keep the workspace green while you edit — the free autofix hooks, and delegating the check-and-fix loop to the tiered camas-fixer ladder so residuals never spend your reasoning. Read after a batch of edits and before declaring work done.
---

The camas gate keeps the workspace green as you edit, in two layers — both driven by what the
project declares in `Config.agent`.

**Fix (automatic, free).** A `PostToolBatch` hook runs `camas mcp fix` after each edit batch, and
a `Stop` hook runs it again as you finish your turn — the node the project registered as
`Config.agent.fix` (its mutating, behavior-preserving auto-fixers: formatters, `--fix` linters),
scoped to the just-changed files. Both run at zero model tokens and never ask you anything; with
no fix registered they are a no-op.

**Check (you delegate — to a tiered ladder).** The check node (`Config.agent.check`, else the
default task) is read-only: it runs the project's checks and classifies the result `green` or
`needs_reasoning`. After a batch of edits, and before you declare work done, **delegate the
check-and-fix loop to the camas-fixer ladder** rather than running the checks and chasing
residuals in your own context. Subagents cannot nest — a haiku subagent cannot spawn a sonnet
one — so you orchestrate the escalation yourself:

- For a **lint/format** residual (the diagnostic names formatters or `--fix` linters): spawn
  `camas-lint-fixer-haiku` with the changed paths as its scope. It takes one pass and hands back.
  If it hands back not green, escalate by spawning `camas-lint-fixer-sonnet` on the same scope —
  it also takes one pass. If *that* hands back not green, the residual is yours: take it from
  there.
- For a **test/coverage** residual (the diagnostic names a test runner or coverage tool): spawn
  `camas-test-fixer` directly — it fixes tests and coverage on a capable model in one delegation,
  ending by re-gating to confirm green. If it hands back not green, or says the fix is ambiguous
  (test wrong vs. behavior wrong), take it from there yourself.
- When diagnostics span both kinds, or you can't tell from the names, start with the lint ladder
  (cheaper) and send whatever remains to `camas-test-fixer`.

Run each fixer in the background and keep working; for independent changed scopes, spawn one per
scope so they run in parallel. Each fixer hands back only what it could not settle: a green
result means that scope is done; a result quoting remaining diagnostics is the residual that
needs your reasoning.

**The Stop-hook nudge.** If your turn ends before you delegated (or a fixer's residual never got
picked up), a background check runs after you stop and, if the workspace isn't green, wakes you
with a reminder to launch the fixer ladder — so a skipped check-and-fix loop doesn't go silently
unnoticed. It wakes you at most once per prompt and never for a configuration gap (no check node
registered), so it cannot loop your turn. Prefer delegating proactively per the ladder above;
treat the nudge as a backstop, not your primary signal.

Never mask a residual — yours or a fixer's: do not suppress, disable, or loosen a check to make
the gate pass. A green gate must mean the work is actually correct.
