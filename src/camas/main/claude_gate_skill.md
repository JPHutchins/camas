---
name: gate
description: Keep the workspace green while you edit — the deterministic autofix, and delegating the check-and-fix loop to the tiered camas-fixer ladder so residuals never spend your reasoning. Read after a batch of edits and before declaring work done.
---

The camas gate keeps the workspace green as you edit, in two layers — both driven by what the
project declares in `Config.agent`.

**Fix (automatic, deterministic, free).** The project's registered autofix (`Config.agent.fix`:
its mutating, behavior-preserving formatters and `--fix` linters) runs at zero model tokens and
never asks you anything: inside every `camas_gate` call, over the gated paths before the checks
run, and at every turn end from a `Stop` hook. Nothing rewrites your files between edits, so a
half-finished edit (an import not yet used) survives until you gate. With no fix registered, it is
a no-op.

**Check (you delegate — to a tiered ladder).** `camas_gate` checks the check node
(`Config.agent.check`, else the default task) and classifies the residual `green` or
`needs_reasoning`. After a batch of edits, and before you declare work done, **delegate the
check-and-fix loop to the camas-fixer ladder** rather than running the checks and chasing
residuals in your own context. Subagents cannot nest — a haiku subagent cannot spawn a sonnet
one — so you orchestrate the escalation yourself:

- For a **lint/format** residual (the diagnostic names formatters or `--fix` linters): spawn
  `camas-lint-fixer-haiku` with the changed paths as its scope. It works the scope for up to 3
  rounds and hands back only what it could not settle. If it hands back not green, escalate by
  spawning `camas-lint-fixer-sonnet` on the same scope. If *that* hands back not green, the
  residual is yours: take it from there.
- For a **test/coverage** residual (the diagnostic names a test runner or coverage tool): spawn
  `camas-test-fixer` directly — it fixes tests and coverage on a capable model for up to 3 rounds.
  If it hands back not green, or says the fix is ambiguous (test wrong vs. behavior wrong), take
  it from there yourself.
- When diagnostics span both kinds, or you can't tell from the names, start with the lint ladder
  (cheaper) and send whatever remains to `camas-test-fixer`.

Run each fixer in the background and keep working; for independent changed scopes, spawn one per
scope so they run in parallel. Each fixer hands back only what it could not settle: a green
result means that scope is done; a result quoting remaining diagnostics is the residual that
needs your reasoning. A fixer result marked **partial** ran out of turns before it finished — it
is neither green nor a hand-back: resume that fixer (it keeps its context) instead of re-spawning
it or escalating.

Re-read a file before editing it after a gate or a fixer: the autofix may have rewritten it.

**The Stop-hook nudge.** At every turn end, after the `Stop` autofix has run, a background
check runs. If your turn ends before you delegated (or a fixer's residual never got picked up)
and the workspace isn't green, it wakes you with a reminder to launch the fixer ladder — so a
skipped check-and-fix loop doesn't go silently unnoticed. It wakes you at most once per prompt and never
for a configuration gap (no check node registered), so it cannot loop your turn. Prefer
delegating proactively per the ladder above; treat the nudge as a backstop, not your primary
signal.

Never mask a residual — yours or a fixer's: do not suppress, disable, or loosen a check to make
the gate pass. A green gate must mean the work is actually correct.
