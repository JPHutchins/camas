---
name: gate
description: How the camas static-analysis gate works — when it runs, how to read a needs_reasoning residual, and the no-masking rule.
---

The camas SA-delegation gate runs after each batch of edits (a `PostToolBatch` hook calls the
`camas_gate` MCP tool). The gate applies camas's deterministic autofix (formatters, import
sorting) for free, runs the project's checks over the fixed workspace, and classifies the
residual: `autofixed` (nothing left) or `needs_reasoning` (a check still fails).

When the gate surfaces a `needs_reasoning` residual, fix the underlying problem — do not mask
it. No `# type: ignore`, no `# noqa`, no disabling or loosening a check to make the gate pass.
A green gate must mean the code is actually correct.

The gate runs the project's default camas task. To scope or time-box it, edit the hook's
`input` in `hooks/hooks.json` (a faster `task`, or an `under` budget), or point it at a check
task whose leaves declare `agent_format` so diagnostics arrive in a structured standard.
