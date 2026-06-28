# /// script
# requires-python = ">=3.10"
# dependencies = ["camas"]
# ///
"""Project tasks — run with ``camas``.

Every task below is a cross-platform placeholder (``python -c ...``);
replace the placeholders with your real commands:

	lint = Task("ruff check .")
	test = Task("cargo test", cwd=Path("rust"))
	build = Task("npm run build")

The ``# /// script`` block above is optional PEP 723 inline metadata. With
it (plus the ``__main__`` block at the bottom), PEP 723-aware tools can run
this file standalone — ``uv run tasks.py``, ``pipx run tasks.py`` — building
a throwaway env with camas installed; handy for non-Python projects. It is
also where camas gets version-pinned (``dependencies = ["camas>=X.Y"]``).
Both are inert under plain ``camas``; delete them if you don't want the
standalone path.
"""

from camas import Claude, Config, Parallel, Sequential, Task, run_cli

# A leaf is any shell command, shlex-split (so quote the -c payload). A bare
# string inside Sequential/Parallel coerces to an anonymous Task; tuple form
# Task(("python", "-c", "...")) skips shlex when quoting gets hairy.
hello = Task("python -c \"print('hello from camas')\"")

# matrix= clones the subtree per axis value, interpolating {NAME} into
# cmd/env/cwd; env= is scoped to the subtree. Pin an axis from the CLI:
# camas greet --NAME Grace
greet = Parallel(
	Task("python -c \"import os; print(os.environ['GREETING'] + ', {NAME}!')\""),
	matrix={"NAME": ("Ada", "Grace")},
	env={"GREETING": "hello"},
	help="say hello to everyone at once",
)

# Sequential runs in order, short-circuiting on the first failure; Parallel
# runs concurrently. They nest freely. The binding name is the task name
# (this defines `ci`); pass name= only to rename or to name a nested,
# anonymous group like the Parallel below.
ci = Sequential(
	hello,
	Parallel(greet, "python --version"),
)

# Your deterministic, behavior-preserving auto-fixers, with {paths} so a run scopes to the
# changed files. Register the node (named anything) to Config.agent.fix below; the camas Claude
# Code plugin's FileChanged hook runs it on every edit via `camas mcp fix --paths <file>`, for
# free. Replace the placeholder with your real fixers, e.g.:
#   autofix = Task("ruff check --fix {paths}", mutates=True, paths=".")
#   autofix = Parallel(Task("ruff format {paths}", mutates=True), Task("ruff check --fix {paths}", mutates=True), paths=".")
#
# Mark mutating leaves mutates=True so --under runs them first in sequence (fast inner loop).
# paths= lets {paths} be injected with the changed file — the scope the FileChanged hook passes.
autofix = Task('python -c "" {paths}', name="autofix", mutates=True, paths=".")

# AgentFormat appends structured-output flags when the GATE runs a leaf — the agent reads
# machine-readable diagnostics (sarif, junit, etc.) instead of scraped text. A human
# `camas lint` never sees these extra flags, so the cmd shown in --list stays clean.
#   from camas.v0.task import AgentFormat
#   lint = Task("ruff check .", agent_format=AgentFormat("--output-format sarif", "sarif"))

# Config is discovered by type, under any binding name (here `_`): bare `camas` runs
# default_task — or github_task under GitHub Actions, falling back to default_task when unset.
#
# agent= wires the Claude Code plugin's two-hook gate:
#   fix:   the deterministic autofix node above — the FileChanged hook runs it on every edit
#          (`camas mcp fix --paths <file>`). Declared explicitly, never derived from mutates=
#          (a mutating leaf may be codegen, not a fixer).
#   check: the node the PostToolBatch gate checks (`camas mcp gate`). None (the default)
#          defers to default_task; set it to a different node for a faster gate subset.
#
# The gate scopes check leaves to the changed paths, appends their agent_format args (if any),
# time-boxes by --under, and classifies the residual green vs needs_reasoning. The gate never
# mutates — all mutation lives in the fix node on FileChanged.
_ = Config(default_task=ci, agent=Claude(fix=autofix))

# The PEP 723 standalone flow (see the docstring): running this file directly
# (`uv run tasks.py <task>`) dispatches through camas. Inert when the `camas`
# command discovers this file itself.
if __name__ == "__main__":
	run_cli(globals())
