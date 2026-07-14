# /// script
# requires-python = ">=3.10"
# dependencies = ["camas"]
# ///
"""Project tasks — run with ``camas``. Kitchen-sink edition: every Task / Sequential /
Parallel / Config option, worked and explained, in one file — scaffolded by
``camas --init --verbose`` (the MCP ``camas_init`` defaults to this template; the plain
``camas --init`` writes the short one instead).

Read this top to bottom once, then delete everything you don't need — it is a reference,
not a project skeleton. Every command below is a cross-platform placeholder
(``python -c ...``); replace them with your real ones, same as the short starter:

	lint = Task("ruff check .")
	test = Task("cargo test", cwd=Path("rust"))
	build = Task("npm run build")

The ``# /// script`` block above is optional PEP 723 inline metadata; see the short
starter's docstring (``camas --init``, no ``--verbose``) for what it buys you. Both it and
the ``__main__`` block at the bottom are inert under plain ``camas``; delete them if you
don't want the standalone path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from camas import (
	AgentFormat,
	Claude,
	Config,
	Effect,
	Parallel,
	Sequential,
	Task,
	by_suffix,
	run_cli,
)
from camas.v0.task_event import CompletedEvent

if TYPE_CHECKING:
	from collections.abc import Sequence

	from camas.v0.leaf_state import LeafState
	from camas.v0.task import TaskNode
	from camas.v0.task_event import TaskEvent

# A leaf is any shell command, shlex-split (so quote the -c payload); the tuple form skips
# shlex entirely, which is why it's the safer choice once a command's own quoting gets hairy.
hello = Task("python -c \"print('hello from the kitchen-sink starter')\"")

# name= on a *top-level* binding does not change how you dispatch it — that's always the
# Python variable (`camas compile_step` below, never `camas compile`) — it only relabels the
# task in --list, --tree, and while it runs. Use it to give a verbose binding a short display
# name, or (further down) to name a *nested*, anonymous group, which has no binding of its own.
compile_step = Task(("python", "-c", "print('quoting-safe via the tuple form')"), name="compile")

# help= overrides the one-line summary `camas --list` and `camas <task> --help` print for a
# task, in place of the raw command — rarely needed (the command usually speaks for itself),
# but handy for a cryptic one.
documented = Task(
	"python -c \"print('see --list / <task> --help for this summary instead of the command')\"",
	help="One-line summary shown by --list and <task> --help",
)

# when= scopes a leaf whose command can't take {paths} (a compiler, `nix flake check`, ...):
# a directory-prefix string, a Path (coerced to its POSIX prefix), or a tuple of prefixes,
# OR'd together. It only ever prunes on a *scoped* run (camas <task> --paths ...) — a full run
# (plain `camas`, as below) never consults it, so both of these always execute here.
native = Task("python -c \"print('native build')\"", when=("src", "include"))
flake = Task("python -c \"print('flake check')\"", when="flake.nix")


def touches_docs(changed: tuple[str, ...]) -> bool:
	"""A ``when=`` callable — the general form behind the prefix-string/tuple shorthand above:
	receives the changed set (never called for a full run) and decides whether this leaf
	applies.
	"""
	return any(c.startswith("docs/") for c in changed)


docs_build = Task("python -c \"print('build docs')\"", name="docs", when=touches_docs)


def web_paths(changed: tuple[str, ...]) -> tuple[str, ...]:
	"""A hand-written ``paths=`` callable — the shape ``by_suffix`` builds for you below: map
	the changed set to the args {paths} injects, and return a default target — never () —
	when ``changed`` is empty (a full run), so a formatter reading stdin on no args never
	hangs.
	"""
	return tuple(c for c in changed if c.endswith((".ts", ".tsx"))) if changed else ("web",)


# mutates=True marks a leaf that writes the workspace (a formatter, an auto-fixer). It has two
# jobs: `camas --under=<duration>` runs every mutates=True leaf first, in sequence, then the
# read-only rest in parallel, so a formatter never races a checker over the same files; and
# it's the natural node to register as Config(agent=Claude(fix=...)) below. paths="." plus the
# {paths} token in the command is what makes it narrowable: `camas fmt --paths a.py` rewrites
# the command to touch only a.py; with no --paths (a full run) {paths} becomes ".".
fmt = Task('python -c "" {paths}', mutates=True, paths=".")

# by_suffix(suffixes, default=...) builds a paths= callable: it filters the changed files by
# extension on a scoped run, and returns default on a full run (never empty, for the reason
# above). A command with no {paths} token can't be narrowed at all — its paths= is then a
# no-op and it always runs, which is why paths= always pairs with a {paths} placeholder.
lint = Task('python -c "" {paths}', paths=by_suffix((".py",), default=(".",)))

# The hand-written callable from above, wired to a leaf the same way.
web_lint = Task('python -c "" {paths}', paths=web_paths)

# agent_format=AgentFormat(args, kind) is agent-only: the gate appends args to the command and
# tags the resulting diagnostics as kind, but a human run leaves the command untouched. kind is
# one of the standards camas passes through verbatim, never parsing: sarif, rdjson, lsp, junit,
# tap, or raw (the default when agent_format is unset). The tuple shorthand
# agent_format=("--output-format sarif", "sarif") coerces to the same thing.
checked = Task(
	"python -c \"print('checked')\"",
	agent_format=AgentFormat("--output-format sarif", "sarif"),
)

# cwd runs the leaf from that directory (accepts a bare str too, e.g. cwd="rust"); Path() —
# the current directory — is used here only to keep this file's own placeholders working with
# no real subproject around. A leaf with a relative cwd but no when= falls back to gating on
# that directory when scoped (the monorepo default); set when="." to opt back into always-run.
subproject = Task("python -c \"print('subproject build')\"", cwd=Path())

# matrix= clones the subtree per axis value, interpolating {NAME} into cmd/env/cwd; env= is
# scoped to the subtree. Pin an axis from the CLI: camas greet --NAME Grace
greet = Parallel(
	Task("python -c \"import os; print(os.environ['GREETING'] + ', {NAME}!')\""),
	matrix={"NAME": ("Ada", "Grace")},
	env={"GREETING": "hello"},
	help="say hello to everyone at once",
)

# A second axis multiplies the cross-product; override either (or both) from the CLI with a
# comma-separated list: camas meet --NAME1 Clara,Dolores
meet = Parallel(
	Task("python -c \"print('hi {NAME1}, meet {NAME2}')\""),
	matrix={"NAME1": ("Jane", "Ada"), "NAME2": ("Wendy", "Alyssa")},
	help="cross a two-axis matrix",
)

# cwd=/env=/paths=/when= set on a Sequential or Parallel — not just a Task — are the default
# for any descendant leaf that sets none of its own, the same propagation as matrix= above.
# Sequential runs its children in order, short-circuiting on the first failure; Parallel runs
# them concurrently (wall-clock max, not sum) — reach for Sequential only for real ordering (a
# mutating step, or one that consumes a prior's output), never to fake an independent-work "&&"
# chain.
frontend = Sequential(
	Task('python -c "" {paths}', mutates=True),
	Task('python -c "" {paths}'),
	cwd=Path(),
	env={"NODE_ENV": "production"},
	paths=".",
	help="cwd=/env=/paths= here are inherited by both children, which set none of their own",
)


def python_versions_from(version_file: Path) -> tuple[str, ...]:
	"""An authoring idiom the engine can't enforce, straight from camas's own ``tasks.py``:
	source a matrix axis from your project's single source of truth instead of hardcoding a
	list that drifts out from under it. This one reads a ``.python-version`` file — the same
	recipe reads a ``rust-toolchain.toml``, an ``.nvmrc``, whatever your project's SSOT is;
	``tasks.py`` is real Python, so read it. The missing-file fallback exists only so this
	template keeps loading in a brand-new project; a real project usually deletes it and lets
	a missing SSOT raise.
	"""
	if not version_file.exists():
		return ("3.13",)
	return tuple(
		stripped
		for line in version_file.read_text(encoding="utf-8").splitlines()
		if (stripped := line.strip()) and not stripped.startswith("#")
	)


PY_VERSIONS: Final = python_versions_from(Path(__file__).parent / ".python-version")

versions = Parallel(
	Task("python -c \"print('checked on {PY}')\""),
	matrix={"PY": PY_VERSIONS},
	help="axis sourced from this project's own .python-version — see python_versions_from",
)


class Announce(Effect[None]):
	"""A custom output renderer, discovered automatically from this file: ``camas --effects``
	lists it beside the built-ins and ``camas --effects='(Announce(),)'`` activates it. These
	three methods are the whole Effect protocol; the generic parameter is a per-leaf context
	this stateless example doesn't need (``None``). Import the event/state types from the
	versioned namespace (``camas.v0.…``, at the top) to pin the API generation your plugin is
	written against.
	"""

	async def setup(self, task: TaskNode) -> None: ...

	async def on_event(self, event: TaskEvent, states: Sequence[LeafState], ctx: None) -> None:
		if isinstance(event, CompletedEvent):
			print(f"announce: {event.task.name or event.task.cmd} is done")

	async def teardown(self, ctxs: tuple[None, ...]) -> None: ...


# Monorepos: Project("libs") — imported from camas like everything above, not demonstrated
# live here since it needs a real child directory to resolve — binds another tasks.py by path
# (relative to this file) as a single task node, mounted under the binding's name (camas libs,
# camas libs.search.lint, ...). A reference composes the child's *matching* Config field: the
# same libs in default_task grabs the child's default_task, in agent.fix its fix node, and so
# on. See the README's Monorepos section for the full picture.
#
# The CLI (and [tool.camas.tasks] in pyproject.toml, camas's TOML authoring surface) also
# accepts typed expressions that reference this file's own bindings by bare name — e.g.
# `camas '{lint, checked}'` or `camas 'Parallel(lint, checked)'` — resolved the same way a
# monorepo's dotted names are.
ci = Sequential(
	hello,
	Parallel(compile_step, documented, native, flake, docs_build, "python --version"),
	Parallel(greet, meet, versions),
	frontend,
	# name= on a *nested*, anonymous group (no binding of its own) is the other half of the
	# lesson at the top: it has no dispatch key at all, only this display label.
	Sequential(fmt, lint, web_lint, checked, subproject, name="checks"),
	name="ci",
)

# Config is discovered by type, under any binding name (here `_`): bare `camas` runs
# default_task, or github_task under GitHub Actions (falling back to default_task when unset).
# default_effects/default_github_effects override the engine's chosen renderer per environment
# — e.g. default_effects=(Announce(),) would make the plugin above the default; None (shown
# here) defers to the engine, an explicit () would mean no effects at all. camas_dir relocates
# the .camas/ run-log and timing-cache directory (default shown here); delete that directory
# any time to reset the cache.
#
# agent= takes an Agent — today that union is just Claude, wiring the Claude Code plugin: fix
# is the registered PostToolBatch autofix node (fmt, above); check is what the gate validates
# (None would defer to default_task/github_task, scoped by --paths and time-boxed by --under);
# default is what a no-task `camas_run` runs (None would defer to check, then
# github_task/default_task).
_ = Config(
	default_task=ci,
	github_task=Sequential(ci, "python -c \"print('extra check only under GitHub Actions')\""),
	default_effects=None,
	default_github_effects=None,
	camas_dir=".camas",
	agent=Claude(fix=fmt, check=ci, default=hello),
)

# The PEP 723 standalone flow (see the module docstring): running this file directly
# (`uv run tasks.py <task>`) dispatches through camas. Inert when the `camas` command
# discovers this file itself.
if __name__ == "__main__":
	run_cli(globals())
