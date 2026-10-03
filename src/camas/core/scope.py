# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Path-scoping: narrow a leaf's ``{paths}`` command to the changed files under its scope.

``{paths}`` in a command marks it narrowable; the changed files replace the placeholder. The
scope is ``Task.paths`` — a directory-prefix string (``"."``, ``"frontend"``) or a
``(changed) -> args`` callable — or, when a leaf sets none, the ``paths`` inherited from its
enclosing ``Sequential``/``Parallel``: a group's ``paths`` is the default target for its
descendants, baked into leaves by :func:`camas.core.matrix.expand_matrix` (own wins, else
inherit) the same way ``env``/``cwd`` propagate. On a full run ``{paths}`` becomes the scope's
prefix (or the callable's default); on a scoped run it becomes the changed files the scope
covers, and a ``{paths}`` leaf covering none of them is dropped.

A command with **no** ``{paths}`` can't be narrowed, so it always runs — unless a ``when=``
predicate (own or inherited) excludes the changed set — and its ``paths`` (own or inherited) is
a no-op regardless; camas errs on correctness otherwise: a tool that can't narrow might be
affected by the edit. ``paths`` only ever prunes a ``{paths}`` command; ``when`` can prune either
kind of leaf, on a scoped run, gating before any ``paths`` narrowing — never on a full run. A
leaf with a ``cwd`` but no explicit ``when`` gates on its ``cwd`` directory (baked by
:func:`camas.core.matrix.expand_matrix`); ``when="."`` opts back into always-run.

:func:`with_default_paths` resolves the full-run default and is applied before every run.
:func:`scope_to_changed` resolves and prunes against a changed set — the entry point for
the gate (#67/#69). Detecting the changed set is the caller's job; :func:`to_changed`
normalizes an externally-supplied set (absolute hook paths, CLI ``--paths``) to the
repo-relative POSIX form the matcher expects.

:func:`scope_warnings` walks a task tree for authoring mistakes — ``paths=`` set on a leaf
whose command can't use it, or a ``paths`` callable that goes empty on a full run — surfaced
by ``camas --check``.
"""

from __future__ import annotations

import shlex
import sys
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, cast

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..v0.task import Group, Pipe, Task, rebuilt
from .task import task_label
from .traversal import flatten_leaves

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Sequence

	from ..v0.task import PathScope, TaskNode, WhenPredicate


PATHS_TOKEN: Final = "{paths}"


def to_changed(raw: Iterable[str], base: Path) -> tuple[str, ...]:
	"""Normalize externally-supplied changed paths to the repo-relative POSIX form
	:func:`scope_to_changed` matches: split comma-separated entries, drop blanks, resolve each
	against ``base`` (a hook's stdin paths are absolute), and drop any that fall outside it. The
	single boundary every changed set passes through — the CLI ``--paths``, the ``camas_gate``
	request, and a hook's files all normalize here, so identical input scopes identically.

	>>> import tempfile, os
	>>> d = Path(tempfile.mkdtemp()); _ = (d / "src").mkdir()
	>>> _ = (d / "src" / "a.py").write_text(""); _ = (d / "b.py").write_text("")
	>>> to_changed([str(d / "src" / "a.py"), "b.py", "/elsewhere/x.py"], d)
	('src/a.py', 'b.py')
	>>> to_changed(["src/a.py,b.py"], d)
	('src/a.py', 'b.py')
	>>> to_changed(["", "  ", "b.py"], d)
	('b.py',)
	"""
	root = base.resolve()
	return tuple(
		rp.relative_to(root).as_posix()
		for r in raw
		for e in r.split(",")
		if (entry := e.strip())
		if (rp := (root / entry).resolve()).is_relative_to(root)
	)


def requested_but_unusable(paths: Sequence[str], changed: Sequence[str]) -> bool:
	"""Whether paths were named and none of them survived normalization — every one outside the repo.

	Distinct from naming none, which means run everything, and the distinction is only visible to a
	caller holding both: :func:`to_changed` returns the same empty tuple either way. Told apart wrong,
	an edit outside the repo runs the whole tree — the opposite of what scoping was asked for.

	>>> requested_but_unusable(["/etc/passwd"], ()), requested_but_unusable([], ())
	(True, False)
	>>> requested_but_unusable(["a.py"], ("a.py",))
	False
	"""
	return bool(paths) and not changed


def coverage_message(paths: Iterable[str]) -> str:
	"""Why a path-scoped run is empty.

	>>> coverage_message(("docs/x.md", "README.md"))
	'No task leaf covers docs/x.md, README.md — nothing to run.'
	>>> coverage_message(("",))
	'No task leaf covers (no paths given) — nothing to run.'
	"""
	return f"No task leaf covers {', '.join(paths) or '(no paths given)'} — nothing to run."


def _within(path: str, prefix: str) -> bool:
	"""True when POSIX ``path`` lies under ``prefix`` (segment-wise, so ``frontend`` does
	not cover ``frontendx``); ``"."`` covers everything.

	>>> _within("frontend/app.ts", "frontend"), _within("frontendx/app.ts", "frontend")
	(True, False)
	>>> _within("anywhere/at/all", ".")
	True
	"""
	base = PurePosixPath(prefix).parts
	return PurePosixPath(path).parts[: len(base)] == base


def _when_matches(when: str | tuple[str, ...] | WhenPredicate, changed: tuple[str, ...]) -> bool:
	"""True when ``when`` matches the ``changed`` set: a prefix string or tuple of prefixes
	(OR'd) matches segment-wise via :func:`_within`; a callable is asked directly.

	>>> _when_matches("src", ("src/a.py",)), _when_matches("src", ("docs/x.md",))
	(True, False)
	>>> _when_matches(("src", "include"), ("include/h.h",))
	True
	>>> _when_matches(lambda c: "x" in c, ("x",))
	True
	"""
	match when:
		case str():
			return any(_within(c, when) for c in changed)
		case tuple():
			return any(  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
				_within(c, prefix)  # ty: ignore[invalid-argument-type]
				for c in changed
				for prefix in when
			)
		case _:
			return when(changed)


def _as_scope(paths: str | PathScope) -> PathScope:
	"""Normalize the ``paths`` field to a scope function. A prefix string covers the changed
	files under it, falling back to the prefix itself for a full run.

	>>> _as_scope(".")(())
	('.',)
	>>> _as_scope("src")(("src/app.py", "docs/readme.md"))
	('src/app.py',)
	>>> _as_scope(lambda c: tuple(p for p in c if p.endswith(".py")))(("a.py", "b.rs"))
	('a.py',)
	"""
	if not isinstance(paths, str):
		return paths
	prefix = paths
	return lambda changed: (
		(prefix,) if not changed else tuple(c for c in changed if _within(c, prefix))
	)


def _inject(cmd: str | tuple[str, ...], parts: tuple[str, ...]) -> str | tuple[str, ...]:
	"""Replace the ``{paths}`` placeholder in ``cmd`` with ``parts``: shell-joined into a
	string command, spliced as tokens into a tuple command.

	>>> _inject("ruff format {paths}", ("a.py", "b.py"))
	'ruff format a.py b.py'
	>>> _inject(("ruff", "format", "{paths}"), ("a.py", "b.py"))
	('ruff', 'format', 'a.py', 'b.py')
	>>> _inject("ruff format {paths}", ())
	'ruff format'
	"""
	match cmd:
		case str():
			if not parts:
				return cmd.replace(" " + PATHS_TOKEN, "").replace(PATHS_TOKEN, "")
			return cmd.replace(PATHS_TOKEN, shlex.join(parts))
		case tuple():
			return tuple(p for tok in cmd for p in (parts if tok == PATHS_TOKEN else (tok,)))
		case _:
			assert_never(cmd)


def _rebase_to_cwd(parts: tuple[str, ...], cwd: Path | None) -> tuple[str, ...]:
	"""Rebase repo-relative injected paths into a leaf's ``cwd`` frame, so a tool that runs from
	a subdir (``cargo`` in ``src-tauri``) gets paths relative to where it runs. A part outside
	``cwd`` is left as-is — a prefix/cwd mismatch is the author's to resolve.

	>>> from pathlib import Path
	>>> _rebase_to_cwd(("src-tauri/src/main.rs", "outside/x"), Path("src-tauri"))
	('src/main.rs', 'outside/x')
	>>> _rebase_to_cwd(("a.py",), None)
	('a.py',)
	"""
	if cwd is None:
		return parts
	root = PurePosixPath(cwd.as_posix())
	return tuple(
		PurePosixPath(p).relative_to(root).as_posix()
		if PurePosixPath(p).is_relative_to(root)
		else p
		for p in parts
	)


def _resolve_leaf(task: Task, changed: tuple[str, ...]) -> Task | None:
	posix = tuple(c.replace("\\", "/") for c in changed)
	if posix and task.when is not None and not _when_matches(task.when, posix):
		return None
	if PATHS_TOKEN not in task.cmd:
		return task
	parts = _as_scope(task.paths if task.paths is not None else ".")(posix)
	if changed and not parts:
		return None
	return Task(
		cmd=_inject(task.cmd, _rebase_to_cwd(parts, task.cwd)),
		name=task.name,
		env=task.env,
		cwd=task.cwd,
		help=task.help,
		mutates=task.mutates,
		paths=task.paths,
		when=task.when,
		agent_format=task.agent_format,
	)


def scope_to_changed(node: TaskNode, changed: tuple[str, ...]) -> TaskNode | None:
	"""``node`` with each ``{paths}`` command narrowed to the changed files under its scope,
	leaves whose scope intersects none of them (or whose ``when`` excludes it) pruned, and
	emptied groups dropped (``None`` when nothing remains).

	A command with no ``{paths}`` can't be narrowed, so it always runs — its ``paths`` (own or
	inherited from a group) is a no-op there, unless a ``when`` predicate (own or inherited)
	excludes the changed set. A ``{paths}`` command is additionally pruned when its scope covers
	none of the changed files.

	>>> from camas.v0.task import Parallel
	>>> py = Task("ruff check {paths}", name="lint", paths=".")
	>>> scope_to_changed(py, ("src/app.py",)).cmd
	'ruff check src/app.py'
	>>> scope_to_changed(Task("cargo check", name="cargo"), ("src/app.py",)).cmd
	'cargo check'
	>>> scope_to_changed(Parallel(py), ("README.md",)) == Parallel(Task("ruff check README.md", name="lint", paths="."))
	True
	>>> scope_to_changed(Parallel(Task("ruff {paths}", paths="src")), ("docs/x.md",)) is None
	True
	>>> scope_to_changed(Task("cargo check", name="cargo", when="src"), ("docs/x.md",)) is None
	True
	>>> scope_to_changed(Task("cargo check", name="cargo", when="src"), ("src/a.rs",))
	Task(cmd='cargo check', name='cargo', env={}, cwd=None, when='src')
	"""
	return scoped_walk(node, changed).node


def _suffix_only_prune(pruned: tuple[int, ...], kept: int, total: int) -> bool:
	"""Whether the pruned positions are exactly the last ``total - kept`` — a suffix-only cut
	needs no rewiring; a mid-pipe cut would rewire the pipeline.
	"""
	return pruned == tuple(range(kept, total))


class Pruned(NamedTuple):
	"""A tree after :func:`prune_pipes`, with what its pipe cuts removed."""

	node: TaskNode | None
	pipe_cut: tuple[Task, ...]
	"""The input stages ``keep_stage`` kept that a mid-pipe cut dropped with their pipe."""


def prune_pipes(
	node: TaskNode,
	keep_stage: Callable[[Task], Task | None],
	keep_whole_pipe: Callable[[tuple[Task, ...]], bool] | None = None,
) -> Pruned:
	"""Each leaf mapped through ``keep_stage`` (``None`` prunes it), groups rebuilt around the
	survivors; a pipe's prune follows the cut semantics — a suffix-only cut keeps the prefix,
	a mid-pipe cut drops the whole pipe — unless ``keep_whole_pipe`` keeps it wholesale.
	"""
	match node:
		case Task():
			return Pruned(keep_stage(node), ())
		case Pipe(tasks=stages):
			stage_tasks: Final = cast("tuple[Task, ...]", stages)
			if keep_whole_pipe is not None and keep_whole_pipe(stage_tasks):
				return Pruned(node, ())
			mapped = tuple(keep_stage(t) for t in stage_tasks)
			kept = tuple(stage for stage in mapped if stage is not None)
			pruned_indices = tuple(i for i, stage in enumerate(mapped) if stage is None)
			if kept and _suffix_only_prune(pruned_indices, len(kept), len(stage_tasks)):
				return Pruned(rebuilt(node, *kept), ())
			return Pruned(
				None,
				tuple(t for t, stage in zip(stage_tasks, mapped, strict=True) if stage is not None),
			)
		case Group() as group:
			pruned_children = tuple(
				prune_pipes(c, keep_stage, keep_whole_pipe) for c in group.tasks
			)
			survivors = tuple(child.node for child in pruned_children if child.node is not None)
			return Pruned(
				rebuilt(group, *survivors) if survivors else None,
				tuple(chain.from_iterable(child.pipe_cut for child in pruned_children)),
			)
		case _:
			assert_never(node)


def with_default_paths(node: TaskNode) -> TaskNode:
	"""``node`` with every ``{paths}`` resolved to its full-run default. Total — the empty
	change set never prunes — so it is safe to apply before any run.

	>>> with_default_paths(Task("ruff format {paths}", paths="."))
	Task(cmd='ruff format .', name=None, env={}, cwd=None, paths='.')
	>>> with_default_paths(Task("mypy ."))
	Task(cmd='mypy .', name=None, env={}, cwd=None)
	"""
	return scope_to_changed(node, ()) or node


class Scoped(NamedTuple):
	"""One scoping walk: the scoped tree, its leaves paired with their originals, and the
	originals a mid-pipe cut dropped.
	"""

	node: TaskNode | None
	pairs: tuple[tuple[Task, Task], ...]
	pipe_cut: tuple[Task, ...]


def scoped_walk(node: TaskNode, changed: tuple[str, ...]) -> Scoped:
	"""``node`` scoped to ``changed``."""
	originals: dict[int, tuple[Task, Task]] = {}

	def resolve_stage(task: Task) -> Task | None:
		scoped = _resolve_leaf(task, changed)
		if scoped is not None:
			# The scoped clone rides in the value so a pipe cut dropping its mapped tuple cannot
			# free it and let a later clone recycle the id this entry keys on.
			originals[id(scoped)] = (task, scoped)
		return scoped

	pruned = prune_pipes(node, resolve_stage)
	pairs = (
		tuple(originals[id(info.task)] for info in flatten_leaves(pruned.node))
		if pruned.node is not None
		else ()
	)
	return Scoped(pruned.node, pairs, pruned.pipe_cut)


def scoped_leaves(node: TaskNode, changed: tuple[str, ...]) -> tuple[tuple[Task, Task], ...]:
	"""Each leaf of ``node`` that survives scoping to ``changed``, paired with its scoped form.

	The pairing is what lets a caller relate what a leaf *reports* when it runs to what it *is*:
	scoping rewrites a ``{paths}`` command, so a leaf with no ``name`` reports a different label for
	every change set. The pairs follow the same pipe cut semantics as :func:`prune_pipes`, so a
	run's tree and its identities are always parallel.

	>>> [(a.cmd, b.cmd) for a, b in scoped_leaves(Task("pylint {paths}", paths="."), ("a.py",))]
	[('pylint {paths}', 'pylint a.py')]
	>>> scoped_leaves(Task("pylint {paths}", paths="src"), ("docs/x.md",))
	()
	"""
	return scoped_walk(node, changed).pairs


def resolve_default_leaf(task: Task) -> Task:
	"""``task`` with its ``{paths}`` resolved to the full-run default — the form a normal run
	records its timing under. The timing lookup keys on this so a ``{paths}``-template leaf
	reuses its recorded (unscoped) estimate instead of missing the cache.

	>>> resolve_default_leaf(Task("ruff check {paths}", paths=".")).cmd
	'ruff check .'
	>>> resolve_default_leaf(Task("mypy .")).cmd
	'mypy .'
	"""
	return _resolve_leaf(task, ()) or task


class ScopeWarning(NamedTuple):
	"""A scope-authoring mistake found by :func:`scope_warnings`: a leaf whose ``paths``
	can't do what its shape suggests.
	"""

	kind: Literal["inert_paths", "empty_full_run_callable"]
	task: str
	message: str


def scope_warnings(node: TaskNode) -> tuple[ScopeWarning, ...]:
	"""Walk the raw task tree — **before** :func:`camas.core.matrix.expand_matrix` bakes an
	inherited ``paths`` onto leaves, since afterward a leaf's own ``paths`` is
	indistinguishable from one it inherited — for two authoring mistakes: a leaf's own
	``paths`` its command can never use, and a ``{paths}`` leaf whose callable scope goes
	empty on a full run.

	>>> from camas.v0.task import Parallel, by_suffix
	>>> scope_warnings(Task("cargo build", name="cargo", paths="."))[0].kind
	'inert_paths'
	>>> scope_warnings(Parallel(Task("cargo build", name="cargo"), paths="."))
	()
	>>> scope_warnings(Task("ruff check {paths}", name="lint", paths=lambda c: c))[0].kind
	'empty_full_run_callable'
	>>> scope_warnings(Task("ruff check {paths}", name="lint", paths=by_suffix((".py",))))
	()
	"""
	match node:
		case Task() as task:
			label = task_label(task)
			inert = (
				(
					ScopeWarning(
						"inert_paths",
						label,
						f"task {label!r} sets paths= but its command has no {PATHS_TOKEN} "
						f"token, so it is never narrowed or pruned; add {PATHS_TOKEN} to the "
						"command, or use when= to gate it on the changed set",
					),
				)
				if task.paths is not None and PATHS_TOKEN not in task.cmd
				else ()
			)
			empty_callable = (
				(
					ScopeWarning(
						"empty_full_run_callable",
						label,
						f"task {label!r}'s paths callable returns () on a full run, so its "
						f"{PATHS_TOKEN} would be stripped entirely — a tool reading stdin on no "
						"args may hang or misbehave; return a default for the empty change set, "
						"e.g. by_suffix(suffixes, default=...)",
					),
				)
				if PATHS_TOKEN in task.cmd
				and task.paths is not None
				and not isinstance(task.paths, str)
				and task.paths(()) == ()
				else ()
			)
			return inert + empty_callable
		case Group() as group:
			return tuple(w for t in group.tasks for w in scope_warnings(t))
		case _:
			assert_never(node)


def scope_warning_messages(nodes: Iterable[TaskNode]) -> tuple[str, ...]:
	"""The :func:`scope_warnings` messages across raw trees, deduplicated preserving order —
	a node shared by two names warns once.

	>>> t = Task("cargo build", name="cargo", paths=".")
	>>> len(scope_warning_messages((t, t)))
	1
	>>> scope_warning_messages((Task("cargo build", name="cargo"),))
	()
	"""
	return tuple(
		w.message for w in dict.fromkeys(w for node in nodes for w in scope_warnings(node))
	)
