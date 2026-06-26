# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Path-scoping: resolve a leaf's ``{paths}`` placeholder against the changed files.

A leaf opts into scoping by putting ``{paths}`` in its command and setting ``Task.paths``
to either a directory-prefix string (``"."``, ``"frontend"``) or a ``(changed) -> args``
callable. On a full run (no changed set) ``{paths}`` becomes the prefix — or the callable's
default — so the task still runs over everything; on a scoped run it becomes the changed
files the leaf covers, and a leaf that covers none of them is dropped.

:func:`with_default_paths` resolves the full-run default and is applied before every run.
:func:`scope_to_changed` resolves and prunes against a changed set — the entry point for
the gate (#67/#69). Detecting the changed set is the caller's job; paths match POSIX,
relative to the same root.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..v0.task import Parallel, Sequential, Task

if TYPE_CHECKING:
	from ..v0.task import PathScope, TaskNode


PATHS_TOKEN: Final = "{paths}"


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
	"""
	match cmd:
		case str():
			return cmd.replace(PATHS_TOKEN, shlex.join(parts))
		case tuple():
			return tuple(p for tok in cmd for p in (parts if tok == PATHS_TOKEN else (tok,)))
		case _:
			assert_never(cmd)


def _resolve_leaf(task: Task, changed: tuple[str, ...]) -> Task | None:
	match task.paths:
		case None:
			return task
		case scope:
			parts = _as_scope(scope)(tuple(c.replace("\\", "/") for c in changed))
			if changed and not parts:
				return None
			return Task(
				cmd=_inject(task.cmd, parts),
				name=task.name,
				env=task.env,
				cwd=task.cwd,
				help=task.help,
				mutates=task.mutates,
				paths=task.paths,
			)


def scope_to_changed(node: TaskNode, changed: tuple[str, ...]) -> TaskNode | None:
	"""``node`` with each leaf's ``{paths}`` resolved for ``changed``, leaves covering none
	of it pruned, and emptied groups dropped (``None`` when nothing remains).

	A leaf whose command omits ``{paths}`` but sets ``paths`` runs whole when its prefix
	changed and is skipped otherwise — the selection a file-list-averse tool (``mypy``,
	``cargo``) needs; here the Rust check drops on a Python-only change.

	>>> py = Task("ruff check {paths}", name="lint", paths=".")
	>>> rs = Task("cargo check", name="cargo", paths="rust")
	>>> scope_to_changed(Parallel(py, rs), ("src/app.py",))
	Parallel(tasks=(Task(cmd='ruff check src/app.py', name='lint', env={}, cwd=None, paths='.'),), name=None, matrix=None, env={}, cwd=None)
	>>> scope_to_changed(Parallel(py), ("README.md",)) == Parallel(Task("ruff check README.md", name="lint", paths="."))
	True
	>>> scope_to_changed(Parallel(Task("ruff {paths}", paths="src")), ("docs/x.md",)) is None
	True
	"""
	match node:
		case Task():
			return _resolve_leaf(node, changed)
		case Sequential(tasks=children, name=name, matrix=matrix, env=env, cwd=cwd, help=help):
			kept = tuple(
				s for s in (scope_to_changed(c, changed) for c in children) if s is not None
			)
			return (
				Sequential(*kept, name=name, matrix=matrix, env=env, cwd=cwd, help=help)
				if kept
				else None
			)
		case Parallel(tasks=children, name=name, matrix=matrix, env=env, cwd=cwd, help=help):
			kept = tuple(
				s for s in (scope_to_changed(c, changed) for c in children) if s is not None
			)
			return (
				Parallel(*kept, name=name, matrix=matrix, env=env, cwd=cwd, help=help)
				if kept
				else None
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
