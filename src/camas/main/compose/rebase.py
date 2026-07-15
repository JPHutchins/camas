# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Re-anchor a task node's ``cwd``/``paths``/``when`` into a referencing directory's frame."""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ...v0.task import Group, Task

if TYPE_CHECKING:
	from ...v0.task import PathScope, TaskNode, WhenPredicate


def qualify(name: str | None, namespace: str) -> str | None:
	"""A task or group ``name`` prefixed with its Project ``namespace`` segment, dotted — the
	composed leaf's display identity, so ``libs``'s ``build`` reads as ``libs.build`` in every
	effect. An empty ``namespace`` or an unnamed node passes through unchanged.

	>>> qualify("build", "libs")
	'libs.build'
	>>> qualify("search.lint", "libs")
	'libs.search.lint'
	>>> qualify("build", "")
	'build'
	>>> qualify(None, "libs") is None
	True
	"""
	if not namespace or name is None:
		return name
	return f"{namespace}.{name}"


def rebase_cwd(cwd: Path | None, rel: PurePosixPath, *, is_root: bool) -> Path | None:
	"""``cwd`` re-anchored into the referencing directory's frame at ``rel``.

	>>> rebase_cwd(None, PurePosixPath("services/api"), is_root=True) == Path("services/api")
	True
	>>> rebase_cwd(None, PurePosixPath("services/api"), is_root=False) is None
	True
	>>> rebase_cwd(Path("rust"), PurePosixPath("services/api"), is_root=True) == Path(
	...     "services/api/rust"
	... )
	True
	>>> cwd = Path.cwd()
	>>> rebase_cwd(cwd, PurePosixPath("services/api"), is_root=True) == cwd
	True
	"""
	if cwd is None:
		return Path(rel) if is_root else None
	if cwd.is_absolute():
		return cwd
	return Path(rel / cwd.as_posix())


def rebase_str_prefix(value: str, rel: PurePosixPath) -> str:
	"""A prefix string re-anchored under ``rel``.

	>>> rebase_str_prefix(".", PurePosixPath("services/api"))
	'services/api'
	>>> rebase_str_prefix("src", PurePosixPath("services/api"))
	'services/api/src'
	"""
	return (rel / value).as_posix()


def _strip_rel(path: str, rel: PurePosixPath) -> str | None:
	"""``path`` with the ``rel`` prefix removed, or ``None`` when it doesn't lie under ``rel``.

	>>> _strip_rel("services/api/src/a.py", PurePosixPath("services/api"))
	'src/a.py'
	>>> _strip_rel("services/other/x", PurePosixPath("services/api")) is None
	True
	"""
	return (
		PurePosixPath(path).relative_to(rel).as_posix()
		if PurePosixPath(path).is_relative_to(rel)
		else None
	)


def rebase_paths(paths: str | PathScope | None, rel: PurePosixPath) -> str | PathScope | None:
	"""``paths`` re-anchored under ``rel``.

	>>> rebase_paths(None, PurePosixPath("api")) is None
	True
	>>> rebase_paths(".", PurePosixPath("api"))
	'api'
	>>> rebase_paths("src", PurePosixPath("api"))
	'api/src'
	>>> rebase_paths(lambda c: ("x",) if not c else c, PurePosixPath("api"))(())
	('api/x',)
	"""
	match paths:
		case None:
			return None
		case str():
			return rebase_str_prefix(paths, rel)
		case _:
			return wrap_pathscope(paths, rel)


def wrap_pathscope(inner: PathScope, rel: PurePosixPath) -> PathScope:
	"""``inner`` wrapped to operate in the referencing directory's frame.

	>>> scoped = wrap_pathscope(lambda c: c or (".",), PurePosixPath("api"))
	>>> scoped(())
	('api',)
	>>> scoped(("api/src/a.py", "other/b.py"))
	('api/src/a.py',)
	"""

	def scoped(changed: tuple[str, ...]) -> tuple[str, ...]:
		if not changed:
			return tuple(rebase_str_prefix(target, rel) for target in inner(()))
		under = tuple(stripped for c in changed if (stripped := _strip_rel(c, rel)) is not None)
		return tuple(rebase_str_prefix(part, rel) for part in inner(under))

	return scoped


def rebase_when(
	when: str | tuple[str, ...] | WhenPredicate | None, rel: PurePosixPath
) -> str | tuple[str, ...] | WhenPredicate | None:
	"""``when`` re-anchored under ``rel``.

	>>> rebase_when(None, PurePosixPath("api")) is None
	True
	>>> rebase_when("src", PurePosixPath("api"))
	'api/src'
	>>> rebase_when(("src", "include"), PurePosixPath("api"))
	('api/src', 'api/include')
	>>> rebase_when(lambda c: bool(c), PurePosixPath("api"))(("api/x",))
	True
	"""
	match when:
		case None:
			return None
		case str():
			return rebase_str_prefix(when, rel)
		case tuple():
			return tuple(
				rebase_str_prefix(w, rel)  # ty: ignore[invalid-argument-type]
				for w in when
			)
		case _:
			return wrap_when(when, rel)


def wrap_when(inner: WhenPredicate, rel: PurePosixPath) -> WhenPredicate:
	"""``inner`` wrapped to receive only the changed entries under ``rel``, stripped of it.

	>>> wrapped = wrap_when(lambda c: "x" in c, PurePosixPath("api"))
	>>> wrapped(("api/x", "other/y"))
	True
	>>> wrapped(("other/y",))
	False
	"""

	def predicate(changed: tuple[str, ...]) -> bool:
		under = tuple(stripped for c in changed if (stripped := _strip_rel(c, rel)) is not None)
		return inner(under)

	return predicate


def rebase_tree(node: TaskNode, rel: PurePosixPath, namespace: str, *, is_root: bool) -> TaskNode:
	"""``node`` rebuilt with its ``cwd``/``paths``/``when`` re-anchored under ``rel`` and every
	node's ``name`` prefixed with the Project ``namespace`` segment (:func:`qualify`); descendants
	rebase with ``is_root=False`` so only the tree's own top node inherits an unset ``cwd``.

	>>> rebase_tree(
	...     Task("ruff {paths}", paths="."), PurePosixPath("services/api"), "api", is_root=True
	... ) == Task("ruff {paths}", cwd=Path("services/api"), paths="services/api")
	True
	>>> rebase_tree(
	...     Task("cargo build", cwd="rust"), PurePosixPath("services/api"), "api", is_root=False
	... ) == Task("cargo build", cwd=Path("services/api/rust"))
	True
	>>> from camas.v0.task import Sequential
	>>> rebase_tree(
	...     Sequential(Task("cargo build", name="c"), name="ci"),
	...     PurePosixPath("services/api"),
	...     "api",
	...     is_root=True,
	... ) == Sequential(Task("cargo build", name="api.c"), cwd=Path("services/api"), name="api.ci")
	True
	"""
	match node:
		case Task(
			cmd=cmd,
			name=name,
			env=env,
			cwd=cwd,
			help=help,
			mutates=mutates,
			paths=paths,
			when=when,
			agent_format=agent_format,
		):
			return Task(
				cmd=cmd,
				name=qualify(name, namespace),
				env=env,
				cwd=rebase_cwd(cwd, rel, is_root=is_root),
				help=help,
				mutates=mutates,
				paths=rebase_paths(paths, rel),
				when=rebase_when(when, rel),
				agent_format=agent_format,
			)
		case Group() as group:
			return type(group)(
				*(rebase_tree(child, rel, namespace, is_root=False) for child in group.tasks),
				name=qualify(group.name, namespace),
				matrix=group.matrix,
				env=group.env,
				cwd=rebase_cwd(group.cwd, rel, is_root=is_root),
				help=group.help,
				paths=rebase_paths(group.paths, rel),
				when=rebase_when(group.when, rel),
			)
		case _:
			assert_never(node)
