# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Monorepo task discovery: compose a ``tasks.py`` with the ``tasks.py`` files found in its
descendant directories, each under a dotted namespace keyed by directory name.

:func:`discover_children` walks a directory depth-first for the nearest descendant directories
containing a ``tasks.py``, skipping hidden and known non-task directories (``node_modules``,
``.venv``, ``venv``, ``__pycache__``) and symlinked directories (cycle safety); a directory that
itself has a ``tasks.py`` is a leaf of the search, so a nested ``tasks.py`` composes only its
own subtree.

:func:`compose_from` recursively composes a loaded ``tasks.py``
(:class:`~camas.main.state.LoadOk`) with each discovered child: the child's tasks are rebased
(:func:`rebase_tree`) into the composing directory's frame and merged under
``f"{segment}.{name}"`` (``segment`` the child directory's own name); the child's
``Config.default_task``, if any, is merged under the bare ``segment`` too. Composition is
opt-out on both ends: a ``Config.discover=False`` parent composes none of its descendants, and
a ``Config.discoverable=False`` child (and everything under it) is never composed into an
ancestor.

Rebasing (:func:`rebase_cwd`, :func:`rebase_paths`, :func:`rebase_when`, :func:`rebase_tree`)
re-anchors a child's ``cwd``/``paths``/``when`` into the composing directory's frame, the same
way :mod:`camas.core.scope` treats a leaf's own scope versus one inherited from an enclosing
group — a leaf that sets none of these keeps deferring, now one level further out.

:func:`composed_view` and :func:`load_py_state` are the ``tasks.py``-file entry points;
:func:`state_from_scope` is the sibling for a ``run_cli(globals())`` (PEP 723) scope.
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..v0.task import Group, Task
from .state import LoadErr, LoadOk
from .tasks import (
	RESERVED_TASK_NAMES,
	load_own,
	name_scope_bindings,
	name_scope_config,
	name_scope_effects,
	reject_reserved_names,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..v0.task import PathScope, TaskNode, WhenPredicate
	from .state import TasksState


IGNORE_DIRS: Final = frozenset({"node_modules", ".venv", "venv", "__pycache__"})
"""Directory basenames :func:`discover_children` never descends into, beyond hidden ones."""


def is_pruned_dir(name: str) -> bool:
	"""True when a directory basename is hidden or a known non-task directory
	(:data:`IGNORE_DIRS`) that :func:`discover_children` never descends into.

	>>> is_pruned_dir(".git"), is_pruned_dir("node_modules"), is_pruned_dir("src")
	(True, True, False)
	"""
	return name.startswith(".") or name in IGNORE_DIRS


def discover_children(root_dir: Path) -> tuple[Path, ...]:
	"""The nearest descendant directories under ``root_dir`` containing a ``tasks.py``, found by
	depth-first, name-sorted traversal. Hidden directories, :data:`IGNORE_DIRS`, and symlinked
	directories are pruned. A directory that itself has a ``tasks.py`` is a leaf of the search —
	its own descendants are never visited, so a nested ``tasks.py`` composes only its own subtree.

	>>> import tempfile
	>>> root = Path(tempfile.mkdtemp())
	>>> _ = (root / "child").mkdir()
	>>> _ = (root / "child" / "tasks.py").write_text("")
	>>> _ = (root / "child" / "deeper").mkdir()
	>>> _ = (root / "child" / "deeper" / "tasks.py").write_text("")
	>>> _ = (root / "gap" / "nested").mkdir(parents=True)
	>>> _ = (root / "gap" / "nested" / "tasks.py").write_text("")
	>>> _ = (root / ".venv").mkdir()
	>>> _ = (root / ".venv" / "tasks.py").write_text("")
	>>> sorted(p.relative_to(root).as_posix() for p in discover_children(root))
	['child', 'gap/nested']
	"""
	entries = sorted((p for p in root_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
	return tuple(
		found
		for entry in entries
		if not entry.is_symlink() and not is_pruned_dir(entry.name)
		for found in ((entry,) if (entry / "tasks.py").is_file() else discover_children(entry))
	)


def rebase_cwd(cwd: Path | None, rel: PurePosixPath, *, is_root: bool) -> Path | None:
	"""``cwd`` re-anchored into the composing directory's frame at ``rel``: an unset ``cwd``
	anchors to ``rel`` only at the tree's own root (elsewhere it stays unset, so
	:func:`camas.core.matrix.expand_matrix` inherits the rebased ancestor's ``cwd`` instead); an
	absolute ``cwd`` is left as-is; a relative ``cwd`` is nested under ``rel``.

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
	"""A prefix string re-anchored under ``rel``; ``PurePosixPath`` drops a bare ``"."``, so the
	cwd-relative default becomes exactly ``rel``.

	>>> rebase_str_prefix(".", PurePosixPath("services/api"))
	'services/api'
	>>> rebase_str_prefix("src", PurePosixPath("services/api"))
	'services/api/src'
	"""
	return (rel / value).as_posix()


def _strip_rel(path: str, rel: PurePosixPath) -> str | None:
	"""``path`` with the ``rel`` prefix removed, or ``None`` when it doesn't lie under ``rel``
	(segment-wise, matching :func:`camas.core.scope._within`).

	>>> _strip_rel("services/api/src/a.py", PurePosixPath("services/api"))
	'src/a.py'
	>>> _strip_rel("services/other/x", PurePosixPath("services/api")) is None
	True
	"""
	parts = PurePosixPath(path).parts
	prefix = rel.parts
	if parts[: len(prefix)] != prefix:
		return None
	return PurePosixPath(*parts[len(prefix) :]).as_posix()


def rebase_paths(paths: str | PathScope | None, rel: PurePosixPath) -> str | PathScope | None:
	"""``paths`` re-anchored under ``rel``: ``None`` stays ``None`` (a leaf's own ``"."``
	default is resolved cwd-relative, and ``cwd`` is already anchored by :func:`rebase_cwd`); a
	prefix string is rebased directly; a callable scope is wrapped by :func:`wrap_pathscope`.

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
	"""``inner`` wrapped to operate in the composing directory's frame: on a full run
	(``changed == ()``) ``inner``'s default targets are re-prefixed with ``rel``; on a scoped
	run, ``changed`` is filtered to entries under ``rel`` and stripped of it before reaching
	``inner``, whose returned args are re-prefixed with ``rel``.

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
	"""``when`` re-anchored under ``rel``: ``None`` stays ``None``; a prefix string or tuple of
	prefixes is rebased directly; a callable predicate is wrapped by :func:`wrap_when`.

	>>> rebase_when(None, PurePosixPath("api")) is None
	True
	>>> rebase_when("src", PurePosixPath("api"))
	'api/src'
	>>> rebase_when(("src", "include"), PurePosixPath("api"))
	('api/src', 'api/include')
	"""
	match when:
		case None:
			return None
		case str():
			return rebase_str_prefix(when, rel)
		case tuple():
			return tuple(rebase_str_prefix(w, rel) for w in when)
		case _:
			return wrap_when(when, rel)


def wrap_when(inner: WhenPredicate, rel: PurePosixPath) -> WhenPredicate:
	"""``inner`` wrapped to receive only the changed entries under ``rel``, stripped of it —
	never called for a full run, so there is no default-target case to re-prefix.

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


def rebase_tree(node: TaskNode, rel: PurePosixPath, *, is_root: bool) -> TaskNode:
	"""``node`` rebuilt with its ``cwd``/``paths``/``when`` re-anchored under ``rel``
	(:func:`rebase_cwd`, :func:`rebase_paths`, :func:`rebase_when`); descendants are always
	rebased with ``is_root=False``, so only the tree's own top node inherits an unset ``cwd``
	from ``rel``. ``name``/``env``/``help``/``matrix``/``mutates``/``agent_format`` pass through.

	>>> rebase_tree(
	...     Task("ruff {paths}", paths="."), PurePosixPath("services/api"), is_root=True
	... ) == Task("ruff {paths}", cwd=Path("services/api"), paths="services/api")
	True
	>>> rebase_tree(
	...     Task("cargo build", cwd="rust"), PurePosixPath("services/api"), is_root=False
	... ) == Task("cargo build", cwd=Path("services/api/rust"))
	True
	>>> from camas.v0.task import Sequential
	>>> rebase_tree(
	...     Sequential(Task("cargo build"), name="ci"), PurePosixPath("services/api"), is_root=True
	... ) == Sequential(Task("cargo build"), cwd=Path("services/api"), name="ci")
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
				name=name,
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
				*(rebase_tree(child, rel, is_root=False) for child in group.tasks),
				name=group.name,
				matrix=group.matrix,
				env=group.env,
				cwd=rebase_cwd(group.cwd, rel, is_root=is_root),
				help=group.help,
				paths=rebase_paths(group.paths, rel),
				when=rebase_when(group.when, rel),
			)
		case _:
			assert_never(node)


def check_segment(segment: str, source: Path) -> None:
	"""Reject a child directory name that would make a bad namespace segment: one reserved for
	a camas subcommand (:data:`camas.main.tasks.RESERVED_TASK_NAMES`), or one containing ``.``,
	the delimiter joining a namespace to a task name.

	Raises:
		ValueError: when ``segment`` is reserved or contains a dot.

	>>> check_segment("api", Path("tasks.py")) is None
	True
	>>> check_segment("mcp", Path("tasks.py"))
	Traceback (most recent call last):
	    ...
	ValueError: tasks.py: directory name 'mcp' is reserved for a camas subcommand; rename it
	>>> check_segment("a.b", Path("tasks.py"))
	Traceback (most recent call last):
	    ...
	ValueError: tasks.py: directory name 'a.b' contains '.', the namespace delimiter; rename it
	"""
	if segment in RESERVED_TASK_NAMES:
		raise ValueError(
			f"{source}: directory name {segment!r} is reserved for a camas subcommand; rename it"
		)
	if "." in segment:
		raise ValueError(
			f"{source}: directory name {segment!r} contains '.', the namespace delimiter; rename it"
		)


class ComposeChildError(Exception):
	"""Any exception raised while loading or composing a child ``tasks.py``, carrying its
	``source`` so the caller can attribute the failure to the child instead of the composing
	parent's own file.
	"""

	def __init__(self, source: Path, cause: Exception) -> None:
		super().__init__(f"{source}: {cause}")
		self.source = source
		self.cause = cause


def compose_from(root_dir: Path, own: LoadOk) -> LoadOk:
	"""``own`` (already loaded from ``root_dir / "tasks.py"``) composed with every discovered
	child (:func:`discover_children`) it is configured to discover: each child's tasks are
	rebased (:func:`rebase_tree`) and merged under ``f"{segment}.{name}"`` — ``segment`` the
	child directory's name — and its ``Config.default_task``, if any, under the bare
	``segment``; a child recurses through this same composition before its own tasks are
	merged, so a grandchild's tasks arrive already namespaced under the child's own segment. A
	child whose ``Config.discoverable`` is ``False`` (and everything below it) is skipped
	entirely. Only ``default_task`` is taken from a child's ``Config`` — its ``scope_effects``
	are not composed.

	Raises:
		ComposeChildError: when loading a child ``tasks.py`` raises.
		ValueError: when two files define the same merged task name, a child directory's name
			is invalid (:func:`check_segment`), or the merged namespace has a reserved name.
	"""
	own_source: Final = root_dir / "tasks.py"
	merged: dict[str, TaskNode] = dict(own.tasks)
	origin: dict[str, Path] = dict.fromkeys(own.tasks, own_source)
	if own.config is None or own.config.discover:
		for child_dir in discover_children(root_dir):
			child_tasks_py = child_dir / "tasks.py"
			try:
				child_own = load_own(child_tasks_py)
			except Exception as e:
				raise ComposeChildError(child_tasks_py, e) from e
			if child_own.config is not None and not child_own.config.discoverable:
				continue
			child_view = compose_from(child_dir, child_own)
			rel = PurePosixPath(child_dir.relative_to(root_dir).as_posix())
			segment = child_dir.name
			check_segment(segment, child_tasks_py)
			for key, node in child_view.tasks.items():
				merged_key = f"{segment}.{key}"
				if merged_key in origin:
					raise ValueError(
						f"task {merged_key!r} defined in both {origin[merged_key]} and "
						f"{child_tasks_py}"
					)
				merged[merged_key] = rebase_tree(node, rel, is_root=True)
				origin[merged_key] = child_tasks_py
			default_task = child_view.config.default_task if child_view.config is not None else None
			if default_task is not None:
				if segment in origin:
					raise ValueError(
						f"task {segment!r} defined in both {origin[segment]} and {child_tasks_py}"
					)
				merged[segment] = rebase_tree(default_task, rel, is_root=True)
				origin[segment] = child_tasks_py
	reject_reserved_names(merged)
	return own._replace(tasks=merged)


def composed_view(tasks_py: Path) -> LoadOk:
	"""``tasks_py`` loaded and composed with its discovered descendants (:func:`compose_from`)."""
	return compose_from(tasks_py.parent, load_own(tasks_py))


def load_py_state(path: Path) -> TasksState:
	"""``path`` loaded and composed (:func:`composed_view`) as a
	:class:`~camas.main.state.TasksState`: any exception during composition becomes a
	:class:`~camas.main.state.LoadErr` attributed to the file that raised — the composing
	parent's own file, or, via :class:`ComposeChildError`, the child that failed to load.
	"""
	try:
		return composed_view(path)
	except ComposeChildError as e:
		return LoadErr(source=e.source, exception=e.cause)
	except Exception as e:
		return LoadErr(source=path, exception=e)


def state_from_scope(scope: Mapping[str, object]) -> TasksState:
	"""A ``run_cli(globals())`` (PEP 723) scope loaded as a
	:class:`~camas.main.state.TasksState`: its own bindings, composed with its discovered
	descendants (:func:`compose_from`) when the scope's ``__file__`` locates it on disk.
	Without a locatable ``__file__`` there is no directory to discover from, so only the
	scope's own reserved-name check applies.
	"""
	source_obj = scope.get("__file__")
	source = Path(source_obj) if isinstance(source_obj, (str, Path)) else None
	own = LoadOk(
		tasks=name_scope_bindings(scope),
		source=source,
		scope_effects=name_scope_effects(scope),
		config=name_scope_config(scope),
	)
	if source is None:
		try:
			reject_reserved_names(own.tasks)
		except ValueError as e:
			return LoadErr(source=Path("tasks.py"), exception=e)
		return own
	try:
		return compose_from(source.parent, own)
	except ComposeChildError as e:
		return LoadErr(source=e.source, exception=e.cause)
	except Exception as e:
		return LoadErr(source=source, exception=e)
