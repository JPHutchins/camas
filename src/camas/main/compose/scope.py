# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Compose a loaded scope's ``Project`` references into its task tree."""

from __future__ import annotations

import os
import runpy
import sys
from enum import Enum, auto
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ...v0.config import Claude, Config
from ...v0.task import Group, Parallel, ProjectRef, Sequential, Task
from ..effects import running_under_agent
from ..state import LoadErr, LoadOk
from ..tasks import (
	name_scope_bindings,
	name_scope_config,
	name_scope_effects,
	reject_reserved_names,
)
from .errors import ProjectLoadError
from .rebase import rebase_tree

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ...v0.task import TaskNode
	from ..state import TasksState


class Field(Enum):
	"""The parent-``Config`` slot a composed ``Project`` reference is assigned to. A reference
	grabs the child's *same* field (:func:`child_node`), so ``fix=Parallel(libs, api)`` composes
	each child's ``fix`` — not whatever a bare ``camas`` happens to run there. ``CONTEXT`` is the
	odd one out: a binding name (``libs = Project(...)``) runs what a bare ``camas`` runs in that
	directory for the current invocation's context.
	"""

	DEFAULT = auto()
	GITHUB = auto()
	FIX = auto()
	CHECK = auto()
	RUN_DEFAULT = auto()
	CONTEXT = auto()


def child_node(field: Field, config: Config, *, github: bool, agent: bool) -> TaskNode | None:
	"""The node the child's ``config`` contributes to a parent ``field``: its matching accessor,
	or, for a binding (``CONTEXT``), the node a bare ``camas`` runs there in this context.

	>>> both = Config(default_task=Task("d"), github_task=Task("g"))
	>>> child_node(Field.DEFAULT, both, github=True, agent=True)
	Task(cmd='d', name=None, env={}, cwd=None)
	>>> child_node(Field.GITHUB, both, github=False, agent=False)
	Task(cmd='g', name=None, env={}, cwd=None)
	>>> child_node(Field.CONTEXT, both, github=True, agent=False)
	Task(cmd='g', name=None, env={}, cwd=None)
	>>> child_node(Field.RUN_DEFAULT, both, github=False, agent=False)
	Task(cmd='g', name=None, env={}, cwd=None)
	>>> child_node(Field.FIX, both, github=False, agent=False) is None
	True
	"""
	match field:
		case Field.DEFAULT:
			return config.bare_task(github=False)
		case Field.GITHUB:
			return config.bare_task(github=True)
		case Field.FIX:
			return config.gate_fix()
		case Field.CHECK:
			return config.gate_check(github=False)
		case Field.RUN_DEFAULT:
			return config.run_default()
		case Field.CONTEXT:
			return config.run_default() if agent else config.bare_task(github=github)
		case _:
			assert_never(field)


def field_role(field: Field) -> str:
	"""The role named in the load error a child raises when its ``field`` accessor yields nothing.

	>>> field_role(Field.FIX)
	'fix'
	>>> field_role(Field.CONTEXT)
	'default'
	>>> [field_role(f) for f in (Field.GITHUB, Field.CHECK, Field.RUN_DEFAULT)]
	['github', 'check', 'agent default']
	"""
	match field:
		case Field.DEFAULT | Field.CONTEXT:
			return "default"
		case Field.GITHUB:
			return "github"
		case Field.FIX:
			return "fix"
		case Field.CHECK:
			return "check"
		case Field.RUN_DEFAULT:
			return "agent default"
		case _:
			assert_never(field)


def _compose_scope(
	scope: Mapping[str, object],
	source: Path | None,
	base_dir: Path | None,
	*,
	seen: frozenset[Path],
) -> LoadOk:
	"""``scope``'s bindings with every ``Project`` resolved and merged: a ``Config`` field grabs
	each child's same field (its :class:`Field`), a binding name runs the child's context default,
	and a child's tasks mount under that name dotted.
	"""
	github: Final = os.environ.get("GITHUB_ACTIONS") == "true"
	agent: Final = running_under_agent()
	child_cache: dict[Path, tuple[PurePosixPath, LoadOk]] = {}
	node_cache: dict[tuple[int, Field], TaskNode] = {}

	def child_view(marker: ProjectRef) -> tuple[PurePosixPath, LoadOk]:
		if base_dir is None:
			raise ValueError(f"Project({marker.path!r}) requires a file-backed tasks.py")
		root = base_dir.resolve()
		target = (root / marker.path).resolve()
		tasks_py = target / "tasks.py" if target.is_dir() else target
		if not tasks_py.is_file():
			raise ValueError(f"Project({marker.path!r}): no tasks.py at {tasks_py}")
		if not tasks_py.parent.is_relative_to(root):
			raise ValueError(f"Project({marker.path!r}) escapes the project root {root}")
		if tasks_py in seen:
			raise ValueError(f"circular Project reference at {tasks_py}")
		if tasks_py not in child_cache:
			rel = PurePosixPath(tasks_py.parent.relative_to(root).as_posix())
			try:
				child = load_scope(tasks_py, seen=seen | {tasks_py})
			except ProjectLoadError:
				raise
			except Exception as e:
				raise ProjectLoadError(tasks_py, e) from e
			child_cache[tasks_py] = (rel, child)
		return child_cache[tasks_py]

	def project_node(marker: ProjectRef, field: Field) -> TaskNode:
		rel, child = child_view(marker)
		node = (
			child_node(field, child.config, github=github, agent=agent)
			if child.config is not None
			else None
		)
		if node is None:
			raise ProjectLoadError(
				child.source if child.source is not None else Path("tasks.py"),
				RuntimeError(f"defines no {field_role(field)} task to run for this context"),
			)
		return rebase_tree(node, rel, is_root=True)

	def resolve(node: TaskNode | ProjectRef, field: Field) -> TaskNode:
		key = (id(node), field)
		if key not in node_cache:
			node_cache[key] = _resolved(node, field)
		return node_cache[key]

	def _resolved(node: TaskNode | ProjectRef, field: Field) -> TaskNode:
		match node:
			case ProjectRef():
				return project_node(node, field)
			case Task():
				return node
			case Group() as group:
				children = tuple(resolve(child, field) for child in group.tasks)
				if all(new is old for new, old in zip(children, group.tasks, strict=True)):
					return group
				return type(group)(
					*children,
					name=group.name,
					matrix=group.matrix,
					env=group.env,
					cwd=group.cwd,
					help=group.help,
					paths=group.paths,
					when=group.when,
				)
			case _:
				assert_never(node)

	def resolve_field(node: TaskNode | None, field: Field) -> TaskNode | None:
		return None if node is None else resolve(node, field)

	def resolve_config(config: Config) -> Config:
		agent_cfg = config.agent
		return Config(
			default_task=resolve_field(config.default_task, Field.DEFAULT),
			github_task=resolve_field(config.github_task, Field.GITHUB),
			default_effects=config.default_effects,
			default_github_effects=config.default_github_effects,
			camas_dir=config.camas_dir,
			agent=None
			if agent_cfg is None
			else Claude(
				fix=resolve(agent_cfg.fix, Field.FIX),
				check=resolve_field(agent_cfg.check, Field.CHECK),
				default=resolve_field(agent_cfg.default, Field.RUN_DEFAULT),
			),
		)

	resolved_scope: dict[str, object] = {}
	namespaces: dict[str, TaskNode] = {}
	for name, value in scope.items():
		if isinstance(value, Config):
			resolved_scope[name] = resolve_config(value)
		elif name.startswith("_") or not isinstance(
			value, (Task, Sequential, Parallel, ProjectRef)
		):
			resolved_scope[name] = value
		elif isinstance(value, ProjectRef):
			resolved_scope[name] = resolve(value, Field.CONTEXT)
			rel, child = child_view(value)
			for key, mounted in child.tasks.items():
				namespaces[f"{name}.{key}"] = rebase_tree(mounted, rel, is_root=True)
		else:
			resolved_scope[name] = resolve(value, Field.CONTEXT)

	own: Final = LoadOk(
		tasks=name_scope_bindings(resolved_scope),
		source=source,
		scope_effects=name_scope_effects(resolved_scope),
		config=name_scope_config(resolved_scope),
	)
	merged: Final = {**own.tasks, **namespaces}
	reject_reserved_names(merged)
	return own._replace(tasks=merged)


def load_scope(path: Path, *, seen: frozenset[Path] = frozenset()) -> LoadOk:
	"""``path`` executed and its ``Project`` references resolved (:func:`_compose_scope`)."""
	return _compose_scope(
		runpy.run_path(str(path)),
		path,
		path.parent,
		seen=seen or frozenset({path.resolve()}),
	)


def load_py_tasks_state(path: Path) -> TasksState:
	"""``path`` loaded (:func:`load_scope`) as a :class:`~camas.main.state.TasksState`, any failure
	captured as a :class:`~camas.main.state.LoadErr` attributed to the file that raised — this
	file, or, via :class:`~camas.main.compose.errors.ProjectLoadError`, the child ``Project`` that
	failed.
	"""
	try:
		return load_scope(path)
	except ProjectLoadError as e:
		return LoadErr(source=e.source, exception=e.cause)
	except Exception as e:
		return LoadErr(source=path, exception=e)


def state_from_scope(scope: Mapping[str, object]) -> TasksState:
	"""A ``run_cli(globals())`` (PEP 723) scope loaded as a
	:class:`~camas.main.state.TasksState`, its ``Project`` references resolved when ``__file__``
	locates it on disk.
	"""
	source_obj = scope.get("__file__")
	source = Path(source_obj) if isinstance(source_obj, (str, Path)) else None
	try:
		return _compose_scope(
			scope,
			source,
			source.parent if source is not None else None,
			seen=frozenset() if source is None else frozenset({source.resolve()}),
		)
	except ProjectLoadError as e:
		return LoadErr(source=e.source, exception=e.cause)
	except Exception as e:
		return LoadErr(source=source if source is not None else Path("tasks.py"), exception=e)
