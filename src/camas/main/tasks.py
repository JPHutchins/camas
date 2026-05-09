# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import runpy
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, TypeGuard, cast

if sys.version_info >= (3, 11):
	from typing import assert_never

	import tomllib
else:  # pragma: no cover
	import tomli as tomllib
	from typing_extensions import assert_never

from ..core.task import Parallel, Sequential, Task, TaskNode
from .expression import Ref, parse_task_value, resolve_refs


def find_pyproject(start: Path) -> Path | None:
	"""Walk upward from ``start`` looking for a pyproject.toml."""
	for candidate in (start, *start.parents):
		if (pyproject := candidate / "pyproject.toml").is_file():
			return pyproject
	return None


def is_str_dict(value: Any) -> TypeGuard[dict[str, Any]]:
	return isinstance(value, dict) and all(isinstance(k, str) for k in value)  # pyright: ignore[reportUnknownVariableType]


def dig(data: Any, key: str) -> Any:
	return data[key] if is_str_dict(data) and key in data else {}


def assign_key_name(node: TaskNode | Ref, key: str) -> TaskNode | Ref:
	"""Set the TOML key as the node's name unless the expression already named it.

	>>> assign_key_name(Task("x"), "foo")
	Task(cmd='x', name='foo', env={}, cwd=None)
	>>> assign_key_name(Task("x", name="explicit"), "foo")
	Task(cmd='x', name='explicit', env={}, cwd=None)
	>>> assign_key_name(Ref("bar"), "foo")
	Ref(name='bar')
	"""
	match node:
		case Task(cmd=cmd, name=None, env=env, cwd=cwd, help=help):
			return Task(cmd=cmd, name=key, env=env, cwd=cwd, help=help)
		case Sequential(tasks=tasks, name=None, matrix=matrix, env=env, cwd=cwd, help=help):
			return Sequential(*tasks, name=key, matrix=matrix, env=env, cwd=cwd, help=help)
		case Parallel(tasks=tasks, name=None, matrix=matrix, env=env, cwd=cwd, help=help):
			return Parallel(*tasks, name=key, matrix=matrix, env=env, cwd=cwd, help=help)
		case _:
			return node


def load_tasks(path: Path) -> dict[str, TaskNode]:
	"""Read [tool.camas.tasks] from a pyproject.toml and resolve all refs."""
	parsed: dict[str, Any] = tomllib.loads(path.read_text())
	raw: Any = dig(dig(dig(parsed, "tool"), "camas"), "tasks")
	if not is_str_dict(raw):
		raise ValueError(f"[tool.camas.tasks] must be a table, got {type(raw).__name__}")
	pre: dict[str, TaskNode | Ref] = {}
	for name, value in raw.items():
		if not isinstance(value, str):
			raise ValueError(f"task {name!r} must be a string, got {type(value).__name__}")
		try:
			pre[name] = assign_key_name(parse_task_value(value), name)
		except ValueError as e:
			raise ValueError(f"task {name!r}: {e}") from e
	return {name: resolve_refs(tree, pre, frozenset({name})) for name, tree in pre.items()}


def find_tasks_py(start: Path) -> Path | None:
	"""Walk upward from ``start`` looking for a ``tasks.py``."""
	for candidate in (start, *start.parents):
		if (tasks_py := candidate / "tasks.py").is_file():
			return tasks_py
	return None


def name_scope_bindings(scope: Mapping[str, object]) -> dict[str, TaskNode]:
	"""Collect public ``Task``/``Sequential``/``Parallel`` bindings from a module's
	globals and propagate each top-level binding's name (by id) into nested
	references — ``Parallel(mypy)`` where ``mypy`` is itself a top-level binding
	inherits ``mypy``'s name, matching ``[tool.camas.tasks]`` naming.
	"""
	bindings: Final = {
		name: val
		for name, val in scope.items()
		if not name.startswith("_") and isinstance(val, Task | Sequential | Parallel)
	}
	named_by_id: Final = {id(val): assign_key_name(val, name) for name, val in bindings.items()}

	def promote(node: TaskNode) -> TaskNode:
		source = cast("TaskNode", named_by_id.get(id(node), node))
		match source:
			case Task():
				return source
			case Sequential(tasks=children, name=n, matrix=m, env=e, cwd=c, help=h):
				return Sequential(
					*(promote(ch) for ch in children), name=n, matrix=m, env=e, cwd=c, help=h
				)
			case Parallel(tasks=children, name=n, matrix=m, env=e, cwd=c, help=h):
				return Parallel(
					*(promote(ch) for ch in children), name=n, matrix=m, env=e, cwd=c, help=h
				)
			case _:
				assert_never(source)

	return {name: promote(val) for name, val in bindings.items()}


def load_tasks_from_py(path: Path) -> dict[str, TaskNode]:
	"""Execute a Python task-definition file and collect module-level TaskNode bindings."""
	return name_scope_bindings(runpy.run_path(str(path)))
