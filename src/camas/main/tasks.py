# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Load task definitions from ``[tool.camas.tasks]`` or a ``tasks.py`` scope."""

from __future__ import annotations

import runpy
import sys
from typing import TYPE_CHECKING, Any, Final, TypeGuard, cast

if sys.version_info >= (3, 11):
	from typing import assert_never

	import tomllib
else:  # pragma: no cover
	import tomli as tomllib
	from typing_extensions import assert_never

from ..v0.config import Config
from ..v0.effect import Effect
from ..v0.task import Parallel, Sequential, Task, TaskNode
from .expression import Ref, parse_task_value, resolve_refs

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


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


def load_tasks(path: Path) -> tuple[dict[str, TaskNode], dict[str, type[Effect[Any]]]]:
	"""Read [tool.camas.tasks] from a pyproject.toml and resolve all refs.

	Returns ``(tasks, scope_effects)``; ``scope_effects`` is always empty for
	pyproject sources (TOML can't host Python classes) — the pair shape is
	for symmetry with :func:`load_tasks_from_py`.

	Raises:
		ValueError: when the table or a task value has the wrong type, or a
			task fails to parse or resolve.
	"""
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
	return {name: resolve_refs(tree, pre, frozenset({name})) for name, tree in pre.items()}, {}


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


def name_scope_effects(scope: Mapping[str, object]) -> dict[str, type[Effect[Any]]]:
	"""Public Effect classes from a tasks-file scope.

	Skips ``Effect`` itself, names starting with ``_``, and anything whose
	``__module__`` already starts with ``camas.effect`` (already covered by
	:func:`camas.main.effects.discover_effects`).
	"""
	return {
		name: val
		for name, val in scope.items()
		if not name.startswith("_")
		and val is not Effect
		and not getattr(val, "__module__", "").startswith("camas.effect")
		and isinstance(val, type)
		and issubclass(val, Effect)
	}


def name_scope_config(scope: Mapping[str, object]) -> Config | None:
	"""The project :class:`Config` discovered in a scope by type — like tasks and
	Effects, it is found by ``isinstance`` rather than a reserved name, so any
	binding (conventionally ``camas_config``) works.

	Each task field is resolved to its promoted (named) binding by identity, so a
	``Config(default_task=ci)`` alias of a top-level ``ci`` runs with ``ci``'s name.

	Raises:
		ValueError: when more than one ``Config`` is bound in the scope.
	"""
	configs: Final = [
		(name, val)
		for name, val in scope.items()
		if not name.startswith("_") and isinstance(val, Config)
	]
	if not configs:
		return None
	if len(configs) > 1:
		names = ", ".join(sorted(name for name, _ in configs))
		raise ValueError(f"multiple Config instances defined ({names}); expected at most one")
	config: Final = configs[0][1]
	promoted: Final = name_scope_bindings(scope)
	name_by_id: Final = {
		id(val): name
		for name, val in scope.items()
		if not name.startswith("_") and isinstance(val, Task | Sequential | Parallel)
	}

	def promote_field(node: TaskNode | None) -> TaskNode | None:
		if node is None:
			return None
		name = name_by_id.get(id(node))
		return promoted[name] if name is not None else node

	return Config(
		default_task=promote_field(config.default_task),
		github_task=promote_field(config.github_task),
	)


def load_tasks_from_py(
	path: Path,
) -> tuple[dict[str, TaskNode], dict[str, type[Effect[Any]]], Config | None]:
	"""Execute a Python task-definition file and collect module-level bindings.

	Returns ``(tasks, scope_effects, config)`` — the TaskNode bindings (with name
	propagation), any Effect classes, and the project :class:`Config` (if defined)
	from the file's module-level scope.
	"""
	scope: Final = runpy.run_path(str(path))
	return name_scope_bindings(scope), name_scope_effects(scope), name_scope_config(scope)
