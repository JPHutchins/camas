# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Load task definitions from ``[tool.camas.tasks]`` or a ``tasks.py`` scope."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Final, TypeGuard, cast

if sys.version_info >= (3, 11):
	from typing import assert_never

	import tomllib
else:  # pragma: no cover
	import tomli as tomllib
	from typing_extensions import assert_never

from ..v0.config import Agent, Claude, Config
from ..v0.effect import Effect
from ..v0.task import Group, Parallel, Sequential, Task, TaskNode
from .expression import Ref, parse_task_value, resolve_refs
from .state import LoadOk

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
		case Task(
			cmd=cmd,
			name=None,
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
				name=key,
				env=env,
				cwd=cwd,
				help=help,
				mutates=mutates,
				paths=paths,
				when=when,
				agent_format=agent_format,
			)
		case Group(name=None) as group:
			return type(group)(
				*group.tasks,
				name=key,
				matrix=group.matrix,
				env=group.env,
				cwd=group.cwd,
				help=group.help,
				paths=group.paths,
				when=group.when,
			)
		case _:
			return node


def redundant_name_warnings(scope: Mapping[str, object]) -> tuple[str, ...]:
	"""Advisory warnings for a top-level binding that passes ``name=`` equal to its own variable
	name: the binding name already *is* the task name (:func:`assign_key_name`), so the ``name=``
	is redundant. Read from the raw scope, where an explicit ``name=`` is still distinguishable
	from the one :func:`assign_key_name` would supply.

	>>> redundant_name_warnings({"fix": Sequential(Task("a"), name="fix")})[0].startswith("task 'fix'")
	True
	>>> redundant_name_warnings({"fix": Sequential(Task("a"))})
	()
	>>> redundant_name_warnings({"checks": Parallel(Task("a"), name="ci")})
	()
	"""
	return tuple(
		f"task {name!r} passes name={name!r}, which just repeats its binding name; the binding "
		"name is already the task name, so drop the redundant name="
		for name, val in scope.items()
		if not name.startswith("_")
		and isinstance(val, Task | Sequential | Parallel)
		and val.name == name
	)


def anonymous_config_field_warnings(scope: Mapping[str, object]) -> tuple[str, ...]:
	"""Advisory warnings for a ``Config`` task field bound to an anonymous inline composite: a
	``Sequential``/``Parallel`` that has no ``name=`` and is not a top-level binding either.
	``task_name`` then has nothing to report, so ``camas_list`` shows that default as ``null`` —
	reading as *no task declared* when one is — leaving it unrunnable by ``camas_run`` / the
	pre-push hook. Bind the node (or pass ``name=``); a bound or named composite is fine — a
	binding is excluded by identity, so ``Config`` fields still resolve correctly across a
	monorepo where ``Project`` composition drops the wrapper's name.

	>>> anonymous_config_field_warnings({"_": Config(github_task=Parallel(Task("a")))})[0].startswith("Config.github_task")
	True
	>>> ci = Parallel(Task("a"))
	>>> anonymous_config_field_warnings({"ci": ci, "_": Config(github_task=ci)})
	()
	>>> anonymous_config_field_warnings({"_": Config(github_task=Parallel(Task("a"), name="ci"))})
	()
	>>> anonymous_config_field_warnings({"_": Config(default_task=Task("a"))})
	()
	>>> anonymous_config_field_warnings({"_": Config(agent=Claude(fix=Parallel(Task("a"))))})[0].startswith("Config.agent.fix")
	True
	>>> anonymous_config_field_warnings({})
	()
	"""
	config = next((val for val in scope.values() if isinstance(val, Config)), None)
	if config is None:
		return ()
	bound_ids = {
		id(val)
		for name, val in scope.items()
		if not name.startswith("_") and isinstance(val, Task | Sequential | Parallel)
	}
	agent = config.agent
	base_fields: tuple[tuple[str, TaskNode | None], ...] = (
		("default_task", config.default_task),
		("github_task", config.github_task),
	)
	agent_fields: tuple[tuple[str, TaskNode | None], ...] = (
		()
		if agent is None
		else (
			("agent.fix", agent.fix),
			("agent.check", agent.check),
			("agent.default", agent.default),
		)
	)
	fields = base_fields + agent_fields
	return tuple(
		f"Config.{field} is an anonymous {type(node).__name__} — bind it to a variable (or pass "
		"name=) so camas_list can name it; an unnamed inline composite reports as null, and "
		"camas_run and the pre-push hook resolve tasks by name"
		for field, node in fields
		if isinstance(node, Group) and node.name is None and id(node) not in bound_ids
	)


RESERVED_TASK_NAMES: Final = frozenset({"mcp"})


def reject_reserved_names(tasks: Mapping[str, object]) -> None:
	"""Reject task names that collide with a reserved ``camas`` subcommand.

	Raises:
		ValueError: when a task is named after a reserved keyword (``mcp``); the CLI
			intercepts those before task dispatch and would otherwise shadow them.

	>>> reject_reserved_names({"build": None, "test": None}) is None
	True
	"""
	clash = sorted(RESERVED_TASK_NAMES.intersection(tasks))
	if clash:
		names = ", ".join(repr(name) for name in clash)
		raise ValueError(f"task name {names} is reserved for a camas subcommand; rename it")


def load_tasks(path: Path) -> LoadOk:
	"""Read [tool.camas.tasks] from a pyproject.toml and resolve all refs.

	Raises:
		ValueError: when the table or a task value has the wrong type, a task fails
			to parse or resolve, or a task uses a reserved name.
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
	reject_reserved_names(pre)
	return LoadOk(
		tasks={name: resolve_refs(tree, pre, frozenset({name})) for name, tree in pre.items()},
		source=path,
		scope_effects={},
	)


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
			case Group() as group:
				return type(group)(
					*(promote(ch) for ch in group.tasks),
					name=group.name,
					matrix=group.matrix,
					env=group.env,
					cwd=group.cwd,
					help=group.help,
					paths=group.paths,
					when=group.when,
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
	"""The scope's :class:`Config`, found by ``isinstance`` under any binding name,
	its task fields resolved to their promoted bindings.

	Raises:
		ValueError: when more than one ``Config`` is bound in the scope.
	"""
	configs: Final = [(name, val) for name, val in scope.items() if isinstance(val, Config)]
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

	def promote_required(node: TaskNode) -> TaskNode:
		name = name_by_id.get(id(node))
		return promoted[name] if name is not None else node

	def promote_field(node: TaskNode | None) -> TaskNode | None:
		return None if node is None else promote_required(node)

	def promote_agent(agent: Agent | None) -> Agent | None:
		if agent is None:
			return None
		return Claude(
			fix=promote_required(agent.fix),
			check=promote_field(agent.check),
			default=promote_field(agent.default),
		)

	return Config(
		default_task=promote_field(config.default_task),
		github_task=promote_field(config.github_task),
		default_effects=config.default_effects,
		default_github_effects=config.default_github_effects,
		camas_dir=config.camas_dir,
		agent=promote_agent(config.agent),
	)


def load_tasks_from_py(path: Path) -> LoadOk:
	"""``path`` loaded with its ``Project`` references resolved (:func:`camas.main.compose.load_scope`)."""
	from .compose import load_scope

	return load_scope(path)
