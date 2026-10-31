# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import sys
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final, TypeGuard, cast

if sys.version_info >= (3, 11):
	from typing import assert_never

	import tomllib
else:  # pragma: no cover
	import tomli as tomllib
	from typing_extensions import assert_never

from ..v0.command import Command, CommandBody
from ..v0.config import Agent, Config
from ..v0.effect import Effect
from ..v0.task import Group, Parallel, Pipe, Sequential, Task, TaskNode, rebuilt, reject_refs
from .expression import Ref, parse_task_value, resolve_refs
from .state import LoadOk

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping
	from pathlib import Path


def is_str_dict(value: Any) -> TypeGuard[dict[str, Any]]:
	return isinstance(value, dict) and all(isinstance(k, str) for k in value)  # pyright: ignore[reportUnknownVariableType]


def dig(data: Any, key: str) -> Any:
	return data[key] if is_str_dict(data) and key in data else {}


def assign_key_name(node: TaskNode | Ref, key: str) -> TaskNode | Ref:
	"""
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
			return rebuilt(group, *group.tasks, name=key)
		case _:
			return node


def redundant_name_warnings(scope: Mapping[str, object]) -> tuple[str, ...]:
	"""
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
		and isinstance(val, Task | Sequential | Parallel | Pipe)
		and val.name == name
	)


def anonymous_config_field_warnings(scope: Mapping[str, object]) -> tuple[str, ...]:
	"""
	>>> anonymous_config_field_warnings({"_": Config(github_task=Parallel(Task("a")))})[0].startswith("Config.github_task")
	True
	>>> ci = Parallel(Task("a"))
	>>> anonymous_config_field_warnings({"ci": ci, "_": Config(github_task=ci)})
	()
	>>> anonymous_config_field_warnings({"_": Config(github_task=Parallel(Task("a"), name="ci"))})
	()
	>>> anonymous_config_field_warnings({"_": Config(default_task=Task("a"))})
	()
	>>> from camas import Claude
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
		if not name.startswith("_") and isinstance(val, Task | Sequential | Parallel | Pipe)
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
	"""
	Raises:
		ValueError: when a task is named after a reserved keyword.

	>>> reject_reserved_names({"build": None, "test": None}) is None
	True
	"""
	clash = sorted(RESERVED_TASK_NAMES.intersection(tasks))
	if clash:
		names = ", ".join(repr(name) for name in clash)
		raise ValueError(f"task name {names} is reserved for a camas subcommand; rename it")


def load_tasks(path: Path) -> LoadOk:
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


def scope_task_bindings(scope: Mapping[str, object]) -> dict[str, TaskNode]:
	return {
		name: val
		for name, val in scope.items()
		if not name.startswith("_") and isinstance(val, Task | Sequential | Parallel | Pipe)
	}


def name_scope_bindings(scope: Mapping[str, object]) -> dict[str, TaskNode]:
	bindings: Final = scope_task_bindings(scope)
	promote: Final = bound_name_promoter(bindings)
	return {name: promote(val) for name, val in bindings.items()}


def name_scope_commands(scope: Mapping[str, object]) -> dict[str, Command]:
	"""
	>>> build = Task("cargo build")
	>>> flash = Command(build + "probe-rs download")
	>>> named = name_scope_commands({"build": build, "flash": flash})["flash"]
	>>> named.name, named.body.name, named.body.tasks[0].name  # type: ignore[union-attr]
	('flash', 'flash', 'build')
	>>> relabeled = name_scope_commands({"flash": Command("probe-rs run", name="board")})["flash"]
	>>> relabeled.name, relabeled.body.name
	('board', 'board')
	"""
	commands: Final = {
		name: val
		for name, val in scope.items()
		if not name.startswith("_") and isinstance(val, Command)
	}
	if not commands:
		return {}
	promote: Final = bound_name_promoter(scope_task_bindings(scope))
	return {
		name: labeled_command(command, command.name if command.name is not None else name, promote)
		for name, command in commands.items()
	}


def labeled_command(
	command: Command, label: str, promote: Callable[[TaskNode], TaskNode]
) -> Command:
	return replace(
		command,
		name=label,
		body=cast("CommandBody", assign_key_name(promote(command.body), label)),
	)


def bound_name_promoter(bindings: Mapping[str, TaskNode]) -> Callable[[TaskNode], TaskNode]:
	named_by_id: Final = {id(val): assign_key_name(val, name) for name, val in bindings.items()}

	def promote(node: TaskNode) -> TaskNode:
		source = cast("TaskNode", named_by_id.get(id(node), node))
		match source:
			case Task():
				return source
			case Group() as group:
				reject_refs(group)
				return rebuilt(group, *(promote(ch) for ch in group.tasks))
			case _:
				assert_never(source)

	return promote


def name_scope_effects(scope: Mapping[str, object]) -> dict[str, type[Effect[Any]]]:
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
		if not name.startswith("_") and isinstance(val, Task | Sequential | Parallel | Pipe)
	}

	def promote_required(node: TaskNode) -> TaskNode:
		name = name_by_id.get(id(node))
		return promoted[name] if name is not None else node

	def promote_field(node: TaskNode | None) -> TaskNode | None:
		return None if node is None else promote_required(node)

	def promote_agent(agent: Agent | None) -> Agent | None:
		if agent is None:
			return None
		return agent._replace(
			fix=promote_required(agent.fix),
			check=promote_field(agent.check),
			default=promote_field(agent.default),
		)

	return config._replace(
		default_task=promote_field(config.default_task),
		github_task=promote_field(config.github_task),
		agent=promote_agent(config.agent),
	)


def load_tasks_from_py(path: Path) -> LoadOk:
	from .compose import load_scope

	return load_scope(path)
