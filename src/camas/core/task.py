# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, TypeAlias

if TYPE_CHECKING:
	from collections.abc import Iterable

	from ..v0.task import Task, TaskNode


class VarBinding(NamedTuple):
	"""
	>>> VarBinding("PY", "3.14")
	VarBinding(name='PY', value='3.14')
	"""

	name: str
	value: str


MatrixBinding: TypeAlias = tuple[VarBinding, ...]


def task_label(task: Task) -> str:
	"""
	>>> from camas import Task
	>>> task_label(Task("echo hi", name="greet"))
	'greet'
	>>> task_label(Task(("python", "-c", "pass")))
	'python -c pass'
	"""
	if task.name is not None:
		return task.name
	return task.cmd if isinstance(task.cmd, str) else " ".join(task.cmd)


def node_label(node: TaskNode) -> str:
	"""
	>>> from camas import Parallel, Task
	>>> node_label(Task("ruff check .")), node_label(Parallel(Task("t"), name="ci"))
	('ruff check .', 'ci')
	>>> node_label(Parallel(Task("t")))
	'<unnamed group>'
	"""
	from ..v0.task import Task

	if isinstance(node, Task):
		return task_label(node)
	return node.name if node.name is not None else "<unnamed group>"


def agent_only_pipe_labels(node: TaskNode) -> tuple[str, ...]:
	"""
	>>> from camas import Pipe, Sequential
	>>> agent_only_pipe_labels(
	...     Sequential(Pipe("a", "b", agent_only=True, name="lint"), Pipe("c", "d", agent_only=True), Pipe("e"))
	... )
	('lint', 'c | d')
	"""
	from ..v0.task import Group, Pipe

	match node:
		case Pipe(agent_only=True, name=None):
			return (" | ".join(node_label(stage) for stage in node.tasks),)
		case Pipe(agent_only=True, name=str(name)):
			return (name,)
		case Group():
			return tuple(label for child in node.tasks for label in agent_only_pipe_labels(child))
		case _:
			return ()


def reject_agent_only_pipes(node: TaskNode) -> None:
	labels = agent_only_pipe_labels(node)
	if labels:
		raise ValueError(
			f"agent_only Pipe {', '.join(repr(label) for label in labels)} reached a run outside "
			"the gate's checks; collapse it with strip_agent_only_pipes, or pass "
			"whole_agent_only_pipes=True as the gate's checks do"
		)


def did_you_mean(name: str, known: Iterable[str]) -> str:
	"""
	>>> did_you_mean("libs-fix", ("libs.fix", "libs.build", "api"))
	"; did you mean 'libs.fix'?"
	>>> did_you_mean("totally-unrelated", ("libs.fix", "api"))
	''
	"""
	from difflib import get_close_matches

	closest = get_close_matches(name, sorted(known), n=1)
	return f"; did you mean {closest[0]!r}?" if closest else ""
