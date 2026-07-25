# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Engine-side task helpers: matrix bindings, the task display label, and name suggestions."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, TypeAlias

if TYPE_CHECKING:
	from collections.abc import Iterable

	from ..v0.task import Task, TaskNode


class VarBinding(NamedTuple):
	"""A single matrix variable bound to a concrete value.

	>>> VarBinding("PY", "3.14")
	VarBinding(name='PY', value='3.14')
	"""

	name: str
	value: str


MatrixBinding: TypeAlias = tuple[VarBinding, ...]


def task_label(task: Task) -> str:
	"""Return a task's label: the explicit ``name``, else the full command string.

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
	"""Any node's label for a message: a leaf's :func:`task_label`, else a group's ``name`` — or
	``<unnamed group>`` for a group with none, which has nothing else to identify it by.

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


def did_you_mean(name: str, known: Iterable[str]) -> str:
	"""Return a ``; did you mean 'x'?`` clause for the closest ``known`` name, else ``""``.

	>>> did_you_mean("libs-fix", ("libs.fix", "libs.build", "api"))
	"; did you mean 'libs.fix'?"
	>>> did_you_mean("totally-unrelated", ("libs.fix", "api"))
	''
	"""
	from difflib import get_close_matches

	closest = get_close_matches(name, sorted(known), n=1)
	return f"; did you mean {closest[0]!r}?" if closest else ""
