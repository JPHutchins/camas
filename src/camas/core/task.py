# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Engine-side task helpers: matrix bindings and the task display label."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, TypeAlias

if TYPE_CHECKING:
	from ..v0.task import Task


class VarBinding(NamedTuple):
	"""A single matrix variable bound to a concrete value.

	>>> VarBinding("PY", "3.14")
	VarBinding(name='PY', value='3.14')
	"""

	name: str
	value: str


MatrixBinding: TypeAlias = tuple[VarBinding, ...]


def task_label(task: Task) -> str:
	"""Return a task's identifying label: the explicit `name` or the full command string.

	This is a data accessor with no concept of display width — callers that render
	into a column-constrained terminal are responsible for truncation.

	>>> from camas import Task
	>>> task_label(Task("echo hi", name="greet"))
	'greet'
	>>> task_label(Task("echo hi"))
	'echo hi'
	>>> task_label(Task(("python", "-c", "pass")))
	'python -c pass'
	"""
	if task.name is not None:
		return task.name
	return task.cmd if isinstance(task.cmd, str) else " ".join(task.cmd)
