# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import sys
from typing import NamedTuple, TypeAlias

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from .completion import Completion
from .task import Task
from .task_event import CompletedEvent, OutputEvent, StartedEvent, TaskEvent


class ChainLink(NamedTuple):
	"""One ancestor's position in a tree-walk chain.

	>>> ChainLink(True, False)
	ChainLink(is_last=True, parent_is_parallel=False)
	"""

	is_last: bool
	"""Whether the current node is the last child in its sibling group. Used for display formatting."""
	parent_is_parallel: bool
	"""Whether the containing group is a ``Parallel``. Used for display formatting."""


class LeafInfo(NamedTuple):
	"""A leaf's position in the task tree: depth and chain of ancestor links.

	>>> LeafInfo(Task("echo hi"), 0, ())
	LeafInfo(task=Task(cmd='echo hi', name=None, env={}, cwd=None), depth=0, is_last_chain=())
	"""

	task: Task
	depth: int
	is_last_chain: tuple[ChainLink, ...]


class Waiting(NamedTuple):
	"""Leaf state: task has not started yet.

	>>> Waiting(Task("echo hi"))
	Waiting(task=Task(cmd='echo hi', name=None, env={}, cwd=None))
	"""

	task: Task


class Running(NamedTuple):
	"""Leaf state: task is currently executing.

	>>> Running(Task("echo hi"), 100.0, b"output")
	Running(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=100.0, last_line=b'output')
	"""

	task: Task
	start_time: float
	last_line: bytes


class Completed(NamedTuple):
	"""Leaf state: task is done — either ran to exit or was skipped.

	The `completion` payload is a sum type: pattern-match on `Finished(...)`
	vs `Skipped(...)` to distinguish the two cases.

	>>> from camas.core.completion import Finished, Skipped
	>>> Completed(Task("echo hi"), Finished(0, 0.5, ()))
	Completed(task=Task(cmd='echo hi', name=None, env={}, cwd=None), completion=Finished(returncode=0, elapsed=0.5, output=()))
	>>> Completed(Task("echo hi"), Skipped(1))
	Completed(task=Task(cmd='echo hi', name=None, env={}, cwd=None), completion=Skipped(returncode=1))
	"""

	task: Task
	completion: Completion


LeafState: TypeAlias = Waiting | Running | Completed


def next_state(state: LeafState, event: TaskEvent) -> LeafState:
	"""Pure state machine: apply a TaskEvent to a LeafState to produce the next state.

	>>> from camas.core.completion import Finished, Skipped
	>>> t = Task("echo hi")
	>>> next_state(Waiting(t), StartedEvent(0, 100.0))
	Running(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=100.0, last_line=b'')
	>>> next_state(Running(t, 100.0, b""), OutputEvent(0, b"hi", 100.5))
	Running(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=100.0, last_line=b'hi')
	>>> next_state(Running(t, 100.0, b""), CompletedEvent(0, Finished(0, 0.5, (b"done",))))
	Completed(task=Task(cmd='echo hi', name=None, env={}, cwd=None), completion=Finished(returncode=0, elapsed=0.5, output=(b'done',)))
	>>> next_state(Waiting(t), CompletedEvent(0, Skipped(1)))
	Completed(task=Task(cmd='echo hi', name=None, env={}, cwd=None), completion=Skipped(returncode=1))
	"""
	match state:
		case Waiting(task=task):
			match event:
				case StartedEvent(timestamp=ts):
					return Running(task, ts, b"")
				case OutputEvent(line=line, timestamp=ts):  # pragma: no cover
					return Running(task, ts, line)
				case CompletedEvent(completion=completion):
					return Completed(task, completion)
				case _:
					assert_never(event)
		case Running(task=task, start_time=start):
			match event:
				case OutputEvent(line=line):
					return Running(task, start, line)
				case CompletedEvent(completion=completion):
					return Completed(task, completion)
				case StartedEvent():  # pragma: no cover
					return state
				case _:
					assert_never(event)
		case Completed():  # pragma: no cover
			return state
		case _:
			assert_never(state)
