# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Per-leaf state machine: ``Waiting`` → ``Running`` → ``Completed``."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from .task_event import CompletedEvent, OutputEvent, StartedEvent, TaskEvent

if TYPE_CHECKING:
	from datetime import datetime

	from .completion import Completion
	from .task import Task


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

	>>> from camas.core.task import Task
	>>> LeafInfo(Task("echo hi"), 0, ())
	LeafInfo(task=Task(cmd='echo hi', name=None, env={}, cwd=None), depth=0, is_last_chain=())
	"""

	task: Task
	depth: int
	is_last_chain: tuple[ChainLink, ...]


class Waiting(NamedTuple):
	"""Leaf state: task has not started yet.

	>>> from camas.core.task import Task
	>>> Waiting(Task("echo hi"))
	Waiting(task=Task(cmd='echo hi', name=None, env={}, cwd=None))
	"""

	task: Task


class Running(NamedTuple):
	"""Leaf state: task is currently executing.

	>>> from datetime import datetime
	>>> from camas.core.task import Task
	>>> Running(Task("echo hi"), datetime(2026, 1, 1, 12, 0, 0), b"output")
	Running(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=datetime.datetime(2026, 1, 1, 12, 0), last_line=b'output')
	"""

	task: Task
	start_time: datetime
	last_line: bytes


class Completed(NamedTuple):
	"""Leaf state: task is done — either ran to exit or was skipped.

	The `completion` payload is a sum type: pattern-match on `Finished(...)`
	vs `Skipped(...)` to distinguish the two cases.

	>>> from camas.core.completion import Finished, Skipped
	>>> from camas.core.task import Task
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

	>>> from datetime import datetime
	>>> from camas.core.completion import Finished, Skipped
	>>> from camas.core.task import Task
	>>> t = Task("echo hi")
	>>> t0 = datetime(2026, 1, 1, 12, 0, 0)
	>>> t1 = datetime(2026, 1, 1, 12, 0, 1)
	>>> next_state(Waiting(t), StartedEvent(t, 0, t0))
	Running(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=datetime.datetime(2026, 1, 1, 12, 0), last_line=b'')
	>>> next_state(Running(t, t0, b""), OutputEvent(t, 0, b"hi", t1))
	Running(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=datetime.datetime(2026, 1, 1, 12, 0), last_line=b'hi')
	>>> next_state(Running(t, t0, b""), CompletedEvent(t, 0, Finished(0, 0.5, (b"done",)), t1))
	Completed(task=Task(cmd='echo hi', name=None, env={}, cwd=None), completion=Finished(returncode=0, elapsed=0.5, output=(b'done',)))
	>>> next_state(Waiting(t), CompletedEvent(t, 0, Skipped(1), t0))
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
