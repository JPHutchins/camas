# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Engine-side leaf machinery: tree-position display info and the pure state
machine over the per-leaf states.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Final, NamedTuple

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..v0.leaf_state import Completed, Interrupting, LeafState, Running, Waiting
from ..v0.task_event import CompletedEvent, OutputEvent, StartedEvent, TaskEvent

if TYPE_CHECKING:
	from ..v0.task import Task


KILL_PRESSES: Final = 3
"""Ctrl-C count at which a leaf is force-killed; its row reads ``KILL`` until it dies."""


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

	>>> from camas import Task
	>>> LeafInfo(Task("echo hi"), 0, ())
	LeafInfo(task=Task(cmd='echo hi', name=None, env={}, cwd=None), depth=0, is_last_chain=())
	"""

	task: Task
	depth: int
	is_last_chain: tuple[ChainLink, ...]


def to_interrupting(state: LeafState, presses: int) -> LeafState:
	"""Mark a running leaf as interrupting at ``presses`` Ctrl-C; non-running states pass through.

	>>> from datetime import datetime
	>>> from camas import Task
	>>> t = Task("echo hi")
	>>> t0 = datetime(2026, 1, 1, 12, 0, 0)
	>>> to_interrupting(Running(t, t0, b"out"), 2)
	Interrupting(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=datetime.datetime(2026, 1, 1, 12, 0), last_line=b'out', presses=2)
	>>> to_interrupting(Waiting(t), 1)
	Waiting(task=Task(cmd='echo hi', name=None, env={}, cwd=None))
	"""
	match state:
		case (
			Running(task=task, start_time=start, last_line=last)
			| Interrupting(task=task, start_time=start, last_line=last)
		):
			return Interrupting(task, start, last, presses)
		case Waiting() | Completed():
			return state


def next_state(state: LeafState, event: TaskEvent) -> LeafState:
	"""Pure state machine: apply a TaskEvent to a LeafState to produce the next state.

	A leaf enters ``Interrupting`` only via :func:`to_interrupting`, never an event here.

	>>> from datetime import datetime
	>>> from camas import Task
	>>> from camas.v0.completion import Finished, Skipped
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
	>>> next_state(Interrupting(t, t0, b"", 2), OutputEvent(t, 0, b"bye", t1))
	Interrupting(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=datetime.datetime(2026, 1, 1, 12, 0), last_line=b'bye', presses=2)
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
		case Interrupting(task=task, start_time=start, presses=presses):
			match event:
				case OutputEvent(line=line):
					return Interrupting(task, start, line, presses)
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
