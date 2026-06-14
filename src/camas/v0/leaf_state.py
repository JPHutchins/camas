# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Per-leaf states: ``Waiting`` → ``Running`` → ``Completed``."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, TypeAlias

if TYPE_CHECKING:
	from datetime import datetime

	from .completion import Completion
	from .task import Task


class Waiting(NamedTuple):
	"""Leaf state: task has not started yet.

	>>> from camas.v0 import Task
	>>> Waiting(Task("echo hi"))
	Waiting(task=Task(cmd='echo hi', name=None, env={}, cwd=None))
	"""

	task: Task


class Running(NamedTuple):
	"""Leaf state: task is currently executing.

	>>> from datetime import datetime
	>>> from camas.v0 import Task
	>>> Running(Task("echo hi"), datetime(2026, 1, 1, 12, 0, 0), b"output")
	Running(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=datetime.datetime(2026, 1, 1, 12, 0), last_line=b'output')
	"""

	task: Task
	start_time: datetime
	last_line: bytes


class Interrupting(NamedTuple):
	"""Leaf state: a signal was forwarded; the task has not exited yet.

	>>> from datetime import datetime
	>>> from camas.v0 import Task
	>>> Interrupting(Task("echo hi"), datetime(2026, 1, 1, 12, 0, 0), b"output", 1)
	Interrupting(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=datetime.datetime(2026, 1, 1, 12, 0), last_line=b'output', presses=1)
	"""

	task: Task
	start_time: datetime
	last_line: bytes
	presses: int
	"""Ctrl-C presses this leaf has absorbed: 1/2 forwarded SIGINT, 3+ force-killed."""


class Completed(NamedTuple):
	"""Leaf state: task is done — ran to exit, was skipped, or was stopped by a signal.

	The `completion` payload is a sum type: pattern-match on `Finished(...)`,
	`Skipped(...)`, or `Stopped(...)` to distinguish the cases.

	>>> from camas.v0 import Task
	>>> from camas.v0.completion import Finished, Skipped
	>>> Completed(Task("echo hi"), Finished(0, 0.5, ()))
	Completed(task=Task(cmd='echo hi', name=None, env={}, cwd=None), completion=Finished(returncode=0, elapsed=0.5, output=()))
	>>> Completed(Task("echo hi"), Skipped(1))
	Completed(task=Task(cmd='echo hi', name=None, env={}, cwd=None), completion=Skipped(returncode=1))
	"""

	task: Task
	completion: Completion


LeafState: TypeAlias = Waiting | Running | Interrupting | Completed
