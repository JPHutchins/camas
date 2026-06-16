# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Event sum type emitted by execution: started, output, completed."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, TypeAlias

if TYPE_CHECKING:
	from datetime import datetime

	from .completion import Completion
	from .task import Task


class StartedEvent(NamedTuple):
	"""Event: a task has started execution.

	>>> from datetime import datetime
	>>> from camas.v0 import Task
	>>> StartedEvent(Task("hi"), 0, datetime(2026, 1, 1, 12, 0, 0))
	StartedEvent(task=Task(cmd='hi', name=None, env={}, cwd=None), leaf_index=0, timestamp=datetime.datetime(2026, 1, 1, 12, 0))
	"""

	task: Task
	leaf_index: int
	timestamp: datetime


class OutputEvent(NamedTuple):
	"""Event: a task produced an output line.

	>>> from datetime import datetime
	>>> from camas.v0 import Task
	>>> OutputEvent(Task("hi"), 0, b"hello", datetime(2026, 1, 1, 12, 0, 1))
	OutputEvent(task=Task(cmd='hi', name=None, env={}, cwd=None), leaf_index=0, line=b'hello', timestamp=datetime.datetime(2026, 1, 1, 12, 0, 1))
	"""

	task: Task
	leaf_index: int
	line: bytes
	timestamp: datetime


class CompletedEvent(NamedTuple):
	"""Event: a task finished execution (either ran or was skipped).

	>>> from datetime import datetime
	>>> from camas.v0 import Task
	>>> from camas.v0.completion import Finished, Skipped
	>>> CompletedEvent(Task("hi"), 0, Finished(0, 1.0, (b"done",)), datetime(2026, 1, 1, 12, 0, 2))
	CompletedEvent(task=Task(cmd='hi', name=None, env={}, cwd=None), leaf_index=0, completion=Finished(returncode=0, elapsed=1.0, output=(b'done',)), timestamp=datetime.datetime(2026, 1, 1, 12, 0, 2))
	>>> CompletedEvent(Task("hi"), 0, Skipped(1), datetime(2026, 1, 1, 12, 0, 0))
	CompletedEvent(task=Task(cmd='hi', name=None, env={}, cwd=None), leaf_index=0, completion=Skipped(returncode=1, blocked_by=None), timestamp=datetime.datetime(2026, 1, 1, 12, 0))
	"""

	task: Task
	leaf_index: int
	completion: Completion
	timestamp: datetime


TaskEvent: TypeAlias = StartedEvent | OutputEvent | CompletedEvent
