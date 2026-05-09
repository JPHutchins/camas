# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import NamedTuple, TypeAlias

from .completion import Completion
from .task import Task


class StartedEvent(NamedTuple):
	"""Internal event: a task has started execution.

	>>> StartedEvent(Task("hi"), 0, 100.0)
	StartedEvent(task=Task(cmd='hi', name=None, env={}, cwd=None), leaf_index=0, timestamp=100.0)
	"""

	task: Task
	leaf_index: int
	timestamp: float


class OutputEvent(NamedTuple):
	"""Internal event: a task produced an output line.

	>>> OutputEvent(Task("hi"), 0, b"hello", 100.5)
	OutputEvent(task=Task(cmd='hi', name=None, env={}, cwd=None), leaf_index=0, line=b'hello', timestamp=100.5)
	"""

	task: Task
	leaf_index: int
	line: bytes
	timestamp: float


class CompletedEvent(NamedTuple):
	"""Internal event: a task finished execution (either ran or was skipped).

	>>> from camas.core.completion import Finished, Skipped
	>>> CompletedEvent(Task("hi"), 0, Finished(0, 1.0, (b"done",)))
	CompletedEvent(task=Task(cmd='hi', name=None, env={}, cwd=None), leaf_index=0, completion=Finished(returncode=0, elapsed=1.0, output=(b'done',)))
	>>> CompletedEvent(Task("hi"), 0, Skipped(1))
	CompletedEvent(task=Task(cmd='hi', name=None, env={}, cwd=None), leaf_index=0, completion=Skipped(returncode=1))
	"""

	task: Task
	leaf_index: int
	completion: Completion


TaskEvent: TypeAlias = StartedEvent | OutputEvent | CompletedEvent
