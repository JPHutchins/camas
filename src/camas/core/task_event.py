# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import NamedTuple, TypeAlias

from .completion import Completion


class StartedEvent(NamedTuple):
	"""Internal event: a task has started execution.

	>>> StartedEvent(0, 100.0)
	StartedEvent(leaf_index=0, timestamp=100.0)
	"""

	leaf_index: int
	timestamp: float


class OutputEvent(NamedTuple):
	"""Internal event: a task produced an output line.

	>>> OutputEvent(0, b"hello", 100.5)
	OutputEvent(leaf_index=0, line=b'hello', timestamp=100.5)
	"""

	leaf_index: int
	line: bytes
	timestamp: float


class CompletedEvent(NamedTuple):
	"""Internal event: a task finished execution (either ran or was skipped).

	>>> from camas.core.completion import Finished, Skipped
	>>> CompletedEvent(0, Finished(0, 1.0, (b"done",)))
	CompletedEvent(leaf_index=0, completion=Finished(returncode=0, elapsed=1.0, output=(b'done',)))
	>>> CompletedEvent(0, Skipped(1))
	CompletedEvent(leaf_index=0, completion=Skipped(returncode=1))
	"""

	leaf_index: int
	completion: Completion


TaskEvent: TypeAlias = StartedEvent | OutputEvent | CompletedEvent
