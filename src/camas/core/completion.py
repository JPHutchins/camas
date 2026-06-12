# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Per-run result aggregates over a run's completion outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
	from ..v0.completion import Completion


class TaskResult(NamedTuple):
	"""Result of a single completed task.

	>>> from camas.v0.completion import Finished
	>>> TaskResult("lint", Finished(0, 1.234, (b"all clean",)))
	TaskResult(name='lint', completion=Finished(returncode=0, elapsed=1.234, output=(b'all clean',)))
	"""

	name: str
	completion: Completion


class RunResult(NamedTuple):
	"""Result of running an entire task tree.

	>>> from camas.v0.completion import Finished
	>>> RunResult(0, (TaskResult("a", Finished(0, 0.1, ())),), 0.1)
	RunResult(returncode=0, results=(TaskResult(name='a', completion=Finished(returncode=0, elapsed=0.1, output=())),), elapsed=0.1)
	"""

	returncode: int
	results: tuple[TaskResult, ...]
	elapsed: float
