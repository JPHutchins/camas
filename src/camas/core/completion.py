# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Completion outcomes (``Finished`` | ``Skipped``) and the per-run result aggregates."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, TypeAlias

if TYPE_CHECKING:
	from collections.abc import Sequence


class Finished(NamedTuple):
	"""Completion outcome: task ran to exit with a returncode.

	>>> Finished(0, 1.234, (b"all clean",))
	Finished(returncode=0, elapsed=1.234, output=(b'all clean',))
	"""

	returncode: int
	elapsed: float
	output: Sequence[bytes]


class Skipped(NamedTuple):
	"""Completion outcome: task was skipped due to a prior Sequential failure.

	Carries the returncode of the task that caused the skip — an Either-like
	propagation so callers that need an rc (e.g. the overall run's exit code)
	can read it uniformly across completion variants.

	>>> Skipped(1)
	Skipped(returncode=1)
	"""

	returncode: int


Completion: TypeAlias = Finished | Skipped


class TaskResult(NamedTuple):
	"""Result of a single completed task.

	>>> TaskResult("lint", Finished(0, 1.234, (b"all clean",)))
	TaskResult(name='lint', completion=Finished(returncode=0, elapsed=1.234, output=(b'all clean',)))
	"""

	name: str
	completion: Completion


class RunResult(NamedTuple):
	"""Result of running an entire task tree.

	>>> RunResult(0, (TaskResult("a", Finished(0, 0.1, ())),), 0.1)
	RunResult(returncode=0, results=(TaskResult(name='a', completion=Finished(returncode=0, elapsed=0.1, output=())),), elapsed=0.1)
	"""

	returncode: int
	results: tuple[TaskResult, ...]
	elapsed: float
