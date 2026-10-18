# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from .timings import (
	CacheKey,  # noqa: TC001  # runtime name get_type_hints resolves; TYPE_CHECKING-only would NameError
)

if TYPE_CHECKING:
	from ..v0.completion import Completion


class TaskResult(NamedTuple):
	"""
	>>> from camas.v0.completion import Finished
	>>> TaskResult("lint", Finished(0, 1.234, (b"all clean",)))
	TaskResult(name='lint', completion=Finished(returncode=0, elapsed=1.234, output=(b'all clean',)), identity=None)
	"""

	name: str
	completion: Completion
	identity: CacheKey | None = None


class RunResult(NamedTuple):
	"""
	>>> from camas.v0.completion import Finished
	>>> RunResult(0, (TaskResult("a", Finished(0, 0.1, ())),), 0.1)
	RunResult(returncode=0, results=(TaskResult(name='a', completion=Finished(returncode=0, elapsed=0.1, output=()), identity=None),), elapsed=0.1, interrupt_count=0)
	"""

	returncode: int
	results: tuple[TaskResult, ...]
	elapsed: float
	interrupt_count: int = 0
