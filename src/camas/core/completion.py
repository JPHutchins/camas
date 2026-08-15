# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Per-run result aggregates over a run's completion outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from .timings import (
	CacheKey,  # noqa: TC001  # runtime name get_type_hints resolves; TYPE_CHECKING-only would NameError
)

if TYPE_CHECKING:
	from ..v0.completion import Completion


class TaskResult(NamedTuple):
	"""Result of a single completed task.

	>>> from camas.v0.completion import Finished
	>>> TaskResult("lint", Finished(0, 1.234, (b"all clean",)))
	TaskResult(name='lint', completion=Finished(returncode=0, elapsed=1.234, output=(b'all clean',)), identity=None)
	"""

	name: str
	completion: Completion
	identity: CacheKey | None = None
	"""The cache key this leaf's timing belongs under, computed before any command rewrite —
	scoping, ``agent_format``, ``--`` passthrough — so recording never reconstructs identity
	from the label the leaf happened to report. ``None`` for runs threaded without identities.
	Compare and unpack element-wise: this field is third, so 2-tuple equality no longer holds."""


class RunResult(NamedTuple):
	"""Result of running an entire task tree.

	>>> from camas.v0.completion import Finished
	>>> RunResult(0, (TaskResult("a", Finished(0, 0.1, ())),), 0.1)
	RunResult(returncode=0, results=(TaskResult(name='a', completion=Finished(returncode=0, elapsed=0.1, output=()), identity=None),), elapsed=0.1, interrupt_count=0)
	"""

	returncode: int
	results: tuple[TaskResult, ...]
	elapsed: float
	interrupt_count: int = 0
	"""Ctrl-C presses the run received; non-zero drives the CLI's exit banner."""
