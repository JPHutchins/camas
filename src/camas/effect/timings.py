# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Effect: on teardown, record the run's per-task duration to the ``.camas`` cache."""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

from ..core import timings
from ..core.traversal import flatten_leaves
from ..v0.leaf_state import Completed, LeafState, Waiting
from ..v0.task import Task, TaskNode
from ..v0.task_event import TaskEvent


@dataclass
class _Latest:
	"""Mutable slot holding the most recent per-leaf states view."""

	states: Sequence[LeafState]


class _Context(NamedTuple):
	task: str | None
	base: Path
	start_mono: float
	latest: _Latest


class Timings:
	"""Records each run's wall-clock duration and slowest leaf to ``.camas/timings``.

	Default-on for a project that has a ``.camas`` directory, off under CI; silent
	otherwise. Anonymous (unnamed) runs are not recorded.
	"""

	def __init__(self, base: Path | None = None) -> None:
		self._base: Final = base

	async def setup(self, task: TaskNode) -> _Context:
		return _Context(
			task=task.name,
			base=self._base if self._base is not None else Path.cwd(),
			start_mono=time.perf_counter(),
			latest=_Latest(states=tuple(Waiting(info.task) for info in flatten_leaves(task))),
		)

	async def on_event(
		self, event: TaskEvent, states: Sequence[LeafState], ctx: _Context
	) -> _Context:
		ctx.latest.states = states
		return ctx

	async def teardown(self, ctxs: tuple[_Context, ...]) -> None:
		ctx: Final = ctxs[0]  # zuban: ignore[misc] # zuban defies PEP591
		if ctx.task is None:
			return
		leaves = [
			(_leaf_name(state.task), elapsed)
			for state in ctx.latest.states
			if isinstance(state, Completed)
			and (elapsed := timings.elapsed_of(state.completion)) is not None
		]
		timings.record(ctx.base, ctx.task, time.perf_counter() - ctx.start_mono, leaves)


def _leaf_name(task: Task) -> str:
	"""The leaf's name, or its command when anonymous."""
	if task.name is not None:
		return task.name
	return task.cmd if isinstance(task.cmd, str) else " ".join(task.cmd)
