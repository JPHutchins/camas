# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Effect: on teardown, record the run's per-leaf durations to the ``.camas`` cache."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

from ..core import timings
from ..core.task import task_label
from ..core.traversal import flatten_leaves
from ..v0.leaf_state import Completed, LeafState, Waiting
from ..v0.task import TaskNode
from ..v0.task_event import TaskEvent


@dataclass
class _Latest:
	"""Mutable slot holding the most recent per-leaf states view."""

	states: Sequence[LeafState]


class _Context(NamedTuple):
	task: str | None
	base: Path
	latest: _Latest


class Timings:
	"""Records each task's observed duration to ``.camas/timings``."""

	def __init__(self, base: Path | None = None) -> None:
		self._base: Final = base

	async def setup(self, task: TaskNode) -> _Context:
		return _Context(
			task=task.name,
			base=self._base if self._base is not None else Path.cwd(),
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
			(task_label(state.task), elapsed)
			for state in ctx.latest.states
			if isinstance(state, Completed)
			and (elapsed := timings.elapsed_of(state.completion)) is not None
		]
		await asyncio.to_thread(timings.record, ctx.base, leaves)
