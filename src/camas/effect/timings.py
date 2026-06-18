# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Effect: on teardown, record the run's per-leaf durations to ``<camas_dir>/timings.txt``."""

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
class TimingsState:
	"""Mutable slot holding the latest per-leaf states view for the Timings effect."""

	states: Sequence[LeafState]


class TimingsContext(NamedTuple):
	"""Immutable context: the run's camas directory and its latest per-leaf states."""

	camas_dir: Path
	state: TimingsState


class Timings:
	"""Records each leaf's observed duration to ``<camas_dir>/timings.txt`` at teardown.

	``camas_dir`` is required and assumed to exist: the caller enables this effect only
	when the project's camas directory is present, so the effect always records.
	"""

	def __init__(self, camas_dir: Path) -> None:
		self._camas_dir: Final = camas_dir

	async def setup(self, task: TaskNode) -> TimingsContext:
		return TimingsContext(
			camas_dir=self._camas_dir,
			state=TimingsState(tuple(Waiting(info.task) for info in flatten_leaves(task))),
		)

	async def on_event(
		self, event: TaskEvent, states: Sequence[LeafState], ctx: TimingsContext
	) -> TimingsContext:
		ctx.state.states = states
		return ctx

	async def teardown(self, ctxs: tuple[TimingsContext, ...]) -> None:
		ctx: Final = ctxs[0]  # zuban: ignore[misc] # zuban defies PEP591
		leaves = [
			(task_label(state.task), elapsed)
			for state in ctx.state.states
			if isinstance(state, Completed)
			and (elapsed := timings.elapsed_of(state.completion)) is not None
		]
		await asyncio.to_thread(timings.record, ctx.camas_dir, leaves)
