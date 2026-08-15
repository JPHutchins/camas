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
	"""Immutable context: the run's camas directory, its scope, the carried identities, and its
	latest leaf states.
	"""

	camas_dir: Path
	scope: int
	identities: tuple[timings.CacheKey, ...] | None
	state: TimingsState


class Timings:
	"""Records each leaf's observed duration to ``<camas_dir>/timings.txt`` at teardown.

	``camas_dir`` is required and assumed to exist: the caller enables this effect only
	when the project's camas directory is present, so the effect always records.

	``scope`` is how many changed paths narrowed this run, keying each observation to the size of
	change it was measured on — see :class:`camas.core.timings.CacheKey`. ``identities`` are the
	per-leaf keys a scoped run reports under, computed from the same pre-rewrite tree. Neither is
	knowable at construction when this effect is written out by hand in ``--effects`` or a
	``Config``, so :func:`for_run` supplies them.
	"""

	def __init__(
		self,
		camas_dir: Path,
		scope: int = 0,
		identities: tuple[timings.CacheKey, ...] | None = None,
	) -> None:
		self._camas_dir: Final = camas_dir
		self._scope: Final = scope
		self._identities: Final = identities

	def for_run(
		self,
		scope: int,
		identities: tuple[timings.CacheKey, ...] | None = None,
	) -> "Timings":
		"""This effect keyed to one run, for a caller that knows what the run is scoped to."""
		return Timings(self._camas_dir, scope, identities)

	async def setup(self, task: TaskNode) -> TimingsContext:
		leaves = flatten_leaves(task)
		if self._identities is not None and len(self._identities) != len(leaves):
			raise ValueError(
				f"identities must be parallel to the run's leaves: "
				f"{len(self._identities)} keys for {len(leaves)} leaves"
			)
		return TimingsContext(
			camas_dir=self._camas_dir,
			scope=self._scope,
			identities=self._identities,
			state=TimingsState(tuple(Waiting(info.task) for info in leaves)),
		)

	async def on_event(
		self, event: TaskEvent, states: Sequence[LeafState], ctx: TimingsContext
	) -> TimingsContext:
		ctx.state.states = states
		return ctx

	async def teardown(self, ctxs: tuple[TimingsContext, ...]) -> None:
		ctx: Final = ctxs[0]  # zuban: ignore[misc] # zuban defies PEP591
		completed = [
			(idx, task_label(state.task), state.completion)
			for idx, state in enumerate(ctx.state.states)
			if isinstance(state, Completed)
		]
		leaves = (
			[
				(ctx.identities[idx], elapsed)
				for idx, _, completion in completed
				if (elapsed := timings.elapsed_of(completion)) is not None
			]
			if ctx.identities is not None
			else timings.observations(
				((label, completion) for _, label, completion in completed),
				ctx.scope,
			)
		)
		await asyncio.to_thread(timings.record_observed, ctx.camas_dir, leaves)
