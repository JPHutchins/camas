# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

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
	states: Sequence[LeafState]


class TimingsContext(NamedTuple):
	camas_dir: Path
	scope: int
	identities: tuple[timings.CacheKey, ...] | None
	state: TimingsState


class Timings:
	"""Each leaf's observed duration, recorded for later runs."""

	camas_dir: Final[Path]
	"""Where the timing cache is."""
	scope: Final[int]
	"""The run's scope, recorded with each duration."""
	identities: Final[tuple[timings.CacheKey, ...] | None]
	"""One cache key per leaf, in leaf order, in place of the keys derived from labels."""

	def __init__(
		self,
		camas_dir: Path,
		scope: int = 0,
		identities: tuple[timings.CacheKey, ...] | None = None,
	) -> None:
		timings.reject_non_tuple_identities(identities)
		self.camas_dir = camas_dir
		self.scope = scope
		self.identities = identities

	def for_run(
		self,
		scope: int,
		identities: tuple[timings.CacheKey, ...] | None = None,
	) -> "Timings":
		return Timings(self.camas_dir, scope, identities)

	async def setup(self, task: TaskNode) -> TimingsContext:
		leaves = flatten_leaves(task)
		if self.identities is not None and len(self.identities) != len(leaves):
			timings.raise_identities_mismatch(len(self.identities), len(leaves))
		return TimingsContext(
			camas_dir=self.camas_dir,
			scope=self.scope,
			identities=self.identities,
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
