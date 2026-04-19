# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, NamedTuple

from camas import LeafState, TaskEvent, TaskNode, Waiting, flatten_leaves
from camas.effect.termtree import (
	STATUS_COL_WIDTH,
	DisplayRow,
	flatten_rows,
	print_failures,
	render_lines,
)


class SummaryOptions(NamedTuple):
	"""Configuration for the summary Effect.

	>>> SummaryOptions()
	SummaryOptions()
	"""


@dataclass
class SummaryState:
	"""Mutable slot holding the latest states view for the Summary effect."""

	states: Sequence[LeafState]


class SummaryContext(NamedTuple):
	"""Immutable context threaded through the summary Effect's lifecycle."""

	rows: tuple[DisplayRow, ...]
	term_width: int
	display_width: int
	wall_start: float
	state: SummaryState


class Summary:
	"""Post-run Effect: renders the final tree once at teardown. No animation.

	Intended for CI and other non-interactive terminals where the live
	cursor-repositioning used by ``Termtree`` either doesn't render or
	produces garbage. End-state output matches ``Termtree``'s final frame.
	"""

	def __init__(self, options: SummaryOptions) -> None:
		self.options: Final = options

	async def setup(self, task: TaskNode) -> SummaryContext:
		term_width: Final = (  # zuban: ignore[misc] # zuban defies PEP591
			shutil.get_terminal_size().columns
		)
		return SummaryContext(
			rows=flatten_rows(task),
			term_width=term_width,
			display_width=term_width - STATUS_COL_WIDTH - 1,
			wall_start=time.perf_counter(),
			state=SummaryState(
				states=tuple(Waiting(info.task) for info in flatten_leaves(task)),
			),
		)

	async def on_event(
		self, event: TaskEvent, states: Sequence[LeafState], ctx: SummaryContext
	) -> SummaryContext:
		ctx.state.states = states
		return ctx

	async def teardown(self, ctxs: tuple[SummaryContext, ...]) -> None:
		ctx: Final = ctxs[0]  # zuban: ignore[misc] # zuban defies PEP591
		lines: Final = render_lines(  # zuban: ignore[misc] # zuban defies PEP591
			ctx.rows,
			ctx.state.states,
			ctx.term_width,
			ctx.display_width,
			time.perf_counter(),
			ctx.wall_start,
		)
		sys.stdout.write("\n".join(lines) + "\n")
		sys.stdout.flush()
		print_failures(ctx.state.states)
