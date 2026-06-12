# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Effect: silent during the run, then one final tree render at teardown."""

import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, NamedTuple, TypeAlias

if sys.version_info >= (3, 11):
	from typing import assert_never
else:
	from typing_extensions import assert_never

from ..core.render import DisplayRow, flatten_rows
from ..core.traversal import flatten_leaves
from ..v0.leaf_state import LeafState, Waiting
from ..v0.task import TaskNode
from ..v0.task_event import TaskEvent
from .termtree import (
	CLEAR_LINE,
	STATUS_COL_WIDTH,
	print_failures,
	print_passes,
	render_lines,
)


class Auto(NamedTuple):
	"""Detect terminal width from the environment (``shutil.get_terminal_size``).

	>>> Auto()
	Auto()
	"""


class Fixed(NamedTuple):
	"""Fixed terminal width in columns, overriding environment detection.

	>>> Fixed(200)
	Fixed(columns=200)
	"""

	columns: int


TermWidth: TypeAlias = Auto | Fixed


class SummaryOptions(NamedTuple):
	"""Configuration for the summary Effect.

	>>> SummaryOptions()
	SummaryOptions(term_width=Auto(), show_passing=False)
	>>> SummaryOptions(term_width=Fixed(120)).term_width
	Fixed(columns=120)
	>>> SummaryOptions(show_passing=True).show_passing
	True
	"""

	term_width: TermWidth = Auto()
	show_passing: bool = False


@dataclass
class SummaryState:
	"""Mutable slot holding the latest states view for the Summary effect."""

	states: Sequence[LeafState]


class SummaryContext(NamedTuple):
	"""Immutable context threaded through the summary Effect's lifecycle.

	``wall_start`` is the wall-clock setup time (for human-facing timestamps if
	ever surfaced); ``wall_start_mono`` is the corresponding monotonic reading
	(``time.perf_counter()``) used to compute the displayed elapsed total so
	the summary line is immune to NTP/DST steps mid-run.
	"""

	rows: tuple[DisplayRow, ...]
	term_width: int
	display_width: int
	wall_start: datetime
	wall_start_mono: float
	state: SummaryState


class Summary:
	"""Post-run Effect: renders the final tree once at teardown. No animation.

	Intended for CI and other non-interactive terminals where the live
	cursor-repositioning used by ``Termtree`` either doesn't render or
	produces garbage. End-state output matches ``Termtree``'s final frame.
	"""

	def __init__(self, options: SummaryOptions = SummaryOptions()) -> None:
		self.options: Final = options

	async def setup(self, task: TaskNode) -> SummaryContext:
		match self.options.term_width:
			case Auto():
				term_width = shutil.get_terminal_size().columns
			case Fixed(columns=columns):
				term_width = columns
			case _:
				assert_never(self.options.term_width)
		return SummaryContext(
			rows=flatten_rows(task),
			term_width=term_width,
			display_width=term_width - STATUS_COL_WIDTH - 1,
			wall_start=datetime.now(),
			wall_start_mono=time.perf_counter(),
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
			datetime.now(),
			time.perf_counter() - ctx.wall_start_mono,
		)
		cleaned: Final = tuple(  # zuban: ignore[misc] # zuban defies PEP591
			line.removeprefix("\r").replace(CLEAR_LINE, "") for line in lines
		)
		sys.stdout.buffer.write(("\n".join(cleaned) + "\n").encode("utf-8", errors="replace"))
		sys.stdout.flush()
		print_failures(ctx.state.states, ctx.term_width)
		if self.options.show_passing:
			print_passes(ctx.state.states, ctx.term_width)
