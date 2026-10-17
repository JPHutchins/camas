# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

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
	"""Terminal width detected from the environment.

	>>> Auto()
	Auto()
	"""


class Fixed(NamedTuple):
	"""A fixed terminal width.

	>>> Fixed(200)
	Fixed(columns=200)
	"""

	columns: int


TermWidth: TypeAlias = Auto | Fixed


@dataclass
class SummaryState:
	states: Sequence[LeafState]


class SummaryContext(NamedTuple):
	rows: tuple[DisplayRow, ...]
	term_width: int
	display_width: int
	wall_start: datetime
	wall_start_mono: float
	state: SummaryState


class Summary:
	"""The final task tree, rendered once after the run."""

	def __init__(self, term_width: TermWidth = Auto(), show_passing: bool = False) -> None:
		self._term_width: Final = term_width
		self._show_passing: Final = show_passing

	async def setup(self, task: TaskNode) -> SummaryContext:
		match self._term_width:
			case Auto():
				term_width = shutil.get_terminal_size().columns
			case Fixed(columns=columns):
				term_width = columns
			case _:
				assert_never(self._term_width)
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
		if self._show_passing:
			print_passes(ctx.state.states, ctx.term_width)
