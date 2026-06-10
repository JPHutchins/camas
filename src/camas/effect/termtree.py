# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Effect: live terminal tree that repaints leaf states in place each frame."""

import asyncio
import contextlib
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, NamedTuple

from ..core.leaf_state import LeafInfo
from ..core.render import (
	BOLD,
	CYAN,
	GREY,
	RESET,
	DisplayRow,
	GroupHeader,
	color_on,
	flatten_rows,
	render_tree_lines,
	render_tree_prefix,
	strip_ansi,
)
from ..core.task import task_label
from ..core.traversal import flatten_leaves
from ..v0 import (
	Completed,
	Finished,
	LeafState,
	Running,
	Skipped,
	Task,
	TaskEvent,
	TaskNode,
	Waiting,
)


def truncate_middle(text: str, max_width: int) -> str:
	"""Truncate text in the middle with '...' if it exceeds max_width.

	Always returns a string of length min(len(text), max(max_width, 0)).

	>>> truncate_middle("hello", 10)
	'hello'
	>>> truncate_middle("hello world!", 9)
	'hel...ld!'
	>>> truncate_middle("built", 2)
	'..'
	>>> truncate_middle("built", 4)
	'...t'
	>>> truncate_middle("built", 0)
	''
	"""
	if len(text) <= max_width:
		return text
	if max_width < 3:
		return "..."[: max(max_width, 0)]
	side: Final = (max_width - 3) // 2
	return text[:side] + "..." + text[len(text) - (max_width - 3 - side) :]


def fit_label(text: str, max_width: int) -> str:
	"""Return ``text`` unchanged if it fits; otherwise middle-truncate to ``max_width``.

	>>> fit_label("uv run ruff check .", 40)
	'uv run ruff check .'
	>>> fit_label("uv run ruff format --check .", 15)
	'uv run...heck .'
	"""
	return text if len(text) <= max_width else truncate_middle(text, max(max_width, 3))


class TermtreeOptions(NamedTuple):
	"""Configuration for the termtree Effect.

	>>> TermtreeOptions()
	TermtreeOptions(frame_interval_ms=16.667, show_passing=False)
	>>> TermtreeOptions(frame_interval_ms=50).frame_interval_ms
	50
	>>> TermtreeOptions(show_passing=True).show_passing
	True
	"""

	frame_interval_ms: float = 16.667
	show_passing: bool = False


GREEN: Final = "\033[32m"
YELLOW: Final = "\033[33m"
RED: Final = "\033[31m"
CLEAR_LINE: Final = "\033[K"
SPINNER: Final = (
	" ▌    ",
	" ▀    ",
	"  ▀   ",
	"   ▀  ",
	"    ▀ ",
	"    ▐ ",
	"    ▄ ",
	"   ▄  ",
	"  ▄   ",
	" ▄    ",
)

STATUS_COL_WIDTH: Final = 18

PROG_WAITING: Final = " "
PROG_RUNNING: Final = "┄"
PROG_DONE: Final = "─"
PROG_MAX_CELL_WIDTH: Final = 12
PROG_MIN_MARGIN: Final = 2


def bucket_glyph_color(states: Sequence[LeafState]) -> tuple[str, str]:
	"""Reduce a bucket of leaf states to (glyph, ANSI color) for a progress cell.

	Worst-progress wins so the bar never overstates progress: any waiting →
	blank; else any running → dashed line; else a solid line. Color
	distinguishes the active state (cyan running) from the settled outcome
	(red on any failure, grey if all skipped, otherwise green).

	>>> t = Task("x")
	>>> t0 = datetime(2026, 1, 1)
	>>> bucket_glyph_color([Waiting(t)]) == (' ', GREY)
	True
	>>> bucket_glyph_color([Running(t, t0, b"")]) == ('┄', CYAN)
	True
	>>> bucket_glyph_color([Completed(t, Finished(0, 0.1, ()))]) == ('─', GREEN)
	True
	>>> bucket_glyph_color([Completed(t, Finished(1, 0.1, ()))]) == ('─', RED)
	True
	>>> bucket_glyph_color([Completed(t, Skipped(1))]) == ('─', GREY)
	True
	>>> bucket_glyph_color([Waiting(t), Running(t, t0, b"")]) == (' ', GREY)
	True
	"""
	if any(isinstance(s, Waiting) for s in states):
		return PROG_WAITING, GREY
	if any(isinstance(s, Running) for s in states):
		return PROG_RUNNING, CYAN
	if any(
		isinstance(s, Completed)
		and isinstance(s.completion, Finished)
		and s.completion.returncode != 0
		for s in states
	):
		return PROG_DONE, RED
	all_skipped: Final = all(
		isinstance(s, Completed) and isinstance(s.completion, Skipped) for s in states
	)
	return PROG_DONE, GREY if all_skipped else GREEN


def render_progress_bar(states: Sequence[LeafState], width: int) -> str:
	"""Render a fragmented progress bar for the result-line summary slot.

	Cells are equal width (capped at ``PROG_MAX_CELL_WIDTH``) and centered in
	``width`` visible columns. When ``len(states)`` exceeds ``width``, leaves
	are bucketed across exactly ``width`` 1-column cells. Returned string
	always has visible width ``width``.

	>>> t = Task("x")
	>>> t0 = datetime(2026, 1, 1)
	>>> bar = render_progress_bar([Waiting(t), Waiting(t)], 10)
	>>> len(strip_ansi(bar))
	10
	>>> strip_ansi(bar)
	'          '
	>>> bar = render_progress_bar([Running(t, t0, b"")], 10)
	>>> strip_ansi(bar)
	'  ┄┄┄┄┄┄  '
	>>> render_progress_bar([], 5)
	'     '
	>>> render_progress_bar([Waiting(t)], 0)
	''
	>>> '─' in render_progress_bar([Completed(t, Finished(0, 0.1, ()))], 8)
	True
	>>> '─' in render_progress_bar(
	...     [Completed(t, Finished(0, 0.1, ())), Completed(t, Finished(1, 0.1, ()))], 12
	... )
	True
	>>> render_progress_bar([Waiting(t)], 3)
	'   '
	>>> strip_ansi(render_progress_bar([Running(t, t0, b"")] * 10, 10))
	'  ┄┄┄┄┄┄  '
	"""
	if width <= 0 or len(states) == 0:
		return " " * max(width, 0)
	usable: Final = max(width - 2 * PROG_MIN_MARGIN, 0)
	if usable == 0:
		return " " * width
	if len(states) <= usable:
		cell_width = min(PROG_MAX_CELL_WIDTH, usable // len(states))
		buckets: tuple[Sequence[LeafState], ...] = tuple((s,) for s in states)
	else:
		cell_width = 1
		buckets = tuple(
			tuple(states[i * len(states) // usable : (i + 1) * len(states) // usable])
			for i in range(usable)
		)
	pad: Final = width - len(buckets) * cell_width
	pad_left: Final = pad // 2
	pad_right: Final = pad - pad_left
	cells: Final = "".join(
		f"{color}{glyph * cell_width}{RESET}"
		for glyph, color in (bucket_glyph_color(b) for b in buckets)
	)
	return f"{' ' * pad_left}{cells}{' ' * pad_right}"


def decode_line(line: bytes) -> str:
	r"""Decode a captured output line (stripped of trailing whitespace).

	>>> decode_line(b"hello\n")
	'hello'
	>>> decode_line(b"")
	''
	"""
	return line.rstrip().decode(errors="replace")


def last_line_display(output: Sequence[bytes]) -> str:
	r"""Decode the final non-empty output line for inline display.

	>>> last_line_display([b"first\n", b"last\n"])
	'last'
	>>> last_line_display([])
	''
	>>> last_line_display([b"\n", b"  \n"])
	''
	"""
	for line in reversed(output):
		if line.rstrip():
			return decode_line(line)
	return ""


def render_lines(
	rows: tuple[DisplayRow, ...],
	states: Sequence[LeafState],
	term_width: int,
	display_width: int,
	now: datetime,
	wall_elapsed: float,
) -> list[str]:
	"""Build the list of ANSI-formatted lines for one frame (leaves + summary).

	``wall_elapsed`` is supplied as monotonic seconds (``time.perf_counter()``
	delta) by the caller so the displayed total time is immune to wall-clock
	jumps (NTP, DST); ``now`` is still a wall-clock ``datetime`` because the
	per-leaf running counter is derived from ``Running.start_time``.
	"""
	lines: list[str] = []
	leaf_idx = 0
	for row in rows:
		prefix = render_tree_prefix(row.depth, row.is_last_chain)
		match row:
			case GroupHeader(label=label):
				lines.append(
					f"\r{GREY}{truncate_middle(f'{prefix}{label}', term_width - 1)}{CLEAR_LINE}{RESET}"
				)
			case LeafInfo(task=task):
				name = fit_label(task_label(task), max(display_width - len(prefix), 0))
				state = states[leaf_idx]
				leaf_idx += 1
				gap = max(display_width - len(prefix) - len(name), 0)
				header = f"{GREY}{prefix}{RESET}{BOLD}{name}{RESET}"
				match state:
					case Completed(completion=completion):
						match completion:
							case Finished(returncode=rc, elapsed=elapsed, output=output):
								color = GREEN if rc == 0 else RED
								status = " PASS " if rc == 0 else " FAIL "
								stream_line = strip_ansi(last_line_display(output))
								stream = (
									f"  {truncate_middle(stream_line, gap - 2)}"
									if gap > 2 and stream_line
									else ""
								)
								padding = " " * max(gap - len(stream), 0)
								lines.append(
									f"\r{header}{GREY}{stream}{RESET}{padding} [{color}{status}{RESET}] {elapsed:7.3f}s{CLEAR_LINE}"
								)
							case Skipped():  # pragma: no branch
								padding = " " * gap
								lines.append(
									f"\r{header}{padding} [{GREY} SKIP {RESET}]{CLEAR_LINE}"
								)
					case Running(start_time=start_time, last_line=last_line):
						elapsed = (now - start_time).total_seconds()
						spin = SPINNER[int(elapsed * 10) % len(SPINNER)]
						stream_line = strip_ansi(decode_line(last_line)) if last_line else ""
						stream = (
							f"  {truncate_middle(stream_line, gap - 2)}"
							if gap > 2 and stream_line
							else ""
						)
						padding = " " * max(gap - len(stream), 0)
						lines.append(
							f"\r{header}{GREY}{stream}{RESET}{padding} [{CYAN}{spin}{RESET}] {elapsed:7.3f}s{CLEAR_LINE}"
						)
					case Waiting():
						padding = " " * gap
						lines.append(f"\r{header}{padding} [{GREY} WAIT {RESET}]{CLEAR_LINE}")
	alldone: Final = all(isinstance(s, Completed) for s in states)
	summary_pad: Final = render_progress_bar(states, max(display_width - 6, 0))
	if alldone:
		failed: Final = any(
			isinstance(s, Completed)
			and isinstance(s.completion, Finished)
			and s.completion.returncode != 0
			for s in states
		)
		summary_color: Final = RED if failed else GREEN
		summary_label: Final = " FAIL " if failed else " PASS "
		lines.append(
			f"\r{BOLD}result{RESET}{summary_pad} [{summary_color}{summary_label}{RESET}] {wall_elapsed:7.3f}s{CLEAR_LINE}"
		)
	else:
		running_spin: Final = SPINNER[int(wall_elapsed * 10) % len(SPINNER)]
		lines.append(
			f"\r{BOLD}result{RESET}{summary_pad} [{CYAN}{running_spin}{RESET}] {wall_elapsed:7.3f}s{CLEAR_LINE}"
		)
	return lines


def render_frame(
	rows: tuple[DisplayRow, ...],
	states: Sequence[LeafState],
	term_width: int,
	display_width: int,
	now: datetime,
	wall_elapsed: float,
) -> str:
	r"""Render one positioned frame as an ANSI string, for live animation.

	Starts with CPL (`\033[NF`) to move the cursor up N lines to column 1,
	then writes the frame. Callers must first reserve N+1 rows of vertical
	space (via `"\n" * N`) so the frame has room to render inline without
	scrolling the viewport.
	"""
	lines: Final = render_lines(rows, states, term_width, display_width, now, wall_elapsed)
	return f"\033[{len(lines) - 1}F" + "\n".join(lines)


def print_task_output(
	task: Task,
	output: Sequence[bytes],
	label: str,
	color: str,
	returncode: int,
	term_width: int,
) -> None:
	rule: Final = "=" * term_width
	sys.stdout.write(f"\n{color}{rule}{RESET}\n")
	sys.stdout.write(f"{color}{BOLD} {label}: {task_label(task)} {RESET}\n")
	sys.stdout.write(f" {task.cmd if isinstance(task.cmd, str) else ' '.join(task.cmd)}\n")
	sys.stdout.write(f" exit code: {returncode}\n")
	sys.stdout.write(f"{color}{rule}{RESET}\n")
	sys.stdout.flush()
	for line in output:
		sys.stdout.buffer.write(line)
	sys.stdout.buffer.flush()
	sys.stdout.write("\n")


def print_failures(states: Sequence[LeafState], term_width: int) -> None:
	"""Print detailed output for each failed task."""
	for state in states:
		match state:
			case Completed(task=task, completion=Finished(returncode=rc, output=output)) if rc > 0:
				print_task_output(task, output, "FAILED", RED, rc, term_width)
			case _:
				pass


def print_passes(states: Sequence[LeafState], term_width: int) -> None:
	"""Print detailed output for each successfully-completed task."""
	for state in states:
		match state:
			case Completed(task=task, completion=Finished(returncode=0, output=output)):
				print_task_output(task, output, "PASSED", GREEN, 0, term_width)
			case _:
				pass


def print_tree(task: TaskNode, show_cmd: bool = False) -> None:
	"""Print the task tree structure to stdout without executing.

	When ``show_cmd`` is True, leaf tasks with a distinct name show ``name: cmd``;
	env entries are shown only at the deepest ancestor that introduces them,
	so matrix expansions annotate their group header and leaves stay clean.
	ANSI colors are emitted when stdout is a TTY and NO_COLOR is unset.

	>>> print_tree(Task("echo hi"))
	echo hi
	>>> print_tree(Task("echo hi", name="greet"), show_cmd=True)
	greet: echo hi
	"""
	for line in render_tree_lines(task, show_cmd=show_cmd, color=color_on()):
		print(line)


@dataclass
class TermtreeState:
	"""Mutable slots of a termtree run: the latest states view and the tick task handle."""

	states: Sequence[LeafState]
	tick_task: asyncio.Task[None] | None = None


class TermtreeContext(NamedTuple):
	"""Immutable context threaded through the termtree Effect's lifecycle.

	``wall_start`` is the wall-clock setup time (for human-facing timestamps if
	ever surfaced); ``wall_start_mono`` is the corresponding monotonic reading
	(``time.perf_counter()``) used to compute the displayed elapsed total so
	the TUI never sees a backward/forward NTP step.
	"""

	rows: tuple[DisplayRow, ...]
	term_width: int
	display_width: int
	wall_start: datetime
	wall_start_mono: float
	state: TermtreeState


class Termtree:
	"""Live terminal Effect: renders each leaf's status and a summary line.

	The background tick task is the sole renderer during a run, so the
	effective frame rate is capped at 1 / frame_interval_ms regardless of
	how fast events arrive.
	"""

	def __init__(self, options: TermtreeOptions = TermtreeOptions()) -> None:
		self.options: Final = options

	async def setup(self, task: TaskNode) -> TermtreeContext:
		term_width: Final = (  # zuban: ignore[misc] # zuban defies PEP591
			shutil.get_terminal_size().columns
		)
		rows: Final = flatten_rows(task)  # zuban: ignore[misc] # zuban defies PEP591
		ctx: Final = TermtreeContext(  # zuban: ignore[misc] # zuban defies PEP591
			rows=rows,
			term_width=term_width,
			display_width=term_width - STATUS_COL_WIDTH - 1,
			wall_start=datetime.now(),
			wall_start_mono=time.perf_counter(),
			state=TermtreeState(
				states=tuple(Waiting(info.task) for info in flatten_leaves(task)),
			),
		)
		sys.stdout.write("\n" * len(rows))
		sys.stdout.flush()
		draw(ctx)
		ctx.state.tick_task = asyncio.create_task(
			tick_loop(ctx, self.options.frame_interval_ms / 1000)
		)
		return ctx

	async def on_event(
		self, event: TaskEvent, states: Sequence[LeafState], ctx: TermtreeContext
	) -> TermtreeContext:
		ctx.state.states = states
		return ctx

	async def teardown(self, ctxs: tuple[TermtreeContext, ...]) -> None:
		ctx: Final = ctxs[0]  # zuban: ignore[misc] # zuban defies PEP591
		if ctx.state.tick_task is not None:  # pragma: no branch
			ctx.state.tick_task.cancel()
			with contextlib.suppress(asyncio.CancelledError):
				await ctx.state.tick_task
		draw(ctx)
		sys.stdout.write("\n")
		sys.stdout.flush()
		print_failures(ctx.state.states, ctx.term_width)
		if self.options.show_passing:
			print_passes(ctx.state.states, ctx.term_width)


def draw(ctx: TermtreeContext) -> None:
	# Write bytes directly: the box-drawing chars can't encode to cp1252 (Windows
	# default in non-TTY contexts like captured subprocesses / piped CI logs).
	frame = render_frame(
		ctx.rows,
		ctx.state.states,
		ctx.term_width,
		ctx.display_width,
		datetime.now(),
		time.perf_counter() - ctx.wall_start_mono,
	)
	sys.stdout.buffer.write(frame.encode("utf-8", errors="replace"))
	sys.stdout.flush()


async def tick_loop(ctx: TermtreeContext, interval: float) -> None:
	while True:
		await asyncio.sleep(interval)
		draw(ctx)
