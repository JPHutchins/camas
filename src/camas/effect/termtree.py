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

from ..core.color import (
	BOLD,
	BOLD_YELLOW,
	CAMAS_VIOLET,
	CYAN,
	DARK_RED,
	GREEN,
	GREY,
	RED,
	RESET,
	YELLOW,
)
from ..core.leaf_state import KILL_PRESSES, LeafInfo
from ..core.render import (
	DisplayRow,
	GroupHeader,
	flatten_rows,
	render_tree_prefix,
	strip_ansi,
	take_cols,
	visual_width,
)
from ..core.task import task_label
from ..core.traversal import flatten_leaves
from ..v0.completion import Errored, Finished, Skipped, Stopped
from ..v0.leaf_state import Completed, Interrupting, LeafState, Running, Waiting
from ..v0.task import Task, TaskNode
from ..v0.task_event import TaskEvent


def truncate_middle(text: str, max_width: int) -> str:
	"""Middle-truncate with '...' so the result's :func:`visual_width` is ``<= max_width``.

	Budgets terminal columns, not code points, so a wide glyph (CJK, emoji) can't push the
	result past ``max_width`` and wrap the row (issue #64).

	>>> truncate_middle("hello world!", 9)
	'hel...ld!'
	>>> truncate_middle("built", 2)
	'..'
	>>> truncate_middle("你好世界你好", 7)
	'你...好'
	"""
	if visual_width(text) <= max_width:
		return text
	if max_width < 3:
		return "..."[: max(max_width, 0)]
	left: Final = (max_width - 3) // 2
	return take_cols(text, left) + "..." + take_cols(text, max_width - 3 - left, from_end=True)


def fit_label(text: str, max_width: int) -> str:
	"""Return ``text`` unchanged if it fits; otherwise middle-truncate to ``max_width``."""
	return text if visual_width(text) <= max_width else truncate_middle(text, max(max_width, 3))


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


def interrupt_status(presses: int) -> str:
	"""Colored 6-col status for an interrupting row, by Ctrl-C count.

	>>> interrupt_status(1) == f"{YELLOW}  ^C  "
	True
	>>> interrupt_status(2) == f"{YELLOW} ^C^C "
	True
	>>> interrupt_status(3) == f"{BOLD_YELLOW} KILL "
	True
	"""
	if presses == 1:
		return f"{YELLOW}  ^C  "
	if presses < KILL_PRESSES:
		return f"{YELLOW} ^C^C "
	return f"{BOLD_YELLOW} KILL "


def bucket_glyph_color(states: Sequence[LeafState]) -> tuple[str, str]:
	"""Reduce a bucket of leaf states to (glyph, ANSI color) for a progress cell.

	Worst-progress wins (any waiting → blank; else any running → dashed; else
	solid), and color reflects the settled outcome (red on any failure, grey if
	all skipped, else green).

	>>> t = Task("x")
	>>> t0 = datetime(2026, 1, 1)
	>>> bucket_glyph_color([Waiting(t)]) == (' ', GREY)
	True
	>>> bucket_glyph_color([Running(t, t0, b"")]) == ('┄', CAMAS_VIOLET)
	True
	>>> bucket_glyph_color([Completed(t, Finished(0, 0.1, ()))]) == ('─', GREEN)
	True
	>>> bucket_glyph_color([Completed(t, Finished(1, 0.1, ()))]) == ('─', RED)
	True
	>>> bucket_glyph_color([Completed(t, Skipped(1))]) == ('─', GREY)
	True
	>>> bucket_glyph_color([Waiting(t), Running(t, t0, b"")]) == (' ', GREY)
	True
	>>> bucket_glyph_color([Interrupting(t, t0, b"", 1)]) == ('┄', YELLOW)
	True
	>>> bucket_glyph_color([Completed(t, Stopped(130, 0.1, ()))]) == ('─', YELLOW)
	True
	>>> bucket_glyph_color([Completed(t, Errored(127, "no such file or directory: x"))]) == ('─', RED)
	True
	"""
	if any(isinstance(s, Waiting) for s in states):
		return PROG_WAITING, GREY
	if any(isinstance(s, Interrupting) for s in states):
		return PROG_RUNNING, YELLOW
	if any(isinstance(s, Running) for s in states):
		return PROG_RUNNING, CAMAS_VIOLET
	if any(
		isinstance(s, Completed)
		and (
			(isinstance(s.completion, Finished) and s.completion.returncode != 0)
			or isinstance(s.completion, Errored)
		)
		for s in states
	):
		return PROG_DONE, RED
	if any(isinstance(s, Completed) and isinstance(s.completion, Stopped) for s in states):
		return PROG_DONE, YELLOW
	all_skipped: Final = all(
		isinstance(s, Completed) and isinstance(s.completion, Skipped) for s in states
	)
	return PROG_DONE, GREY if all_skipped else GREEN


def render_progress_bar(states: Sequence[LeafState], width: int) -> str:
	"""Render a fragmented progress bar of visible width ``width`` for the summary slot.

	Cells are equal width (capped at ``PROG_MAX_CELL_WIDTH``) and centered; when
	``len(states)`` exceeds the usable span, leaves are bucketed into 1-column cells.

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
	"""Decode a captured output line, stripped of trailing whitespace."""
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
				name = fit_label(task_label(task), max(display_width - visual_width(prefix), 0))
				state = states[leaf_idx]
				leaf_idx += 1
				gap = max(display_width - visual_width(prefix) - visual_width(name), 0)
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
								padding = " " * max(gap - visual_width(stream), 0)
								lines.append(
									f"\r{header}{GREY}{stream}{RESET}{padding} {CYAN}|{color}{status}{CYAN}|{RESET} {elapsed:7.3f}s{CLEAR_LINE}"
								)
							case Skipped():
								padding = " " * gap
								lines.append(
									f"\r{header}{padding} {CYAN}|{GREY} SKIP {CYAN}|{RESET}{CLEAR_LINE}"
								)
							case Stopped(elapsed=elapsed, output=output):
								stream_line = strip_ansi(last_line_display(output))
								stream = (
									f"  {truncate_middle(stream_line, gap - 2)}"
									if gap > 2 and stream_line
									else ""
								)
								padding = " " * max(gap - visual_width(stream), 0)
								lines.append(
									f"\r{header}{GREY}{stream}{RESET}{padding} {CYAN}|{DARK_RED} STOP {CYAN}|{RESET} {elapsed:7.3f}s{CLEAR_LINE}"
								)
							case Errored(message=message):
								stream = f"  {truncate_middle(message, gap - 2)}" if gap > 2 else ""
								padding = " " * max(gap - visual_width(stream), 0)
								lines.append(
									f"\r{header}{GREY}{stream}{RESET}{padding} {CYAN}|{RED} ERROR{CYAN}|{RESET}{CLEAR_LINE}"
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
						padding = " " * max(gap - visual_width(stream), 0)
						lines.append(
							f"\r{header}{GREY}{stream}{RESET}{padding} {CYAN}|{CAMAS_VIOLET}{spin}{CYAN}|{RESET} {elapsed:7.3f}s{CLEAR_LINE}"
						)
					case Interrupting(start_time=start_time, last_line=last_line, presses=presses):
						elapsed = (now - start_time).total_seconds()
						stream_line = strip_ansi(decode_line(last_line)) if last_line else ""
						stream = (
							f"  {truncate_middle(stream_line, gap - 2)}"
							if gap > 2 and stream_line
							else ""
						)
						padding = " " * max(gap - visual_width(stream), 0)
						lines.append(
							f"\r{header}{GREY}{stream}{RESET}{padding} {CYAN}|{interrupt_status(presses)}{CYAN}|{RESET} {elapsed:7.3f}s{CLEAR_LINE}"
						)
					case Waiting():
						padding = " " * gap
						lines.append(
							f"\r{header}{padding} {CYAN}|{GREY} WAIT {CYAN}|{RESET}{CLEAR_LINE}"
						)
	alldone: Final = all(isinstance(s, Completed) for s in states)
	summary_pad: Final = render_progress_bar(states, max(display_width - 6, 0))
	if alldone:
		stopped: Final = any(
			isinstance(s, Completed) and isinstance(s.completion, Stopped) for s in states
		)
		failed: Final = any(
			isinstance(s, Completed)
			and (
				(isinstance(s.completion, Finished) and s.completion.returncode != 0)
				or isinstance(s.completion, Errored)
			)
			for s in states
		)
		summary_color: Final = DARK_RED if stopped else (RED if failed else GREEN)
		summary_label: Final = " STOP " if stopped else (" FAIL " if failed else " PASS ")
		lines.append(
			f"\r{BOLD}result{RESET}{summary_pad} {CYAN}|{summary_color}{summary_label}{CYAN}|{RESET} {wall_elapsed:7.3f}s{CLEAR_LINE}"
		)
	elif interrupting := [s.presses for s in states if isinstance(s, Interrupting)]:
		lines.append(
			f"\r{BOLD}result{RESET}{summary_pad} {CYAN}|{interrupt_status(max(interrupting))}{CYAN}|{RESET} {wall_elapsed:7.3f}s{CLEAR_LINE}"
		)
	else:
		running_spin: Final = SPINNER[int(wall_elapsed * 10) % len(SPINNER)]
		lines.append(
			f"\r{BOLD}result{RESET}{summary_pad} {CYAN}|{CAMAS_VIOLET}{running_spin}{CYAN}|{RESET} {wall_elapsed:7.3f}s{CLEAR_LINE}"
		)
	return lines


class FrameResult(NamedTuple):
	"""A positioned frame: the bytes to emit and the live-region height it occupies.

	>>> FrameResult("", 0)
	FrameResult(text='', visible=0)
	"""

	text: str
	visible: int


def clamp_lines(lines: tuple[str, ...], height: int) -> tuple[str, ...]:
	r"""Reduce a frame to at most ``height`` lines, always keeping the summary (last) line.

	The summary is the most information-dense line (aggregate progress + total
	elapsed), so it survives; the overflowing leaf rows above it collapse into one
	elision marker. Bounding the live region to the viewport is what stops the
	repaint from scrolling: cursor-up (``\033[NF``) clamps at the top of the
	screen, so a frame taller than the window can never be rewritten in place.

	>>> clamp_lines(("a", "b", "sum"), 5)
	('a', 'b', 'sum')
	>>> clamp_lines((), 5)
	()
	>>> clamp_lines(("sum",), 0)
	()
	>>> clamp_lines(("a", "b", "sum"), 1)
	('sum',)
	>>> over = tuple(str(i) for i in range(10)) + ("sum",)
	>>> result = clamp_lines(over, 4)
	>>> (len(result), result[0], result[-1])
	(4, '0', 'sum')
	"""
	if height <= 0:
		return ()
	if len(lines) <= height:
		return lines
	if height == 1:
		return (lines[-1],)
	visible_rows: Final = height - 2
	hidden: Final = len(lines) - 1 - visible_rows
	marker: Final = f"\r{GREY}... {hidden} more{CLEAR_LINE}{RESET}"
	return (*lines[:visible_rows], marker, lines[-1])


def render_frame(
	rows: tuple[DisplayRow, ...],
	states: Sequence[LeafState],
	term_width: int,
	term_height: int,
	now: datetime,
	wall_elapsed: float,
	prev_visible: int,
	*,
	final: bool = False,
) -> FrameResult:
	r"""Render one positioned frame for live animation, clamped to ``term_height``.

	Repaints in place by moving the cursor up ``prev_visible - 1`` lines (CPL,
	``\033[NF``) to the top of the previously-drawn region, then rewriting it.
	``prev_visible`` is the height the *previous* frame occupied (0 on the first
	frame, which just writes where the cursor is and lets the terminal scroll to
	make room) — tracking that, not the current frame's height, keeps the cursor
	math right when the line count changes across a resize, and lets a now-shorter
	frame erase the rows the previous one left behind (``\033[0J``).

	``final`` skips the clamp to emit the whole tree once at teardown so the
	completed run persists in the scrollback.
	"""
	full: Final = tuple(
		render_lines(rows, states, term_width, term_width - STATUS_COL_WIDTH - 1, now, wall_elapsed)
	)
	lines: Final = full if final else clamp_lines(full, term_height)
	reposition: Final = f"\033[{prev_visible - 1}F" if prev_visible > 1 else ""
	erase: Final = "\033[0J" if len(lines) < prev_visible else ""
	return FrameResult(reposition + "\n".join(lines) + erase, len(lines))


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
			case Completed(task=task, completion=Errored(returncode=rc, message=message)):
				print_task_output(task, (message.encode(),), "FAILED", RED, rc, term_width)
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


def print_stops(states: Sequence[LeafState], term_width: int) -> None:
	"""Print captured output for each task stopped by a forwarded signal."""
	for state in states:
		match state:
			case Completed(task=task, completion=Stopped(returncode=rc, output=output)):
				print_task_output(task, output, "STOPPED", DARK_RED, rc, term_width)
			case _:
				pass


@dataclass
class TermtreeState:
	"""Mutable slots of a termtree run: the latest states view and the tick task handle."""

	states: Sequence[LeafState]
	tick_task: asyncio.Task[None] | None = None
	visible: int = 0
	"""Lines the previous frame occupied; :func:`draw` reads it to reposition, then writes it back."""


class TermtreeContext(NamedTuple):
	"""Immutable context threaded through the termtree Effect's lifecycle.

	``wall_start`` is the wall-clock setup time (for human-facing timestamps if
	ever surfaced); ``wall_start_mono`` is the corresponding monotonic reading
	(``time.perf_counter()``) used to compute the displayed elapsed total so
	the TUI never sees a backward/forward NTP step. Terminal dimensions are not
	cached here — :func:`draw` re-reads them every frame so the view tracks resizes.
	"""

	rows: tuple[DisplayRow, ...]
	wall_start: datetime
	wall_start_mono: float
	state: TermtreeState


class Termtree:
	"""Live terminal Effect: renders each leaf's status and a summary line.

	The background tick task is the sole renderer during a run, so the
	effective frame rate is capped at 1 / frame_interval_ms regardless of
	how fast events arrive.
	"""

	def __init__(
		self,
		frame_interval_ms: float = 16.667,
		show_passing: bool = False,
		output_ctrl_c: bool = False,
	) -> None:
		self._frame_interval_ms: Final = frame_interval_ms
		self._show_passing: Final = show_passing
		self._output_ctrl_c: Final = output_ctrl_c

	async def setup(self, task: TaskNode) -> TermtreeContext:
		ctx: Final = TermtreeContext(  # zuban: ignore[misc] # zuban defies PEP591
			rows=flatten_rows(task),
			wall_start=datetime.now(),
			wall_start_mono=time.perf_counter(),
			state=TermtreeState(
				states=tuple(Waiting(info.task) for info in flatten_leaves(task)),
			),
		)
		draw(ctx)
		ctx.state.tick_task = asyncio.create_task(tick_loop(ctx, self._frame_interval_ms / 1000))
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
		draw(ctx, final=True)
		sys.stdout.write("\n")
		sys.stdout.flush()
		term_width: Final = (  # zuban: ignore[misc] # zuban defies PEP591
			shutil.get_terminal_size().columns
		)
		print_failures(ctx.state.states, term_width)
		if self._output_ctrl_c:
			print_stops(ctx.state.states, term_width)
		if self._show_passing:
			print_passes(ctx.state.states, term_width)


def draw(ctx: TermtreeContext, *, final: bool = False) -> None:
	"""Render and emit one frame, re-reading the terminal size so the view tracks resizes.

	``final`` emits the whole tree (no height clamp) once the live region stops, so
	the completed tree persists in the scrollback.
	"""
	# Write bytes directly: the box-drawing chars can't encode to cp1252 (Windows
	# default in non-TTY contexts like captured subprocesses / piped CI logs).
	size: Final = shutil.get_terminal_size()
	frame: Final = render_frame(
		ctx.rows,
		ctx.state.states,
		size.columns,
		size.lines,
		datetime.now(),
		time.perf_counter() - ctx.wall_start_mono,
		ctx.state.visible,
		final=final,
	)
	sys.stdout.buffer.write(frame.text.encode("utf-8", errors="replace"))
	sys.stdout.flush()
	ctx.state.visible = frame.visible


async def tick_loop(ctx: TermtreeContext, interval: float) -> None:
	while True:
		await asyncio.sleep(interval)
		draw(ctx)
