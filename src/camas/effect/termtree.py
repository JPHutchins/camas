# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

import asyncio
import os
import re
import shutil
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple, TypeAlias

if sys.version_info >= (3, 11):
	from typing import assert_never
else:
	from typing_extensions import assert_never

from camas import (
	ChainLink,
	Completed,
	Finished,
	LeafInfo,
	LeafState,
	Parallel,
	Running,
	Sequential,
	Skipped,
	Task,
	TaskEvent,
	TaskNode,
	Waiting,
	expand_matrix,
	flatten_leaves,
	task_label,
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


class GroupHeader(NamedTuple):
	"""Display row for a Sequential or Parallel group header.

	>>> GroupHeader("ci", 0, ())
	GroupHeader(label='ci', depth=0, is_last_chain=())
	"""

	label: str
	depth: int
	is_last_chain: tuple[ChainLink, ...]


DisplayRow: TypeAlias = LeafInfo | GroupHeader


ANSI_ESCAPE: Final = re.compile(
	r"\x1b(?:"
	r"\[[0-?]*[ -/]*[@-~]"  # CSI sequences (colors, cursor movement, etc.)
	r"|\][^\x07]*\x07"  # OSC terminated by BEL (hyperlinks, window title)
	r"|\][^\x1b]*\x1b\\"  # OSC terminated by ST
	r"|[@-Z\\-_]"  # two-character Fe sequences (after OSC: ] is in range)
	r")"
	r"|[\x00-\x1f\x7f]"  # remaining ASCII control characters (e.g. \r from tools)
)


def strip_ansi(text: str) -> str:
	"""Remove ANSI escape sequences and ASCII control characters from a string.

	>>> strip_ansi("\x1b[32mgreen\x1b[0m text")
	'green text'
	>>> strip_ansi("\x1b]8;;https://example.com\x07link\x1b]8;;\x07 text")
	'link text'
	>>> strip_ansi("no escapes")
	'no escapes'
	"""
	return ANSI_ESCAPE.sub("", text)


BOLD: Final = "\033[1m"
GREEN: Final = "\033[32m"
YELLOW: Final = "\033[33m"
RED: Final = "\033[31m"
CYAN: Final = "\033[36m"
GREY: Final = "\033[90m"
RESET: Final = "\033[0m"
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
PROG_MAX_CELL_WIDTH: Final = 6
PROG_MIN_MARGIN: Final = 2


def _bucket_glyph_color(states: Sequence[LeafState]) -> tuple[str, str]:
	"""Reduce a bucket of leaf states to (glyph, ANSI color) for a progress cell.

	Worst-progress wins so the bar never overstates progress: any waiting →
	blank; else any running → dashed line; else a solid line. Color
	distinguishes the active state (cyan running) from the settled outcome
	(red on any failure, grey if all skipped, otherwise green).

	>>> from camas import Completed, Finished, Running, Skipped, Task, Waiting
	>>> t = Task("x")
	>>> _bucket_glyph_color([Waiting(t)]) == (' ', GREY)
	True
	>>> _bucket_glyph_color([Running(t, 0.0, b"")]) == ('┄', CYAN)
	True
	>>> _bucket_glyph_color([Completed(t, Finished(0, 0.1, ()))]) == ('─', GREEN)
	True
	>>> _bucket_glyph_color([Completed(t, Finished(1, 0.1, ()))]) == ('─', RED)
	True
	>>> _bucket_glyph_color([Completed(t, Skipped(1))]) == ('─', GREY)
	True
	>>> _bucket_glyph_color([Waiting(t), Running(t, 0.0, b"")]) == (' ', GREY)
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

	>>> from camas import Completed, Finished, Running, Task, Waiting
	>>> t = Task("x")
	>>> bar = render_progress_bar([Waiting(t), Waiting(t)], 10)
	>>> len(strip_ansi(bar))
	10
	>>> strip_ansi(bar)
	'          '
	>>> bar = render_progress_bar([Running(t, 0.0, b"")], 10)
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
		for glyph, color in (_bucket_glyph_color(b) for b in buckets)
	)
	return f"{' ' * pad_left}{cells}{' ' * pad_right}"


def group_display_name(tasks: tuple[TaskNode, ...], separator: str) -> str:
	"""Derive a display label for a group by joining children's names.

	>>> group_display_name((Task("a"), Task("b")), " | ")
	'a | b'
	>>> group_display_name((Task("build"), Task("test")), " → ")
	'build → test'
	"""
	parts: list[str] = []
	for t in tasks:
		match t:
			case Task():
				parts.append(task_label(t))
			case Sequential(name=name) | Parallel(name=name):
				parts.append(
					name
					if name is not None
					else f"({group_display_name(t.tasks, ' | ' if isinstance(t, Parallel) else ' → ')})"
				)
			case _:
				assert_never(t)
	return separator.join(parts)


def render_tree_prefix(depth: int, is_last_chain: tuple[ChainLink, ...]) -> str:
	"""Reconstitute the ASCII tree prefix from structural position data.

	Children of a ``Sequential`` get ``├─`` / ``└─`` branches with ``│`` continuations —
	the sequence has an ordering and a terminator. Children of a ``Parallel`` get a
	plain ``┃`` column with no ``├``/``└`` distinction, since parallel siblings have
	no order.

	>>> render_tree_prefix(0, ())
	''
	>>> render_tree_prefix(1, (ChainLink(True, False),))
	'└─ '
	>>> render_tree_prefix(1, (ChainLink(False, False),))
	'├─ '
	>>> render_tree_prefix(1, (ChainLink(False, True),))
	'┃ '
	>>> render_tree_prefix(1, (ChainLink(True, True),))
	'┃ '
	>>> render_tree_prefix(2, (ChainLink(False, False), ChainLink(True, True)))
	'│ ┃ '
	>>> render_tree_prefix(2, (ChainLink(True, False), ChainLink(False, False)))
	'  ├─ '
	"""
	if depth == 0:
		return ""
	parts: list[str] = []
	for link in is_last_chain[:-1]:
		if link.parent_is_parallel:
			parts.append("┃ ")
		else:
			parts.append("  " if link.is_last else "│ ")
	last: Final = is_last_chain[-1]
	if last.parent_is_parallel:
		parts.append("┃ ")
	else:
		parts.append("└─ " if last.is_last else "├─ ")
	return "".join(parts)


SEQ_SUFFIX: Final = " →"
PAR_SUFFIX: Final = " ∥"


def iter_rows(
	node: TaskNode,
	depth: int = 0,
	is_last_chain: tuple[ChainLink, ...] = (),
) -> Iterator[DisplayRow]:
	"""Walk a task tree depth-first, yielding one DisplayRow per node (groups + leaves)."""
	match node:
		case Task():
			yield LeafInfo(node, depth, is_last_chain)
		case Sequential(tasks=children, name=name):
			seq_label = (
				f"{name}{SEQ_SUFFIX}" if name is not None else group_display_name(children, " → ")
			)
			yield GroupHeader(seq_label, depth, is_last_chain)
			seq_last = len(children) - 1
			for i, child in enumerate(children):
				link = ChainLink(is_last=i == seq_last, parent_is_parallel=False)
				yield from iter_rows(child, depth + 1, (*is_last_chain, link))
		case Parallel(tasks=children, name=name):
			par_label = (
				f"{name}{PAR_SUFFIX}" if name is not None else group_display_name(children, " | ")
			)
			yield GroupHeader(par_label, depth, is_last_chain)
			par_last = len(children) - 1
			for i, child in enumerate(children):
				link = ChainLink(is_last=i == par_last, parent_is_parallel=True)
				yield from iter_rows(child, depth + 1, (*is_last_chain, link))
		case _:
			assert_never(node)


def flatten_rows(task: TaskNode) -> tuple[DisplayRow, ...]:
	"""Flatten a task tree into display rows (GroupHeaders + LeafInfos) in DFS order.

	>>> rows = flatten_rows(Parallel(tasks=(Task("a"), Task("b"))))
	>>> len(rows)
	3
	>>> isinstance(rows[0], GroupHeader)
	True
	"""
	return tuple(iter_rows(task))


def decode_line(line: bytes) -> str:
	"""Decode a captured output line (stripped of trailing whitespace).

	>>> decode_line(b"hello\\n")
	'hello'
	>>> decode_line(b"")
	''
	"""
	return line.rstrip().decode(errors="replace")


def last_line_display(output: Sequence[bytes]) -> str:
	"""Decode the final non-empty output line for inline display.

	>>> last_line_display([b"first\\n", b"last\\n"])
	'last'
	>>> last_line_display([])
	''
	>>> last_line_display([b"\\n", b"  \\n"])
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
	now: float,
	wall_start: float,
) -> list[str]:
	"""Build the list of ANSI-formatted lines for one frame (leaves + summary)."""
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
						elapsed = now - start_time
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
	wall_elapsed: Final = now - wall_start
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
	now: float,
	wall_start: float,
) -> str:
	"""Render one positioned frame as an ANSI string, for live animation.

	Starts with CPL (`\\033[NF`) to move the cursor up N lines to column 1,
	then writes the frame. Callers must first reserve N+1 rows of vertical
	space (via `"\\n" * N`) so the frame has room to render inline without
	scrolling the viewport.
	"""
	lines: Final = render_lines(rows, states, term_width, display_width, now, wall_start)
	return f"\033[{len(lines) - 1}F" + "\n".join(lines)


def _print_task_output(task: Task, output: Sequence[bytes], label: str, color: str) -> None:
	sys.stdout.write(f"\n{color}{'=' * 60}{RESET}\n")
	sys.stdout.write(f"{color}{BOLD} {label}: {task_label(task)} {RESET}\n")
	sys.stdout.write(f"{color}{'=' * 60}{RESET}\n")
	sys.stdout.flush()
	for line in output:
		sys.stdout.buffer.write(line)
	sys.stdout.buffer.flush()
	sys.stdout.write("\n")


def print_failures(states: Sequence[LeafState]) -> None:
	"""Print detailed output for each failed task."""
	for state in states:
		match state:
			case Completed(task=task, completion=Finished(returncode=rc, output=output)) if rc > 0:
				_print_task_output(task, output, "FAILED", RED)
			case _:
				pass


def print_passes(states: Sequence[LeafState]) -> None:
	"""Print detailed output for each successfully-completed task."""
	for state in states:
		match state:
			case Completed(task=task, completion=Finished(returncode=0, output=output)):
				_print_task_output(task, output, "PASSED", GREEN)
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
	color = _color_on()
	for row, env_new, cwd_new in _walk_with_context(expand_matrix(task)):
		prefix = render_tree_prefix(row.depth, row.is_last_chain)
		meta: list[str] = []
		if show_cmd and cwd_new is not None:
			meta.append(_c(f"(cwd: {cwd_new})", GREY, color))
		if show_cmd and env_new:
			meta.append(_c(" ".join(f"{k}={v}" for k, v in env_new.items()), GREY, color))
		meta_str = f"  {' '.join(meta)}" if meta else ""
		match row:
			case GroupHeader(label=label):
				print(f"{_c(prefix, GREY, color)}{label}{meta_str}")
			case LeafInfo(task=leaf_task):
				print(
					f"{_c(prefix, GREY, color)}{_leaf_label(leaf_task, show_cmd, color)}{meta_str}"
				)
			case _:
				assert_never(row)


def _walk_with_context(
	node: TaskNode,
	depth: int = 0,
	is_last_chain: tuple[ChainLink, ...] = (),
	ancestor_env: Mapping[str, str] = {},
	ancestor_cwd: Path | None = None,
) -> Iterator[tuple[DisplayRow, dict[str, str], Path | None]]:
	"""Walk the expanded tree yielding (row, env_introduced_here, cwd_introduced_here).

	Env entries and cwd are each reported only at the node that introduces or
	changes them, so they render exactly once in the tree.
	"""
	match node:
		case Task(env=env, cwd=cwd):
			yield (
				LeafInfo(node, depth, is_last_chain),
				_env_diff(env, ancestor_env),
				cwd if cwd != ancestor_cwd else None,
			)
		case Sequential(tasks=children, name=name, env=env, cwd=cwd):
			here_env = _env_diff(env, ancestor_env)
			here_cwd = cwd if cwd is not None and cwd != ancestor_cwd else None
			label = (
				f"{name}{SEQ_SUFFIX}" if name is not None else group_display_name(children, " → ")
			)
			yield GroupHeader(label, depth, is_last_chain), here_env, here_cwd
			last_i = len(children) - 1
			new_env = {**ancestor_env, **env}
			new_cwd = cwd if cwd is not None else ancestor_cwd
			for i, child in enumerate(children):
				link = ChainLink(is_last=i == last_i, parent_is_parallel=False)
				yield from _walk_with_context(
					child, depth + 1, (*is_last_chain, link), new_env, new_cwd
				)
		case Parallel(tasks=children, name=name, env=env, cwd=cwd):
			here_env = _env_diff(env, ancestor_env)
			here_cwd = cwd if cwd is not None and cwd != ancestor_cwd else None
			label = (
				f"{name}{PAR_SUFFIX}" if name is not None else group_display_name(children, " | ")
			)
			yield GroupHeader(label, depth, is_last_chain), here_env, here_cwd
			last_i = len(children) - 1
			new_env = {**ancestor_env, **env}
			new_cwd = cwd if cwd is not None else ancestor_cwd
			for i, child in enumerate(children):
				link = ChainLink(is_last=i == last_i, parent_is_parallel=True)
				yield from _walk_with_context(
					child, depth + 1, (*is_last_chain, link), new_env, new_cwd
				)
		case _:
			assert_never(node)


def _env_diff(env: Mapping[str, str], ancestor_env: Mapping[str, str]) -> dict[str, str]:
	return {k: v for k, v in env.items() if ancestor_env.get(k) != v}


def _color_on() -> bool:
	return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, code: str, on: bool) -> str:
	return f"{code}{text}{RESET}" if on and text else text


def _leaf_label(task: Task, show_cmd: bool, color: bool) -> str:
	label = task_label(task)
	base = _c(label, BOLD, color)
	if show_cmd and task.name is not None:
		cmd = task.cmd if isinstance(task.cmd, str) else " ".join(task.cmd)
		if cmd != task.name:
			base = f"{base}: {_c(cmd, CYAN, color)}"
	return base


@dataclass
class TermtreeState:
	"""Mutable slots of a termtree run: the latest states view and the tick task handle."""

	states: Sequence[LeafState]
	tick_task: asyncio.Task[None] | None = None


class TermtreeContext(NamedTuple):
	"""Immutable context threaded through the termtree Effect's lifecycle."""

	rows: tuple[DisplayRow, ...]
	term_width: int
	display_width: int
	wall_start: float
	state: TermtreeState


class Termtree:
	"""Live terminal Effect: renders each leaf's status and a summary line.

	The background tick task is the sole renderer during a run, so the
	effective frame rate is capped at 1 / frame_interval_ms regardless of
	how fast events arrive.
	"""

	def __init__(self, options: TermtreeOptions | None = None) -> None:
		self.options: Final = options if options is not None else TermtreeOptions()

	async def setup(self, task: TaskNode) -> TermtreeContext:
		term_width: Final = (  # zuban: ignore[misc] # zuban defies PEP591
			shutil.get_terminal_size().columns
		)
		rows: Final = flatten_rows(task)  # zuban: ignore[misc] # zuban defies PEP591
		ctx: Final = TermtreeContext(  # zuban: ignore[misc] # zuban defies PEP591
			rows=rows,
			term_width=term_width,
			display_width=term_width - STATUS_COL_WIDTH - 1,
			wall_start=time.perf_counter(),
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
			try:
				await ctx.state.tick_task
			except asyncio.CancelledError:
				pass
		draw(ctx)
		sys.stdout.write("\n")
		sys.stdout.flush()
		print_failures(ctx.state.states)
		if self.options.show_passing:
			print_passes(ctx.state.states)


def draw(ctx: TermtreeContext) -> None:
	# Write bytes directly: the box-drawing chars can't encode to cp1252 (Windows
	# default in non-TTY contexts like captured subprocesses / piped CI logs).
	frame = render_frame(
		ctx.rows,
		ctx.state.states,
		ctx.term_width,
		ctx.display_width,
		time.perf_counter(),
		ctx.wall_start,
	)
	sys.stdout.buffer.write(frame.encode("utf-8", errors="replace"))
	sys.stdout.flush()


async def tick_loop(ctx: TermtreeContext, interval: float) -> None:
	while True:
		await asyncio.sleep(interval)
		draw(ctx)
