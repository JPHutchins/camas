import asyncio
import shutil
import sys
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Final, NamedTuple, assert_never

from camas import (
	Done,
	Effect,
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
	next_state,
	task_display_name,
	truncate_middle,
)


class TermtreeOptions(NamedTuple):
	"""Configuration for the termtree Effect.

	>>> TermtreeOptions()
	TermtreeOptions(frame_interval_ms=100)
	>>> TermtreeOptions(frame_interval_ms=50).frame_interval_ms
	50
	"""

	frame_interval_ms: int = 100


class GroupHeader(NamedTuple):
	"""Display row for a Sequential or Parallel group header.

	>>> GroupHeader("ci", 0, ())
	GroupHeader(label='ci', depth=0, is_last_chain=())
	"""

	label: str
	depth: int
	is_last_chain: tuple[bool, ...]


type DisplayRow = LeafInfo | GroupHeader


BOLD: Final = "\033[1m"
GREEN: Final = "\033[32m"
YELLOW: Final = "\033[33m"
RED: Final = "\033[31m"
GREY: Final = "\033[90m"
RESET: Final = "\033[0m"
CLEAR_LINE: Final = "\033[K"
SPINNER: Final = (
	" ▄    ",
	"  ▄   ",
	"   ▄  ",
	"    ▄ ",
	"    ▐ ",
	"    ▀ ",
	"   ▀  ",
	"  ▀   ",
	" ▀    ",
	" ▌    ",
)

STATUS_COL_WIDTH: Final = 18
MIN_DETAIL_WIDTH: Final = 10


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
				parts.append(task_display_name(t))
			case Sequential(name=name) | Parallel(name=name):
				parts.append(
					name
					if name is not None
					else f"({group_display_name(t.tasks, ' | ' if isinstance(t, Parallel) else ' → ')})"
				)
			case _:
				assert_never(t)
	return separator.join(parts)


def render_tree_prefix(depth: int, is_last_chain: tuple[bool, ...]) -> str:
	"""Reconstitute the ASCII tree prefix from structural position data.

	>>> render_tree_prefix(0, ())
	''
	>>> render_tree_prefix(1, (True,))
	'└─ '
	>>> render_tree_prefix(1, (False,))
	'├─ '
	>>> render_tree_prefix(2, (False, True))
	'│ └─ '
	>>> render_tree_prefix(2, (True, False))
	'  ├─ '
	"""
	if depth == 0:
		return ""
	continuations: Final = "".join("  " if last else "│ " for last in is_last_chain[:-1])
	connector: Final = "└─ " if is_last_chain[-1] else "├─ "
	return continuations + connector


def iter_rows(
	node: TaskNode,
	depth: int = 0,
	is_last_chain: tuple[bool, ...] = (),
) -> Iterator[DisplayRow]:
	"""Walk a task tree depth-first, yielding one DisplayRow per node (groups + leaves)."""
	match node:
		case Task():
			yield LeafInfo(node, depth, is_last_chain)
		case Sequential(tasks=children, name=name):
			seq_label = name if name is not None else group_display_name(children, " → ")
			yield GroupHeader(seq_label, depth, is_last_chain)
			seq_last = len(children) - 1
			for i, child in enumerate(children):
				yield from iter_rows(child, depth + 1, (*is_last_chain, i == seq_last))
		case Parallel(tasks=children, name=name):
			par_label = name if name is not None else group_display_name(children, " | ")
			yield GroupHeader(par_label, depth, is_last_chain)
			par_last = len(children) - 1
			for i, child in enumerate(children):
				yield from iter_rows(child, depth + 1, (*is_last_chain, i == par_last))
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
	states: tuple[LeafState, ...],
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
				name = truncate_middle(
					task_display_name(task),
					max(display_width - len(prefix) - MIN_DETAIL_WIDTH, 3),
				)
				state = states[leaf_idx]
				leaf_idx += 1
				gap = max(display_width - len(prefix) - len(name), 0)
				header = f"{GREY}{prefix}{RESET}{BOLD}{name}{RESET}"
				match state:
					case Done(result=result):
						color = GREEN if result.returncode == 0 else RED
						status = " PASS " if result.returncode == 0 else " FAIL "
						tail = last_line_display(result.output)
						detail = f"  {truncate_middle(tail, gap - 2)}" if gap > 2 and tail else ""
						padding = " " * max(gap - len(detail), 0)
						lines.append(
							f"\r{header}{GREY}{detail}{RESET}{padding} [{color}{status}{RESET}] {result.elapsed:7.3f}s{CLEAR_LINE}"
						)
					case Running(start_time=start_time, last_line=last_line):
						elapsed = now - start_time
						spin = SPINNER[int(elapsed * 10) % len(SPINNER)]
						tail = decode_line(last_line) if last_line else ""
						detail = f"  {truncate_middle(tail, gap - 2)}" if gap > 2 and tail else ""
						padding = " " * max(gap - len(detail), 0)
						lines.append(
							f"\r{header}{GREY}{detail}{RESET}{padding} [{YELLOW}{spin}{RESET}] {elapsed:7.3f}s{CLEAR_LINE}"
						)
					case Skipped():
						padding = " " * gap
						lines.append(f"\r{header}{padding} [{GREY} SKIP {RESET}]{CLEAR_LINE}")
					case Waiting():
						padding = " " * gap
						lines.append(f"\r{header}{padding} [{GREY} WAIT {RESET}]{CLEAR_LINE}")
	wall_elapsed: Final = now - wall_start
	alldone: Final = all(isinstance(s, Done | Skipped) for s in states)
	summary_pad: Final = " " * max(display_width - 6, 0)
	if alldone:
		failed: Final = any(isinstance(s, Done) and s.result.returncode != 0 for s in states)
		summary_color: Final = RED if failed else GREEN
		summary_label: Final = " FAIL " if failed else " PASS "
		lines.append(
			f"\r{BOLD}result{RESET}{summary_pad} [{summary_color}{summary_label}{RESET}] {wall_elapsed:7.3f}s{CLEAR_LINE}"
		)
	else:
		running_spin: Final = SPINNER[int(wall_elapsed * 10) % len(SPINNER)]
		lines.append(
			f"\r{BOLD}result{RESET}{summary_pad} [{YELLOW}{running_spin}{RESET}] {wall_elapsed:7.3f}s{CLEAR_LINE}"
		)
	return lines


def render_frame(
	rows: tuple[DisplayRow, ...],
	states: tuple[LeafState, ...],
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


def print_failures(states: tuple[LeafState, ...]) -> None:
	"""Print detailed output for each failed task."""
	for state in states:
		match state:
			case Done(task=task, result=result) if result.returncode > 0:
				sys.stdout.write(f"\n{RED}{'=' * 60}{RESET}\n")
				sys.stdout.write(f"{RED}{BOLD} FAILED: {task_display_name(task)} {RESET}\n")
				sys.stdout.write(f"{RED}{'=' * 60}{RESET}\n")
				sys.stdout.flush()
				for line in result.output:
					sys.stdout.buffer.write(line)
				sys.stdout.buffer.flush()
				sys.stdout.write("\n")
			case _:
				pass


def print_tree(task: TaskNode) -> None:
	"""Print the task tree structure to stdout without executing.

	>>> print_tree(Task("echo hi"))
	echo hi
	"""
	for row in flatten_rows(expand_matrix(task)):
		prefix = render_tree_prefix(row.depth, row.is_last_chain)
		match row:
			case GroupHeader(label=label):
				print(f"{prefix}{label}")
			case LeafInfo(task=leaf_task):
				print(f"{prefix}{task_display_name(leaf_task)}")
			case _:
				assert_never(row)


def termtree(options: TermtreeOptions) -> Effect:
	"""Build a live terminal Effect that renders each leaf's status and a summary.

	Returns an Effect configured with the given options; the returned Effect
	receives the matrix-expanded task tree and an AsyncIterator of TaskEvents.
	Animation ticks at `options.frame_interval_ms` while execution is in progress.
	"""
	frame_interval: Final = options.frame_interval_ms / 1000

	async def effect(task: TaskNode, events: AsyncIterator[TaskEvent]) -> None:
		rows = flatten_rows(task)
		leaves = tuple(info.task for info in flatten_leaves(task))
		term_size = shutil.get_terminal_size()
		term_width = term_size.columns
		display_width = term_width - STATUS_COL_WIDTH - 1
		states: list[LeafState] = [Waiting(leaf) for leaf in leaves]
		wall_start = time.perf_counter()
		animate = len(rows) + 1 <= term_size.lines

		if animate:
			wake = asyncio.Event()
			done = asyncio.Event()

			sys.stdout.write("\n" * len(rows))
			sys.stdout.flush()

			def draw() -> None:
				sys.stdout.write(
					render_frame(
						rows,
						tuple(states),
						term_width,
						display_width,
						time.perf_counter(),
						wall_start,
					)
				)
				sys.stdout.flush()

			async def consume() -> None:
				async for event in events:
					states[event.leaf_index] = next_state(states[event.leaf_index], event)
					wake.set()
				done.set()
				wake.set()

			async def render() -> None:
				while not done.is_set():
					draw()
					wake.clear()
					try:
						await asyncio.wait_for(wake.wait(), timeout=frame_interval)
					except TimeoutError:
						pass
				draw()

			async with asyncio.TaskGroup() as tg:
				tg.create_task(consume())
				tg.create_task(render())
		else:
			async for event in events:
				states[event.leaf_index] = next_state(states[event.leaf_index], event)
			sys.stdout.write(
				"\n".join(
					render_lines(
						rows,
						tuple(states),
						term_width,
						display_width,
						time.perf_counter(),
						wall_start,
					)
				)
			)
			sys.stdout.flush()

		sys.stdout.write("\n")
		sys.stdout.flush()
		print_failures(tuple(states))

	return effect
