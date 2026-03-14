from __future__ import annotations

import asyncio
import itertools
import os
import platform
import shlex
import shutil
import sys
import time
from subprocess import STDOUT
from typing import Final, NamedTuple, assert_never


class Task(NamedTuple):
	cmd: str | tuple[str, ...]
	name: str | None = None
	env: dict[str, str] | None = None


class Sequential(NamedTuple):
	tasks: tuple[TaskNode, ...]
	name: str | None = None
	matrix: dict[str, tuple[str, ...]] | None = None


class Parallel(NamedTuple):
	tasks: tuple[TaskNode, ...]
	name: str | None = None
	matrix: dict[str, tuple[str, ...]] | None = None


type TaskNode = Task | Sequential | Parallel


class _LeafCmd(NamedTuple):
	name: str
	cmd: tuple[str, ...]
	env: dict[str, str] | None
	tree_prefix: str


class _GroupHeader(NamedTuple):
	label: str
	tree_prefix: str


type _DisplayRow = _LeafCmd | _GroupHeader


class OutputLine(NamedTuple):
	name: str
	line: str


class TaskStarted(NamedTuple):
	name: str


class TaskResult(NamedTuple):
	name: str
	returncode: int
	elapsed: float
	output: str


type TaskEvent = TaskStarted | OutputLine | TaskResult


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

_SKIPPED_RETURNCODE: Final = -1


def _resolve_cmd(cmd: str | tuple[str, ...]) -> tuple[str, ...]:
	match cmd:
		case str():
			return tuple(shlex.split(cmd)) if platform.system() != "Windows" else (cmd,)
		case tuple():
			return cmd
		case _:
			assert_never(cmd)


def _substitute_in_tuple(parts: tuple[str, ...], combo: dict[str, str]) -> tuple[str, ...]:
	result = parts
	for key, val in combo.items():
		result = tuple(part.replace(f"{{{key}}}", val) for part in result)
	return result


def _merge_env(base: dict[str, str] | None, overlay: dict[str, str]) -> dict[str, str]:
	return {**base, **overlay} if base else dict(overlay)


def _apply_combo_to_task_leaf(cmd: Task, combo: dict[str, str], suffix: str) -> Task:
	match cmd.cmd:
		case str() as s:
			new_cmd: str | tuple[str, ...] = s
			for key, val in combo.items():
				new_cmd = new_cmd.replace(f"{{{key}}}", val)  # type: ignore[union-attr]
		case tuple() as t:
			new_cmd = _substitute_in_tuple(t, combo)
		case _:
			assert_never(cmd.cmd)
	base_name = (
		cmd.name
		if cmd.name is not None
		else (cmd.cmd if isinstance(cmd.cmd, str) else " ".join(cmd.cmd))
	)
	return Task(cmd=new_cmd, name=f"{base_name} {suffix}", env=_merge_env(cmd.env, combo))


def _apply_combo_to_task(task: TaskNode, combo: dict[str, str], suffix: str) -> TaskNode:
	match task:
		case Task():
			return _apply_combo_to_task_leaf(task, combo, suffix)
		case Sequential(tasks=tasks, name=name):
			return Sequential(
				tasks=tuple(_apply_combo_to_task(t, combo, suffix) for t in tasks),
				name=f"{name} {suffix}" if name is not None else None,
			)
		case Parallel(tasks=tasks, name=name):
			return Parallel(
				tasks=tuple(_apply_combo_to_task(t, combo, suffix) for t in tasks),
				name=f"{name} {suffix}" if name is not None else None,
			)
		case _:
			assert_never(task)


def _combo_suffix(combo: dict[str, str]) -> str:
	return "[" + ", ".join(f"{k}={v}" for k, v in combo.items()) + "]"


def expand_matrix(task: TaskNode) -> TaskNode:
	match task:
		case Task():
			return task
		case Sequential(tasks=tasks, matrix=matrix):
			expanded_children = tuple(expand_matrix(t) for t in tasks)
			if matrix is None:
				return Sequential(tasks=expanded_children, name=task.name)
			keys = tuple(matrix.keys())
			combos = tuple(dict(zip(keys, vals)) for vals in itertools.product(*matrix.values()))
			cloned_sequences = tuple(
				Sequential(
					tasks=tuple(
						_apply_combo_to_task(child, combo, _combo_suffix(combo))
						for child in expanded_children
					),
					name=(f"{task.name} {_combo_suffix(combo)}" if task.name is not None else None),
				)
				for combo in combos
			)
			return Parallel(tasks=cloned_sequences, name=task.name)
		case Parallel(tasks=tasks, matrix=matrix):
			expanded_children = tuple(expand_matrix(t) for t in tasks)
			if matrix is None:
				return Parallel(tasks=expanded_children, name=task.name)
			keys = tuple(matrix.keys())
			combos = tuple(dict(zip(keys, vals)) for vals in itertools.product(*matrix.values()))
			return Parallel(
				tasks=tuple(
					_apply_combo_to_task(child, combo, _combo_suffix(combo))
					for combo in combos
					for child in expanded_children
				),
				name=task.name,
			)
		case _:
			assert_never(task)


def _task_display_name(task: Task) -> str:
	return (
		task.name
		if task.name is not None
		else (task.cmd if isinstance(task.cmd, str) else " ".join(task.cmd))
	)


def _flatten_with_tree(
	task: TaskNode,
	ancestor_continuations: tuple[str, ...] = (),
	is_last: bool = True,
	is_root: bool = True,
) -> tuple[_DisplayRow, ...]:
	match task:
		case Task():
			if is_root:
				prefix = ""
			else:
				connector = "└── " if is_last else "├── "
				prefix = "".join(ancestor_continuations) + connector
			return (
				_LeafCmd(
					name=_task_display_name(task),
					cmd=_resolve_cmd(task.cmd),
					env=task.env,
					tree_prefix=prefix,
				),
			)
		case Sequential(tasks=tasks) | Parallel(tasks=tasks):
			label = "seq" if isinstance(task, Sequential) else "par"
			rows: list[_DisplayRow] = []
			if is_root:
				header_prefix = "┬── "
				child_continuations: tuple[str, ...] = ()
			else:
				connector = "└── " if is_last else "├── "
				header_prefix = "".join(ancestor_continuations) + connector
				continuation = "    " if is_last else "│   "
				child_continuations = (*ancestor_continuations, continuation)
			rows.append(_GroupHeader(label=label, tree_prefix=header_prefix))
			for i, child in enumerate(tasks):
				rows.extend(
					_flatten_with_tree(
						child,
						child_continuations,
						is_last=i == len(tasks) - 1,
						is_root=False,
					)
				)
			return tuple(rows)
		case _:
			assert_never(task)


def _build_display(task: TaskNode) -> tuple[_DisplayRow, ...]:
	return _flatten_with_tree(task)


def flatten_leaves(task: TaskNode) -> tuple[_LeafCmd, ...]:
	return tuple(row for row in _build_display(task) if isinstance(row, _LeafCmd))


def truncate_middle(text: str, max_width: int) -> str:
	if len(text) <= max_width:
		return text
	side = (max_width - 3) // 2
	return text[:side] + "..." + text[len(text) - (max_width - 3 - side) :]


async def _run_cmd(leaf: _LeafCmd, queue: asyncio.Queue[TaskEvent]) -> TaskResult:
	await queue.put(TaskStarted(leaf.name))
	start = time.perf_counter()
	env = {**os.environ, **leaf.env} if leaf.env else None
	proc = await asyncio.create_subprocess_exec(
		*leaf.cmd,
		stdout=asyncio.subprocess.PIPE,
		stderr=STDOUT,
		env=env,
	)
	output_lines: list[str] = []
	if proc.stdout is not None:
		async for raw in proc.stdout:
			line = raw.decode(errors="replace").rstrip()
			output_lines.append(line)
			if line.strip():
				await queue.put(OutputLine(leaf.name, line))
	await proc.wait()
	result = TaskResult(
		leaf.name,
		proc.returncode or 0,
		time.perf_counter() - start,
		"\n".join(output_lines),
	)
	await queue.put(result)
	return result


async def _execute(
	task: TaskNode,
	queue: asyncio.Queue[TaskEvent],
	leaves: tuple[_LeafCmd, ...],
) -> tuple[TaskResult, ...]:
	match task:
		case Task():
			leaf = _LeafCmd(
				name=_task_display_name(task),
				cmd=_resolve_cmd(task.cmd),
				env=task.env,
				tree_prefix="",
			)
			return (await _run_cmd(leaf, queue),)
		case Parallel(tasks=tasks):
			results: list[TaskResult] = []
			async with asyncio.TaskGroup() as tg:
				futures = [tg.create_task(_execute(child, queue, leaves)) for child in tasks]
			for future in futures:
				results.extend(future.result())
			return tuple(results)
		case Sequential(tasks=tasks):
			results = []
			failed = False
			for child in tasks:
				if failed:
					for leaf in flatten_leaves(child):
						skip_result = TaskResult(leaf.name, _SKIPPED_RETURNCODE, 0.0, "")
						await queue.put(skip_result)
						results.append(skip_result)
				else:
					child_results = await _execute(child, queue, leaves)
					results.extend(child_results)
					if any(r.returncode != 0 for r in child_results):
						failed = True
			return tuple(results)
		case _:
			assert_never(task)


def _render_frame(
	rows: tuple[_DisplayRow, ...],
	leaves: tuple[_LeafCmd, ...],
	last_lines: dict[str, str],
	start_times: dict[str, float],
	completed: dict[str, TaskResult],
	term_width: int,
	display_width: int,
	now: float,
	wall_start: float,
) -> str:
	total_lines = len(rows) + 1
	status_width = display_width + 23
	lines: list[str] = []
	for row in rows:
		match row:
			case _GroupHeader(label=label, tree_prefix=prefix):
				pad_len = display_width - len(prefix) - len(label)
				dash_len = max(pad_len + 8, 3)
				lines.append(f"\r{GREY}{prefix}{label} {'─' * dash_len}{CLEAR_LINE}{RESET}")
			case _LeafCmd(name=name, tree_prefix=prefix):
				display = f"{GREY}{prefix}{RESET}{BOLD}{name}{RESET}"
				pad_len = display_width - len(prefix) - len(name)
				padding = " " * max(pad_len, 0)
				if name in completed:
					done = completed[name]
					if done.returncode == _SKIPPED_RETURNCODE:
						lines.append(f"\r{display}{padding} [{GREY} SKIP {RESET}]{CLEAR_LINE}")
					else:
						color = GREEN if done.returncode == 0 else RED
						status = " PASS " if done.returncode == 0 else " FAIL "
						lines.append(
							f"\r{display}{padding} [{color}{status}{RESET}] {done.elapsed:7.3f}s{CLEAR_LINE}"
						)
				elif name in start_times:
					elapsed = now - start_times[name]
					spin = SPINNER[int(elapsed * 10) % len(SPINNER)]
					detail = truncate_middle(last_lines.get(name, ""), term_width - status_width)
					lines.append(
						f"\r{display}{padding} [{YELLOW}{spin}{RESET}] {elapsed:7.3f}s  {detail}{CLEAR_LINE}"
					)
				else:
					lines.append(f"\r{display}{padding} [{GREY} WAIT {RESET}]{CLEAR_LINE}")
	wall_elapsed = now - wall_start
	all_done = len(completed) == len(leaves)
	summary_pad = " " * max(display_width - 6, 0)
	if all_done:
		failed = any(d.returncode != 0 for d in completed.values())
		color = RED if failed else GREEN
		label = " FAIL " if failed else " PASS "
		lines.append(
			f"\r{BOLD}result{RESET}{summary_pad} [{color}{label}{RESET}] {wall_elapsed:7.3f}s{CLEAR_LINE}"
		)
	else:
		spin = SPINNER[int(wall_elapsed * 10) % len(SPINNER)]
		lines.append(
			f"\r{BOLD}result{RESET}{summary_pad} [{YELLOW}{spin}{RESET}] {wall_elapsed:7.3f}s{CLEAR_LINE}"
		)
	return f"\033[{total_lines}F" + "\n".join(lines) + "\n"


def _process_event(
	event: TaskEvent,
	last_lines: dict[str, str],
	completed: dict[str, TaskResult],
	start_times: dict[str, float],
) -> None:
	match event:
		case TaskStarted(name):
			if name not in start_times:
				start_times[name] = time.perf_counter()
		case OutputLine(name, line):
			last_lines[name] = line
			if name not in start_times:
				start_times[name] = time.perf_counter()
		case TaskResult() as done:
			completed[done.name] = done
		case _:
			assert_never(event)


async def _render_loop(
	rows: tuple[_DisplayRow, ...],
	leaves: tuple[_LeafCmd, ...],
	queue: asyncio.Queue[TaskEvent],
	wall_start: float,
) -> dict[str, TaskResult]:
	term_width = shutil.get_terminal_size().columns
	display_width = max(
		(len(row.tree_prefix) + len(row.name if isinstance(row, _LeafCmd) else row.label))
		for row in rows
	)
	last_lines: dict[str, str] = {}
	start_times: dict[str, float] = {}
	completed: dict[str, TaskResult] = {}

	sys.stdout.write("\n" * (len(rows) + 1))
	sys.stdout.flush()

	while len(completed) < len(leaves):
		try:
			_process_event(
				await asyncio.wait_for(queue.get(), timeout=0.1),
				last_lines,
				completed,
				start_times,
			)
		except TimeoutError:
			pass
		while not queue.empty():
			_process_event(queue.get_nowait(), last_lines, completed, start_times)
		sys.stdout.write(
			_render_frame(
				rows,
				leaves,
				last_lines,
				start_times,
				completed,
				term_width,
				display_width,
				time.perf_counter(),
				wall_start,
			)
		)
		sys.stdout.flush()

	return completed


def _print_failures(leaves: tuple[_LeafCmd, ...], completed: dict[str, TaskResult]) -> None:
	for leaf in leaves:
		done = completed[leaf.name]
		if done.returncode > 0:
			sys.stdout.write(f"\n{RED}{'=' * 60}{RESET}\n")
			sys.stdout.write(f"{RED}{BOLD} FAILED: {done.name} {RESET}\n")
			sys.stdout.write(f"{RED}{'=' * 60}{RESET}\n")
			sys.stdout.write(done.output)
			sys.stdout.write("\n")


async def _run(task: TaskNode) -> int:
	expanded = expand_matrix(task)
	rows = _build_display(expanded)
	leaves = tuple(row for row in rows if isinstance(row, _LeafCmd))
	queue: asyncio.Queue[TaskEvent] = asyncio.Queue()

	wall_start = time.perf_counter()
	async with asyncio.TaskGroup() as tg:
		tg.create_task(_execute(expanded, queue, leaves))
		completed = await _render_loop(rows, leaves, queue, wall_start)

	_print_failures(leaves, completed)
	return 1 if any(d.returncode != 0 for d in completed.values()) else 0


def run(task: TaskNode) -> int:
	return asyncio.run(_run(task))
