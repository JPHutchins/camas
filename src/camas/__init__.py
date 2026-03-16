from __future__ import annotations

import asyncio
import functools
import itertools
import os
import shutil
import sys
import time
from subprocess import STDOUT
from typing import Final, NamedTuple, assert_never


class VarBinding(NamedTuple):
	"""A single matrix variable bound to a concrete value.

	>>> VarBinding("PY", "3.14")
	VarBinding(name='PY', value='3.14')
	"""

	name: str
	value: str


type MatrixBinding = tuple[VarBinding, ...]


class Task(NamedTuple):
	"""A leaf task that executes a shell command.

	>>> Task("echo hi")
	Task(cmd='echo hi', name=None, env={})
	>>> Task(("ruff", "check", "."), name="lint")
	Task(cmd=('ruff', 'check', '.'), name='lint', env={})
	"""

	cmd: str | tuple[str, ...]
	name: str | None = None
	env: dict[str, str] = {}


class Sequential(NamedTuple):
	"""A group of tasks that run one after another, short-circuiting on failure.

	>>> Sequential(tasks=(Task("build"), Task("test")), name="ci")
	Sequential(tasks=(Task(cmd='build', name=None, env={}), Task(cmd='test', name=None, env={})), name='ci', matrix=None)
	"""

	tasks: tuple[TaskNode, ...]
	name: str | None = None
	matrix: dict[str, tuple[str, ...]] | None = None


class Parallel(NamedTuple):
	"""A group of tasks that run concurrently.

	>>> Parallel(tasks=(Task("lint"), Task("typecheck")))
	Parallel(tasks=(Task(cmd='lint', name=None, env={}), Task(cmd='typecheck', name=None, env={})), name=None, matrix=None)
	"""

	tasks: tuple[TaskNode, ...]
	name: str | None = None
	matrix: dict[str, tuple[str, ...]] | None = None


type TaskNode = Task | Sequential | Parallel


class TaskResult(NamedTuple):
	"""Result of a single completed task.

	>>> TaskResult("lint", 0, 1.234, "all clean")
	TaskResult(name='lint', returncode=0, elapsed=1.234, output='all clean')
	"""

	name: str
	returncode: int
	elapsed: float
	output: str


class RunResult(NamedTuple):
	"""Result of running an entire task tree.

	>>> RunResult(0, (TaskResult("a", 0, 0.1, ""),), 0.1)
	RunResult(returncode=0, results=(TaskResult(name='a', returncode=0, elapsed=0.1, output=''),), elapsed=0.1)
	"""

	returncode: int
	results: tuple[TaskResult, ...]
	elapsed: float


class _LeafCmd(NamedTuple):
	"""Internal display representation of an executable leaf node.

	>>> _LeafCmd("test", ("pytest",), {}, "├── ")
	_LeafCmd(name='test', cmd=('pytest',), env={}, tree_prefix='├── ')
	"""

	name: str
	cmd: tuple[str, ...]
	env: dict[str, str]
	tree_prefix: str


class _GroupHeader(NamedTuple):
	"""Internal display header for a Sequential or Parallel group.

	>>> _GroupHeader("ci", "┌── ")
	_GroupHeader(label='ci', tree_prefix='┌── ')
	"""

	label: str
	tree_prefix: str


type _DisplayRow = _LeafCmd | _GroupHeader


class _StartedEvent(NamedTuple):
	"""Internal event: a task has started execution.

	>>> _StartedEvent(0, 100.0)
	_StartedEvent(leaf_index=0, timestamp=100.0)
	"""

	leaf_index: int
	timestamp: float


class _OutputEvent(NamedTuple):
	"""Internal event: a task produced an output line.

	>>> _OutputEvent(0, "hello", 100.5)
	_OutputEvent(leaf_index=0, line='hello', timestamp=100.5)
	"""

	leaf_index: int
	line: str
	timestamp: float


class _CompletedEvent(NamedTuple):
	"""Internal event: a task finished execution.

	>>> _CompletedEvent(0, 0, 1.0, "done")
	_CompletedEvent(leaf_index=0, returncode=0, elapsed=1.0, output='done')
	"""

	leaf_index: int
	returncode: int
	elapsed: float
	output: str


type _TaskEvent = _StartedEvent | _OutputEvent | _CompletedEvent


class _Waiting(NamedTuple):
	"""Leaf state: task has not started yet.

	>>> _Waiting(_LeafCmd("t", ("echo",), {}, ""))
	_Waiting(leaf=_LeafCmd(name='t', cmd=('echo',), env={}, tree_prefix=''))
	"""

	leaf: _LeafCmd


class _Running(NamedTuple):
	"""Leaf state: task is currently executing.

	>>> _Running(_LeafCmd("t", ("echo",), {}, ""), 100.0, "output...")
	_Running(leaf=_LeafCmd(name='t', cmd=('echo',), env={}, tree_prefix=''), start_time=100.0, last_line='output...')
	"""

	leaf: _LeafCmd
	start_time: float
	last_line: str


class _Done(NamedTuple):
	"""Leaf state: task completed with a result.

	>>> _Done(_LeafCmd("t", ("echo",), {}, ""), TaskResult("t", 0, 0.5, ""))
	_Done(leaf=_LeafCmd(name='t', cmd=('echo',), env={}, tree_prefix=''), result=TaskResult(name='t', returncode=0, elapsed=0.5, output=''))
	"""

	leaf: _LeafCmd
	result: TaskResult


class _Skipped(NamedTuple):
	"""Leaf state: task was skipped due to a prior failure in a Sequential.

	>>> _Skipped(_LeafCmd("t", ("echo",), {}, ""))
	_Skipped(leaf=_LeafCmd(name='t', cmd=('echo',), env={}, tree_prefix=''))
	"""

	leaf: _LeafCmd


type _LeafState = _Waiting | _Running | _Done | _Skipped


class _LeafIndexMap(NamedTuple):
	"""Mapping from Task object identity to leaf position in the flattened display.

	>>> _LeafIndexMap({}, 0)
	_LeafIndexMap(mapping={}, size=0)
	"""

	mapping: dict[int, int]
	size: int


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


if sys.platform == "win32":  # pragma: no cover

	def _resolve_cmd(cmd: str | tuple[str, ...]) -> tuple[str, ...]:
		"""Resolve a command to a tuple of strings (Windows: no shell splitting).

		>>> _resolve_cmd(("echo", "hi"))
		('echo', 'hi')
		"""
		match cmd:
			case str():
				return (cmd,)
			case tuple():
				return cmd
			case _:
				assert_never(cmd)

else:
	import shlex

	def _resolve_cmd(cmd: str | tuple[str, ...]) -> tuple[str, ...]:
		"""Resolve a command to a tuple of strings, splitting shell strings with shlex.

		>>> _resolve_cmd(("echo", "hi"))
		('echo', 'hi')
		>>> _resolve_cmd("echo hi")
		('echo', 'hi')
		"""
		match cmd:
			case str():
				return tuple(shlex.split(cmd))
			case tuple():
				return cmd
			case _:
				assert_never(cmd)


def truncate_middle(text: str, max_width: int) -> str:
	"""Truncate text in the middle with '...' if it exceeds max_width.

	>>> truncate_middle("hello", 10)
	'hello'
	>>> truncate_middle("hello world!", 9)
	'hel...ld!'
	"""
	if len(text) <= max_width:
		return text
	side: Final = (max_width - 3) // 2
	return text[:side] + "..." + text[len(text) - (max_width - 3 - side) :]


def _derive_name(cmd: str | tuple[str, ...], width: int) -> str:
	"""Derive a display name from a command, truncated to _AUTO_NAME_MAX_WIDTH.

	>>> _derive_name("echo hi", 20)
	'echo hi'
	>>> _derive_name(("python", "-c", "pass"), 20)
	'python -c pass'
	>>> _derive_name("a very long command that exceeds twenty characters", 20)
	'a very l...haracters'
	"""
	return truncate_middle(
		cmd if isinstance(cmd, str) else " ".join(cmd),
		width,
	)


def _task_display_name(task: Task) -> str:
	"""Return the display name for a Task: explicit name or auto-derived.

	>>> _task_display_name(Task("echo hi", name="greet"))
	'greet'
	>>> _task_display_name(Task("echo hi"))
	'echo hi'
	"""
	return task.name if task.name is not None else _derive_name(task.cmd, 20)


def _group_display_name(tasks: tuple[TaskNode, ...], separator: str) -> str:
	"""Derive a display name for a group by joining children's names.

	>>> _group_display_name((Task("a"), Task("b")), " | ")
	'a | b'
	>>> _group_display_name((Task("build"), Task("test")), " → ")
	'build → test'
	"""
	parts: list[str] = []
	for t in tasks:
		match t:
			case Task():
				parts.append(_task_display_name(t))
			case Sequential(name=name) | Parallel(name=name):
				parts.append(
					name
					if name is not None
					else f"({_group_display_name(t.tasks, ' | ' if isinstance(t, Parallel) else ' → ')})"
				)
			case _:
				assert_never(t)
	return separator.join(parts)


def _substitute_in_str(text: str, binding: MatrixBinding) -> str:
	"""Replace {name} placeholders in a string with binding values.

	>>> _substitute_in_str("test --python {PY}", (VarBinding("PY", "3.14"),))
	'test --python 3.14'
	>>> _substitute_in_str("no placeholders", (VarBinding("X", "1"),))
	'no placeholders'
	"""
	return functools.reduce(
		lambda acc, vb: acc.replace(f"{{{vb.name}}}", vb.value),
		binding,
		text,
	)


def _substitute_in_tuple(parts: tuple[str, ...], binding: MatrixBinding) -> tuple[str, ...]:
	"""Replace {name} placeholders in each element of a tuple.

	>>> _substitute_in_tuple(("test", "--python", "{PY}"), (VarBinding("PY", "3.14"),))
	('test', '--python', '3.14')
	"""
	return functools.reduce(
		lambda acc, vb: tuple(p.replace(f"{{{vb.name}}}", vb.value) for p in acc),
		binding,
		parts,
	)


def _specialize_task(task: Task, binding: MatrixBinding, suffix: str) -> Task:
	"""Specialize a leaf Task with concrete variable values from a matrix binding.

	>>> _specialize_task(Task("test {PY}"), (VarBinding("PY", "3.14"),), "[PY=3.14]")
	Task(cmd='test 3.14', name='test 3.14 [PY=3.14]', env={'PY': '3.14'})
	"""
	match task.cmd:
		case str():
			new_cmd: str | tuple[str, ...] = _substitute_in_str(task.cmd, binding)
		case tuple():
			new_cmd = _substitute_in_tuple(task.cmd, binding)
		case _:
			assert_never(task.cmd)
	return Task(
		cmd=new_cmd,
		name=f"{_substitute_in_str(task.name if task.name is not None else _derive_name(task.cmd, 20), binding)} {suffix}",
		env=task.env | dict(binding),
	)


def _specialize_node(task: TaskNode, binding: MatrixBinding, suffix: str) -> TaskNode:
	"""Recursively specialize an entire task tree with concrete variable values.

	>>> _specialize_node(Task("test {X}"), (VarBinding("X", "1"),), "[X=1]")
	Task(cmd='test 1', name='test 1 [X=1]', env={'X': '1'})
	"""
	match task:
		case Task():
			return _specialize_task(task, binding, suffix)
		case Sequential(tasks=tasks, name=name):
			return Sequential(
				tasks=tuple(_specialize_node(t, binding, suffix) for t in tasks),
				name=f"{name} {suffix}" if name is not None else None,
			)
		case Parallel(tasks=tasks, name=name):
			return Parallel(
				tasks=tuple(_specialize_node(t, binding, suffix) for t in tasks),
				name=f"{name} {suffix}" if name is not None else None,
			)
		case _:
			assert_never(task)


def _binding_suffix(binding: MatrixBinding) -> str:
	"""Format a matrix binding as a bracketed suffix.

	>>> _binding_suffix((VarBinding("PY", "3.14"),))
	'[PY=3.14]'
	>>> _binding_suffix((VarBinding("OS", "linux"), VarBinding("PY", "3.14")))
	'[OS=linux, PY=3.14]'
	"""
	return "[" + ", ".join(f"{vb.name}={vb.value}" for vb in binding) + "]"


def _matrix_bindings(matrix: dict[str, tuple[str, ...]]) -> tuple[MatrixBinding, ...]:
	"""Generate all cartesian product bindings from a matrix definition.

	>>> _matrix_bindings({"PY": ("3.12", "3.13")})
	((VarBinding(name='PY', value='3.12'),), (VarBinding(name='PY', value='3.13'),))
	>>> len(_matrix_bindings({"A": ("1", "2"), "B": ("x", "y")}))
	4
	"""
	keys: Final = tuple(matrix.keys())
	return tuple(
		tuple(VarBinding(k, v) for k, v in zip(keys, vals))
		for vals in itertools.product(*matrix.values())
	)


def _expand_sequential_matrix(
	children: tuple[TaskNode, ...],
	matrix: dict[str, tuple[str, ...]],
	name: str | None,
) -> Parallel:
	"""Expand a Sequential's matrix into a Parallel of cloned Sequentials.

	>>> result = _expand_sequential_matrix((Task("build"), Task("test")), {"X": ("1", "2")}, "ci")
	>>> len(result.tasks)
	2
	>>> all(isinstance(t, Sequential) for t in result.tasks)
	True
	"""
	return Parallel(
		tasks=tuple(
			Sequential(
				tasks=tuple(
					_specialize_node(child, binding, _binding_suffix(binding)) for child in children
				),
				name=(f"{name} {_binding_suffix(binding)}" if name is not None else None),
			)
			for binding in _matrix_bindings(matrix)
		),
		name=name,
	)


def _expand_parallel_matrix(
	children: tuple[TaskNode, ...],
	matrix: dict[str, tuple[str, ...]],
	name: str | None,
) -> Parallel:
	"""Expand a Parallel's matrix into a flat Parallel of all binding × child products.

	>>> result = _expand_parallel_matrix((Task("test"),), {"PY": ("3.12", "3.13")}, None)
	>>> len(result.tasks)
	2
	"""
	return Parallel(
		tasks=tuple(
			_specialize_node(child, binding, _binding_suffix(binding))
			for binding in _matrix_bindings(matrix)
			for child in children
		),
		name=name,
	)


def expand_matrix(task: TaskNode) -> TaskNode:
	"""Recursively expand all matrix parameters in a task tree.

	>>> expand_matrix(Task("echo hi"))
	Task(cmd='echo hi', name=None, env={})
	>>> result = expand_matrix(Parallel(tasks=(Task("test"),), matrix={"X": ("1", "2")}))
	>>> len(result.tasks)  # type: ignore[union-attr]
	2
	"""
	match task:
		case Task():
			return task
		case Sequential(tasks=tasks, matrix=matrix):
			expanded = tuple(expand_matrix(t) for t in tasks)
			if matrix is None:
				return Sequential(tasks=expanded, name=task.name)
			return _expand_sequential_matrix(expanded, matrix, task.name)
		case Parallel(tasks=tasks, matrix=matrix):
			expanded = tuple(expand_matrix(t) for t in tasks)
			if matrix is None:
				return Parallel(tasks=expanded, name=task.name)
			return _expand_parallel_matrix(expanded, matrix, task.name)
		case _:
			assert_never(task)


def _tree_connector(is_last: bool) -> str:
	"""Return the tree drawing connector for a node.

	>>> _tree_connector(True)
	'└── '
	>>> _tree_connector(False)
	'├── '
	"""
	return "└── " if is_last else "├── "


def _flatten_leaf(
	task: Task,
	ancestor_continuations: tuple[str, ...],
	is_last: bool,
	is_root: bool,
) -> tuple[_LeafCmd]:
	"""Flatten a Task into a single-element tuple of _LeafCmd.

	>>> _flatten_leaf(Task("echo hi"), (), True, True)[0].name
	'echo hi'
	>>> _flatten_leaf(Task("echo hi"), (), True, False)[0].tree_prefix
	'└── '
	"""
	prefix: Final = "" if is_root else "".join(ancestor_continuations) + _tree_connector(is_last)
	return (
		_LeafCmd(
			name=_task_display_name(task),
			cmd=_resolve_cmd(task.cmd),
			env=task.env,
			tree_prefix=prefix,
		),
	)


def _flatten_group(
	label: str,
	tasks: tuple[TaskNode, ...],
	ancestor_continuations: tuple[str, ...],
	is_last: bool,
	is_root: bool,
) -> tuple[_DisplayRow, ...]:
	"""Flatten a group (Sequential/Parallel) into display rows with tree prefixes.

	>>> rows = _flatten_group("ci", (Task("a"), Task("b")), (), True, True)
	>>> rows[0]
	_GroupHeader(label='ci', tree_prefix='')
	>>> len(rows)
	3
	"""
	if is_root:
		header_prefix = ""
		child_continuations: tuple[str, ...] = ()
	else:
		header_prefix = "".join(ancestor_continuations) + _tree_connector(is_last)
		child_continuations = (*ancestor_continuations, "    " if is_last else "│   ")
	return (
		_GroupHeader(label=label, tree_prefix=header_prefix),
		*(
			row
			for i, child in enumerate(tasks)
			for row in _flatten_with_tree(
				child,
				child_continuations,
				is_last=i == len(tasks) - 1,
				is_root=False,
			)
		),
	)


def _flatten_with_tree(
	task: TaskNode,
	ancestor_continuations: tuple[str, ...] = (),
	is_last: bool = True,
	is_root: bool = True,
) -> tuple[_DisplayRow, ...]:
	"""Flatten a TaskNode tree into display rows with tree prefixes.

	>>> rows = _flatten_with_tree(Parallel(tasks=(Task("a"), Task("b"))))
	>>> len(rows)
	3
	"""
	match task:
		case Task():
			return _flatten_leaf(task, ancestor_continuations, is_last, is_root)
		case Sequential(tasks=tasks, name=name):
			return _flatten_group(
				name if name is not None else _group_display_name(tasks, " → "),
				tasks,
				ancestor_continuations,
				is_last,
				is_root,
			)
		case Parallel(tasks=tasks, name=name):
			return _flatten_group(
				name if name is not None else _group_display_name(tasks, " | "),
				tasks,
				ancestor_continuations,
				is_last,
				is_root,
			)
		case _:
			assert_never(task)


def flatten_leaves(task: TaskNode) -> tuple[_LeafCmd, ...]:
	"""Extract all executable leaf nodes from a task tree.

	>>> [leaf.name for leaf in flatten_leaves(Parallel(tasks=(Task("a"), Task("b"))))]
	['a', 'b']
	"""
	return tuple(row for row in _flatten_with_tree(task) if isinstance(row, _LeafCmd))


def _build_leaf_index_map(task: TaskNode, offset: int = 0) -> _LeafIndexMap:
	"""Build a mapping from Task object identity to leaf index for execution.

	>>> t1, t2 = Task("a"), Task("b")
	>>> result = _build_leaf_index_map(Parallel(tasks=(t1, t2)))
	>>> result.size
	2
	>>> result.mapping[id(t1)], result.mapping[id(t2)]
	(0, 1)
	"""
	match task:
		case Task():
			return _LeafIndexMap({id(task): offset}, offset + 1)
		case Sequential(tasks=tasks) | Parallel(tasks=tasks):
			mapping: dict[int, int] = {}
			for child in tasks:
				child_result = _build_leaf_index_map(child, offset)
				mapping.update(child_result.mapping)
				offset = child_result.size
			return _LeafIndexMap(mapping, offset)
		case _:
			assert_never(task)


def _subtree_leaf_indices(task: TaskNode, index_map: dict[int, int]) -> tuple[int, ...]:
	"""Collect all leaf indices within a subtree.

	>>> t1, t2 = Task("a"), Task("b")
	>>> tree = Parallel(tasks=(t1, t2))
	>>> _subtree_leaf_indices(tree, _build_leaf_index_map(tree).mapping)
	(0, 1)
	"""
	match task:
		case Task():
			return (index_map[id(task)],)
		case Sequential(tasks=tasks) | Parallel(tasks=tasks):
			return tuple(i for child in tasks for i in _subtree_leaf_indices(child, index_map))
		case _:
			assert_never(task)


async def _run_cmd(leaf: _LeafCmd, leaf_index: int, queue: asyncio.Queue[_TaskEvent]) -> TaskResult:
	"""Execute a single leaf command as a subprocess, streaming events to the queue."""
	start: Final = time.perf_counter()
	await queue.put(_StartedEvent(leaf_index, start))
	env: Final = {**os.environ, **leaf.env} if leaf.env else None
	proc = await asyncio.create_subprocess_exec(
		*leaf.cmd,
		stdout=asyncio.subprocess.PIPE,
		stderr=STDOUT,
		env=env,
	)
	output_lines: list[str] = []
	if proc.stdout is not None:  # pragma: no branch
		async for raw in proc.stdout:
			line = raw.decode(errors="replace").rstrip()
			output_lines.append(line)
			if line.strip():
				await queue.put(_OutputEvent(leaf_index, line, time.perf_counter()))
	await proc.wait()
	elapsed: Final = time.perf_counter() - start
	output: Final = "\n".join(output_lines)
	await queue.put(_CompletedEvent(leaf_index, proc.returncode or 0, elapsed, output))
	return TaskResult(leaf.name, proc.returncode or 0, elapsed, output)


async def _execute(
	task: TaskNode,
	queue: asyncio.Queue[_TaskEvent],
	leaves: tuple[_LeafCmd, ...],
	index_map: dict[int, int],
) -> tuple[TaskResult, ...]:
	"""Recursively execute a task tree, respecting Sequential/Parallel semantics."""
	match task:
		case Task():
			idx = index_map[id(task)]
			return (await _run_cmd(leaves[idx], idx, queue),)
		case Parallel(tasks=tasks):
			results: list[TaskResult] = []
			async with asyncio.TaskGroup() as tg:
				futures: Final = [
					tg.create_task(_execute(child, queue, leaves, index_map)) for child in tasks
				]
			for future in futures:
				results.extend(future.result())
			return tuple(results)
		case Sequential(tasks=tasks):
			results = []
			failed = False
			for child in tasks:
				if failed:
					for idx in _subtree_leaf_indices(child, index_map):
						await queue.put(_CompletedEvent(idx, _SKIPPED_RETURNCODE, 0.0, ""))
						results.append(TaskResult(leaves[idx].name, _SKIPPED_RETURNCODE, 0.0, ""))
				else:
					child_results = await _execute(child, queue, leaves, index_map)
					results.extend(child_results)
					if any(r.returncode != 0 for r in child_results):
						failed = True
			return tuple(results)
		case _:
			assert_never(task)


class _RenderContext(NamedTuple):
	"""Snapshot of all state needed to render one TUI frame.

	>>> ctx = _RenderContext((), (), 80, 20, 100.0, 99.0)
	>>> ctx.term_width
	80
	"""

	rows: tuple[_DisplayRow, ...]
	states: tuple[_LeafState, ...]
	term_width: int
	display_width: int
	now: float
	wall_start: float


_STATUS_COL_WIDTH: Final = 18


def _render_frame(ctx: _RenderContext) -> str:
	"""Render a single TUI frame as an ANSI string from the current render context."""
	total_lines: Final = len(ctx.rows) + 1
	lines: list[str] = []
	leaf_idx = 0
	for row in ctx.rows:
		match row:
			case _GroupHeader(label=label, tree_prefix=prefix):
				lines.append(
					f"\r{GREY}{truncate_middle(f'{prefix}{label}', ctx.term_width - 1)}{CLEAR_LINE}{RESET}"
				)
			case _LeafCmd(name=name, tree_prefix=prefix):
				state = ctx.states[leaf_idx]
				leaf_idx += 1
				gap = max(ctx.display_width - len(prefix) - len(name), 0)
				display = f"{GREY}{prefix}{RESET}{BOLD}{name}{RESET}"
				match state:
					case _Done(result=result):
						color = GREEN if result.returncode == 0 else RED
						status = " PASS " if result.returncode == 0 else " FAIL "
						detail = (
							f"  {truncate_middle(result.output, gap - 2)}"
							if gap > 2 and result.output
							else ""
						)
						padding = " " * max(gap - len(detail), 0)
						lines.append(
							f"\r{display}{GREY}{detail}{RESET}{padding} [{color}{status}{RESET}] {result.elapsed:7.3f}s{CLEAR_LINE}"
						)
					case _Running(start_time=start_time, last_line=last_line):
						elapsed = ctx.now - start_time
						spin = SPINNER[int(elapsed * 10) % len(SPINNER)]
						detail = (
							f"  {truncate_middle(last_line, gap - 2)}"
							if gap > 2 and last_line
							else ""
						)
						padding = " " * max(gap - len(detail), 0)
						lines.append(
							f"\r{display}{GREY}{detail}{RESET}{padding} [{YELLOW}{spin}{RESET}] {elapsed:7.3f}s{CLEAR_LINE}"
						)
					case _Skipped():
						padding = " " * gap
						lines.append(f"\r{display}{padding} [{GREY} SKIP {RESET}]{CLEAR_LINE}")
					case _Waiting():
						padding = " " * gap
						lines.append(f"\r{display}{padding} [{GREY} WAIT {RESET}]{CLEAR_LINE}")
	wall_elapsed: Final = ctx.now - ctx.wall_start
	all_done: Final = all(isinstance(s, _Done | _Skipped) for s in ctx.states)
	summary_pad: Final = " " * max(ctx.display_width - 6, 0)
	if all_done:
		failed = any(isinstance(s, _Done) and s.result.returncode != 0 for s in ctx.states)
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
	return f"\033[{total_lines - 1}F" + "\n".join(lines)


def _next_state(state: _LeafState, event: _TaskEvent) -> _LeafState:
	"""Pure state machine: given current state and an event, return the next state.

	>>> leaf = _LeafCmd("t", ("echo",), {}, "")
	>>> _next_state(_Waiting(leaf), _StartedEvent(0, 100.0))
	_Running(leaf=_LeafCmd(name='t', cmd=('echo',), env={}, tree_prefix=''), start_time=100.0, last_line='')
	>>> _next_state(_Running(leaf, 100.0, ""), _OutputEvent(0, "hi", 100.5))
	_Running(leaf=_LeafCmd(name='t', cmd=('echo',), env={}, tree_prefix=''), start_time=100.0, last_line='hi')
	>>> _next_state(_Running(leaf, 100.0, ""), _CompletedEvent(0, 0, 0.5, "done"))
	_Done(leaf=_LeafCmd(name='t', cmd=('echo',), env={}, tree_prefix=''), result=TaskResult(name='t', returncode=0, elapsed=0.5, output='done'))
	"""
	match state:
		case _Waiting(leaf=leaf):
			match event:
				case _StartedEvent(timestamp=ts):
					return _Running(leaf, ts, "")
				case _OutputEvent(line=line, timestamp=ts):  # pragma: no cover
					return _Running(leaf, ts, line)
				case _CompletedEvent(returncode=rc, elapsed=elapsed, output=output):
					if rc == _SKIPPED_RETURNCODE:
						return _Skipped(leaf)
					return _Done(
						leaf, TaskResult(leaf.name, rc, elapsed, output)
					)  # pragma: no cover
				case _:
					assert_never(event)
		case _Running(leaf=leaf, start_time=start, last_line=last_line):
			match event:
				case _OutputEvent(line=line):
					return _Running(leaf, start, line)
				case _CompletedEvent(returncode=rc, elapsed=elapsed, output=output):
					return _Done(
						leaf, TaskResult(leaf.name, rc, elapsed, output if output else last_line)
					)
				case _StartedEvent():  # pragma: no cover
					return state
				case _:
					assert_never(event)

		case _Done() | _Skipped():  # pragma: no cover
			return state
		case _:
			assert_never(state)


async def _render_loop(
	rows: tuple[_DisplayRow, ...],
	leaves: tuple[_LeafCmd, ...],
	queue: asyncio.Queue[_TaskEvent],
	wall_start: float,
) -> list[_LeafState]:
	"""Process events and render TUI frames until all tasks complete."""
	term_width: Final = shutil.get_terminal_size().columns
	display_width: Final = term_width - _STATUS_COL_WIDTH - 1
	states: list[_LeafState] = [_Waiting(leaf) for leaf in leaves]

	sys.stdout.write("\n" * len(rows))
	sys.stdout.flush()

	def _apply_event(event: _TaskEvent) -> None:
		states[event.leaf_index] = _next_state(states[event.leaf_index], event)

	while not all(isinstance(s, _Done | _Skipped) for s in states):
		try:
			_apply_event(await asyncio.wait_for(queue.get(), timeout=0.1))
		except TimeoutError:
			pass
		while not queue.empty():
			_apply_event(queue.get_nowait())
		sys.stdout.write(
			_render_frame(
				_RenderContext(
					rows=rows,
					states=tuple(states),
					term_width=term_width,
					display_width=display_width,
					now=time.perf_counter(),
					wall_start=wall_start,
				)
			)
		)
		sys.stdout.flush()

	return states


def _print_failures(states: tuple[_LeafState, ...]) -> None:
	"""Print detailed output for each failed task."""
	for state in states:
		match state:
			case _Done(leaf=leaf, result=result) if result.returncode > 0:
				sys.stdout.write(f"\n{RED}{'=' * 60}{RESET}\n")
				sys.stdout.write(f"{RED}{BOLD} FAILED: {leaf.name} {RESET}\n")
				sys.stdout.write(f"{RED}{'=' * 60}{RESET}\n")
				sys.stdout.write(result.output)
				sys.stdout.write("\n")
			case _:
				pass


def print_tree(task: TaskNode) -> None:
	"""Print the task tree structure to stdout without executing.

	>>> print_tree(Task("echo hi"))
	echo hi
	"""
	for row in _flatten_with_tree(expand_matrix(task)):
		match row:
			case _GroupHeader(label=label, tree_prefix=prefix):
				print(f"{prefix}{label}")
			case _LeafCmd(name=name, tree_prefix=prefix):
				print(f"{prefix}{name}")
			case _:
				assert_never(row)


async def run(task: TaskNode) -> RunResult:
	"""Execute a task tree with TUI output and return structured results."""
	expanded: Final = expand_matrix(task)
	rows: Final = _flatten_with_tree(expanded)
	leaves: Final = tuple(row for row in rows if isinstance(row, _LeafCmd))
	index_map: Final = _build_leaf_index_map(expanded)
	queue: asyncio.Queue[_TaskEvent] = asyncio.Queue()

	wall_start: Final = time.perf_counter()
	async with asyncio.TaskGroup() as tg:
		tg.create_task(_execute(expanded, queue, leaves, index_map.mapping))
		final_states: Final = await _render_loop(rows, leaves, queue, wall_start)

	_print_failures(tuple(final_states))
	results: Final = tuple(s.result for s in final_states if isinstance(s, _Done))
	elapsed: Final = time.perf_counter() - wall_start
	return RunResult(
		returncode=1 if any(r.returncode != 0 for r in results) else 0,
		results=results,
		elapsed=elapsed,
	)
