# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import functools
import itertools
import os
import sys
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from subprocess import STDOUT
from typing import Any, Final, NamedTuple, Protocol, assert_never


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

	>>> TaskResult("lint", 0, 1.234, (b"all clean",))
	TaskResult(name='lint', returncode=0, elapsed=1.234, output=(b'all clean',))
	"""

	name: str
	returncode: int
	elapsed: float
	output: Sequence[bytes]


class RunResult(NamedTuple):
	"""Result of running an entire task tree.

	>>> RunResult(0, (TaskResult("a", 0, 0.1, ()),), 0.1)
	RunResult(returncode=0, results=(TaskResult(name='a', returncode=0, elapsed=0.1, output=()),), elapsed=0.1)
	"""

	returncode: int
	results: tuple[TaskResult, ...]
	elapsed: float


class StartedEvent(NamedTuple):
	"""Internal event: a task has started execution.

	>>> StartedEvent(0, 100.0)
	StartedEvent(leaf_index=0, timestamp=100.0)
	"""

	leaf_index: int
	timestamp: float


class OutputEvent(NamedTuple):
	"""Internal event: a task produced an output line.

	>>> OutputEvent(0, b"hello", 100.5)
	OutputEvent(leaf_index=0, line=b'hello', timestamp=100.5)
	"""

	leaf_index: int
	line: bytes
	timestamp: float


class CompletedEvent(NamedTuple):
	"""Internal event: a task finished execution.

	>>> CompletedEvent(0, 0, 1.0, (b"done",))
	CompletedEvent(leaf_index=0, returncode=0, elapsed=1.0, output=(b'done',))
	"""

	leaf_index: int
	returncode: int
	elapsed: float
	output: Sequence[bytes]


type TaskEvent = StartedEvent | OutputEvent | CompletedEvent


class LeafInfo(NamedTuple):
	"""A leaf's position in the task tree: depth and last-sibling chain up to it.

	>>> LeafInfo(Task("echo hi"), 0, ())
	LeafInfo(task=Task(cmd='echo hi', name=None, env={}), depth=0, is_last_chain=())
	"""

	task: Task
	depth: int
	is_last_chain: tuple[bool, ...]


class Waiting(NamedTuple):
	"""Leaf state: task has not started yet.

	>>> Waiting(Task("echo hi"))
	Waiting(task=Task(cmd='echo hi', name=None, env={}))
	"""

	task: Task


class Running(NamedTuple):
	"""Leaf state: task is currently executing.

	>>> Running(Task("echo hi"), 100.0, b"output")
	Running(task=Task(cmd='echo hi', name=None, env={}), start_time=100.0, last_line=b'output')
	"""

	task: Task
	start_time: float
	last_line: bytes


class Done(NamedTuple):
	"""Leaf state: task completed with a result.

	>>> Done(Task("echo hi"), TaskResult("echo hi", 0, 0.5, ()))
	Done(task=Task(cmd='echo hi', name=None, env={}), result=TaskResult(name='echo hi', returncode=0, elapsed=0.5, output=()))
	"""

	task: Task
	result: TaskResult


class Skipped(NamedTuple):
	"""Leaf state: task was skipped due to a prior failure in a Sequential.

	>>> Skipped(Task("echo hi"))
	Skipped(task=Task(cmd='echo hi', name=None, env={}))
	"""

	task: Task


type LeafState = Waiting | Running | Done | Skipped


type EventSink = Callable[[int, TaskEvent], Awaitable[None]]
"""Per-leaf event dispatcher: await sink(leaf_idx, event)."""


class Effect[T](Protocol):
	"""Observer over a run's event stream, with a per-leaf context of type T.

	Each leaf owns an independent chain: setup seeds every leaf's slot,
	on_event advances the slot for the affected leaf, and teardown receives
	one final T per leaf in leaf-index order. Different leaves' on_event
	calls may run concurrently.
	"""

	async def setup(self, task: TaskNode) -> T:
		"""Return the initial per-leaf context."""
		...

	async def on_event(self, event: TaskEvent, states: Sequence[LeafState], ctx: T) -> T:
		"""Return the next context for the leaf identified by event.leaf_index."""
		...

	async def teardown(self, ctxs: tuple[T, ...]) -> None:
		"""Receive every leaf's final context, in leaf-index order."""


SKIPPED_RETURNCODE: Final = -1
AUTO_NAME_MAX_WIDTH: Final = 20


if sys.platform == "win32":  # pragma: no cover

	def resolve_cmd(cmd: str | tuple[str, ...]) -> tuple[str, ...]:
		"""Resolve a command to a tuple of strings (Windows: no shell splitting).

		>>> resolve_cmd(("echo", "hi"))
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

	def resolve_cmd(cmd: str | tuple[str, ...]) -> tuple[str, ...]:
		"""Resolve a command to a tuple of strings, splitting shell strings with shlex.

		>>> resolve_cmd(("echo", "hi"))
		('echo', 'hi')
		>>> resolve_cmd("echo hi")
		('echo', 'hi')
		"""
		match cmd:
			case str():
				return tuple(shlex.split(cmd))
			case tuple():
				return cmd
			case _:
				assert_never(cmd)


def substitute_in_str(text: str, binding: MatrixBinding) -> str:
	"""Replace {name} placeholders in a string with binding values.

	>>> substitute_in_str("test --python {PY}", (VarBinding("PY", "3.14"),))
	'test --python 3.14'
	>>> substitute_in_str("no placeholders", (VarBinding("X", "1"),))
	'no placeholders'
	"""
	return functools.reduce(
		lambda acc, vb: acc.replace(f"{{{vb.name}}}", vb.value),
		binding,
		text,
	)


def substitute_in_tuple(parts: tuple[str, ...], binding: MatrixBinding) -> tuple[str, ...]:
	"""Replace {name} placeholders in each element of a tuple.

	>>> substitute_in_tuple(("test", "--python", "{PY}"), (VarBinding("PY", "3.14"),))
	('test', '--python', '3.14')
	"""
	return functools.reduce(
		lambda acc, vb: tuple(p.replace(f"{{{vb.name}}}", vb.value) for p in acc),
		binding,
		parts,
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


def derive_name(cmd: str | tuple[str, ...], width: int) -> str:
	"""Derive a display name from a command, truncated to `width`.

	>>> derive_name("echo hi", 20)
	'echo hi'
	>>> derive_name(("python", "-c", "pass"), 20)
	'python -c pass'
	>>> derive_name("a very long command that exceeds twenty characters", 20)
	'a very l...haracters'
	"""
	return truncate_middle(
		cmd if isinstance(cmd, str) else " ".join(cmd),
		width,
	)


def task_display_name(task: Task) -> str:
	"""Return the display name for a Task: explicit `name` or auto-derived.

	>>> task_display_name(Task("echo hi", name="greet"))
	'greet'
	>>> task_display_name(Task("echo hi"))
	'echo hi'
	"""
	return task.name if task.name is not None else derive_name(task.cmd, AUTO_NAME_MAX_WIDTH)


def specialize_task(task: Task, binding: MatrixBinding, suffix: str) -> Task:
	"""Specialize a leaf Task with concrete variable values from a matrix binding.

	>>> specialize_task(Task("test {PY}"), (VarBinding("PY", "3.14"),), "[PY=3.14]")
	Task(cmd='test 3.14', name='test 3.14 [PY=3.14]', env={'PY': '3.14'})
	"""
	match task.cmd:
		case str():
			new_cmd: str | tuple[str, ...] = substitute_in_str(task.cmd, binding)
		case tuple():
			new_cmd = substitute_in_tuple(task.cmd, binding)
		case _:
			assert_never(task.cmd)
	return Task(
		cmd=new_cmd,
		name=f"{substitute_in_str(task.name if task.name is not None else derive_name(task.cmd, AUTO_NAME_MAX_WIDTH), binding)} {suffix}",
		env=task.env | dict(binding),
	)


def specialize_node(task: TaskNode, binding: MatrixBinding, suffix: str) -> TaskNode:
	"""Recursively specialize an entire task tree with concrete variable values.

	>>> specialize_node(Task("test {X}"), (VarBinding("X", "1"),), "[X=1]")
	Task(cmd='test 1', name='test 1 [X=1]', env={'X': '1'})
	"""
	match task:
		case Task():
			return specialize_task(task, binding, suffix)
		case Sequential(tasks=tasks, name=name):
			return Sequential(
				tasks=tuple(specialize_node(t, binding, suffix) for t in tasks),
				name=f"{name} {suffix}" if name is not None else None,
			)
		case Parallel(tasks=tasks, name=name):
			return Parallel(
				tasks=tuple(specialize_node(t, binding, suffix) for t in tasks),
				name=f"{name} {suffix}" if name is not None else None,
			)
		case _:
			assert_never(task)


def matrix_bindings(matrix: dict[str, tuple[str, ...]]) -> tuple[MatrixBinding, ...]:
	"""Generate all cartesian product bindings from a matrix definition.

	>>> matrix_bindings({"PY": ("3.12", "3.13")})
	((VarBinding(name='PY', value='3.12'),), (VarBinding(name='PY', value='3.13'),))
	>>> len(matrix_bindings({"A": ("1", "2"), "B": ("x", "y")}))
	4
	"""
	keys: Final = tuple(matrix.keys())
	return tuple(
		tuple(VarBinding(k, v) for k, v in zip(keys, vals))
		for vals in itertools.product(*matrix.values())
	)


def binding_suffix(binding: MatrixBinding) -> str:
	"""Format a matrix binding as a bracketed suffix.

	>>> binding_suffix((VarBinding("PY", "3.14"),))
	'[PY=3.14]'
	>>> binding_suffix((VarBinding("OS", "linux"), VarBinding("PY", "3.14")))
	'[OS=linux, PY=3.14]'
	"""
	return "[" + ", ".join(f"{vb.name}={vb.value}" for vb in binding) + "]"


def expand_sequential_matrix(
	children: tuple[TaskNode, ...],
	matrix: dict[str, tuple[str, ...]],
	name: str | None,
) -> Parallel:
	"""Expand a Sequential's matrix into a Parallel of cloned Sequentials.

	>>> result = expand_sequential_matrix((Task("build"), Task("test")), {"X": ("1", "2")}, "ci")
	>>> len(result.tasks)
	2
	>>> all(isinstance(t, Sequential) for t in result.tasks)
	True
	"""
	return Parallel(
		tasks=tuple(
			Sequential(
				tasks=tuple(
					specialize_node(child, binding, binding_suffix(binding)) for child in children
				),
				name=(f"{name} {binding_suffix(binding)}" if name is not None else None),
			)
			for binding in matrix_bindings(matrix)
		),
		name=name,
	)


def expand_parallel_matrix(
	children: tuple[TaskNode, ...],
	matrix: dict[str, tuple[str, ...]],
	name: str | None,
) -> Parallel:
	"""Expand a Parallel's matrix into a flat Parallel of all binding × child products.

	>>> result = expand_parallel_matrix((Task("test"),), {"PY": ("3.12", "3.13")}, None)
	>>> len(result.tasks)
	2
	"""
	return Parallel(
		tasks=tuple(
			specialize_node(child, binding, binding_suffix(binding))
			for binding in matrix_bindings(matrix)
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
			return expand_sequential_matrix(expanded, matrix, task.name)
		case Parallel(tasks=tasks, matrix=matrix):
			expanded = tuple(expand_matrix(t) for t in tasks)
			if matrix is None:
				return Parallel(tasks=expanded, name=task.name)
			return expand_parallel_matrix(expanded, matrix, task.name)
		case _:
			assert_never(task)


def iter_leaves(
	node: TaskNode,
	depth: int,
	is_last_chain: tuple[bool, ...],
) -> Iterator[LeafInfo]:
	"""Walk a task tree depth-first, yielding LeafInfo for each leaf."""
	match node:
		case Task():
			yield LeafInfo(node, depth, is_last_chain)
		case Sequential(tasks=children) | Parallel(tasks=children):
			last_i: Final = len(children) - 1
			for i, child in enumerate(children):
				yield from iter_leaves(child, depth + 1, (*is_last_chain, i == last_i))
		case _:
			assert_never(node)


def flatten_leaves(task: TaskNode) -> tuple[LeafInfo, ...]:
	"""Flatten a task tree into a tuple of LeafInfo in depth-first order.

	>>> [info.task.cmd for info in flatten_leaves(Parallel(tasks=(Task("a"), Task("b"))))]
	['a', 'b']
	>>> flatten_leaves(Task("echo hi"))[0].depth
	0
	>>> flatten_leaves(Parallel(tasks=(Task("a"), Task("b"))))[0].is_last_chain
	(False,)
	>>> flatten_leaves(Parallel(tasks=(Task("a"), Task("b"))))[1].is_last_chain
	(True,)
	"""
	return tuple(iter_leaves(task, depth=0, is_last_chain=()))


def build_leaf_index_map(task: TaskNode) -> dict[int, int]:
	"""Map `id(Task)` to leaf index (depth-first position) for the whole tree.

	>>> t1, t2 = Task("a"), Task("b")
	>>> m = build_leaf_index_map(Parallel(tasks=(t1, t2)))
	>>> m[id(t1)], m[id(t2)]
	(0, 1)
	"""
	return {id(info.task): i for i, info in enumerate(flatten_leaves(task))}


def subtree_leaf_indices(task: TaskNode, index_map: dict[int, int]) -> tuple[int, ...]:
	"""Collect all leaf indices within a subtree.

	>>> t1, t2 = Task("a"), Task("b")
	>>> tree = Parallel(tasks=(t1, t2))
	>>> subtree_leaf_indices(tree, build_leaf_index_map(tree))
	(0, 1)
	"""
	match task:
		case Task():
			return (index_map[id(task)],)
		case Sequential(tasks=tasks) | Parallel(tasks=tasks):
			return tuple(i for child in tasks for i in subtree_leaf_indices(child, index_map))
		case _:
			assert_never(task)


def next_state(state: LeafState, event: TaskEvent) -> LeafState:
	"""Pure state machine: apply a TaskEvent to a LeafState to produce the next state.

	>>> t = Task("echo hi")
	>>> next_state(Waiting(t), StartedEvent(0, 100.0))
	Running(task=Task(cmd='echo hi', name=None, env={}), start_time=100.0, last_line=b'')
	>>> next_state(Running(t, 100.0, b""), OutputEvent(0, b"hi", 100.5))
	Running(task=Task(cmd='echo hi', name=None, env={}), start_time=100.0, last_line=b'hi')
	>>> next_state(Running(t, 100.0, b""), CompletedEvent(0, 0, 0.5, (b"done",)))
	Done(task=Task(cmd='echo hi', name=None, env={}), result=TaskResult(name='echo hi', returncode=0, elapsed=0.5, output=(b'done',)))
	>>> next_state(Waiting(t), CompletedEvent(0, SKIPPED_RETURNCODE, 0.0, ()))
	Skipped(task=Task(cmd='echo hi', name=None, env={}))
	"""
	match state:
		case Waiting(task=task):
			match event:
				case StartedEvent(timestamp=ts):
					return Running(task, ts, b"")
				case OutputEvent(line=line, timestamp=ts):  # pragma: no cover
					return Running(task, ts, line)
				case CompletedEvent(returncode=rc, elapsed=elapsed, output=output):
					if rc == SKIPPED_RETURNCODE:
						return Skipped(task)
					return Done(
						task, TaskResult(task_display_name(task), rc, elapsed, output)
					)  # pragma: no cover
				case _:
					assert_never(event)
		case Running(task=task, start_time=start):
			match event:
				case OutputEvent(line=line):
					return Running(task, start, line)
				case CompletedEvent(returncode=rc, elapsed=elapsed, output=output):
					if rc == SKIPPED_RETURNCODE:  # pragma: no cover
						return Skipped(task)
					return Done(task, TaskResult(task_display_name(task), rc, elapsed, output))
				case StartedEvent():  # pragma: no cover
					return state
				case _:
					assert_never(event)
		case Done() | Skipped():  # pragma: no cover
			return state
		case _:
			assert_never(state)


async def run_cmd(task: Task, leaf_index: int, dispatch: EventSink) -> TaskResult:
	"""Run one leaf as a subprocess, dispatching Started/Output/Completed events."""
	start: Final = time.perf_counter()
	await dispatch(leaf_index, StartedEvent(leaf_index, start))
	proc: Final = await asyncio.create_subprocess_exec(
		*resolve_cmd(task.cmd),
		stdout=asyncio.subprocess.PIPE,
		stderr=STDOUT,
		env=os.environ | task.env,
	)
	output: Final[list[bytes]] = []
	if proc.stdout is not None:  # pragma: no branch
		async for line in proc.stdout:
			output.append(line)
			await dispatch(leaf_index, OutputEvent(leaf_index, line, time.perf_counter()))
	await proc.wait()
	elapsed: Final = time.perf_counter() - start
	rc: Final = proc.returncode or 0
	await dispatch(leaf_index, CompletedEvent(leaf_index, rc, elapsed, output))
	return TaskResult(task_display_name(task), rc, elapsed, output)


async def execute(
	node: TaskNode,
	dispatch: EventSink,
	leaves: tuple[Task, ...],
	index_map: dict[int, int],
) -> tuple[TaskResult, ...]:
	"""Walk a task subtree, returning one TaskResult per leaf in DFS order."""
	match node:
		case Task():
			return (await run_cmd(node, index_map[id(node)], dispatch),)
		case Parallel(tasks=children):
			async with asyncio.TaskGroup() as tg:
				futures: Final = tuple(
					tg.create_task(execute(child, dispatch, leaves, index_map))
					for child in children
				)
			return tuple(r for f in futures for r in f.result())
		case Sequential(tasks=children):
			seq_results: tuple[TaskResult, ...] = ()
			failed = False
			for child in children:
				if failed:
					for skipped_idx in subtree_leaf_indices(child, index_map):
						await dispatch(
							skipped_idx, CompletedEvent(skipped_idx, SKIPPED_RETURNCODE, 0.0, ())
						)
						seq_results = (
							*seq_results,
							TaskResult(
								task_display_name(leaves[skipped_idx]), SKIPPED_RETURNCODE, 0.0, ()
							),
						)
				else:
					child_results = await execute(child, dispatch, leaves, index_map)
					seq_results = (*seq_results, *child_results)
					if any(r.returncode != 0 for r in child_results):
						failed = True
			return seq_results
		case _:
			assert_never(node)


async def run(task: TaskNode, effects: Sequence[Effect[Any]] = ()) -> RunResult:
	"""Execute a task tree, dispatching events to every effect.

	>>> import asyncio
	>>> asyncio.run(run(Task(("python", "-c", "pass")))).returncode
	0
	>>> asyncio.run(run(Task(("python", "-c", "raise SystemExit(1)")))).returncode
	1
	"""
	expanded: Final = expand_matrix(task)
	leaf_infos: Final = flatten_leaves(expanded)
	leaves: Final = tuple(info.task for info in leaf_infos)
	index_map: Final = {id(info.task): i for i, info in enumerate(leaf_infos)}

	wall_start: Final = time.perf_counter()
	setup_results: Final = await asyncio.gather(
		*(effect.setup(expanded) for effect in effects),
		return_exceptions=True,
	)
	active_effects: Final = tuple(
		e for e, r in zip(effects, setup_results) if not isinstance(r, BaseException)
	)
	setup_errors: Final = tuple(r for r in setup_results if isinstance(r, BaseException))
	ctx_grid: Final[list[list[Any]]] = [
		[r for r in setup_results if not isinstance(r, BaseException)] for _ in leaf_infos
	]
	states: Final[list[LeafState]] = [Waiting(info.task) for info in leaf_infos]

	async def dispatch(leaf_idx: int, event: TaskEvent) -> None:
		states[leaf_idx] = next_state(states[leaf_idx], event)
		slot: Final = ctx_grid[leaf_idx]
		for effect_idx, effect_ctx in enumerate(
			await asyncio.gather(
				*(
					effect.on_event(event, states, ctx)
					for effect, ctx in zip(active_effects, slot, strict=True)
				)
			)
		):
			slot[effect_idx] = effect_ctx

	results: tuple[TaskResult, ...] = ()
	try:
		if setup_errors:
			raise BaseExceptionGroup("setup errors", setup_errors)
		results = await execute(expanded, dispatch, leaves, index_map)
	finally:
		teardown_errors: Final = tuple(
			r
			for r in await asyncio.gather(
				*(
					effect.teardown(tuple(row[effect_idx] for row in ctx_grid))
					for effect_idx, effect in enumerate(active_effects)
				),
				return_exceptions=True,
			)
			if isinstance(r, BaseException)
		)
		if teardown_errors:
			raise BaseExceptionGroup("teardown errors", teardown_errors)
	return RunResult(
		returncode=1 if any(r.returncode != 0 for r in results) else 0,
		results=results,
		elapsed=time.perf_counter() - wall_start,
	)
