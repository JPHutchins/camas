from __future__ import annotations

import asyncio
import functools
import itertools
import os
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Sequence
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
	"""Structural information about a leaf Task in its position within a tree.

	`is_last_chain[i]` is whether the node at level `i + 1` (root is level 0) is
	the last sibling at its level. Length equals `depth`; the final entry
	corresponds to the leaf itself. Effects reconstitute tree presentation
	(ASCII prefixes, indentation, hierarchical keys, ...) from these fields.

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


type EventSink = Callable[[TaskEvent], Awaitable[None]]


type Effect = Callable[[TaskNode, AsyncIterator[TaskEvent]], Coroutine[object, object, None]]
"""An Effect is an async callable consuming the event stream of a run.

Effects run concurrently inside the runner's TaskGroup alongside task execution;
the event stream terminates (via StopAsyncIteration) once execution finishes.

To configure an effect with options, define a factory that returns an Effect:
the options are closed over, and the outer factory encodes the shape of the
configuration as its parameter type."""


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

	Always returns a string of length ``min(len(text), max(max_width, 0))``.

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


async def run_cmd(task: Task, leaf_index: int, sink: EventSink) -> TaskResult:
	"""Execute a single leaf command as a subprocess, streaming events to the sink."""
	start: Final = time.perf_counter()
	await sink(StartedEvent(leaf_index, start))
	proc: Final = await asyncio.create_subprocess_exec(
		*resolve_cmd(task.cmd),
		stdout=asyncio.subprocess.PIPE,
		stderr=STDOUT,
		env=os.environ | task.env,
	)
	output: list[bytes] = []
	if proc.stdout is not None:  # pragma: no branch
		async for line in proc.stdout:
			output.append(line)
			await sink(OutputEvent(leaf_index, line, time.perf_counter()))
	await proc.wait()
	elapsed: Final = time.perf_counter() - start
	rc: Final = proc.returncode or 0
	await sink(CompletedEvent(leaf_index, rc, elapsed, output))
	return TaskResult(task_display_name(task), rc, elapsed, output)


async def execute(
	node: TaskNode,
	sink: EventSink,
	leaves: tuple[Task, ...],
	index_map: dict[int, int],
) -> tuple[TaskResult, ...]:
	"""Recursively execute a task tree, respecting Sequential/Parallel semantics."""
	match node:
		case Task():
			idx: Final = index_map[id(node)]
			return (await run_cmd(node, idx, sink),)
		case Parallel(tasks=children):
			parallel_results: list[TaskResult] = []
			async with asyncio.TaskGroup() as tg:
				futures: Final = [
					tg.create_task(execute(child, sink, leaves, index_map)) for child in children
				]
			for future in futures:
				parallel_results.extend(future.result())
			return tuple(parallel_results)
		case Sequential(tasks=children):
			seq_results: list[TaskResult] = []
			failed = False
			for child in children:
				if failed:
					for skipped_idx in subtree_leaf_indices(child, index_map):
						leaf = leaves[skipped_idx]
						await sink(CompletedEvent(skipped_idx, SKIPPED_RETURNCODE, 0.0, ()))
						seq_results.append(
							TaskResult(task_display_name(leaf), SKIPPED_RETURNCODE, 0.0, ())
						)
				else:
					child_results = await execute(child, sink, leaves, index_map)
					seq_results.extend(child_results)
					if any(r.returncode != 0 for r in child_results):
						failed = True
			return tuple(seq_results)
		case _:
			assert_never(node)


async def queue_iter(q: asyncio.Queue[TaskEvent | None]) -> AsyncIterator[TaskEvent]:
	"""Drain a queue as an async iterator, terminating on the `None` sentinel."""
	while True:
		event = await q.get()
		if event is None:
			return
		yield event


async def run(task: TaskNode, effects: Sequence[Effect] = ()) -> RunResult:
	"""Execute a task tree, fanning events out to each effect in parallel.

	The engine itself is silent — stdout is untouched unless an effect writes.
	Each effect receives the matrix-expanded tree and an async iterator of
	`TaskEvent`s in emission order; iteration terminates when execution finishes.
	"""
	expanded: Final = expand_matrix(task)
	leaf_infos: Final = flatten_leaves(expanded)
	leaves: Final = tuple(info.task for info in leaf_infos)
	index_map: Final = {id(info.task): i for i, info in enumerate(leaf_infos)}
	queues: Final[tuple[asyncio.Queue[TaskEvent | None], ...]] = tuple(
		asyncio.Queue() for _ in effects
	)

	async def sink(event: TaskEvent) -> None:
		for q in queues:
			await q.put(event)

	wall_start: Final = time.perf_counter()
	async with asyncio.TaskGroup() as tg:
		for effect, q in zip(effects, queues, strict=True):
			tg.create_task(effect(expanded, queue_iter(q)))

		async def drive() -> tuple[TaskResult, ...]:
			try:
				return await execute(expanded, sink, leaves, index_map)
			finally:
				for q in queues:
					await q.put(None)

		driver: Final = tg.create_task(drive())

	results: Final = driver.result()
	elapsed: Final = time.perf_counter() - wall_start
	return RunResult(
		returncode=1 if any(r.returncode != 0 for r in results) else 0,
		results=results,
		elapsed=elapsed,
	)
