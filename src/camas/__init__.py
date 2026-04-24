# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import functools
import itertools
import os
import shlex
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from subprocess import STDOUT
from typing import Any, Final, NamedTuple, Protocol, TypeAlias, TypeVar

if sys.version_info >= (3, 11):
	from asyncio import TaskGroup
	from builtins import BaseExceptionGroup
	from typing import assert_never
else:  # pragma: no cover
	from exceptiongroup import BaseExceptionGroup
	from taskgroup import TaskGroup
	from typing_extensions import assert_never


class VarBinding(NamedTuple):
	"""A single matrix variable bound to a concrete value.

	>>> VarBinding("PY", "3.14")
	VarBinding(name='PY', value='3.14')
	"""

	name: str
	value: str


MatrixBinding: TypeAlias = tuple[VarBinding, ...]


class Task(NamedTuple):
	"""A leaf task that executes a shell command.

	>>> Task("echo hi")
	Task(cmd='echo hi', name=None, env={}, cwd=None)
	>>> Task(("ruff", "check", "."), name="lint")
	Task(cmd=('ruff', 'check', '.'), name='lint', env={}, cwd=None)
	>>> Task("cargo test", cwd=Path("src-tauri")).cwd
	PosixPath('src-tauri')
	"""

	cmd: str | tuple[str, ...]
	name: str | None = None
	env: dict[str, str] = {}
	cwd: Path | None = None


class Sequential(NamedTuple):
	"""A group of tasks that run one after another, short-circuiting on failure.

	>>> Sequential(tasks=(Task("build"), Task("test")), name="ci")
	Sequential(tasks=(Task(cmd='build', name=None, env={}, cwd=None), Task(cmd='test', name=None, env={}, cwd=None)), name='ci', matrix=None, env={}, cwd=None)
	"""

	tasks: tuple[TaskNode, ...]
	name: str | None = None
	matrix: dict[str, tuple[str, ...]] | None = None
	env: dict[str, str] = {}
	cwd: Path | None = None


class Parallel(NamedTuple):
	"""A group of tasks that run concurrently.

	>>> Parallel(tasks=(Task("lint"), Task("typecheck")))
	Parallel(tasks=(Task(cmd='lint', name=None, env={}, cwd=None), Task(cmd='typecheck', name=None, env={}, cwd=None)), name=None, matrix=None, env={}, cwd=None)
	"""

	tasks: tuple[TaskNode, ...]
	name: str | None = None
	matrix: dict[str, tuple[str, ...]] | None = None
	env: dict[str, str] = {}
	cwd: Path | None = None


TaskNode: TypeAlias = Task | Sequential | Parallel


class Finished(NamedTuple):
	"""Completion outcome: task ran to exit with a returncode.

	>>> Finished(0, 1.234, (b"all clean",))
	Finished(returncode=0, elapsed=1.234, output=(b'all clean',))
	"""

	returncode: int
	elapsed: float
	output: Sequence[bytes]


class Skipped(NamedTuple):
	"""Completion outcome: task was skipped due to a prior Sequential failure.

	Carries the returncode of the task that caused the skip — an Either-like
	propagation so callers that need an rc (e.g. the overall run's exit code)
	can read it uniformly across completion variants.

	>>> Skipped(1)
	Skipped(returncode=1)
	"""

	returncode: int


Completion: TypeAlias = Finished | Skipped


class TaskResult(NamedTuple):
	"""Result of a single completed task.

	>>> TaskResult("lint", Finished(0, 1.234, (b"all clean",)))
	TaskResult(name='lint', completion=Finished(returncode=0, elapsed=1.234, output=(b'all clean',)))
	"""

	name: str
	completion: Completion


class RunResult(NamedTuple):
	"""Result of running an entire task tree.

	>>> RunResult(0, (TaskResult("a", Finished(0, 0.1, ())),), 0.1)
	RunResult(returncode=0, results=(TaskResult(name='a', completion=Finished(returncode=0, elapsed=0.1, output=())),), elapsed=0.1)
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
	"""Internal event: a task finished execution (either ran or was skipped).

	>>> CompletedEvent(0, Finished(0, 1.0, (b"done",)))
	CompletedEvent(leaf_index=0, completion=Finished(returncode=0, elapsed=1.0, output=(b'done',)))
	>>> CompletedEvent(0, Skipped(1))
	CompletedEvent(leaf_index=0, completion=Skipped(returncode=1))
	"""

	leaf_index: int
	completion: Completion


TaskEvent: TypeAlias = StartedEvent | OutputEvent | CompletedEvent


class ChainLink(NamedTuple):
	"""One ancestor's position in a tree-walk chain.

	>>> ChainLink(True, False)
	ChainLink(is_last=True, parent_is_parallel=False)
	"""

	is_last: bool
	"""Whether the current node is the last child in its sibling group. Used for display formatting."""
	parent_is_parallel: bool
	"""Whether the containing group is a ``Parallel``. Used for display formatting."""


class LeafInfo(NamedTuple):
	"""A leaf's position in the task tree: depth and chain of ancestor links.

	>>> LeafInfo(Task("echo hi"), 0, ())
	LeafInfo(task=Task(cmd='echo hi', name=None, env={}, cwd=None), depth=0, is_last_chain=())
	"""

	task: Task
	depth: int
	is_last_chain: tuple[ChainLink, ...]


class Waiting(NamedTuple):
	"""Leaf state: task has not started yet.

	>>> Waiting(Task("echo hi"))
	Waiting(task=Task(cmd='echo hi', name=None, env={}, cwd=None))
	"""

	task: Task


class Running(NamedTuple):
	"""Leaf state: task is currently executing.

	>>> Running(Task("echo hi"), 100.0, b"output")
	Running(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=100.0, last_line=b'output')
	"""

	task: Task
	start_time: float
	last_line: bytes


class Completed(NamedTuple):
	"""Leaf state: task is done — either ran to exit or was skipped.

	The `completion` payload is a sum type: pattern-match on `Finished(...)`
	vs `Skipped(...)` to distinguish the two cases.

	>>> Completed(Task("echo hi"), Finished(0, 0.5, ()))
	Completed(task=Task(cmd='echo hi', name=None, env={}, cwd=None), completion=Finished(returncode=0, elapsed=0.5, output=()))
	>>> Completed(Task("echo hi"), Skipped(1))
	Completed(task=Task(cmd='echo hi', name=None, env={}, cwd=None), completion=Skipped(returncode=1))
	"""

	task: Task
	completion: Completion


LeafState: TypeAlias = Waiting | Running | Completed


EventSink: TypeAlias = Callable[[int, TaskEvent], Awaitable[None]]
"""Per-leaf event dispatcher: await sink(leaf_idx, event)."""


T = TypeVar("T")


class Effect(Protocol[T]):
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


def resolve_cmd(cmd: str | tuple[str, ...]) -> tuple[str, ...]:
	"""Resolve a command to a tuple of argv tokens, splitting shell strings with shlex.

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


def task_label(task: Task) -> str:
	"""Return a task's identifying label: the explicit `name` or the full command string.

	This is a data accessor with no concept of display width — callers that render
	into a column-constrained terminal are responsible for truncation.

	>>> task_label(Task("echo hi", name="greet"))
	'greet'
	>>> task_label(Task("echo hi"))
	'echo hi'
	>>> task_label(Task(("python", "-c", "pass")))
	'python -c pass'
	"""
	if task.name is not None:
		return task.name
	return task.cmd if isinstance(task.cmd, str) else " ".join(task.cmd)


def _substitute_cwd(cwd: Path | None, binding: MatrixBinding) -> Path | None:
	return Path(substitute_in_str(str(cwd), binding)) if cwd is not None else None


def specialize_task(task: Task, binding: MatrixBinding, suffix: str) -> Task:
	"""Specialize a leaf Task with concrete variable values from a matrix binding.

	>>> specialize_task(Task("test {PY}"), (VarBinding("PY", "3.14"),), "[PY=3.14]")
	Task(cmd='test 3.14', name='test 3.14 [PY=3.14]', env={'PY': '3.14'}, cwd=None)
	>>> specialize_task(Task("go", env={"VENV": ".venv-{PY}"}), (VarBinding("PY", "3.14"),), "[PY=3.14]").env
	{'VENV': '.venv-3.14', 'PY': '3.14'}
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
		name=f"{substitute_in_str(task_label(task), binding)} {suffix}",
		env={k: substitute_in_str(v, binding) for k, v in task.env.items()} | dict(binding),
		cwd=_substitute_cwd(task.cwd, binding),
	)


def specialize_node(task: TaskNode, binding: MatrixBinding, suffix: str) -> TaskNode:
	"""Recursively specialize an entire task tree with concrete variable values.

	>>> specialize_node(Task("test {X}"), (VarBinding("X", "1"),), "[X=1]")
	Task(cmd='test 1', name='test 1 [X=1]', env={'X': '1'}, cwd=None)
	"""
	match task:
		case Task():
			return specialize_task(task, binding, suffix)
		case Sequential(tasks=tasks, name=name, cwd=cwd):
			return Sequential(
				tasks=tuple(specialize_node(t, binding, suffix) for t in tasks),
				name=f"{name} {suffix}" if name is not None else None,
				cwd=_substitute_cwd(cwd, binding),
			)
		case Parallel(tasks=tasks, name=name, cwd=cwd):
			return Parallel(
				tasks=tuple(specialize_node(t, binding, suffix) for t in tasks),
				name=f"{name} {suffix}" if name is not None else None,
				cwd=_substitute_cwd(cwd, binding),
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
	container_env: dict[str, str],
	container_cwd: Path | None,
) -> Parallel:
	"""Expand a Sequential's matrix into a Parallel of cloned Sequentials.

	Each per-binding Sequential carries the binding-scope env (container env with
	matrix values substituted, plus the binding itself) so the display can show
	it once at the group header instead of on every leaf.

	>>> result = expand_sequential_matrix((Task("build"), Task("test")), {"X": ("1", "2")}, "ci", {}, None)
	>>> len(result.tasks)
	2
	>>> all(isinstance(t, Sequential) for t in result.tasks)
	True
	>>> result.tasks[0].env  # type: ignore[union-attr]
	{'X': '1'}
	"""
	return Parallel(
		tasks=tuple(
			Sequential(
				tasks=tuple(
					specialize_node(child, binding, binding_suffix(binding)) for child in children
				),
				name=(f"{name} {binding_suffix(binding)}" if name is not None else None),
				env={k: substitute_in_str(v, binding) for k, v in container_env.items()}
				| dict(binding),
				cwd=_substitute_cwd(container_cwd, binding),
			)
			for binding in matrix_bindings(matrix)
		),
		name=name,
	)


def expand_parallel_matrix(
	children: tuple[TaskNode, ...],
	matrix: dict[str, tuple[str, ...]],
	name: str | None,
	container_env: dict[str, str],
	container_cwd: Path | None,
) -> Parallel:
	"""Expand a Parallel's matrix into a flat Parallel of all binding × child products.

	The shared ``container_env`` is kept on the outer Parallel only when it has the
	same value across every binding; per-binding pieces land on the individual
	specialized leaves via ``specialize_node``.

	>>> result = expand_parallel_matrix((Task("test"),), {"PY": ("3.12", "3.13")}, None, {}, None)
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
		env={k: v for k, v in container_env.items() if "{" not in v},
		cwd=container_cwd if container_cwd is not None and "{" not in str(container_cwd) else None,
	)


def expand_matrix(
	task: TaskNode,
	ancestor_env: Mapping[str, str] | None = None,
	ancestor_cwd: Path | None = None,
) -> TaskNode:
	"""Recursively expand all matrix parameters and propagate container env/cwd into leaves.

	Container env and cwd are also retained on the expanded group nodes (for display
	purposes); execution still reads the accumulated values from leaves. A child's
	``cwd`` takes precedence over an ancestor's.

	>>> expand_matrix(Task("echo hi"))
	Task(cmd='echo hi', name=None, env={}, cwd=None)
	>>> result = expand_matrix(Parallel(tasks=(Task("test"),), matrix={"X": ("1", "2")}))
	>>> len(result.tasks)  # type: ignore[union-attr]
	2
	>>> expand_matrix(Parallel(tasks=(Task("hi"),), env={"K": "v"})).tasks[0].env  # type: ignore[union-attr]
	{'K': 'v'}
	>>> expand_matrix(Parallel(tasks=(Task("hi"),), cwd=Path("w"))).tasks[0].cwd  # type: ignore[union-attr]
	PosixPath('w')
	"""
	parent_env: Final = dict(ancestor_env) if ancestor_env else {}
	match task:
		case Task():
			return Task(
				cmd=task.cmd,
				name=task.name,
				env=parent_env | task.env,
				cwd=task.cwd if task.cwd is not None else ancestor_cwd,
			)
		case Sequential(tasks=tasks, matrix=matrix, env=env, cwd=cwd):
			seq_env: Final = parent_env | env
			seq_cwd: Final = cwd if cwd is not None else ancestor_cwd
			seq_expanded: Final = tuple(expand_matrix(t, seq_env, seq_cwd) for t in tasks)
			if matrix is None:
				return Sequential(tasks=seq_expanded, name=task.name, env=env, cwd=cwd)
			return expand_sequential_matrix(seq_expanded, matrix, task.name, env, cwd)
		case Parallel(tasks=tasks, matrix=matrix, env=env, cwd=cwd):
			par_env: Final = parent_env | env
			par_cwd: Final = cwd if cwd is not None else ancestor_cwd
			par_expanded: Final = tuple(expand_matrix(t, par_env, par_cwd) for t in tasks)
			if matrix is None:
				return Parallel(tasks=par_expanded, name=task.name, env=env, cwd=cwd)
			return expand_parallel_matrix(par_expanded, matrix, task.name, env, cwd)
		case _:
			assert_never(task)


def iter_leaves(
	node: TaskNode,
	depth: int,
	is_last_chain: tuple[ChainLink, ...],
) -> Iterator[LeafInfo]:
	"""Walk a task tree depth-first, yielding LeafInfo for each leaf."""
	match node:
		case Task():
			yield LeafInfo(node, depth, is_last_chain)
		case Sequential(tasks=children) | Parallel(tasks=children):
			parent_is_par: Final = isinstance(node, Parallel)
			last_i: Final = len(children) - 1
			for i, child in enumerate(children):
				link = ChainLink(is_last=i == last_i, parent_is_parallel=parent_is_par)
				yield from iter_leaves(child, depth + 1, (*is_last_chain, link))
		case _:
			assert_never(node)


def flatten_leaves(task: TaskNode) -> tuple[LeafInfo, ...]:
	"""Flatten a task tree into a tuple of LeafInfo in depth-first order.

	>>> [info.task.cmd for info in flatten_leaves(Parallel(tasks=(Task("a"), Task("b"))))]
	['a', 'b']
	>>> flatten_leaves(Task("echo hi"))[0].depth
	0
	>>> flatten_leaves(Parallel(tasks=(Task("a"), Task("b"))))[0].is_last_chain
	(ChainLink(is_last=False, parent_is_parallel=True),)
	>>> flatten_leaves(Parallel(tasks=(Task("a"), Task("b"))))[1].is_last_chain
	(ChainLink(is_last=True, parent_is_parallel=True),)
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
	Running(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=100.0, last_line=b'')
	>>> next_state(Running(t, 100.0, b""), OutputEvent(0, b"hi", 100.5))
	Running(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=100.0, last_line=b'hi')
	>>> next_state(Running(t, 100.0, b""), CompletedEvent(0, Finished(0, 0.5, (b"done",))))
	Completed(task=Task(cmd='echo hi', name=None, env={}, cwd=None), completion=Finished(returncode=0, elapsed=0.5, output=(b'done',)))
	>>> next_state(Waiting(t), CompletedEvent(0, Skipped(1)))
	Completed(task=Task(cmd='echo hi', name=None, env={}, cwd=None), completion=Skipped(returncode=1))
	"""
	match state:
		case Waiting(task=task):
			match event:
				case StartedEvent(timestamp=ts):
					return Running(task, ts, b"")
				case OutputEvent(line=line, timestamp=ts):  # pragma: no cover
					return Running(task, ts, line)
				case CompletedEvent(completion=completion):
					return Completed(task, completion)
				case _:
					assert_never(event)
		case Running(task=task, start_time=start):
			match event:
				case OutputEvent(line=line):
					return Running(task, start, line)
				case CompletedEvent(completion=completion):
					return Completed(task, completion)
				case StartedEvent():  # pragma: no cover
					return state
				case _:
					assert_never(event)
		case Completed():  # pragma: no cover
			return state
		case _:
			assert_never(state)


def _color_env(merged: dict[str, str]) -> dict[str, str]:
	"""Inject FORCE_COLOR/CLICOLOR_FORCE unless NO_COLOR is set (which also strips them)."""
	if "NO_COLOR" in merged:
		return {k: v for k, v in merged.items() if k not in {"FORCE_COLOR", "CLICOLOR_FORCE"}}
	return merged | {"FORCE_COLOR": "1", "CLICOLOR_FORCE": "1"}


async def run_cmd(task: Task, leaf_index: int, dispatch: EventSink) -> TaskResult:
	"""Run one leaf as a subprocess, dispatching Started/Output/Completed events."""
	start: Final = time.perf_counter()
	await dispatch(leaf_index, StartedEvent(leaf_index, start))
	proc: Final = await asyncio.create_subprocess_exec(
		*resolve_cmd(task.cmd),
		stdout=asyncio.subprocess.PIPE,
		stderr=STDOUT,
		env=_color_env(os.environ | task.env),
		cwd=task.cwd,
	)
	output: Final[list[bytes]] = []
	if proc.stdout is not None:  # pragma: no branch
		async for line in proc.stdout:
			output.append(line)
			await dispatch(leaf_index, OutputEvent(leaf_index, line, time.perf_counter()))
	await proc.wait()
	elapsed: Final = time.perf_counter() - start
	rc: Final = proc.returncode or 0
	completion: Final = Finished(rc, elapsed, output)
	await dispatch(leaf_index, CompletedEvent(leaf_index, completion))
	return TaskResult(task_label(task), completion)


async def skip_subtree(
	child: TaskNode,
	skip: Skipped,
	dispatch: EventSink,
	leaves: tuple[Task, ...],
	index_map: dict[int, int],
) -> tuple[TaskResult, ...]:
	"""Dispatch a Skipped completion for every leaf in a subtree, in DFS order."""
	results: tuple[TaskResult, ...] = ()
	for idx in subtree_leaf_indices(child, index_map):
		await dispatch(idx, CompletedEvent(idx, skip))
		results = (*results, TaskResult(task_label(leaves[idx]), skip))
	return results


def first_failure_rc(results: Iterable[TaskResult]) -> int | None:
	"""Return the first non-zero returncode in ``results``, or ``None`` if all are zero.

	>>> first_failure_rc((TaskResult("a", Finished(0, 0.1, ())),))
	>>> first_failure_rc((TaskResult("a", Finished(0, 0.1, ())), TaskResult("b", Finished(2, 0.1, ()))))
	2
	"""
	return next(
		(r.completion.returncode for r in results if r.completion.returncode != 0),
		None,
	)


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
			async with TaskGroup() as tg:
				futures: Final = tuple(
					tg.create_task(execute(child, dispatch, leaves, index_map))
					for child in children
				)
			return tuple(r for f in futures for r in f.result())
		case Sequential(tasks=children):
			seq_results: tuple[TaskResult, ...] = ()
			failed_rc: int | None = None
			for child in children:
				child_results = (
					await skip_subtree(child, Skipped(failed_rc), dispatch, leaves, index_map)
					if failed_rc is not None
					else await execute(child, dispatch, leaves, index_map)
				)
				seq_results = (*seq_results, *child_results)
				if failed_rc is None:
					failed_rc = first_failure_rc(child_results)
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
		returncode=1 if any(r.completion.returncode != 0 for r in results) else 0,
		results=results,
		elapsed=time.perf_counter() - wall_start,
	)
