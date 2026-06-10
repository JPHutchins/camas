# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Async tree execution: leaves run as subprocesses, events fan out to Effects."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime
from subprocess import STDOUT
from typing import TYPE_CHECKING, Any, Final

if sys.version_info >= (3, 11):
	from asyncio import TaskGroup
	from builtins import BaseExceptionGroup
	from typing import assert_never
else:  # pragma: no cover
	from exceptiongroup import BaseExceptionGroup
	from taskgroup import TaskGroup
	from typing_extensions import assert_never

from .completion import Finished, RunResult, Skipped, TaskResult
from .leaf_state import LeafState, Waiting, next_state
from .matrix import expand_matrix, resolve_cmd
from .task import Parallel, Sequential, Task, TaskNode, task_label
from .task_event import CompletedEvent, OutputEvent, StartedEvent, TaskEvent
from .traversal import flatten_leaves, subtree_leaf_indices

if TYPE_CHECKING:
	from collections.abc import Iterable, Sequence

	from .effect import Effect, EventSink


def subprocess_env(merged: dict[str, str]) -> dict[str, str]:
	"""Leaf-subprocess env: defaults merged underneath ``merged``; ``NO_COLOR`` strips color forces."""
	base = {"PYTHONUNBUFFERED": "1"} | merged
	if "NO_COLOR" in base:
		return {k: v for k, v in base.items() if k not in {"FORCE_COLOR", "CLICOLOR_FORCE"}}
	return {"FORCE_COLOR": "1", "CLICOLOR_FORCE": "1"} | base


async def run_cmd(task: Task, leaf_index: int, dispatch: EventSink) -> TaskResult:
	"""Run one leaf as a subprocess, dispatching Started/Output/Completed events."""
	start_pc: Final = time.perf_counter()
	await dispatch(leaf_index, StartedEvent(task, leaf_index, datetime.now()))
	proc: Final = await asyncio.create_subprocess_exec(
		*resolve_cmd(task.cmd),
		stdout=asyncio.subprocess.PIPE,
		stderr=STDOUT,
		env=subprocess_env({**os.environ, **task.env}),
		cwd=task.cwd,
	)
	output: Final[list[bytes]] = []
	if proc.stdout is not None:  # pragma: no branch
		async for line in proc.stdout:
			output.append(line)
			await dispatch(leaf_index, OutputEvent(task, leaf_index, line, datetime.now()))
	await proc.wait()
	elapsed: Final = time.perf_counter() - start_pc
	rc: Final = proc.returncode or 0
	completion: Final = Finished(rc, elapsed, output)
	await dispatch(leaf_index, CompletedEvent(task, leaf_index, completion, datetime.now()))
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
		await dispatch(idx, CompletedEvent(leaves[idx], idx, skip, datetime.now()))
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

	Raises:
		BaseExceptionGroup: every error raised by Effects during setup,
			on_event, or teardown, collected per phase.

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
		e for e, r in zip(effects, setup_results, strict=True) if not isinstance(r, BaseException)
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
