# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Async tree execution: leaves run as subprocesses, events fan out to Effects."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from subprocess import STDOUT
from typing import TYPE_CHECKING, Any, Final, Protocol, TypeAlias

if sys.version_info >= (3, 11):
	from asyncio import TaskGroup
	from builtins import BaseExceptionGroup
	from typing import assert_never
else:  # pragma: no cover
	from exceptiongroup import BaseExceptionGroup
	from taskgroup import TaskGroup
	from typing_extensions import assert_never

from ..v0.completion import INTERRUPT_RC, Finished, Skipped, Stopped
from ..v0.leaf_state import LeafState, Waiting
from ..v0.task import Parallel, Sequential, Task, TaskNode
from ..v0.task_event import (
	AbortedEvent,
	CompletedEvent,
	InterruptedEvent,
	OutputEvent,
	StartedEvent,
	TaskEvent,
)
from .completion import RunResult, TaskResult
from .leaf_state import next_state
from .matrix import expand_matrix, resolve_cmd
from .task import task_label
from .traversal import flatten_leaves, subtree_leaf_indices

if TYPE_CHECKING:
	from collections.abc import Callable, Sequence

	from ..v0.effect import Effect
	from .effect import EventSink


Limiter: TypeAlias = "asyncio.Semaphore | nullcontext[None]"
"""Throttles concurrent leaf subprocesses under ``--jobs``; a no-op when unbounded."""


class Signalable(Protocol):
	"""The subset of ``asyncio.subprocess.Process`` the interrupt path drives."""

	def send_signal(self, sig: int, /) -> None: ...
	def kill(self) -> None: ...


@dataclass
class Interrupts:
	"""Mutable Ctrl-C coordinator threaded through a run, in the style of ``dispatch``."""

	procs: dict[int, Signalable]
	"""Live leaf subprocesses by leaf index; populated/popped by ``run_cmd``."""
	signaled: set[int]
	"""Leaves that were forwarded a signal, so their completion is tagged ``Stopped``."""
	pending: list[asyncio.Task[None]]
	"""Dispatch tasks the handler spawns, drained before teardown."""
	count: int = 0
	main_task: asyncio.Task[tuple[TaskResult, ...]] | None = None


def silence_dead(action: Callable[[], None]) -> None:
	"""Run ``action`` (a signal/kill call), ignoring the race where the process already exited."""
	with suppress(ProcessLookupError):
		action()


def step_interrupt(
	interrupts: Interrupts,
	leaves: tuple[Task, ...],
	now: datetime,
	dispatch_event: Callable[[int, TaskEvent], None],
) -> None:
	"""Advance the escalation by one Ctrl-C press — forward, forward, kill, cancel."""
	interrupts.count += 1
	running: Final = tuple(interrupts.procs.items())
	match interrupts.count:
		case 1:
			for idx, proc in running:
				silence_dead(partial(proc.send_signal, signal.SIGINT))
				interrupts.signaled.add(idx)
				dispatch_event(idx, InterruptedEvent(leaves[idx], idx, now))
		case 2:
			for _idx, proc in running:
				silence_dead(partial(proc.send_signal, signal.SIGINT))
		case 3:
			for idx, proc in running:
				silence_dead(proc.kill)
				dispatch_event(idx, AbortedEvent(leaves[idx], idx, now))
		case _:
			if interrupts.main_task is not None:  # pragma: no branch
				interrupts.main_task.cancel()


async def await_run(
	main_task: asyncio.Task[tuple[TaskResult, ...]], interrupts: Interrupts
) -> tuple[TaskResult, ...]:
	"""Await the run task; a 4th Ctrl-C cancels it, a Windows ``KeyboardInterrupt`` kills tracked leaves."""
	try:
		return await main_task
	except asyncio.CancelledError:
		return ()
	except KeyboardInterrupt:  # pragma: no cover  (Windows single-stage fallback)
		interrupts.count += 1
		for proc in tuple(interrupts.procs.values()):
			silence_dead(proc.kill)
		return ()


def subprocess_env(merged: dict[str, str]) -> dict[str, str]:
	"""Leaf-subprocess env: defaults merged underneath ``merged``; ``NO_COLOR`` strips color forces."""
	base = {"PYTHONUNBUFFERED": "1"} | merged
	if "NO_COLOR" in base:
		return {k: v for k, v in base.items() if k not in {"FORCE_COLOR", "CLICOLOR_FORCE"}}
	return {"FORCE_COLOR": "1", "CLICOLOR_FORCE": "1"} | base


async def run_cmd(
	task: Task, leaf_index: int, dispatch: EventSink, limiter: Limiter, interrupts: Interrupts
) -> TaskResult:
	"""Run one leaf as a subprocess, dispatching Started/Output/Completed events."""
	async with limiter:
		if interrupts.count:
			stopped: Final = Stopped(INTERRUPT_RC, 0.0, ())
			await dispatch(leaf_index, CompletedEvent(task, leaf_index, stopped, datetime.now()))
			return TaskResult(task_label(task), stopped)
		start_pc: Final = time.perf_counter()
		await dispatch(leaf_index, StartedEvent(task, leaf_index, datetime.now()))
		proc: Final = await asyncio.create_subprocess_exec(
			*resolve_cmd(task.cmd),
			stdout=asyncio.subprocess.PIPE,
			stderr=STDOUT,
			env=subprocess_env({**os.environ, **task.env}),
			cwd=task.cwd,
		)
		interrupts.procs[leaf_index] = proc
		output: Final[list[bytes]] = []
		try:
			if proc.stdout is not None:  # pragma: no branch
				async for line in proc.stdout:
					output.append(line)
					await dispatch(leaf_index, OutputEvent(task, leaf_index, line, datetime.now()))
			await proc.wait()
		finally:
			interrupts.procs.pop(leaf_index, None)
		elapsed: Final = time.perf_counter() - start_pc
		rc: Final = proc.returncode or 0
		completion: Final = (
			Stopped(rc, elapsed, output)
			if leaf_index in interrupts.signaled
			else Finished(rc, elapsed, output)
		)
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


async def execute(
	node: TaskNode,
	dispatch: EventSink,
	leaves: tuple[Task, ...],
	index_map: dict[int, int],
	limiter: Limiter,
	interrupts: Interrupts,
) -> tuple[TaskResult, ...]:
	"""Walk a task subtree, returning one TaskResult per leaf in DFS order."""
	match node:
		case Task():
			return (await run_cmd(node, index_map[id(node)], dispatch, limiter, interrupts),)
		case Parallel(tasks=children):
			async with TaskGroup() as tg:
				futures: Final = tuple(
					tg.create_task(execute(child, dispatch, leaves, index_map, limiter, interrupts))
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
					else await execute(child, dispatch, leaves, index_map, limiter, interrupts)
				)
				seq_results = (*seq_results, *child_results)
				if failed_rc is None:
					failed_rc = next(
						(
							r.completion.returncode
							for r in child_results
							if r.completion.returncode != 0
						),
						None,
					)
			return seq_results
		case _:
			assert_never(node)


async def run(
	task: TaskNode, effects: Sequence[Effect[Any]] = (), jobs: int | None = None
) -> RunResult:
	"""Execute a task tree, dispatching events to every effect.

	Raises:
		ValueError: when ``jobs`` is provided and less than 1.
		BaseExceptionGroup: every error raised by Effects during setup,
			on_event, or teardown, collected per phase.

	>>> import asyncio
	>>> asyncio.run(run(Task(("python", "-c", "pass")), jobs=1)).returncode
	0
	>>> asyncio.run(run(Task(("python", "-c", "raise SystemExit(1)")))).returncode
	1
	"""
	if jobs is not None and jobs < 1:
		raise ValueError(f"jobs must be >= 1, got {jobs}")
	limiter: Final[Limiter] = asyncio.Semaphore(jobs) if jobs is not None else nullcontext()
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

	interrupts: Final = Interrupts(procs={}, signaled=set(), pending=[])
	loop: Final = asyncio.get_running_loop()

	def dispatch_event(idx: int, event: TaskEvent) -> None:
		interrupts.pending.append(loop.create_task(dispatch(idx, event)))

	def on_sigint() -> None:
		step_interrupt(interrupts, leaves, datetime.now(), dispatch_event)

	sigint_handled = False
	try:
		loop.add_signal_handler(signal.SIGINT, on_sigint)
		sigint_handled = True
	except NotImplementedError:  # pragma: no cover  (Windows ProactorEventLoop)
		pass

	results: tuple[TaskResult, ...] = ()
	try:
		if setup_errors:
			raise BaseExceptionGroup("setup errors", setup_errors)
		main_task: Final = loop.create_task(
			execute(expanded, dispatch, leaves, index_map, limiter, interrupts)
		)
		interrupts.main_task = main_task
		results = await await_run(main_task, interrupts)
	finally:
		if sigint_handled:  # pragma: no branch
			loop.remove_signal_handler(signal.SIGINT)
		if interrupts.pending:
			await asyncio.gather(*interrupts.pending, return_exceptions=True)
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
		returncode=INTERRUPT_RC
		if interrupts.count
		else (1 if any(r.completion.returncode != 0 for r in results) else 0),
		results=results,
		elapsed=time.perf_counter() - wall_start,
	)
