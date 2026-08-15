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
from subprocess import DEVNULL, STDOUT
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol, TypeAlias

if sys.version_info >= (3, 11):
	from asyncio import TaskGroup
	from builtins import BaseExceptionGroup
	from typing import assert_never
else:  # pragma: no cover
	from exceptiongroup import BaseExceptionGroup
	from taskgroup import TaskGroup
	from typing_extensions import assert_never

from ..v0.completion import INTERRUPT_RC, NOT_FOUND_RC, Errored, Finished, Skipped, Stopped
from ..v0.leaf_state import Interrupting, LeafState, Waiting
from ..v0.task import Parallel, Sequential, Task, TaskNode
from ..v0.task_event import CompletedEvent, OutputEvent, StartedEvent, TaskEvent
from .completion import RunResult, TaskResult
from .leaf_state import KILL_PRESSES, next_state, to_interrupting
from .matrix import expand_matrix, resolve_cmd
from .scope import with_default_paths
from .task import task_label
from .timings import (
	CacheKey,  # noqa: TC001  # runtime name get_type_hints resolves; TYPE_CHECKING-only would NameError
)
from .traversal import flatten_leaves, subtree_leaf_indices

if TYPE_CHECKING:
	from collections.abc import Sequence
	from pathlib import Path

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
	count: int = 0
	main_task: asyncio.Task[tuple[TaskResult, ...]] | None = None

	def landed(self) -> bool:
		"""Whether any Ctrl-C has arrived — askable again after an await, where the signal
		handler may have run.
		"""
		return self.count > 0


class RunContext(NamedTuple):
	"""Run-invariant context threaded through the ``execute`` recursion."""

	dispatch: EventSink
	leaves: tuple[Task, ...]
	index_map: dict[int, int]
	limiter: Limiter
	interrupts: Interrupts
	states: list[LeafState]
	base: Path | None
	"""The frame a leaf's ``cwd`` is spawned relative to (:func:`spawn_cwd`); ``None`` when the
	tasks source has no on-disk location to anchor to (a scope run without a ``__file__``),
	leaving a relative ``cwd`` to resolve against the process working directory."""
	child_stdin: int | None
	"""Why a served run denies leaves the parent's stdin instead of inheriting it: under the MCP
	stdio server that stdin is a pipe the transport's reader thread holds open, and on Windows an
	inherited copy stalls the Proactor loop's read of the leaf's *own* stdout, so EOF never
	arrives and the run hangs (#253). A leaf is a batch command with no terminal to want, so
	``DEVNULL`` also turns a stray stdin read into immediate EOF rather than a block."""
	leaf_color: bool
	"""``Config.leaf_color``: whether camas forces color on in the leaf environment
	(:func:`subprocess_env`)."""
	identities: tuple[CacheKey, ...] | None
	"""Per-leaf cache keys, parallel to ``leaves``, computed before any command rewrite; ``None``
	when the run was threaded without them."""


if sys.platform != "win32":
	import termios

	def suppress_ctrl_c_echo() -> list[Any] | None:
		"""Clear ``ECHOCTL`` on the controlling tty for the run; return prior attrs to restore.

		A tty echoes the interrupt char as ``^C`` at the cursor when Ctrl-C is pressed;
		on some terminals (notably the WSL pty under Windows Terminal) that echo carries
		a newline, which slides the live tree's repaint anchor down and strands rows.
		Returns ``None`` (a no-op) off a tty (piped / captured).
		"""
		try:
			fd = sys.stdin.fileno()
			saved = termios.tcgetattr(fd)
		except (OSError, ValueError, termios.error):
			return None
		updated = termios.tcgetattr(fd)
		updated[3] &= ~termios.ECHOCTL
		termios.tcsetattr(fd, termios.TCSADRAIN, updated)
		return saved

	def restore_tty(saved: list[Any] | None) -> None:
		"""Restore tty attributes captured by :func:`suppress_ctrl_c_echo`."""
		if saved is not None:
			termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)

else:  # pragma: no cover

	def suppress_ctrl_c_echo() -> list[Any] | None:
		"""Ctrl-C echo suppression is POSIX-only; a no-op on Windows."""
		return None

	def restore_tty(saved: list[Any] | None) -> None:
		"""No tty state to restore on Windows."""


def interrupt_proc(
	states: list[LeafState], leaf_index: int, proc: Signalable, presses: int
) -> None:
	"""Bring one proc in line with the escalation at ``presses`` Ctrl-C — SIGINT, or kill once
	the kill press has been reached — and mark its leaf Interrupting. ``step_interrupt`` applies
	this to every registered proc; ``run_cmd`` replays the missed presses to a proc whose
	registration a landed interrupt missed (#265).
	"""
	if presses >= KILL_PRESSES:
		proc.kill()
	else:
		proc.send_signal(signal.SIGINT)
	states[leaf_index] = to_interrupting(states[leaf_index], presses)


def step_interrupt(interrupts: Interrupts, states: list[LeafState]) -> None:
	"""Advance escalation one Ctrl-C press: forward SIGINT, again, kill, then cancel the run."""
	interrupts.count += 1
	if interrupts.count > KILL_PRESSES:
		if interrupts.main_task is not None:  # pragma: no branch
			interrupts.main_task.cancel()
		return
	for leaf_index, proc in tuple(interrupts.procs.items()):
		interrupt_proc(states, leaf_index, proc, interrupts.count)


async def await_run(
	main_task: asyncio.Task[tuple[TaskResult, ...]], interrupts: Interrupts
) -> tuple[TaskResult, ...]:
	"""Await the run task; a 4th Ctrl-C cancels it, a Windows ``KeyboardInterrupt`` kills tracked leaves."""
	try:
		return await main_task
	except asyncio.CancelledError:
		return ()
	except KeyboardInterrupt:  # pragma: no cover
		interrupts.count += 1
		for proc in tuple(interrupts.procs.values()):
			proc.kill()
		return ()


FORCED_COLOR: Final = {"FORCE_COLOR": "1", "CLICOLOR_FORCE": "1"}
"""What camas adds so a piped tool still colors, and — by its keys — what it takes back out when
the environment says no."""


def subprocess_env(merged: dict[str, str], *, color: bool = True) -> dict[str, str]:
	"""Leaf-subprocess env: defaults merged underneath ``merged``. Color is forced by default —
	camas pipes the leaf, so a tool that probes for a terminal would otherwise drop the color the
	tree and an ANSI-rendering CI log both want. ``color=False`` (``Config(leaf_color=False)``)
	leaves that decision to the environment instead, for a leaf whose output something parses or
	asserts on: the forcing env is inherited by the leaf's own children, so it reaches a nested
	command whose stdout the leaf itself captured. ``NO_COLOR`` wins over both — when non-empty,
	the reading no-color.org specifies and :func:`camas.core.render.color_on` already applies to
	camas's own output.

	>>> subprocess_env({})
	{'FORCE_COLOR': '1', 'CLICOLOR_FORCE': '1', 'PYTHONUNBUFFERED': '1'}
	>>> subprocess_env({}, color=False)
	{'PYTHONUNBUFFERED': '1'}
	>>> subprocess_env({"NO_COLOR": "1", "FORCE_COLOR": "1"})
	{'PYTHONUNBUFFERED': '1', 'NO_COLOR': '1'}
	>>> subprocess_env({"NO_COLOR": ""})
	{'FORCE_COLOR': '1', 'CLICOLOR_FORCE': '1', 'PYTHONUNBUFFERED': '1', 'NO_COLOR': ''}
	>>> subprocess_env({"FORCE_COLOR": "1"}, color=False)
	{'PYTHONUNBUFFERED': '1', 'FORCE_COLOR': '1'}
	"""
	base = {"PYTHONUNBUFFERED": "1"} | merged
	if base.get("NO_COLOR"):
		return {k: v for k, v in base.items() if k not in FORCED_COLOR}
	if not color:
		return base
	return FORCED_COLOR | base


def drop_case_variants(overlay: dict[str, str], inherited: dict[str, str]) -> dict[str, str]:
	"""``inherited`` minus entries whose names collide case-insensitively with an ``overlay`` key —
	Windows env blocks are case-insensitive, so a differently-cased pre-existing entry would shadow
	the overlay in the child.

	>>> drop_case_variants({"MYPYPATH": "new"}, {"Mypypath": "old", "OTHER": "kept"})
	{'OTHER': 'kept'}
	"""
	folded = {k.casefold() for k in overlay}
	return {k: v for k, v in inherited.items() if k.casefold() not in folded}


def spawn_cwd(base: Path | None, cwd: Path | None) -> Path | None:
	"""A leaf's spawn-time cwd: ``cwd`` is authored relative to ``base``; an absolute ``cwd``,
	an unset ``cwd``, or an unset ``base`` each pass through unresolved.

	>>> from pathlib import Path
	>>> spawn_cwd(None, None) is None
	True
	>>> spawn_cwd(None, Path("rel")) == Path("rel")
	True
	>>> spawn_cwd(Path("base"), None) == Path("base")
	True
	>>> here = Path.cwd()
	>>> spawn_cwd(Path("base"), here) == here
	True
	>>> spawn_cwd(Path("base"), Path("rel")) == Path("base") / "rel"
	True
	"""
	if base is None:
		return cwd
	if cwd is None:
		return base
	if cwd.is_absolute():
		return cwd
	return base / cwd


def unusable_cwd(cwd: Path | None) -> str | None:
	"""``cwd`` when a leaf could not have been spawned in it, else ``None`` — asked only after a
	spawn has already failed, to attribute a failure the OS declined to attribute itself.

	Answers about the real filesystem, so these ask it about this module's own file and directory —
	which exist, and are what they are, wherever the suite runs from.

	>>> from pathlib import Path
	>>> unusable_cwd(None) is None
	True
	>>> unusable_cwd(Path(__file__).parent) is None
	True
	>>> unusable_cwd(Path(__file__)) == str(Path(__file__))
	True
	"""
	with suppress(OSError):
		if cwd is not None and not cwd.is_dir():
			return str(cwd)
	return None


def spawn_error_message(exc: OSError, argv: Sequence[str], cwd: Path | None) -> str:
	"""The Errored message for a leaf whose spawn raised ``exc``: the canonical
	'no such file or directory' for a missing executable, else the OS ``strerror``.

	Names the path the OS reported, else the leaf's ``cwd`` when that is what it could not have
	run in, else the executable. Naming the executable for a leaf whose ``cwd`` is missing sends
	the reader after a file that is present and fine — and Windows reports exactly that case as
	'The directory name is invalid' with no path attached, so the ``cwd`` has to be filled in
	from what camas passed rather than read out of the error.

	>>> from pathlib import Path
	>>> spawn_error_message(FileNotFoundError(2, "No such file or directory"), ("ghost",), None)
	'no such file or directory: ghost'
	>>> spawn_error_message(
	...     FileNotFoundError(2, "No such file or directory", "gone"), ("echo",), Path("gone")
	... )
	'no such file or directory: gone'

	The OS wins over both when all three differ:

	>>> spawn_error_message(
	...     FileNotFoundError(2, "No such file or directory", "named-by-os"),
	...     ("ghost",),
	...     Path("also-gone"),
	... )
	'no such file or directory: named-by-os'
	>>> spawn_error_message(PermissionError(13, "Permission denied"), ("./script.sh",), None)
	'permission denied: ./script.sh'
	>>> spawn_error_message(OSError(), ("weird",), None)
	'could not start command: weird'
	"""
	target: Final = exc.filename or unusable_cwd(cwd) or argv[0]
	if isinstance(exc, FileNotFoundError):
		return f"no such file or directory: {target}"
	reason: Final = exc.strerror.lower() if exc.strerror else "could not start command"
	return f"{reason}: {target}"


def leaf_identity(ctx: RunContext, leaf_index: int) -> CacheKey | None:
	"""The carried cache key for ``leaf_index``, when the run was threaded with identities."""
	return ctx.identities[leaf_index] if ctx.identities is not None else None


async def run_cmd(task: Task, leaf_index: int, ctx: RunContext) -> TaskResult:
	"""Run one leaf as a subprocess, dispatching Started/Output/Completed events."""
	async with ctx.limiter:
		if ctx.interrupts.count:
			stopped: Final = Stopped(INTERRUPT_RC, 0.0, ())
			await ctx.dispatch(
				leaf_index, CompletedEvent(task, leaf_index, stopped, datetime.now())
			)
			return TaskResult(task_label(task), stopped, leaf_identity(ctx, leaf_index))
		start_pc: Final = time.perf_counter()
		await ctx.dispatch(leaf_index, StartedEvent(task, leaf_index, datetime.now()))
		argv: Final = resolve_cmd(task.cmd)
		cwd: Final = spawn_cwd(ctx.base, task.cwd)
		inherited: Final = (
			drop_case_variants(dict(task.env), dict(os.environ))
			if sys.platform == "win32"
			else dict(os.environ)
		)
		try:
			proc: Final = await asyncio.create_subprocess_exec(
				*argv,
				stdin=ctx.child_stdin,
				stdout=asyncio.subprocess.PIPE,
				stderr=STDOUT,
				env=subprocess_env({**inherited, **task.env}, color=ctx.leaf_color),
				cwd=cwd,
			)
		except OSError as exc:
			errored: Final = Errored(NOT_FOUND_RC, spawn_error_message(exc, argv, cwd))
			await ctx.dispatch(
				leaf_index, CompletedEvent(task, leaf_index, errored, datetime.now())
			)
			return TaskResult(task_label(task), errored, leaf_identity(ctx, leaf_index))
		ctx.interrupts.procs[leaf_index] = proc
		if ctx.interrupts.landed():
			for press in range(1, ctx.interrupts.count + 1):
				with suppress(ProcessLookupError):
					interrupt_proc(ctx.states, leaf_index, proc, press)
		output: Final[list[bytes]] = []
		try:
			if proc.stdout is not None:  # pragma: no branch
				async for line in proc.stdout:
					output.append(line)
					await ctx.dispatch(
						leaf_index, OutputEvent(task, leaf_index, line, datetime.now())
					)
			await proc.wait()
		finally:
			ctx.interrupts.procs.pop(leaf_index, None)
		elapsed: Final = time.perf_counter() - start_pc
		rc: Final = proc.returncode or 0
		completion: Final = (
			Stopped(rc, elapsed, output)
			if isinstance(ctx.states[leaf_index], Interrupting)
			else Finished(rc, elapsed, output)
		)
		await ctx.dispatch(leaf_index, CompletedEvent(task, leaf_index, completion, datetime.now()))
		return TaskResult(task_label(task), completion, leaf_identity(ctx, leaf_index))


async def skip_subtree(child: TaskNode, skip: Skipped, ctx: RunContext) -> tuple[TaskResult, ...]:
	"""Dispatch a Skipped completion for every leaf in a subtree, in DFS order."""
	results: tuple[TaskResult, ...] = ()
	for idx in subtree_leaf_indices(child, ctx.index_map):
		await ctx.dispatch(idx, CompletedEvent(ctx.leaves[idx], idx, skip, datetime.now()))
		results = (*results, TaskResult(task_label(ctx.leaves[idx]), skip, leaf_identity(ctx, idx)))
	return results


async def execute(node: TaskNode, ctx: RunContext) -> tuple[TaskResult, ...]:
	"""Walk a task subtree, returning one TaskResult per leaf in DFS order."""
	match node:
		case Task():
			return (await run_cmd(node, ctx.index_map[id(node)], ctx),)
		case Parallel(tasks=children):
			async with TaskGroup() as tg:
				futures: Final = tuple(tg.create_task(execute(child, ctx)) for child in children)
			return tuple(r for f in futures for r in f.result())
		case Sequential(tasks=children):
			seq_results: tuple[TaskResult, ...] = ()
			blocker: TaskResult | None = None
			for child in children:
				child_results = (
					await skip_subtree(
						child, Skipped(blocker.completion.returncode, blocker.name), ctx
					)
					if blocker is not None
					else await execute(child, ctx)
				)
				seq_results = (*seq_results, *child_results)
				if blocker is None:
					blocker = next(
						(r for r in child_results if r.completion.returncode != 0),
						None,
					)
			return seq_results
		case _:
			assert_never(node)


async def run(
	task: TaskNode,
	effects: Sequence[Effect[Any]] = (),
	jobs: int | None = None,
	*,
	interactive: bool = True,
	base: Path | None = None,
	leaf_color: bool = True,
	identities: tuple[CacheKey, ...] | None = None,
) -> RunResult:
	"""Execute a task tree, dispatching events to every effect.

	Raises:
		ValueError: when ``jobs`` is provided and less than 1, or when ``identities``
			is provided and not parallel to the run's leaves.
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
	expanded: Final = with_default_paths(expand_matrix(task))
	leaf_infos: Final = flatten_leaves(expanded)
	leaves: Final = tuple(info.task for info in leaf_infos)
	if identities is not None and len(identities) != len(leaves):
		raise ValueError(
			f"identities must be parallel to the run's leaves: "
			f"{len(identities)} keys for {len(leaves)} leaves"
		)
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
	active_ctxs: Final[list[Any]] = [r for r in setup_results if not isinstance(r, BaseException)]
	ctx_grid: Final[list[list[Any]]] = [list(active_ctxs) for _ in leaf_infos]
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

	interrupts: Final = Interrupts(procs={})
	ctx: Final = RunContext(
		dispatch,
		leaves,
		index_map,
		limiter,
		interrupts,
		states,
		base,
		None if interactive else DEVNULL,
		leaf_color,
		identities,
	)
	loop: Final = asyncio.get_running_loop()

	def on_sigint() -> None:
		step_interrupt(interrupts, states)

	saved_tty: Final = suppress_ctrl_c_echo() if interactive else None
	sigint_handled = False
	if interactive:
		try:
			loop.add_signal_handler(signal.SIGINT, on_sigint)
			sigint_handled = True
		except NotImplementedError:  # pragma: no cover
			pass

	results: tuple[TaskResult, ...] = ()
	try:
		if setup_errors:
			raise BaseExceptionGroup("setup errors", setup_errors)
		main_task: Final = loop.create_task(execute(expanded, ctx))
		interrupts.main_task = main_task
		results = await await_run(main_task, interrupts)
	finally:
		if sigint_handled:  # pragma: no branch
			loop.remove_signal_handler(signal.SIGINT)
		teardown_errors: Final = tuple(
			r
			for r in await asyncio.gather(
				*(
					effect.teardown(tuple(row[effect_idx] for row in (ctx_grid or [active_ctxs])))
					for effect_idx, effect in enumerate(active_effects)
				),
				return_exceptions=True,
			)
			if isinstance(r, BaseException)
		)
		restore_tty(saved_tty)
		if teardown_errors:
			raise BaseExceptionGroup("teardown errors", teardown_errors)
	return RunResult(
		returncode=INTERRUPT_RC
		if interrupts.count
		else (1 if any(r.completion.returncode != 0 for r in results) else 0),
		results=results,
		elapsed=time.perf_counter() - wall_start,
		interrupt_count=interrupts.count,
	)
