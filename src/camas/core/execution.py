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
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol, TypeAlias, cast

if sys.version_info >= (3, 11):
	from asyncio import TaskGroup
	from builtins import BaseExceptionGroup
	from typing import assert_never
else:  # pragma: no cover
	from exceptiongroup import BaseExceptionGroup
	from taskgroup import TaskGroup
	from typing_extensions import assert_never

from ..v0.completion import (
	INTERRUPT_RC,
	NOT_FOUND_RC,
	Completion,
	Errored,
	Finished,
	Skipped,
	Stopped,
)
from ..v0.leaf_state import Completed, Interrupting, LeafState, Running, Waiting
from ..v0.task import Parallel, Pipe, Sequential, Task, TaskNode
from ..v0.task_event import CompletedEvent, OutputEvent, StartedEvent, TaskEvent
from .completion import RunResult, TaskResult
from .leaf_state import KILL_PRESSES, next_state, to_interrupting
from .matrix import expand_matrix, resolve_cmd
from .scope import with_default_paths
from .task import task_label
from .timings import (
	CacheKey,  # runtime name get_type_hints resolves; TYPE_CHECKING-only would NameError
	raise_identities_mismatch,
	reject_non_tuple_identities,
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

	@property
	def returncode(self) -> int | None: ...

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

	def register(self, states: list[LeafState], leaf_index: int, proc: Signalable) -> None:
		"""Store a live leaf proc; a landed interrupt that missed its registration — the
		spawn-window Ctrl-C — replays the missed presses now. A child reaped in that window
		owns the Interrupting mark only when its returncode is SIGINT's death signature
		(:func:`died_of_sigint`); any other reaped child exited on its own and keeps its own
		attribution. A natural death reaped but not yet delivered by the watcher's callback
		still reads ``None`` and is marked — that sub-window cannot be told from a group kill.
		On Windows SIGINT is not a sendable signal, so the replayed presses 1-2 are
		suppressed no-ops; the kill press delivers a real kill. The mark lands either way —
		the interrupted spawn reads Stopped.
		"""
		self.procs[leaf_index] = proc
		if not self.landed():
			return
		returncode = proc.returncode
		presses = min(self.count, KILL_PRESSES)
		if not owns_mark(returncode):
			return
		if returncode is None:
			for press in range(1, presses + 1):
				signal_press(proc, press)
		states[leaf_index] = to_interrupting(states[leaf_index], presses)


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


def signal_press(proc: Signalable, presses: int) -> None:
	"""One escalation step — SIGINT, or kill once the kill press has been reached — best
	effort: a gone transport's raise (or Windows rejecting SIGINT) is suppressed here, the
	one place the delivery policy owns it. ``presses`` is the 1-based press number this step
	delivers — SIGINT below ``KILL_PRESSES``, kill from it on.
	"""
	with suppress(OSError, ValueError):
		if presses >= KILL_PRESSES:
			proc.kill()
		else:
			proc.send_signal(signal.SIGINT)


WINDOWS_CTRL_C_EXIT: Final = 0xC000013A
"""STATUS_CONTROL_C_EXIT — the Windows console's death signature for the same group kill the
POSIX signatures name."""

KILL_DEATH_RC: Final = -9 if sys.platform != "win32" else 1
"""The returncode a child killed by the kill press reports — SIGKILL's negative signal on
POSIX, TerminateProcess's exit code on Windows."""

SIGINT_DEATH_SIGNATURES: Final = (
	(WINDOWS_CTRL_C_EXIT,) if sys.platform == "win32" else (-signal.SIGINT, INTERRUPT_RC)
)
"""The platform's SIGINT death signatures — the reaped-child returncodes a landed interrupt
honestly owns."""


def died_of_sigint(returncode: int | None) -> bool:
	"""Whether a reaped child's returncode is one of :data:`SIGINT_DEATH_SIGNATURES` — the
	negative signal and the 128 + signal KeyboardInterrupt teardown exit on POSIX, the console
	kill code on Windows — the only reaped children a landed interrupt honestly owns. A child
	that exits 130 on its own is indistinguishable from that teardown and reads Stopped in the
	same window.
	"""
	return returncode in SIGINT_DEATH_SIGNATURES


def owns_mark(returncode: int | None) -> bool:
	"""Whether a proc's returncode allows the Interrupting mark — live (``None``), or reaped
	with SIGINT's death signature (:func:`died_of_sigint`).
	"""
	return returncode is None or died_of_sigint(returncode)


def step_interrupt(interrupts: Interrupts, states: list[LeafState]) -> None:
	"""Advance escalation one Ctrl-C press: forward SIGINT, again, kill, then cancel the run.
	A child reaped without SIGINT's death signature exited on its own and keeps its own
	attribution; one reaped with it owns the mark without signals — the press's group kill.
	A reaped-but-undelivered natural death reads ``None`` at the probe and is marked too —
	the sub-window the two cannot be told apart.
	"""
	interrupts.count += 1
	if interrupts.count > KILL_PRESSES:
		if interrupts.main_task is not None:  # pragma: no branch
			interrupts.main_task.cancel()
		return
	for leaf_index, proc in tuple(interrupts.procs.items()):
		returncode = proc.returncode
		if not owns_mark(returncode):
			continue
		if returncode is None:
			signal_press(proc, interrupts.count)
		states[leaf_index] = to_interrupting(states[leaf_index], interrupts.count)


async def await_run(
	main_task: asyncio.Task[tuple[TaskResult, ...]],
	interrupts: Interrupts,
	states: list[LeafState],
) -> tuple[TaskResult, ...] | None:
	"""Await the run task; a 4th Ctrl-C cancels it, returning ``None`` for the caller to
	rebuild the results from the states. A cancellation of the awaiting task
	itself — a client dropping a served run — kills the tracked leaves, cancels the run, and
	re-raises; the count having passed ``KILL_PRESSES`` is what tells the two apart (3.11+
	cancels the awaited task when the waiter is cancelled, so ``main_task.cancelled()`` is
	true in both cases). A ``KeyboardInterrupt`` caught here is a host outside
	``asyncio.run`` — whose own handler owns Ctrl-C on 3.11+ — interrupting the coroutine
	directly; it kills the tracked leaves, marks them like the other sites do, and awaits the
	run's unwind before returning.

	Raises:
		asyncio.CancelledError: when the awaiting task is cancelled before the 4th press —
			the client's cancellation, re-raised, not the press's.
	"""
	try:
		return await main_task
	except asyncio.CancelledError:
		if interrupts.count <= KILL_PRESSES:
			for proc in tuple(interrupts.procs.values()):
				signal_press(proc, KILL_PRESSES)
			main_task.cancel()
			raise
		return None
	except KeyboardInterrupt:  # pragma: no cover
		interrupts.count += 1
		for leaf_index, proc in tuple(interrupts.procs.items()):
			returncode = proc.returncode
			if not owns_mark(returncode):
				continue
			if returncode is None:
				signal_press(proc, KILL_PRESSES)
			states[leaf_index] = to_interrupting(states[leaf_index], interrupts.count)
		main_task.cancel()
		await asyncio.gather(main_task, return_exceptions=True)
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
	on a case-insensitive env block, a differently-cased pre-existing entry would shadow the
	overlay in the child.

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


def leaf_result(ctx: RunContext, leaf_index: int, completion: Completion) -> TaskResult:
	"""The TaskResult a leaf's completion produces — the one construction every completion
	path shares.
	"""
	return TaskResult(
		task_label(ctx.leaves[leaf_index]), completion, leaf_identity(ctx, leaf_index)
	)


async def _spawn_stage(
	task: Task,
	*,
	stdin: int | None,
	stdout: int,
	stderr: int,
	base: Path | None,
	leaf_color: bool,
) -> asyncio.subprocess.Process:
	"""Spawn ``task`` with the shared env inheritance; ``OSError`` propagates for the caller to
	classify (a missing executable or a failed exec).
	"""
	inherited = (
		drop_case_variants(dict(task.env), dict(os.environ))
		if sys.platform == "win32"
		else dict(os.environ)
	)
	return await asyncio.create_subprocess_exec(
		*resolve_cmd(task.cmd),
		stdin=stdin,
		stdout=stdout,
		stderr=stderr,
		env=subprocess_env({**inherited, **task.env}, color=leaf_color),
		cwd=spawn_cwd(base, task.cwd),
	)


async def run_cmd(task: Task, leaf_index: int, ctx: RunContext) -> TaskResult:
	"""Run one leaf as a subprocess, dispatching Started/Output/Completed events.

	Raises:
		asyncio.CancelledError: when the run is cancelled — re-raised after the child is
			killed and its reap awaited, so no transport outlives the loop.
	"""
	async with ctx.limiter:
		if ctx.interrupts.count:
			stopped: Final = Stopped(INTERRUPT_RC, 0.0, ())
			await ctx.dispatch(
				leaf_index, CompletedEvent(task, leaf_index, stopped, datetime.now())
			)
			return leaf_result(ctx, leaf_index, stopped)
		start_pc: Final = time.perf_counter()
		await ctx.dispatch(leaf_index, StartedEvent(task, leaf_index, datetime.now()))
		argv: Final = resolve_cmd(task.cmd)
		cwd: Final = spawn_cwd(ctx.base, task.cwd)
		proc: asyncio.subprocess.Process | None = None
		output: Final[list[bytes]] = []
		try:
			try:
				proc = await _spawn_stage(
					task,
					stdin=ctx.child_stdin,
					stdout=asyncio.subprocess.PIPE,
					stderr=STDOUT,
					base=ctx.base,
					leaf_color=ctx.leaf_color,
				)
			except OSError as exc:
				errored: Final = Errored(NOT_FOUND_RC, spawn_error_message(exc, argv, cwd))
				await ctx.dispatch(
					leaf_index, CompletedEvent(task, leaf_index, errored, datetime.now())
				)
				return leaf_result(ctx, leaf_index, errored)
			ctx.interrupts.register(ctx.states, leaf_index, proc)
			if proc.stdout is not None:  # pragma: no branch
				async for line in proc.stdout:
					output.append(line)
					await ctx.dispatch(
						leaf_index, OutputEvent(task, leaf_index, line, datetime.now())
					)
			await proc.wait()
			elapsed: Final = time.perf_counter() - start_pc
			rc: Final = proc.returncode or 0
			completion: Final = (
				Stopped(rc, elapsed, output)
				if isinstance(ctx.states[leaf_index], Interrupting)
				else Finished(rc, elapsed, output)
			)
			await ctx.dispatch(
				leaf_index, CompletedEvent(task, leaf_index, completion, datetime.now())
			)
			return leaf_result(ctx, leaf_index, completion)
		except asyncio.CancelledError:
			if proc is not None:
				with suppress(ProcessLookupError, OSError):
					proc.kill()
				await proc.wait()
			raise
		finally:
			if proc is not None:
				ctx.interrupts.procs.pop(leaf_index, None)


async def run_pipe(stages: tuple[TaskNode, ...], ctx: RunContext) -> tuple[TaskResult, ...]:
	"""Run a Pipe's stages concurrently, each stage's stdout wired into the next's stdin — the
	last stage's stdout is the pipeline's output, its stderr merged in like a leaf's. Every
	stage runs to completion (a dying stage feeds EOF downstream), and each stage's own exit is
	its leaf's result, so ``pipefail`` holds: any non-zero stage fails the run. Stages
	deliberately bypass the leaf limiter — a pipeline is one unit whose stages must all be live
	at once, or a full pipe deadlocks its writer. Each stage's completion dispatches as it
	reaps, like a leaf's. The ``os.pipe()`` fd wiring is exercised on Windows by the wheels
	suite's run of the pipe tests. Whatever fails — a cancel, a spawn error, a reader overflow,
	a landed interrupt — the readers are cancelled and every child is killed and awaited before
	the failure propagates, so no transport outlives the loop.
	"""
	leaves: Final = cast("tuple[Task, ...]", stages)
	procs: Final[dict[int, asyncio.subprocess.Process]] = {}
	readers: Final[list[asyncio.Task[None]]] = []
	waiters: Final[list[asyncio.Task[None]]] = []
	stage_readers: Final[dict[int, tuple[asyncio.Task[None], ...]]] = {}
	started_pc: Final[dict[int, float]] = {}
	outputs: Final[dict[int, list[bytes]]] = {ctx.index_map[id(s)]: [] for s in stages}
	completions: Final[dict[int, Completion]] = {}
	spawn_failure: tuple[int, str, int, str] | None = None
	prev_read: int | None = ctx.child_stdin
	pending_read: int | None = None
	pending_write: int = -1
	results: list[TaskResult] = []

	async def read_into(leaf_index: int, stream: asyncio.StreamReader) -> None:
		async for line in stream:
			outputs[leaf_index].append(line)
			await ctx.dispatch(
				leaf_index, OutputEvent(ctx.leaves[leaf_index], leaf_index, line, datetime.now())
			)

	async def kill_all(leaked_read: int | None) -> None:
		"""Close a leaked pipe end, kill and reap every spawned stage, then let the readers
		drain — the reaped procs EOF their streams, so a reader awaiting ``readline`` gets the
		buffered tail instead of losing it to a cancel — cancel the waiters, and drop the
		interrupt registrations.
		"""
		if leaked_read is not None and leaked_read != ctx.child_stdin:
			with suppress(OSError):
				os.close(leaked_read)
		for proc in procs.values():
			with suppress(ProcessLookupError, OSError):
				proc.kill()
		for proc in procs.values():
			with suppress(ProcessLookupError, OSError):
				await proc.wait()
		for reader in readers:
			with suppress(BaseException):
				await reader
		for waiter in waiters:
			waiter.cancel()
		for waiter in waiters:
			with suppress(BaseException):
				await waiter
		for leaf_index in procs:
			ctx.interrupts.procs.pop(leaf_index, None)

	async def wait_and_complete(
		stage: Task, leaf_index: int, proc: asyncio.subprocess.Process
	) -> None:
		"""Dispatch the stage's completion as soon as it reaps — after its own readers drained,
		so the completion's output never misses the tail — the per-leaf streaming every other
		execution path has.
		"""
		await proc.wait()
		for reader in stage_readers[leaf_index]:
			await reader
		elapsed = time.perf_counter() - started_pc[leaf_index]
		rc = proc.returncode or 0
		completion: Completion = (
			Stopped(rc, elapsed, tuple(outputs[leaf_index]))
			if isinstance(ctx.states[leaf_index], Interrupting)
			else Finished(rc, elapsed, tuple(outputs[leaf_index]))
		)
		completions[leaf_index] = completion
		await ctx.dispatch(
			leaf_index, CompletedEvent(stage, leaf_index, completion, datetime.now())
		)

	try:
		for pos, stage in enumerate(leaves):
			leaf_index = ctx.index_map[id(stage)]
			if ctx.interrupts.count:
				await kill_all(prev_read)
				for leaf_index, interrupted_proc in procs.items():
					stopped_rc = interrupted_proc.returncode or 0
					spawned_completion: Completion = Stopped(
						stopped_rc,
						time.perf_counter() - started_pc[leaf_index],
						tuple(outputs[leaf_index]),
					)
					await ctx.dispatch(
						leaf_index,
						CompletedEvent(
							ctx.leaves[leaf_index], leaf_index, spawned_completion, datetime.now()
						),
					)
					results.append(leaf_result(ctx, leaf_index, spawned_completion))
				# stages from here on never launched: spawn-failure semantics when one failed,
				# else a landed interrupt
				if spawn_failure is not None:
					rc, message, failed_index, failed_name = spawn_failure
					for remaining_stage in leaves[len(procs) :]:
						remaining_index = ctx.index_map[id(remaining_stage)]
						remaining_completion: Completion = (
							Errored(rc, message)
							if remaining_index == failed_index
							else Skipped(rc, failed_name)
						)
						await ctx.dispatch(
							remaining_index,
							CompletedEvent(
								remaining_stage,
								remaining_index,
								remaining_completion,
								datetime.now(),
							),
						)
						results.append(leaf_result(ctx, remaining_index, remaining_completion))
					return tuple(results)
				stopped = Stopped(INTERRUPT_RC, 0.0, ())
				for remaining_stage in leaves[len(procs) :]:
					remaining_index = ctx.index_map[id(remaining_stage)]
					await ctx.dispatch(
						remaining_index,
						CompletedEvent(remaining_stage, remaining_index, stopped, datetime.now()),
					)
					results.append(leaf_result(ctx, remaining_index, stopped))
				return tuple(results)
			if spawn_failure is not None:
				continue
			started_pc[leaf_index] = time.perf_counter()
			await ctx.dispatch(leaf_index, StartedEvent(stage, leaf_index, datetime.now()))
			argv = resolve_cmd(stage.cmd)
			cwd = spawn_cwd(ctx.base, stage.cwd)
			is_last = pos == len(leaves) - 1
			read_fd: int | None = None
			stdout: int = -1
			if is_last:
				stdout = asyncio.subprocess.PIPE
			else:
				read_fd, stdout = os.pipe()
				pending_read = read_fd
				pending_write = stdout
			try:
				proc = await _spawn_stage(
					stage,
					stdin=prev_read,
					stdout=stdout,
					stderr=STDOUT if is_last else asyncio.subprocess.PIPE,
					base=ctx.base,
					leaf_color=ctx.leaf_color,
				)
			except OSError as exc:
				if read_fd is not None:
					with suppress(OSError):
						os.close(read_fd)
				if stdout >= 0:
					with suppress(OSError):
						os.close(stdout)
				if prev_read is not None and prev_read != ctx.child_stdin:
					with suppress(OSError):
						os.close(prev_read)
				prev_read = None
				spawn_failure = (
					NOT_FOUND_RC,
					spawn_error_message(exc, argv, cwd),
					leaf_index,
					task_label(ctx.leaves[leaf_index]),
				)
				continue
			if not is_last:
				os.close(stdout)
			if prev_read is not None and prev_read != ctx.child_stdin:
				os.close(prev_read)
			prev_read = read_fd
			pending_read = None
			pending_write = -1
			procs[leaf_index] = proc
			ctx.interrupts.register(ctx.states, leaf_index, proc)
			stage_reader_list: list[asyncio.Task[None]] = []
			if proc.stderr is not None:
				stage_reader_list.append(asyncio.create_task(read_into(leaf_index, proc.stderr)))
			if is_last and proc.stdout is not None:  # pragma: no branch
				stage_reader_list.append(asyncio.create_task(read_into(leaf_index, proc.stdout)))
			stage_readers[leaf_index] = tuple(stage_reader_list)
			readers.extend(stage_reader_list)
	except BaseException:
		if pending_read is not None:
			with suppress(OSError):
				os.close(pending_read)
		if pending_write >= 0:
			with suppress(OSError):
				os.close(pending_write)
		await kill_all(prev_read)
		raise

	if spawn_failure is not None:
		rc, message, failed_index, failed_name = spawn_failure
		await kill_all(None)
		for stage in leaves:
			leaf_index = ctx.index_map[id(stage)]
			spawned_proc = procs.get(leaf_index)
			failed_completion: Completion
			if spawned_proc is None:
				failed_completion = (
					Errored(rc, message) if leaf_index == failed_index else Skipped(rc, failed_name)
				)
			else:
				failed_completion = Stopped(
					spawned_proc.returncode or 0,
					time.perf_counter() - started_pc[leaf_index],
					tuple(outputs[leaf_index]),
				)
			await ctx.dispatch(
				leaf_index,
				CompletedEvent(stage, leaf_index, failed_completion, datetime.now()),
			)
			results.append(leaf_result(ctx, leaf_index, failed_completion))
		return tuple(results)

	for leaf_index, proc in procs.items():
		waiters.append(
			asyncio.create_task(wait_and_complete(ctx.leaves[leaf_index], leaf_index, proc))
		)
	try:
		await asyncio.gather(*waiters)
	except BaseException:
		await kill_all(None)
		raise
	for stage in leaves:
		leaf_index = ctx.index_map[id(stage)]
		results.append(leaf_result(ctx, leaf_index, completions[leaf_index]))
	for leaf_index in procs:
		ctx.interrupts.procs.pop(leaf_index, None)
	return tuple(results)


async def skip_subtree(child: TaskNode, skip: Skipped, ctx: RunContext) -> tuple[TaskResult, ...]:
	"""Dispatch a Skipped completion for every leaf in a subtree, in DFS order."""
	results: tuple[TaskResult, ...] = ()
	for idx in subtree_leaf_indices(child, ctx.index_map):
		await ctx.dispatch(idx, CompletedEvent(ctx.leaves[idx], idx, skip, datetime.now()))
		results = (*results, leaf_result(ctx, idx, skip))
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
		case Pipe(tasks=children):
			return await run_pipe(children, ctx)
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


async def recovered_results(ctx: RunContext, states: list[LeafState]) -> tuple[TaskResult, ...]:
	"""Rebuild the leaf results after the 4th press cancelled the task tree, which can no
	longer return them. A ``Completed`` leaf keeps its carried completion; a leaf the cancel
	caught mid-flight — still ``Waiting`` on the limiter, ``Running``, or ``Interrupting`` —
	reads ``Stopped`` and dispatches the ``CompletedEvent`` its cancelled ``run_cmd`` never
	will. An ``Interrupting`` leaf's returncode is the press-derived death code (SIGINT below
	the kill press, the kill's code from it on); a ``Running`` or ``Waiting`` leaf reads
	``INTERRUPT_RC`` like the pre-spawn catch does. Mid-flight output is unrecoverable — the
	cancelled ``run_cmd`` owned the buffer, so the rebuilt completion carries none.
	"""
	results: list[TaskResult] = []
	for leaf_index, state in enumerate(states):
		match state:
			case Completed(completion=completion):
				pass
			case Waiting():
				completion = Stopped(INTERRUPT_RC, 0.0, ())
			case Running(start_time=start_time):
				completion = Stopped(
					INTERRUPT_RC,
					max(0.0, (datetime.now() - start_time).total_seconds()),
					(),
				)
			case Interrupting(start_time=start_time, presses=presses):
				completion = Stopped(
					KILL_DEATH_RC if presses >= KILL_PRESSES else -signal.SIGINT,
					max(0.0, (datetime.now() - start_time).total_seconds()),
					(),
				)
			case _:
				assert_never(state)
		if not isinstance(state, Completed):
			await ctx.dispatch(
				leaf_index,
				CompletedEvent(ctx.leaves[leaf_index], leaf_index, completion, datetime.now()),
			)
		results.append(leaf_result(ctx, leaf_index, completion))
	return tuple(results)


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
			is provided and is not a tuple of per-leaf cache keys parallel to the run's leaves.
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
	reject_non_tuple_identities(identities)
	if identities is not None and len(identities) != len(leaves):
		raise_identities_mismatch(len(identities), len(leaves))
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

	results: tuple[TaskResult, ...] | None = ()
	try:
		if setup_errors:
			raise BaseExceptionGroup("setup errors", setup_errors)
		main_task: Final = loop.create_task(execute(expanded, ctx))
		interrupts.main_task = main_task
		results = await await_run(main_task, interrupts, states)
		if results is None:
			results = await recovered_results(ctx, states)
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
