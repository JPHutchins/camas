# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import errno
import os
import shutil
import signal
import sys
import time
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
from .unwind import unwind, unwind_failure

if TYPE_CHECKING:
	from collections.abc import AsyncIterator, Mapping, Sequence

	from ..v0.effect import Effect
	from .effect import EventSink


Limiter: TypeAlias = "asyncio.Semaphore | nullcontext[None]"


class Signalable(Protocol):
	@property
	def returncode(self) -> int | None: ...

	def send_signal(self, sig: int, /) -> None: ...
	def kill(self) -> None: ...


@dataclass
class Interrupts:
	procs: dict[int, Signalable]
	count: int = 0
	main_task: asyncio.Task[tuple[TaskResult, ...]] | None = None

	def landed(self) -> bool:
		return self.count > 0

	def register(self, states: list[LeafState], leaf_index: int, proc: Signalable) -> None:
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
	dispatch: EventSink
	leaves: tuple[Task, ...]
	index_map: dict[int, int]
	limiter: Limiter
	interrupts: Interrupts
	states: list[LeafState]
	base: Path | None
	child_stdin: int | None
	leaf_color: bool
	identities: tuple[CacheKey, ...] | None


if sys.platform != "win32":
	import termios

	def suppress_ctrl_c_echo() -> list[Any] | None:
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
		if saved is not None:
			termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)

else:  # pragma: no cover

	def suppress_ctrl_c_echo() -> list[Any] | None:
		return None

	def restore_tty(saved: list[Any] | None) -> None: ...


def signal_press(proc: Signalable, presses: int) -> None:
	with suppress(OSError, ValueError):
		if presses >= KILL_PRESSES:
			proc.kill()
		else:
			proc.send_signal(signal.SIGINT)


WINDOWS_CTRL_C_EXIT: Final = 0xC000013A

KILL_DEATH_RC: Final = -9 if sys.platform != "win32" else 1

SIGINT_DEATH_SIGNATURES: Final = (
	(WINDOWS_CTRL_C_EXIT,) if sys.platform == "win32" else (-signal.SIGINT, INTERRUPT_RC)
)


def unwound_rc(unreaped: frozenset[int], leaf_index: int, returncode: int | None) -> int:
	"""
	>>> unwound_rc(frozenset({1}), 1, None) == KILL_DEATH_RC
	True
	>>> unwound_rc(frozenset(), 0, 3)
	3
	"""
	return KILL_DEATH_RC if leaf_index in unreaped else (returncode or 0)


def died_of_sigint(returncode: int | None) -> bool:
	return returncode in SIGINT_DEATH_SIGNATURES


def owns_mark(returncode: int | None) -> bool:
	return returncode is None or died_of_sigint(returncode)


def step_interrupt(interrupts: Interrupts, states: list[LeafState]) -> None:
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


def subprocess_env(merged: dict[str, str], *, color: bool = True) -> dict[str, str]:
	"""
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
	"""
	>>> drop_case_variants({"MYPYPATH": "new"}, {"Mypypath": "old", "OTHER": "kept"})
	{'OTHER': 'kept'}
	"""
	folded = {k.casefold() for k in overlay}
	return {k: v for k, v in inherited.items() if k.casefold() not in folded}


BATCH_SUFFIXES: Final = (".bat", ".cmd")

CMD_METACHARACTERS: Final = frozenset('&|<>^%"\r\n')


def is_batch_program(name: str) -> bool:
	"""
	>>> is_batch_program("npx.CMD"), is_batch_program("npx.cmd. "), is_batch_program("npx.exe")
	(True, True, False)
	"""
	return name.rstrip(". ").lower().endswith(BATCH_SUFFIXES)


def resolve_program(argv: tuple[str, ...], env: Mapping[str, str]) -> tuple[str, ...]:
	return refuse_cmd_reparse(_shim_on_path(argv, env) or argv)


def _shim_on_path(argv: tuple[str, ...], env: Mapping[str, str]) -> tuple[str, ...] | None:
	path: Final = next((value for key, value in env.items() if key.casefold() == "path"), None)
	if not argv or path is None or Path(argv[0]).name != argv[0]:
		return None
	program: Final = argv[0].rstrip(". ") if is_batch_program(argv[0]) else argv[0]
	found: Final = shutil.which(program, path=path)
	if found is None or not is_batch_program(found):
		return None
	return (str(Path(found).absolute()), *argv[1:])


def refuse_cmd_reparse(argv: tuple[str, ...]) -> tuple[str, ...]:
	batch: Final = bool(argv) and is_batch_program(argv[0])
	unsafe: Final = next((token for token in argv if CMD_METACHARACTERS.intersection(token)), None)
	if batch and unsafe is not None:
		raise OSError(
			errno.EINVAL,
			'refusing to run a batch script with a token holding a cmd.exe metacharacter (& | < > ^ % "'
			" or a line break), which cmd.exe would run as a command",
			unsafe,
		)
	return argv


def spawn_cwd(base: Path | None, cwd: Path | None) -> Path | None:
	"""
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
	"""
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


class UnstartableCommandError(Exception): ...


def command_argv(cmd: str | tuple[str, ...]) -> tuple[str, ...]:
	"""
	Raises:
		UnstartableCommandError: for a command with no program, or one ``shlex`` cannot split.

	>>> command_argv('echo "hi there"')
	('echo', 'hi there')
	>>> command_argv('echo "unclosed')
	Traceback (most recent call last):
	...
	camas.core.execution.UnstartableCommandError: malformed command: no closing quotation
	>>> command_argv(("",))
	Traceback (most recent call last):
	...
	camas.core.execution.UnstartableCommandError: empty command
	"""
	try:
		argv: Final = resolve_cmd(cmd)
	except ValueError as exc:
		raise UnstartableCommandError(f"malformed command: {str(exc).lower()}") from exc
	if not argv or not argv[0]:
		raise UnstartableCommandError("empty command")
	return argv


def spawn_error_message(exc: OSError | ValueError, program: str, cwd: Path | None) -> str:
	"""
	>>> from pathlib import Path
	>>> spawn_error_message(FileNotFoundError(2, "No such file or directory"), "ghost", None)
	'no such file or directory: ghost'
	>>> spawn_error_message(
	...     FileNotFoundError(2, "No such file or directory", "gone"), "echo", Path("gone")
	... )
	'no such file or directory: gone'

	>>> spawn_error_message(
	...     FileNotFoundError(2, "No such file or directory", "named-by-os"),
	...     "ghost",
	...     Path("also-gone"),
	... )
	'no such file or directory: named-by-os'
	>>> spawn_error_message(PermissionError(13, "Permission denied"), "./script.sh", None)
	'permission denied: ./script.sh'
	>>> spawn_error_message(OSError(), "weird", None)
	'could not start command: weird'
	>>> spawn_error_message(ValueError("embedded null byte"), "echo", Path("gone"))
	'could not start command: embedded null byte'
	"""
	if isinstance(exc, ValueError):
		return f"could not start command: {exc}"
	reason: Final = exc.strerror.lower() if exc.strerror else "could not start command"
	target: Final = exc.filename or unusable_cwd(cwd) or program
	if isinstance(exc, FileNotFoundError):
		return f"no such file or directory: {target}"
	return f"{reason}: {target}"


def leaf_identity(ctx: RunContext, leaf_index: int) -> CacheKey | None:
	return ctx.identities[leaf_index] if ctx.identities is not None else None


def leaf_result(ctx: RunContext, leaf_index: int, completion: Completion) -> TaskResult:
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
	argv: Final = command_argv(task.cmd)
	cwd: Final = spawn_cwd(base, task.cwd)
	try:
		inherited = (
			drop_case_variants(dict(task.env), dict(os.environ))
			if sys.platform == "win32"
			else dict(os.environ)
		)
		env: Final = subprocess_env({**inherited, **task.env}, color=leaf_color)
		return await asyncio.create_subprocess_exec(
			*(resolve_program(argv, env) if sys.platform == "win32" else argv),
			stdin=stdin,
			stdout=stdout,
			stderr=stderr,
			env=env,
			cwd=cwd,
		)
	except (OSError, ValueError) as exc:
		raise UnstartableCommandError(spawn_error_message(exc, argv[0], cwd)) from exc


def _completion(state: LeafState, rc: int, elapsed: float, output: Sequence[bytes]) -> Completion:
	return (
		Stopped(rc, elapsed, tuple(output))
		if isinstance(state, Interrupting)
		else Finished(rc, elapsed, tuple(output))
	)


async def _lines(stream: asyncio.StreamReader) -> AsyncIterator[bytes]:
	pieces: list[bytes] = []
	while True:
		try:
			line = await stream.readuntil(b"\n")
		except asyncio.LimitOverrunError as overrun:  # noqa: PERF203  # catching an overrun per read is the loop's purpose
			pieces.append(await stream.read(overrun.consumed))
		except asyncio.IncompleteReadError as eof:
			if pieces or eof.partial:
				yield b"".join((*pieces, eof.partial))
			return
		else:
			yield b"".join((*pieces, line)) if pieces else line
			pieces.clear()


async def run_cmd(task: Task, leaf_index: int, ctx: RunContext) -> TaskResult:
	async with ctx.limiter:
		if ctx.interrupts.count:
			stopped: Final = Stopped(INTERRUPT_RC, 0.0, ())
			await ctx.dispatch(
				leaf_index, CompletedEvent(task, leaf_index, stopped, datetime.now())
			)
			return leaf_result(ctx, leaf_index, stopped)
		start_pc: Final = time.perf_counter()
		await ctx.dispatch(leaf_index, StartedEvent(task, leaf_index, datetime.now()))
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
			except UnstartableCommandError as exc:
				errored: Final = Errored(NOT_FOUND_RC, str(exc))
				await ctx.dispatch(
					leaf_index, CompletedEvent(task, leaf_index, errored, datetime.now())
				)
				return leaf_result(ctx, leaf_index, errored)
			ctx.interrupts.register(ctx.states, leaf_index, proc)
			if proc.stdout is not None:  # pragma: no branch
				async for line in _lines(
					proc.stdout
				):  # pragma: no branch  # sysmon misses this loop's exit over an async generator; ctrace records it
					output.append(line)
					await ctx.dispatch(
						leaf_index, OutputEvent(task, leaf_index, line, datetime.now())
					)
			await proc.wait()
			elapsed: Final = time.perf_counter() - start_pc
			rc: Final = proc.returncode or 0
			completion: Final = _completion(ctx.states[leaf_index], rc, elapsed, output)
			await ctx.dispatch(
				leaf_index, CompletedEvent(task, leaf_index, completion, datetime.now())
			)
			return leaf_result(ctx, leaf_index, completion)
		except BaseException as exc:
			await unwind_failure(exc, {} if proc is None else {leaf_index: proc}, ())
		finally:
			if proc is not None:
				ctx.interrupts.procs.pop(leaf_index, None)


async def run_pipe(stages: tuple[TaskNode, ...], ctx: RunContext) -> tuple[TaskResult, ...]:
	async with ctx.limiter:
		return await _run_pipe_stages(stages, ctx)


async def _run_pipe_stages(stages: tuple[TaskNode, ...], ctx: RunContext) -> tuple[TaskResult, ...]:
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
	interrupted = False
	results: list[TaskResult] = []

	async def read_into(leaf_index: int, stream: asyncio.StreamReader) -> None:
		async for line in _lines(stream):
			outputs[leaf_index].append(line)
			await ctx.dispatch(
				leaf_index, OutputEvent(ctx.leaves[leaf_index], leaf_index, line, datetime.now())
			)

	def detach(leaked_read: int | None) -> tuple[asyncio.Future[None], ...]:
		if leaked_read is not None and leaked_read != ctx.child_stdin:
			with suppress(OSError):
				os.close(leaked_read)
		for waiter in waiters:
			waiter.cancel()
		for leaf_index in procs:
			ctx.interrupts.procs.pop(leaf_index, None)
		return (*readers, *waiters)

	async def wait_and_complete(
		stage: Task, leaf_index: int, proc: asyncio.subprocess.Process
	) -> None:
		await proc.wait()
		for reader in stage_readers[leaf_index]:
			await asyncio.shield(reader)
		elapsed = time.perf_counter() - started_pc[leaf_index]
		rc = proc.returncode or 0
		completion: Completion = _completion(
			ctx.states[leaf_index], rc, elapsed, outputs[leaf_index]
		)
		completions[leaf_index] = completion
		await ctx.dispatch(
			leaf_index, CompletedEvent(stage, leaf_index, completion, datetime.now())
		)

	try:
		for pos, stage in enumerate(leaves):
			leaf_index = ctx.index_map[id(stage)]
			if ctx.interrupts.count:
				interrupted = True
				break
			if spawn_failure is not None:
				continue
			started_pc[leaf_index] = time.perf_counter()
			await ctx.dispatch(leaf_index, StartedEvent(stage, leaf_index, datetime.now()))
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
			except UnstartableCommandError as exc:
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
				pending_read = None
				pending_write = -1
				spawn_failure = (
					NOT_FOUND_RC,
					str(exc),
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
	except BaseException as exc:
		if pending_read is not None:
			with suppress(OSError):
				os.close(pending_read)
		if pending_write >= 0:
			with suppress(OSError):
				os.close(pending_write)
		await unwind_failure(exc, procs, detach(prev_read))

	if interrupted:
		unreaped = await unwind(procs, detach(prev_read))
		for leaf_index, interrupted_proc in procs.items():
			stopped_rc = unwound_rc(unreaped, leaf_index, interrupted_proc.returncode)
			spawned_completion: Completion = _completion(
				ctx.states[leaf_index],
				stopped_rc,
				time.perf_counter() - started_pc[leaf_index],
				outputs[leaf_index],
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
		rc, message, failed_index, failed_name = spawn_failure
		failure_unreaped = await unwind(procs, detach(None))
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
					unwound_rc(failure_unreaped, leaf_index, spawned_proc.returncode),
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
	except BaseException as exc:
		await unwind_failure(exc, procs, detach(None))
	for stage in leaves:
		leaf_index = ctx.index_map[id(stage)]
		results.append(leaf_result(ctx, leaf_index, completions[leaf_index]))
	for leaf_index in procs:
		ctx.interrupts.procs.pop(leaf_index, None)
	return tuple(results)


async def skip_subtree(child: TaskNode, skip: Skipped, ctx: RunContext) -> tuple[TaskResult, ...]:
	results: tuple[TaskResult, ...] = ()
	for idx in subtree_leaf_indices(child, ctx.index_map):
		await ctx.dispatch(idx, CompletedEvent(ctx.leaves[idx], idx, skip, datetime.now()))
		results = (*results, leaf_result(ctx, idx, skip))
	return results


async def execute(node: TaskNode, ctx: RunContext) -> tuple[TaskResult, ...]:
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
	"""
	Raises:
		ValueError: on a non-positive ``jobs`` or mismatched ``identities``.
		BaseExceptionGroup: errors raised by Effects.

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
