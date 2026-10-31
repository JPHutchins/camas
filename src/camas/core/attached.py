# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Run a Command attached to the terminal."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn, TypeAlias

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..v0.command import body_problem
from ..v0.completion import NOT_FOUND_RC
from ..v0.task import STAGE_MESSAGE, Parallel, Pipe, Sequential, Task
from .execution import (
	UnstartableCommandError,
	command_argv,
	inherited_env,
	spawn_argv,
	spawn_cwd,
	spawn_error_message,
)
from .matrix import expand_matrix

if TYPE_CHECKING:
	from collections.abc import Generator, Sequence
	from pathlib import Path
	from types import FrameType

	from ..v0.command import Command
	from ..v0.task import TaskNode


class Exited(NamedTuple):
	code: int


class Signaled(NamedTuple):
	signum: int


Outcome: TypeAlias = Exited | Signaled

Step: TypeAlias = Task | Pipe

SUCCESS: Final = Exited(0)

if sys.platform == "win32":  # pragma: no cover
	RELAYED: Final[tuple[int, ...]] = (signal.SIGINT, signal.SIGBREAK)
	FORWARDED: Final[frozenset[int]] = frozenset()
	FORWARDED_OFF_TTY: Final[frozenset[int]] = frozenset()
else:
	RELAYED: Final[tuple[int, ...]] = (signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGHUP)
	FORWARDED: Final[frozenset[int]] = frozenset({signal.SIGTERM, signal.SIGHUP})
	FORWARDED_OFF_TTY: Final[frozenset[int]] = frozenset({signal.SIGINT, signal.SIGQUIT})

FORWARDABLE: Final = FORWARDED | FORWARDED_OFF_TTY


def scoped_body(command: Command) -> TaskNode:
	"""
	>>> from pathlib import Path
	>>> from camas.v0.command import Command
	>>> leaf = scoped_body(Command(Task("run", env={"A": "1"}), env={"B": "2"}, cwd="fw"))
	>>> leaf.env, leaf.cwd == Path("fw")  # type: ignore[union-attr]
	({'B': '2', 'A': '1'}, True)
	"""
	return expand_matrix(command.body, command.env, command.cwd)


def attached_steps(node: TaskNode) -> tuple[Step, ...]:
	"""
	>>> [type(s).__name__ for s in attached_steps(Sequential("a", Sequential(Pipe("b", "c"))))]
	['Task', 'Pipe']
	>>> attached_steps(Parallel("a"))
	Traceback (most recent call last):
	    ...
	ValueError: a Command body cannot hold a Parallel: its children would share one terminal

	Raises:
		ValueError: on a ``Parallel``.
	"""
	match node:
		case Task() | Pipe():
			return (node,)
		case Sequential(tasks=children):
			return tuple(step for child in children for step in attached_steps(child))
		case Parallel():
			raise ValueError(body_problem(node))
		case _:
			assert_never(node)


def run_attached(node: TaskNode, base: Path | None) -> Outcome:
	"""
	>>> run_attached(Sequential(Task(("python", "-c", "pass")), Task(("python", "-c", "raise SystemExit(3)"))), None)
	Exited(code=3)
	"""
	return next(
		(
			outcome
			for outcome in (run_step(step, base) for step in attached_steps(node))
			if outcome != SUCCESS
		),
		SUCCESS,
	)


def run_step(step: Step, base: Path | None) -> Outcome:
	procs: Final[list[subprocess.Popen[bytes]]] = []
	with relaying(procs) as received, reaping(procs):
		try:
			spawn_stages(procs, stages_of(step), base, received)
		except UnstartableCommandError as exc:
			print(f"error: {exc}", file=sys.stderr)
			return Exited(NOT_FOUND_RC)
		returncode: Final = pipefail(tuple(proc.wait() for proc in procs))
	return outcome_of(returncode, received)


def stages_of(step: Step) -> tuple[Task, ...]:
	"""
	>>> stages_of(Task("a"))
	(Task(cmd='a', name=None, env={}, cwd=None),)
	>>> from camas.v0.ref import Ref
	>>> stages_of(Pipe("a", Ref("b")))
	Traceback (most recent call last):
	    ...
	ValueError: Pipe stages must be Tasks — a nested group would mean several commands sharing one stream

	Raises:
		ValueError: on a pipe stage that is not a ``Task``.
	"""
	match step:
		case Task():
			return (step,)
		case Pipe(tasks=stages):
			if not all(isinstance(stage, Task) for stage in stages):
				raise ValueError(STAGE_MESSAGE)
			return tuple(stage for stage in stages if isinstance(stage, Task))
		case _:
			assert_never(step)


def pipefail(returncodes: Sequence[int]) -> int:
	"""
	>>> pipefail((0, 0)), pipefail((1, 0)), pipefail((1, 2)), pipefail((-13, 0))
	(0, 1, 2, -13)
	"""
	return next((rc for rc in reversed(returncodes) if rc != 0), 0)


def outcome_of(returncode: int, received: Sequence[int]) -> Outcome:
	"""
	>>> outcome_of(3, ()), outcome_of(-2, (2,)), outcome_of(-11, (2,))
	(Exited(code=3), Signaled(signum=2), Exited(code=139))
	"""
	if returncode > 0:
		return Exited(returncode)
	if returncode == 0:
		return next((Signaled(signum) for signum in received if signum in FORWARDED), SUCCESS)
	if -returncode in received:
		return Signaled(-returncode)
	return Exited(128 - returncode)


def spawn_stages(
	procs: list[subprocess.Popen[bytes]],
	stages: tuple[Task, ...],
	base: Path | None,
	received: Sequence[int],
) -> None:
	upstream: int | None = None
	for index, stage in enumerate(stages):
		read_end, write_end = os.pipe() if index < len(stages) - 1 else (None, None)
		try:
			proc = spawn(stage, base, stdin=upstream, stdout=write_end)
		except BaseException:
			close_fd(read_end)
			raise
		finally:
			close_fd(write_end)
			close_fd(upstream)
		procs.append(proc)
		for signum in tuple(received):
			if signum in FORWARDABLE:
				send(proc, signum)
		upstream = read_end


def spawn(
	task: Task, base: Path | None, *, stdin: int | None, stdout: int | None
) -> subprocess.Popen[bytes]:
	argv: Final = command_argv(task.cmd)
	cwd: Final = spawn_cwd(base, task.cwd)
	env: Final = inherited_env(task.env)
	try:
		return subprocess.Popen(
			spawn_argv(argv, env),
			stdin=stdin,
			stdout=stdout,
			env=env,
			cwd=cwd,
		)
	except (OSError, ValueError) as exc:
		raise UnstartableCommandError(spawn_error_message(exc, argv[0], cwd)) from exc


def close_fd(fd: int | None) -> None:
	if fd is not None:
		os.close(fd)


def send(proc: subprocess.Popen[bytes], signum: int) -> None:
	with suppress(OSError):
		proc.send_signal(signum)


@contextmanager
def reaping(procs: Sequence[subprocess.Popen[bytes]]) -> Generator[None]:
	try:
		yield
	finally:
		for proc in procs:
			if proc.poll() is None:
				proc.kill()
			proc.wait()


@contextmanager
def relaying(procs: Sequence[subprocess.Popen[bytes]]) -> Generator[Sequence[int]]:
	received: Final[list[int]] = []

	def relay(signum: int, _frame: FrameType | None) -> None:
		received.append(signum)
		if signum in FORWARDED or (signum in FORWARDED_OFF_TTY and not in_foreground()):
			for proc in procs:
				send(proc, signum)

	previous: Final = {signum: signal.signal(signum, relay) for signum in RELAYED}
	try:
		yield received
	finally:
		for signum, handler in previous.items():
			signal.signal(signum, handler if handler is not None else signal.SIG_DFL)


if sys.platform == "win32":  # pragma: no cover

	def in_foreground() -> bool:
		return False

else:

	def in_foreground() -> bool:
		try:
			terminal = os.open(os.ctermid(), os.O_RDONLY | os.O_NOCTTY)
		except OSError:
			return False
		try:
			return os.tcgetpgrp(terminal) == os.getpgrp()
		except OSError:
			return False
		finally:
			os.close(terminal)


def die_of(signum: int) -> NoReturn:  # pragma: no cover  # the process ends before coverage writes
	signal.signal(signum, signal.SIG_DFL)
	os.kill(os.getpid(), signum)
	sys.exit(128 + signum)
