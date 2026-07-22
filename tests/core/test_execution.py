# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from subprocess import DEVNULL
from typing import TYPE_CHECKING

import pytest

from camas import Parallel, Sequential, Task
from camas.core.execution import (
	Interrupts,
	Signalable,
	await_run,
	restore_tty,
	run,
	spawn_cwd,
	step_interrupt,
	suppress_ctrl_c_echo,
)
from camas.core.leaf_state import KILL_PRESSES
from camas.v0.completion import INTERRUPT_RC, NOT_FOUND_RC, Errored, Finished, Skipped, Stopped
from camas.v0.leaf_state import Interrupting, LeafState, Running
from camas.v0.task_event import CompletedEvent, OutputEvent

if TYPE_CHECKING:
	from collections.abc import Sequence
	from typing import Any, Final

	from camas.core.completion import RunResult, TaskResult
	from camas.v0.task import TaskNode
	from camas.v0.task_event import TaskEvent


def test_force_color_injected_in_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("NO_COLOR", raising=False)
	task = Task(
		("python", "-c", "import os,sys; sys.exit(0 if os.environ.get('FORCE_COLOR')=='1' else 1)")
	)
	assert asyncio.run(run(task)).returncode == 0


def test_no_color_suppresses_force_color(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("NO_COLOR", "1")
	task = Task(
		("python", "-c", "import os,sys; sys.exit(0 if 'FORCE_COLOR' not in os.environ else 1)")
	)
	assert asyncio.run(run(task)).returncode == 0


def test_single_cmd_success() -> None:
	assert asyncio.run(run(Task(("python", "-c", "pass")))).returncode == 0


def test_zero_leaf_run_effects_do_not_crash(tmp_path: Path) -> None:
	pytest.importorskip("msgspec")
	from camas.effect.ctrf import Ctrf
	from camas.effect.summary import Summary
	from camas.effect.termtree import Termtree
	from camas.effect.timings import Timings

	task = Parallel(Task("echo hi", name="x"), name="g", matrix={"X": ()})
	effects = (
		Ctrf(path=str(tmp_path / "ctrf.json")),
		Summary(),
		Termtree(frame_interval_ms=50),
		Timings(camas_dir=tmp_path),
	)
	result = asyncio.run(run(task, effects=effects, interactive=False))
	assert result.returncode == 0
	assert result.results == ()


def test_single_cmd_failure() -> None:
	assert asyncio.run(run(Task(("python", "-c", "raise SystemExit(1)")))).returncode == 1


def test_cmd_string_form() -> None:
	assert asyncio.run(run(Task("python -c pass"))).returncode == 0


def test_cmd_env_passed() -> None:
	task = Task(
		(
			"python",
			"-c",
			"import os; raise SystemExit(0 if os.environ.get('MY_VAR')=='42' else 1)",
		),
		env={"MY_VAR": "42"},
	)
	assert asyncio.run(run(task)).returncode == 0


@pytest.mark.parametrize(
	("returncodes", "expected_exit"),
	[
		((0, 0, 0), 0),
		((0, 1, 0), 1),
		((1, 1, 1), 1),
	],
)
def test_parallel_returncodes(returncodes: tuple[int, ...], expected_exit: int) -> None:
	task = Parallel(
		*(
			Task(("python", "-c", f"raise SystemExit({rc})"), name=f"t{i}")
			for i, rc in enumerate(returncodes)
		)
	)
	assert asyncio.run(run(task)).returncode == expected_exit


@pytest.mark.parametrize(
	("returncodes", "expected_exit"),
	[
		((0, 0, 0), 0),
		((1, 0, 0), 1),
		((0, 2, 0), 1),
		((0, 0, 1), 1),
	],
)
def test_sequential_returncodes(returncodes: tuple[int, ...], expected_exit: int) -> None:
	task = Sequential(
		*(
			Task(("python", "-c", f"raise SystemExit({rc})"), name=f"t{i}")
			for i, rc in enumerate(returncodes)
		)
	)
	assert asyncio.run(run(task)).returncode == expected_exit


def test_nested_parallel_in_sequential() -> None:
	task = Sequential(
		Parallel(
			Task(("python", "-c", "pass"), name="p1"), Task(("python", "-c", "pass"), name="p2")
		),
		Task(("python", "-c", "pass"), name="after"),
	)
	assert asyncio.run(run(task)).returncode == 0


def test_nested_sequential_in_parallel() -> None:
	task = Parallel(
		Sequential(
			Task(("python", "-c", "pass"), name="s1"), Task(("python", "-c", "pass"), name="s2")
		),
		Task(("python", "-c", "pass"), name="par"),
	)
	assert asyncio.run(run(task)).returncode == 0


def test_deeply_nested() -> None:
	task = Sequential(
		Parallel(
			Sequential(
				Task(("python", "-c", "pass"), name="deep1"),
				Task(("python", "-c", "pass"), name="deep2"),
			),
			Task(("python", "-c", "pass"), name="shallow"),
		),
		Task(("python", "-c", "pass"), name="final"),
	)
	assert asyncio.run(run(task)).returncode == 0


def test_matrix_env_reaches_subprocess() -> None:
	task = Parallel(
		Task(
			(
				"python",
				"-c",
				"import os; raise SystemExit(0 if os.environ.get('X')=='1' else 1)",
			),
		),
		matrix={"X": ("1",)},
	)
	assert asyncio.run(run(task)).returncode == 0


def test_matrix_substitution_in_cmd() -> None:
	task = Parallel(
		Task(
			(
				"python",
				"-c",
				"import sys; raise SystemExit(0 if '{V}' == 'hello' else 1)",
			),
		),
		matrix={"V": ("hello",)},
	)
	assert asyncio.run(run(task)).returncode == 0


def test_parallel_concurrency() -> None:
	task = Parallel(
		Task(("python", "-c", "import time; time.sleep(1.0)"), name="a"),
		Task(("python", "-c", "import time; time.sleep(1.0)"), name="b"),
		Task(("python", "-c", "import time; time.sleep(1.0)"), name="c"),
	)
	start = time.perf_counter()
	asyncio.run(run(task))
	elapsed = time.perf_counter() - start
	# Concurrent: ~1.0s + spawn overhead. Sequential would be ≥3.0s.
	assert elapsed < 2.5


def test_jobs_cap_serializes_parallel() -> None:
	task = Parallel(
		*(Task(("python", "-c", "import time; time.sleep(0.3)"), name=f"t{i}") for i in range(3))
	)
	start = time.perf_counter()
	asyncio.run(run(task, jobs=1))
	elapsed = time.perf_counter() - start
	# jobs=1 serializes the three 0.3s sleeps (~0.9s+); unbounded would be ~0.3s.
	assert elapsed >= 0.8


def test_jobs_cap_preserves_results() -> None:
	task = Parallel(
		Task(("python", "-c", "pass"), name="a"),
		Task(("python", "-c", "raise SystemExit(1)"), name="b"),
		Task(("python", "-c", "pass"), name="c"),
	)
	result = asyncio.run(run(task, jobs=2))
	assert result.returncode == 1
	assert len(result.results) == 3


@pytest.mark.parametrize("bad", [0, -1])
def test_jobs_below_one_rejected(bad: int) -> None:
	with pytest.raises(ValueError, match="jobs must be >= 1"):
		asyncio.run(run(Task(("python", "-c", "pass")), jobs=bad))


def test_sequential_ordering() -> None:
	task = Sequential(
		Task(("python", "-c", "import time; time.sleep(0.1)"), name="a"),
		Task(("python", "-c", "import time; time.sleep(0.1)"), name="b"),
	)
	start = time.perf_counter()
	asyncio.run(run(task))
	elapsed = time.perf_counter() - start
	assert elapsed >= 0.2


def test_task_with_stdout_output() -> None:
	task = Task(("python", "-c", "print('hello world')"), name="printer")
	result = asyncio.run(run(task))
	assert result.returncode == 0
	assert any(
		b"hello world" in line
		for r in result.results
		if isinstance(r.completion, Finished)
		for line in r.completion.output
	)


def test_nonexistent_executable_errors_without_traceback() -> None:
	task = Task(("camas-does-not-exist-xyz", "--flag"), name="ghost")
	result = asyncio.run(run(task))
	assert result.returncode == 1
	assert len(result.results) == 1
	completion = result.results[0].completion
	assert isinstance(completion, Errored)
	assert completion.returncode == NOT_FOUND_RC
	assert "camas-does-not-exist-xyz" in completion.message


def test_sequential_skips_sibling_after_nonexistent_executable() -> None:
	task = Sequential(
		Task(("camas-does-not-exist-xyz",), name="ghost"),
		Task(("python", "-c", "pass"), name="after"),
	)
	result = asyncio.run(run(task))
	assert result.returncode == 1
	by_name = {r.name: r.completion for r in result.results}
	assert isinstance(by_name["ghost"], Errored)
	assert by_name["after"] == Skipped(NOT_FOUND_RC, "ghost")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX exec-permission semantics")
def test_non_executable_file_errors_without_traceback(tmp_path: Path) -> None:
	script = tmp_path / "not_executable.sh"
	script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
	script.chmod(0o644)
	result = asyncio.run(run(Task((str(script),), name="denied")))
	assert result.returncode == 1
	completion = result.results[0].completion
	assert isinstance(completion, Errored)
	assert completion.returncode == NOT_FOUND_RC
	assert "permission denied" in completion.message.lower()
	assert str(script) in completion.message


def test_missing_cwd_errors_without_traceback(tmp_path: Path) -> None:
	task = Task(("python", "-c", "pass"), name="lost", cwd=tmp_path / "does-not-exist")
	result = asyncio.run(run(task))
	assert result.returncode == 1
	completion = result.results[0].completion
	assert isinstance(completion, Errored)
	assert completion.returncode == NOT_FOUND_RC
	assert completion.message


def test_sequential_skip_nested_group() -> None:
	task = Sequential(
		Task(("python", "-c", "raise SystemExit(1)"), name="fail"),
		Parallel(
			Task(("python", "-c", "pass"), name="skipped1"),
			Task(("python", "-c", "pass"), name="skipped2"),
		),
	)
	result = asyncio.run(run(task))
	assert result.returncode == 1
	skips = {r.name: r.completion for r in result.results if r.name.startswith("skipped")}
	assert skips == {"skipped1": Skipped(1, "fail"), "skipped2": Skipped(1, "fail")}


def test_interactive_flag_gates_terminal_setup(monkeypatch: pytest.MonkeyPatch) -> None:
	calls: list[int] = []
	monkeypatch.setattr("camas.core.execution.suppress_ctrl_c_echo", lambda: calls.append(1))
	assert asyncio.run(run(Task(("python", "-c", "pass")), interactive=True)).returncode == 0
	assert asyncio.run(run(Task(("python", "-c", "pass")), interactive=False)).returncode == 0
	assert calls == [1]


@pytest.mark.parametrize(("interactive", "expected_stdin"), [(False, DEVNULL), (True, None)])
def test_leaf_stdin_follows_interactive(
	monkeypatch: pytest.MonkeyPatch, interactive: bool, expected_stdin: int | None
) -> None:
	captured: list[int | None] = []
	real = asyncio.create_subprocess_exec

	async def spy(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
		captured.append(kwargs.get("stdin"))
		return await real(*args, **kwargs)

	monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
	assert asyncio.run(run(Task(("python", "-c", "pass")), interactive=interactive)).returncode == 0
	assert captured == [expected_stdin]


def test_matrix_nested_sequential_in_parallel() -> None:
	task = Parallel(
		Sequential(
			Task(("python", "-c", "pass"), name="a"), Task(("python", "-c", "pass"), name="b")
		),
		matrix={"X": ("1", "2")},
	)
	result = asyncio.run(run(task))
	assert result.returncode == 0
	assert len(result.results) == 4


def test_run_result_has_structured_results() -> None:
	task = Parallel(
		Task(("python", "-c", "pass"), name="a"),
		Task(("python", "-c", "raise SystemExit(1)"), name="b"),
	)
	result = asyncio.run(run(task))
	assert result.returncode == 1
	assert len(result.results) == 2
	assert result.elapsed > 0
	names = {r.name for r in result.results}
	assert names == {"a", "b"}


def test_spawn_cwd_resolves_relative_against_base(tmp_path: Path) -> None:
	assert spawn_cwd(tmp_path, Path("sub")) == tmp_path / "sub"


def _printed_cwd(result: RunResult) -> Path:
	completion = result.results[0].completion
	assert isinstance(completion, Finished)
	return Path(completion.output[0].decode().strip())


def test_run_resolves_relative_cwd_against_base(tmp_path: Path) -> None:
	(tmp_path / "sub").mkdir()
	task = Task(("python", "-c", "import os; print(os.getcwd())"), cwd=Path("sub"))
	result = asyncio.run(run(task, base=tmp_path))
	assert result.returncode == 0
	assert _printed_cwd(result).resolve() == (tmp_path / "sub").resolve()


def test_run_cwd_none_runs_in_base(tmp_path: Path) -> None:
	task = Task(("python", "-c", "import os; print(os.getcwd())"))
	result = asyncio.run(run(task, base=tmp_path))
	assert result.returncode == 0
	assert _printed_cwd(result).resolve() == tmp_path.resolve()


def test_run_base_none_preserves_process_cwd() -> None:
	task = Task(("python", "-c", "import os; print(os.getcwd())"))
	result = asyncio.run(run(task))
	assert result.returncode == 0
	assert _printed_cwd(result).resolve() == Path.cwd().resolve()


class FakeProc:
	"""Records the signals an interrupt escalation would send to a leaf subprocess."""

	def __init__(self) -> None:
		self.signals: list[int] = []
		self.killed = False

	def send_signal(self, sig: int, /) -> None:
		self.signals.append(sig)

	def kill(self) -> None:
		self.killed = True


async def _forever() -> tuple[TaskResult, ...]:
	pending: asyncio.Future[tuple[TaskResult, ...]] = asyncio.get_running_loop().create_future()
	return await pending


def test_step_interrupt_forwards_twice_then_kills() -> None:
	a, b = Task("a"), Task("b")
	t0 = datetime(2026, 1, 1)
	p0, p1 = FakeProc(), FakeProc()
	procs: dict[int, Signalable] = {0: p0, 1: p1}
	interrupts = Interrupts(procs=procs)
	states: list[LeafState] = [Running(a, t0, b""), Running(b, t0, b"")]

	step_interrupt(interrupts, states)
	assert (p0.signals, p1.signals) == ([signal.SIGINT], [signal.SIGINT])
	assert states == [Interrupting(a, t0, b"", 1), Interrupting(b, t0, b"", 1)]

	step_interrupt(interrupts, states)
	assert p0.signals == [signal.SIGINT, signal.SIGINT]
	assert states == [Interrupting(a, t0, b"", 2), Interrupting(b, t0, b"", 2)]

	step_interrupt(interrupts, states)
	assert (p0.killed, p1.killed) == (True, True)
	assert states == [
		Interrupting(a, t0, b"", KILL_PRESSES),
		Interrupting(b, t0, b"", KILL_PRESSES),
	]


def test_fourth_press_cancels_run_and_await_run_returns_empty() -> None:
	async def scenario() -> tuple[bool, tuple[TaskResult, ...]]:
		main = asyncio.ensure_future(_forever())
		await asyncio.sleep(0)
		procs: dict[int, Signalable] = {0: FakeProc()}
		interrupts = Interrupts(procs=procs, main_task=main)
		states: list[LeafState] = [Running(Task("x"), datetime(2026, 1, 1), b"")]
		for _ in range(4):
			step_interrupt(interrupts, states)
		results = await await_run(main, interrupts)
		return main.cancelled(), results

	cancelled, results = asyncio.run(scenario())
	assert cancelled is True
	assert results == ()


class _SignalAfterOutputs:
	"""Fire SIGINT once ``count`` distinct leaves have emitted output — a deterministic replacement
	for a fixed wall-clock timer. A leaf's output arrives only after its subprocess is spawned *and*
	registered for interruption (its ``StartedEvent`` fires before the spawn), so a slow CI runner
	can't deliver the signal while a child is still unstarted and unkillable — the macOS-ARM SIGINT
	race fixed for a sibling test in #156/#161, applied here to the other two.
	"""

	def __init__(self, count: int) -> None:
		self.count = count
		self.started: set[int] = set()

	async def setup(self, task: TaskNode) -> None:
		return None

	async def on_event(self, event: TaskEvent, states: Sequence[LeafState], ctx: None) -> None:
		if isinstance(event, OutputEvent) and len(self.started) < self.count:
			self.started.add(event.leaf_index)
			if len(self.started) == self.count:
				os.kill(os.getpid(), signal.SIGINT)

	async def teardown(self, ctxs: tuple[None, ...]) -> None:
		return None


_PRINT_THEN_SLEEP: Final = "print('up', flush=True); import time; time.sleep(5)"
"""Emit a line (so an ``OutputEvent`` marks the subprocess registered), then sleep long enough to
be interrupted while running."""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal handling only")
def test_ctrl_c_forwards_sigint_and_exits_130() -> None:
	async def scenario() -> RunResult:
		task = Parallel(
			Task(("python", "-c", _PRINT_THEN_SLEEP), name="a"),
			Task(("python", "-c", _PRINT_THEN_SLEEP), name="b"),
		)
		return await run(task, effects=(_SignalAfterOutputs(2),))

	result = asyncio.run(scenario())
	assert result.returncode == INTERRUPT_RC
	assert result.results
	assert all(isinstance(r.completion, Stopped) for r in result.results)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal handling only")
def test_ctrl_c_resolves_jobs_queued_leaves_as_stopped() -> None:
	async def scenario() -> RunResult:
		task = Parallel(
			*(Task(("python", "-c", _PRINT_THEN_SLEEP), name=f"t{i}") for i in range(4))
		)
		return await run(task, jobs=1, effects=(_SignalAfterOutputs(1),))

	result = asyncio.run(scenario())
	assert result.returncode == INTERRUPT_RC
	assert len(result.results) == 4
	assert all(isinstance(r.completion, Stopped) for r in result.results)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal handling only")
def test_ctrl_c_keeps_pre_interrupt_completion_finished() -> None:
	class _SignalAfterQuick:
		async def setup(self, task: TaskNode) -> None:
			return None

		async def on_event(self, event: TaskEvent, states: Sequence[LeafState], ctx: None) -> None:
			if isinstance(event, CompletedEvent) and event.task.name == "quick":
				os.kill(os.getpid(), signal.SIGINT)

		async def teardown(self, ctxs: tuple[None, ...]) -> None:
			pass

	async def scenario() -> RunResult:
		task = Parallel(
			Task(("python", "-c", "pass"), name="quick"),
			Task(("python", "-c", "import time; time.sleep(5)"), name="slow"),
		)
		return await run(task, effects=(_SignalAfterQuick(),))

	result = asyncio.run(scenario())
	by_name = {r.name: r.completion for r in result.results}
	assert isinstance(by_name["quick"], Finished)
	assert isinstance(by_name["slow"], Stopped)


class _FakeStdin:
	def __init__(self, fd: int) -> None:
		self._fd = fd

	def fileno(self) -> int:
		return self._fd


def test_suppress_and_restore_ctrl_c_echo(monkeypatch: pytest.MonkeyPatch) -> None:
	if sys.platform != "win32":  # pragma: no branch
		import termios

		master, slave = os.openpty()
		try:
			on = termios.tcgetattr(slave)
			on[3] |= termios.ECHOCTL
			termios.tcsetattr(slave, termios.TCSANOW, on)
			monkeypatch.setattr(sys, "stdin", _FakeStdin(slave))

			saved = suppress_ctrl_c_echo()
			assert saved is not None
			assert not termios.tcgetattr(slave)[3] & termios.ECHOCTL

			restore_tty(saved)
			assert termios.tcgetattr(slave)[3] & termios.ECHOCTL
		finally:
			os.close(master)
			os.close(slave)
