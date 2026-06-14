# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from camas import Parallel, Sequential, Task
from camas.core.execution import Interrupts, Signalable, await_run, run, step_interrupt
from camas.v0.completion import INTERRUPT_RC, Finished, Stopped
from camas.v0.task_event import AbortedEvent, InterruptedEvent, TaskEvent

if TYPE_CHECKING:
	from camas.core.completion import RunResult, TaskResult


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


class FakeProc:
	"""Records the signals an interrupt escalation would send to a leaf subprocess."""

	def __init__(self) -> None:
		self.signals: list[int] = []
		self.killed = False

	def send_signal(self, sig: int, /) -> None:
		self.signals.append(sig)

	def kill(self) -> None:
		self.killed = True


def _ignore_event(_idx: int, _event: TaskEvent) -> None:
	pass


async def _forever() -> tuple[TaskResult, ...]:
	pending: asyncio.Future[tuple[TaskResult, ...]] = asyncio.get_running_loop().create_future()
	return await pending


def test_step_interrupt_forwards_twice_then_kills() -> None:
	leaves = (Task("a"), Task("b"))
	p0, p1 = FakeProc(), FakeProc()
	procs: dict[int, Signalable] = {0: p0, 1: p1}
	events: list[tuple[int, TaskEvent]] = []
	interrupts = Interrupts(procs=procs, signaled=set(), pending=[])
	now = datetime(2026, 1, 1)

	step_interrupt(interrupts, leaves, now, lambda i, e: events.append((i, e)))
	assert p0.signals == [signal.SIGINT]
	assert p1.signals == [signal.SIGINT]
	assert interrupts.signaled == {0, 1}
	assert [type(e) for _, e in events] == [InterruptedEvent, InterruptedEvent]

	step_interrupt(interrupts, leaves, now, lambda i, e: events.append((i, e)))
	assert p0.signals == [signal.SIGINT, signal.SIGINT]
	assert len(events) == 2  # 2nd press forwards only — no new events

	step_interrupt(interrupts, leaves, now, lambda i, e: events.append((i, e)))
	assert p0.killed
	assert p1.killed
	assert [type(e) for _, e in events[2:]] == [AbortedEvent, AbortedEvent]


def test_fourth_press_cancels_run_and_await_run_returns_empty() -> None:
	async def scenario() -> tuple[bool, tuple[TaskResult, ...]]:
		main = asyncio.ensure_future(_forever())
		await asyncio.sleep(0)  # let the run task suspend before we interrupt it
		procs: dict[int, Signalable] = {0: FakeProc()}
		interrupts = Interrupts(procs=procs, signaled=set(), pending=[], main_task=main)
		for _ in range(4):
			step_interrupt(interrupts, (Task("x"),), datetime(2026, 1, 1), _ignore_event)
		results = await await_run(main, interrupts)
		return main.cancelled(), results

	cancelled, results = asyncio.run(scenario())
	assert cancelled is True
	assert results == ()


def _raise_sigint() -> None:
	os.kill(os.getpid(), signal.SIGINT)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal handling only")
def test_ctrl_c_forwards_sigint_and_exits_130() -> None:
	async def scenario() -> RunResult:
		asyncio.get_running_loop().call_later(0.3, _raise_sigint)
		task = Parallel(
			Task(("python", "-c", "import time; time.sleep(5)"), name="a"),
			Task(("python", "-c", "import time; time.sleep(5)"), name="b"),
		)
		return await run(task)

	result = asyncio.run(scenario())
	assert result.returncode == INTERRUPT_RC
	assert result.results
	assert all(isinstance(r.completion, Stopped) for r in result.results)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal handling only")
def test_ctrl_c_resolves_jobs_queued_leaves_as_stopped() -> None:
	async def scenario() -> RunResult:
		asyncio.get_running_loop().call_later(0.3, _raise_sigint)
		task = Parallel(
			*(Task(("python", "-c", "import time; time.sleep(5)"), name=f"t{i}") for i in range(4))
		)
		return await run(task, jobs=1)

	result = asyncio.run(scenario())
	assert result.returncode == INTERRUPT_RC
	assert len(result.results) == 4
	assert all(isinstance(r.completion, Stopped) for r in result.results)
