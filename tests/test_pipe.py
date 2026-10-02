# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Pipe (#276): fd-wired stages, pipefail, and the agent/human runner split."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

import pytest

from camas import Parallel, Pipe, Sequential, Task
from camas.core.budget import Fits, plan_under
from camas.core.execution import run
from camas.core.gate import strip_agent_only_pipes, with_agent_format
from camas.core.matrix import expand_matrix
from camas.core.timings import CacheKey, TaskTiming
from camas.v0.completion import INTERRUPT_RC, Errored, Finished, Skipped, Stopped
from camas.v0.leaf_state import Waiting
from camas.v0.task import AgentFormat

if TYPE_CHECKING:
	import subprocess
	from collections.abc import Sequence
	from pathlib import Path

	from camas.core.completion import TaskResult
	from camas.v0.leaf_state import LeafState
	from camas.v0.task import TaskNode
	from camas.v0.task_event import TaskEvent

ECHO_UPPER: tuple[str, ...] = (
	"python",
	"-c",
	"import sys; sys.stdout.write(sys.stdin.read().upper())",
)


def test_pipe_coerces_stage_strings_and_rejects_nested_groups() -> None:
	assert Pipe("echo a", "echo b").tasks == (Task("echo a"), Task("echo b"))
	with pytest.raises(ValueError, match="stages must be Tasks"):
		Pipe("echo a", Sequential("echo b"))
	stage = Task("echo a")
	with pytest.raises(ValueError, match="stages must be distinct"):
		Pipe(stage, stage)


def test_pipe_equality_includes_agent_only() -> None:
	assert Pipe("a", "b") != Pipe("a", "b", agent_only=True)
	assert Pipe("a", "b", agent_only=True) == Pipe("a", "b", agent_only=True)


def test_pipe_operators_compose_as_an_opaque_group() -> None:
	pipe = Pipe("a", "b")
	assert (pipe | "c").tasks == (pipe, Task("c"))
	assert (pipe + "c").tasks == (pipe, Task("c"))


def test_gt_operator_composes_pipe_stages() -> None:
	"""``>`` pipes leaf stdout into the next stage. Chains of more than two need parens —
	Python's comparison chaining would otherwise drop the head."""
	assert (Task("a") > "b") == Pipe("a", "b")
	assert ((Task("a") > "b") > "c").tasks == (Task("a"), Task("b"), Task("c"))
	assert (Task("a") > Pipe("b", "c")) == Pipe("a", "b", "c")
	assert (Pipe("a") > "b").tasks == (Task("a"), Task("b"))


def test_gt_operator_carries_fields_and_agent_only() -> None:
	extended = Pipe("a", env={"K": "v"}, agent_only=True) > "b"
	assert extended.env == {"K": "v"}
	assert extended.agent_only is True
	assert (Pipe("a") > Pipe("b", env={"K": "v"})).env == {"K": "v"}
	assert (Pipe("a", agent_only=True) > Pipe("b")).agent_only is True
	assert (Pipe("a") > Pipe("b", agent_only=True)).agent_only is True


def test_gt_operator_rejects_a_group_stage() -> None:
	with pytest.raises(ValueError, match="stages must be Tasks"):
		_ = Sequential("a") > "b"
	with pytest.raises(ValueError, match="stages must be Tasks"):
		_ = Parallel("a") > "b"
	with pytest.raises(ValueError, match="stages must be Tasks"):
		_ = Task("a") > Sequential("b")


def test_strip_agent_only_pipes_leaves_a_pipeless_tree_untouched() -> None:
	seq = Sequential(Task("a"), Task("b"))
	assert strip_agent_only_pipes(seq) is seq


def test_strip_agent_only_pipes_collapses_to_the_first_stage() -> None:
	"""The collapse keeps the group's fields on a single-stage Pipe, so env/cwd/matrix still
	reach the human's run of that stage."""
	node = Sequential(
		Pipe("cargo clippy", "clippy-sarif", agent_only=True, env={"A": "1"}),
		Pipe("fmt", "tee"),
	)
	assert strip_agent_only_pipes(node) == Sequential(
		Pipe(Task("cargo clippy"), env={"A": "1"}),
		Pipe("fmt", "tee"),
	)
	assert strip_agent_only_pipes(Pipe("a", "b")) == Pipe("a", "b")


def test_apply_overrides_keeps_agent_only() -> None:
	from camas.core.matrix import apply_overrides

	pipe = Pipe("a {X}", "b {X}", matrix={"X": ("1",)}, agent_only=True)
	overridden = apply_overrides(pipe, {"X": ("1",)})
	assert isinstance(overridden, Pipe)
	assert overridden.agent_only is True


def test_pipe_hash_handles_group_fields() -> None:
	hash(Pipe("a", "b", env={"K": "v"}, matrix={"X": ("1",)}))


def test_pipe_wires_stdout_into_the_next_stage() -> None:
	pipe = Pipe(
		Task(("python", "-c", "import sys; sys.stdout.write('hello')")),
		Task(ECHO_UPPER),
	)
	result = asyncio.run(run(pipe, jobs=1))
	assert result.returncode == 0
	first, last = (r.completion for r in result.results)
	assert isinstance(first, Finished)
	assert first.output == ()
	assert isinstance(last, Finished)
	assert last.output == (b"HELLO",)


def test_pipe_runs_with_a_denied_stdin() -> None:
	"""The gate path denies the parent's stdin — the first stage reads DEVNULL, and the shared
	handle must survive the pipe's fd cleanup."""
	pipe = Pipe(
		Task(("python", "-c", "import sys; sys.stdout.write('x')")),
		Task(ECHO_UPPER),
	)
	result = asyncio.run(run(pipe, jobs=1, interactive=False))
	assert result.returncode == 0


def test_pipe_last_stage_stdout_is_the_pipeline_output() -> None:
	pipe = Pipe(
		Task(("python", "-c", "print('x'); print('y')")),
		Task(
			(
				"python",
				"-c",
				"import sys; sys.stdout.write(str(len(sys.stdin.read().splitlines())))",
			)
		),
	)
	result = asyncio.run(run(pipe, jobs=1))
	last = result.results[1].completion
	assert isinstance(last, Finished)
	assert last.output == (b"2",)


def test_pipe_fails_pipefail_style_when_an_upstream_stage_dies() -> None:
	pipe = Pipe(Task(("python", "-c", "raise SystemExit(3)")), Task(("python", "-c", "pass")))
	result = asyncio.run(run(pipe, jobs=1))
	assert result.returncode == 1
	assert result.results[0].completion.returncode == 3
	assert result.results[1].completion.returncode == 0


def test_pipe_runs_every_stage_when_an_upstream_stage_dies() -> None:
	pipe = Pipe(Task(("python", "-c", "raise SystemExit(3)")), Task(ECHO_UPPER))
	result = asyncio.run(run(pipe, jobs=1))
	assert result.results[1].completion.returncode == 0
	last = result.results[1].completion
	assert isinstance(last, Finished)
	assert last.output == ()


def test_pipe_forwards_stage_stderr_to_that_stages_leaf() -> None:
	pipe = Pipe(
		Task(("python", "-c", "import sys; sys.stderr.write('warn'); print('out')")),
		Task(ECHO_UPPER),
	)
	result = asyncio.run(run(pipe, jobs=1))
	first = result.results[0].completion
	assert isinstance(first, Finished)
	assert first.output == (b"warn",)


def test_with_agent_format_appends_args_to_each_pipe_stage(tmp_path: Path) -> None:
	pipe = Pipe(
		Task("cargo clippy", agent_format=AgentFormat("--message-format=json", "raw")),
		Task("clippy-sarif", agent_format=AgentFormat("", "sarif")),
	)
	formatted = with_agent_format(pipe, tmp_path)
	assert formatted.node.tasks[0].cmd == "cargo clippy --message-format=json"  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
	assert formatted.node.tasks[1].cmd == "clippy-sarif"  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]


def test_plan_under_preserves_pipe_stage_order() -> None:
	gen = Task("cargo clippy", name="gen")
	sarif = Task("clippy-sarif", name="sarif")
	pipe = Pipe(gen, sarif)
	timings = {
		CacheKey("gen", 0): TaskTiming(0.1, 5),
		CacheKey("sarif", 0): TaskTiming(0.1, 5),
	}
	plan = plan_under(pipe, 1.0, timings)
	assert plan.node == pipe
	assert plan.fits == (Fits(gen, 0.1), Fits(sarif, 0.1))
	assert plan.over_budget == ()
	assert plan.untimed == ()


def test_plan_under_drops_the_whole_pipe_when_a_stage_is_over_budget() -> None:
	"""A cut stage would rewire the pipeline — the survivor before the cut feeding the one
	after it — so any dropped stage drops the whole pipe."""
	gen = Task("cargo clippy", name="gen")
	sarif = Task("clippy-sarif", name="sarif")
	pipe = Pipe(gen, sarif)
	timings = {
		CacheKey("gen", 0): TaskTiming(0.1, 5),
		CacheKey("sarif", 0): TaskTiming(9.0, 5),
	}
	plan = plan_under(pipe, 1.0, timings)
	assert plan.node is None
	assert plan.fits == (Fits(gen, 0.1),)
	assert plan.over_budget == (plan.over_budget[0],)


def test_expand_matrix_fans_out_a_pipe_matrix_as_pipe_clones() -> None:
	result = expand_matrix(Pipe("a {X}", "b {X}", matrix={"X": ("1", "2")}))
	assert isinstance(result, Parallel)
	assert all(isinstance(t, Pipe) for t in result.tasks)
	assert result.tasks[0].tasks[0].cmd == "a 1"  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]


def test_github_matrix_rejects_a_pipe_with_the_fd_wiring_message() -> None:
	from camas.main.github_matrix import jobs_emission

	with pytest.raises(ValueError, match="fd-wired"):
		jobs_emission(Pipe("a", "b"), {})


def test_github_matrix_rejects_a_single_stage_pipe_as_a_leaf() -> None:
	"""A collapsed agent_only pipe is one stage — the leaf message, not the fd-wiring one."""
	from camas.main.github_matrix import jobs_emission

	with pytest.raises(ValueError, match="single leaf"):
		jobs_emission(Pipe("a"), {})


def test_pipe_cancel_kills_every_stage() -> None:
	"""Cancelling a pipe run kills and reaps every stage — no transport outlives the loop."""
	pipe = Pipe(
		Task(("python", "-c", "import time; time.sleep(60)")),
		Task(("python", "-c", "import time; time.sleep(60)")),
	)

	async def scenario() -> None:
		main_task = asyncio.ensure_future(run(pipe))
		await asyncio.sleep(0.2)
		main_task.cancel()
		with pytest.raises(asyncio.CancelledError):
			await main_task

	asyncio.run(scenario())


def test_pipe_cancel_during_spawn_kills_registered_stages() -> None:
	"""A cancel landing inside the spawn loop still kills and reaps the spawned stages."""
	pipe = Pipe(
		Task(("python", "-c", "import time; time.sleep(60)")),
		Task(("python", "-c", "pass")),
	)

	class SlowStart:
		def __init__(self) -> None:
			self.started = 0

		async def setup(self, task: TaskNode) -> None:
			return None

		async def on_event(self, event: TaskEvent, states: Sequence[LeafState], ctx: None) -> None:
			from camas.v0.task_event import StartedEvent

			self.started += isinstance(event, StartedEvent)
			if self.started == 2:
				await asyncio.sleep(60)

		async def teardown(self, ctxs: tuple[None, ...]) -> None:
			pass

	async def scenario() -> None:
		main_task = asyncio.ensure_future(run(pipe, effects=(SlowStart(),)))
		await asyncio.sleep(0.2)
		main_task.cancel()
		with pytest.raises(asyncio.CancelledError):
			await main_task

	asyncio.run(scenario())


def test_pipe_spawn_failure_errors_that_stage_and_skips_the_rest() -> None:
	pipe = Pipe(Task("definitely-not-a-command-xyz"), Task(ECHO_UPPER))
	result = asyncio.run(run(pipe, jobs=1))
	assert result.returncode == 1
	assert result.results[0].completion.returncode == 127
	assert result.results[1].completion.returncode == 127


def test_pipe_spawn_failure_mid_pipe_kills_the_spawned_stages() -> None:
	pipe = Pipe(Task(("python", "-c", "import time; time.sleep(60)")), Task("no-such-cmd-xyz"))
	result = asyncio.run(run(pipe, jobs=1))
	assert result.returncode == 1
	assert result.results[1].completion.returncode == 127


def test_pipe_spawn_failure_never_starts_later_stages() -> None:
	"""Stages after a failed spawn never launch — Skipped without a StartedEvent, like
	skip_subtree's leaves."""
	from camas.v0.task_event import StartedEvent

	events: list[TaskEvent] = []

	class Recorder:
		async def setup(self, task: TaskNode) -> None:
			return None

		async def on_event(self, event: TaskEvent, states: Sequence[LeafState], ctx: None) -> None:
			events.append(event)

		async def teardown(self, ctxs: tuple[None, ...]) -> None:
			pass

	pipe = Pipe(
		Task(("python", "-c", "pass")),
		Task("no-such-cmd-xyz"),
		Task(("python", "-c", "pass")),
	)
	result = asyncio.run(run(pipe, jobs=1, effects=(Recorder(),)))
	assert result.results[2].completion.returncode == 127
	started = {e.leaf_index for e in events if isinstance(e, StartedEvent)}
	assert 2 not in started


def test_pipe_spawn_failure_reports_a_finished_earlier_stage_as_stopped() -> None:
	"""A stage that genuinely ran before the failure reads Stopped with its own code — 0 when
	it finished before the kill, a kill code otherwise — never Skipped with the failed
	stage's 127, and it keeps the stderr its reader captured before the kill. The failed
	spawn waits on the reader's OutputEvent, so the capture provably precedes the failure
	instead of racing the stage's interpreter startup."""
	from camas.core import execution as execution_module
	from camas.v0.task_event import OutputEvent

	original_spawn = execution_module._spawn_stage  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # the monkeypatched seam, kept for pass-through
	wrote = asyncio.Event()

	class Recorder:
		async def setup(self, task: TaskNode) -> None:
			return None

		async def on_event(self, event: TaskEvent, states: Sequence[LeafState], ctx: None) -> None:
			if isinstance(event, OutputEvent) and event.leaf_index == 0:
				wrote.set()

		async def teardown(self, ctxs: tuple[None, ...]) -> None:
			pass

	async def failing_spawn(
		task: Task,
		*,
		stdin: int | None,
		stdout: int,
		stderr: int,
		base: Path | None,
		leaf_color: bool,
	) -> asyncio.subprocess.Process:
		if task.cmd == "no-such-cmd-xyz":
			await asyncio.wait_for(wrote.wait(), timeout=10)
			raise FileNotFoundError
		return await original_spawn(
			task, stdin=stdin, stdout=stdout, stderr=stderr, base=base, leaf_color=leaf_color
		)

	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(execution_module, "_spawn_stage", failing_spawn)
	pipe = Pipe(
		Task(
			(
				"python",
				"-c",
				"import sys, time; sys.stderr.buffer.write(b'warn\\n'); sys.stderr.flush(); time.sleep(60)",
			)
		),
		Task("no-such-cmd-xyz"),
	)
	result = asyncio.run(run(pipe, jobs=1, effects=(Recorder(),)))
	monkeypatch.undo()
	first = result.results[0].completion
	assert isinstance(first, Stopped)
	assert first.returncode != 127
	assert first.output == (b"warn\n",)
	assert result.results[1].completion.returncode == 127


def test_pipe_interrupted_before_spawning_stops_the_stages() -> None:
	"""A landed interrupt stops the remaining stages without launching them — the run_cmd
	pre-spawn guard, mirrored."""
	from contextlib import nullcontext

	from camas.core.execution import Interrupts, RunContext, run_pipe
	from camas.v0.task_event import CompletedEvent

	a, b = Task("a"), Task("b")
	events: list[TaskEvent] = []

	async def dispatch(leaf_idx: int, event: TaskEvent) -> None:
		events.append(event)

	async def scenario() -> tuple[TaskResult, ...]:
		leaves = (a, b)
		index_map = {id(a): 0, id(b): 1}
		states: list[LeafState] = [Waiting(a), Waiting(b)]
		interrupts = Interrupts(procs={})
		interrupts.count = 1
		ctx = RunContext(
			dispatch, leaves, index_map, nullcontext(), interrupts, states, None, None, True, None
		)
		return await run_pipe((a, b), ctx)

	results = asyncio.run(scenario())
	assert all(isinstance(r.completion, Stopped) for r in results)
	assert all(r.completion.returncode == INTERRUPT_RC for r in results)
	assert all(isinstance(e, CompletedEvent) for e in events)


def test_pipe_interrupt_cuts_by_position_not_equality() -> None:
	"""Equal-but-distinct stages: the cut slices by position, so each leaf gets exactly one
	completion."""
	from contextlib import nullcontext

	from camas.core.execution import Interrupts, RunContext, run_pipe
	from camas.v0.task_event import CompletedEvent

	a1 = Task("python -c pass")
	a2 = Task("python -c pass")
	events: list[TaskEvent] = []

	async def dispatch(leaf_idx: int, event: TaskEvent) -> None:
		events.append(event)

	async def scenario() -> tuple[TaskResult, ...]:
		leaves = (a1, a2)
		index_map = {id(a1): 0, id(a2): 1}
		states: list[LeafState] = [Waiting(a1), Waiting(a2)]
		interrupts = Interrupts(procs={})
		interrupts.count = 1
		ctx = RunContext(
			dispatch, leaves, index_map, nullcontext(), interrupts, states, None, None, True, None
		)
		return await run_pipe((a1, a2), ctx)

	results = asyncio.run(scenario())
	assert len(results) == 2
	completions = [e for e in events if isinstance(e, CompletedEvent)]
	assert [c.leaf_index for c in completions] == [0, 1]


def test_pipe_interrupt_mid_pipe_unwinds_the_spawned_stages() -> None:
	"""A landed interrupt after a stage spawned kills it and reports its own code, then
	Stops the rest — nothing is abandoned."""
	from contextlib import nullcontext

	from camas.core.execution import Interrupts, RunContext, run_pipe
	from camas.v0.task_event import CompletedEvent, StartedEvent

	a = Task(("python", "-c", "import time; time.sleep(60)"))
	b = Task("b")
	events: list[TaskEvent] = []
	interrupts = Interrupts(procs={})

	async def dispatch(leaf_idx: int, event: TaskEvent) -> None:
		if isinstance(event, StartedEvent):
			interrupts.count = 1
		events.append(event)

	async def scenario() -> tuple[TaskResult, ...]:
		leaves = (a, b)
		index_map = {id(a): 0, id(b): 1}
		states: list[LeafState] = [Waiting(a), Waiting(b)]
		ctx = RunContext(
			dispatch, leaves, index_map, nullcontext(), interrupts, states, None, None, True, None
		)
		return await run_pipe((a, b), ctx)

	results = asyncio.run(scenario())
	assert len(results) == 2
	assert isinstance(results[0].completion, Stopped)
	assert results[0].completion.returncode != 0
	assert results[1].completion.returncode == INTERRUPT_RC
	completions = [e for e in events if isinstance(e, CompletedEvent)]
	assert [c.leaf_index for c in completions] == [0, 1]


def test_pipe_interrupt_after_a_spawn_failure_unwinds_with_the_failure_semantics() -> None:
	"""An interrupt landing after a mid-pipe spawn failure still reports every stage exactly
	once: the spawned one Stopped with its own code, the failed one Errored, the rest
	Skipped — the failure's semantics, not a blanket interrupt."""
	from contextlib import nullcontext

	from camas.core.execution import Interrupts, RunContext, run_pipe
	from camas.v0.task_event import CompletedEvent, StartedEvent

	a = Task(("python", "-c", "import time; time.sleep(60)"))
	bad = Task("no-such-cmd-xyz")
	c = Task("c")
	events: list[TaskEvent] = []
	interrupts = Interrupts(procs={})

	async def dispatch(leaf_idx: int, event: TaskEvent) -> None:
		if isinstance(event, StartedEvent) and event.leaf_index == 1:
			interrupts.count = 1
		events.append(event)

	async def scenario() -> tuple[TaskResult, ...]:
		leaves = (a, bad, c)
		index_map = {id(a): 0, id(bad): 1, id(c): 2}
		states: list[LeafState] = [Waiting(a), Waiting(bad), Waiting(c)]
		ctx = RunContext(
			dispatch, leaves, index_map, nullcontext(), interrupts, states, None, None, True, None
		)
		return await run_pipe((a, bad, c), ctx)

	results = asyncio.run(scenario())
	assert len(results) == 3
	spawned, failed, rest = (r.completion for r in results)
	assert isinstance(spawned, Stopped)
	assert spawned.returncode != 127
	assert isinstance(failed, Errored)
	assert failed.returncode == 127
	assert isinstance(rest, Skipped)
	assert rest.returncode == 127
	completions = [e for e in events if isinstance(e, CompletedEvent)]
	assert [c.leaf_index for c in completions] == [0, 1, 2]
	started = {e.leaf_index for e in events if isinstance(e, StartedEvent)}
	assert started == {0, 1}


def test_pipe_cancel_inside_spawn_closes_the_fresh_pipe_fds() -> None:
	"""A cancel landing after os.pipe() but inside the spawn await closes the fresh pair —
	nothing leaks toward EMFILE in a long-lived server."""
	from camas.core import execution as execution_module

	async def slow_spawn(*args: object, **kwargs: object) -> object:
		await asyncio.sleep(60)
		raise AssertionError("unreachable")

	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(execution_module, "_spawn_stage", slow_spawn)
	pipe = Pipe(
		Task(("python", "-c", "pass")),
		Task(("python", "-c", "pass")),
	)

	async def scenario() -> None:
		main_task = asyncio.ensure_future(run(pipe, jobs=1))
		await asyncio.sleep(0.2)
		main_task.cancel()
		with pytest.raises(asyncio.CancelledError):
			await main_task

	asyncio.run(scenario())
	monkeypatch.undo()


def _cancelled_unwind(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Patch the unwind so a cancel lands on the unwinding task while the real unwind waits —
	the shape of a caller's timeout firing mid-teardown."""
	from typing import Any

	from camas.core import execution as execution_module
	from camas.core.unwind import unwind as real_unwind

	async def cancelled_unwind(*args: Any, **kwargs: Any) -> Any:
		current = asyncio.current_task()
		assert current is not None
		current.cancel()
		return await real_unwind(*args, **kwargs)

	monkeypatch.setattr(execution_module, "unwind", cancelled_unwind)


async def test_a_cancel_inside_a_real_stage_spawn_leaves_no_child(
	forked: list[subprocess.Popen[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
	"""asyncio's own transport kills and reaps a stage whose spawn is cancelled, so the pipe
	needs no shield or detached reaper — pinned against the real spawn, not a fake suspension."""
	from typing import Any

	from camas.core import execution as execution_module

	real_spawn = execution_module._spawn_stage  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # the monkeypatched seam, kept for pass-through

	async def cancelled_mid_spawn(task: Task, **kwargs: Any) -> asyncio.subprocess.Process:
		current = asyncio.current_task()
		assert current is not None
		asyncio.get_running_loop().call_soon(
			current.cancel
		)  # zuban: ignore[call-arg] # zuban drops Task.cancel's optional msg
		return await real_spawn(task, **kwargs)

	monkeypatch.setattr(execution_module, "_spawn_stage", cancelled_mid_spawn)
	sleeper = ("python", "-c", "import time; time.sleep(60)")
	with pytest.raises(asyncio.CancelledError):
		await run(Pipe(Task(sleeper), Task(sleeper)), interactive=False)
	assert len(forked) == 1
	assert forked[0].returncode is not None


async def test_the_spawn_failure_path_rethrows_a_cancel_its_unwind_absorbed(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""The spawn-failure path returns results normally after its unwind — a cancel absorbed
	there must still surface, or a caller's timeout silently becomes a success."""
	from contextlib import nullcontext

	from camas.core.execution import Interrupts, RunContext, run_pipe

	_cancelled_unwind(monkeypatch)
	a = Task(("python", "-c", "import time; time.sleep(60)"))
	b = Task(("camas-no-such-executable-on-path",))

	async def dispatch(leaf_idx: int, event: TaskEvent) -> None:
		pass

	states: list[LeafState] = [Waiting(a), Waiting(b)]
	ctx = RunContext(
		dispatch,
		(a, b),
		{id(a): 0, id(b): 1},
		nullcontext(),
		Interrupts(procs={}),
		states,
		None,
		None,
		True,
		None,
	)
	with pytest.raises(asyncio.CancelledError):
		await run_pipe((a, b), ctx)


async def test_a_timeout_firing_during_the_spawn_failure_unwind_still_times_out(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A caller's ``wait_for`` deadline expiring while the spawn-failure path unwinds must
	surface as ``TimeoutError``, not as the path's normal return — through 3.10's legacy
	``wait_for`` and 3.12+'s ``timeout()``-based one alike."""
	from contextlib import nullcontext
	from typing import Any

	from camas.core import execution as execution_module
	from camas.core.execution import Interrupts, RunContext, run_pipe
	from camas.core.unwind import unwind as real_unwind

	async def slow_unwind(children: Any, tasks: Any, timeout_s: float = 5.0) -> Any:
		lingering = asyncio.ensure_future(asyncio.sleep(0.3))
		return await real_unwind(children, (*tasks, lingering), timeout_s)

	monkeypatch.setattr(execution_module, "unwind", slow_unwind)
	a = Task(("python", "-c", "import time; time.sleep(60)"))
	b = Task(("camas-no-such-executable-on-path",))

	async def dispatch(leaf_idx: int, event: TaskEvent) -> None:
		pass

	states: list[LeafState] = [Waiting(a), Waiting(b)]
	ctx = RunContext(
		dispatch,
		(a, b),
		{id(a): 0, id(b): 1},
		nullcontext(),
		Interrupts(procs={}),
		states,
		None,
		None,
		True,
		None,
	)
	with pytest.raises(asyncio.TimeoutError):
		await asyncio.wait_for(run_pipe((a, b), ctx), 0.1)


async def test_the_interrupt_path_rethrows_a_cancel_its_unwind_absorbed(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""The landed-interrupt path returns results normally after its unwind — a cancel absorbed
	there must still surface."""
	from contextlib import nullcontext

	from camas.core.execution import Interrupts, RunContext, run_pipe

	_cancelled_unwind(monkeypatch)
	a = Task(("python", "-c", "import time; time.sleep(60)"))
	b = Task(("python", "-c", "import time; time.sleep(60)"))
	interrupts = Interrupts(procs={})

	async def dispatch(leaf_idx: int, event: TaskEvent) -> None:
		interrupts.count = 1

	states: list[LeafState] = [Waiting(a), Waiting(b)]
	ctx = RunContext(
		dispatch,
		(a, b),
		{id(a): 0, id(b): 1},
		nullcontext(),
		interrupts,
		states,
		None,
		None,
		True,
		None,
	)
	with pytest.raises(asyncio.CancelledError):
		await run_pipe((a, b), ctx)


@pytest.mark.skipif(sys.platform == "win32", reason="the grandchild handoff is POSIX-only")
async def test_a_grandchild_holding_a_stage_pipe_cannot_wedge_the_unwind(
	tmp_path: Path,
	capsys: pytest.CaptureFixture[str],
	forked: list[subprocess.Popen[bytes]],
) -> None:
	"""A stage's grandchild inherits its stderr and outlives the kill, so that reader never sees
	EOF. The unwind drains it only until its one deadline, then drops the tail and lets the
	cancel propagate — before, the drain waited on it forever."""
	import os
	import signal
	import time
	from contextlib import suppress

	from camas.core.unwind import UNWIND_TIMEOUT_S

	pid_file = tmp_path / "grandchild.pid"
	pid_tmp = tmp_path / "grandchild.pid.tmp"
	grandchild = (
		f"import os, time; open({str(pid_tmp)!r}, 'w').write(str(os.getpid())); "
		f"os.replace({str(pid_tmp)!r}, {str(pid_file)!r}); time.sleep(60)"
	)
	stage = (
		f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
		"time.sleep(60)"
	)
	running = asyncio.ensure_future(
		run(
			Pipe(
				Task(("python", "-c", stage)),
				Task(("python", "-c", "import sys; sys.stdin.read()")),
			),
			interactive=False,
		)
	)
	try:
		deadline = time.monotonic() + 10
		while not pid_file.exists():
			if time.monotonic() > deadline:  # pragma: no cover — only a dead stage never forks
				pytest.fail("the grandchild never announced its pid")
			await asyncio.sleep(0.01)
		started = time.monotonic()
		assert running.cancel()
		with pytest.raises(asyncio.CancelledError):
			await running
		assert time.monotonic() - started < UNWIND_TIMEOUT_S + 2
		assert "1 output reader(s) still open" in capsys.readouterr().err
	finally:
		if pid_file.exists():  # pragma: no branch — the pid is announced before the cancel
			with suppress(ProcessLookupError):
				os.kill(int(pid_file.read_text()), signal.SIGKILL)
		# The stage's transport closes only once the dead grandchild's end of the pipe EOFs;
		# let it, or the transport is collected after the loop closes.
		stage_stderr = forked[0].stderr
		assert stage_stderr is not None
		release = time.monotonic() + 10
		while not stage_stderr.closed:
			if time.monotonic() > release:  # pragma: no cover — only a live grandchild holds it
				pytest.fail("the stage's stderr never closed")
			await asyncio.sleep(0.01)
		await asyncio.sleep(0)


def test_render_shows_a_pipe_with_the_pipe_separator() -> None:
	from camas.core.render import GroupHeader, flatten_rows, render_tree_lines

	assert render_tree_lines(Pipe("a", "b")) == ["a | b", "├─ a", "└─ b"]
	assert render_tree_lines(Pipe("a", "b", name="fmt")) == ["fmt |", "├─ a", "└─ b"]
	header = flatten_rows(Pipe("a", "b"))[0]
	assert isinstance(header, GroupHeader)
	assert header.label == "a | b"
