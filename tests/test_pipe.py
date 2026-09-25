# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Pipe (#276): fd-wired stages, pipefail, and the agent/human runner split."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from camas import Parallel, Pipe, Sequential, Task
from camas.core.budget import Fits, OverBudget, Untimed, plan_under
from camas.core.execution import run
from camas.core.gate import strip_agent_only_pipes, with_agent_format
from camas.core.matrix import expand_matrix
from camas.core.timings import CacheKey, TaskTiming
from camas.v0.completion import INTERRUPT_RC, Errored, Finished, Skipped, Stopped
from camas.v0.leaf_state import Waiting
from camas.v0.task import AgentFormat

if TYPE_CHECKING:
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


def test_plan_under_drops_the_whole_pipe_when_a_mid_stage_is_over_budget() -> None:
	"""A cut mid-pipe would rewire the pipeline — the survivor before the cut feeding the one
	after it — so a mid-pipe drop drops the whole pipe."""
	gen = Task("cargo clippy", name="gen")
	sarif = Task("clippy-sarif", name="sarif")
	pipe = Pipe(gen, sarif)
	timings = {
		CacheKey("gen", 0): TaskTiming(9.0, 5),
		CacheKey("sarif", 0): TaskTiming(0.1, 5),
	}
	plan = plan_under(pipe, 1.0, timings)
	assert plan.node is None
	assert plan.runnable == ()
	assert plan.fits == (Fits(sarif, 0.1),)
	assert plan.over_budget == (OverBudget(gen, 9.0),)


def test_plan_under_keeps_a_pipe_prefix_when_only_the_last_stage_is_over_budget() -> None:
	"""A suffix-only drop needs no rewiring — the surviving prefix runs as the pipeline."""
	gen = Task("cargo clippy", name="gen")
	sarif = Task("clippy-sarif", name="sarif")
	pipe = Pipe(gen, sarif)
	timings = {
		CacheKey("gen", 0): TaskTiming(0.1, 5),
		CacheKey("sarif", 0): TaskTiming(9.0, 5),
	}
	plan = plan_under(pipe, 1.0, timings)
	assert plan.node == Pipe(gen)
	assert plan.fits == (Fits(gen, 0.1),)
	assert isinstance(plan.over_budget[0], OverBudget)


def test_plan_under_keeps_a_pipe_with_an_untimed_stage_whole() -> None:
	"""Dropping the pipe would starve the untimed stage of the first run that measures it,
	so the pipe runs whole — the over-budget stage included — and the timing data lets the
	next budget drop it cleanly."""
	gen = Task("cargo clippy", name="gen")
	sarif = Task("clippy-sarif", name="sarif")
	pipe = Pipe(gen, sarif)
	plan = plan_under(pipe, 1.0, {CacheKey("gen", 0): TaskTiming(9.0, 5)})
	assert plan.node == pipe
	assert plan.runnable == (gen, sarif)
	assert plan.fits == ()
	assert plan.over_budget == ()
	assert isinstance(plan.running_over_budget[0], OverBudget)
	assert isinstance(plan.untimed[0], Untimed)


def test_plan_under_reports_an_untimed_whole_pipe_mutating() -> None:
	"""The untimed-whole-run executes the over-budget stage too — its mutates must reach the
	parent's mutating-first ordering, or a Parallel could run it beside another mutator."""
	gen = Task("cargo clippy", name="gen", mutates=True)
	sarif = Task("clippy-sarif", name="sarif")
	pipe = Pipe(gen, sarif)
	plain = Task("other-mutator", name="plain", mutates=True)
	plan = plan_under(
		Parallel(pipe, plain),
		1.0,
		{CacheKey("gen", 0): TaskTiming(9.0, 5), CacheKey("plain", 0): TaskTiming(0.1, 5)},
	)
	assert plan.node == Sequential(pipe, plain)


def test_budget_summary_notes_over_budget_stages_running_for_untimed_siblings() -> None:
	from camas.main.dispatch import budget_summary_lines

	gen = Task("cargo clippy", name="gen")
	sarif = Task("clippy-sarif", name="sarif")
	plan = plan_under(Pipe(gen, sarif), 1.0, {CacheKey("gen", 0): TaskTiming(9.0, 5)})
	lines = budget_summary_lines(plan)
	assert "running 2 leaf(s) (1 unmeasured), excluded 0 over budget" in lines[0]
	assert "running anyway to measure untimed pipe siblings: gen ~9.00s" in lines[1]
	assert all("  over budget:" not in line for line in lines)


def test_budget_summary_counts_nothing_running_when_the_pipe_drops() -> None:
	"""The running count derives from the runnable schedule — a fits leaf of a dropped pipe
	is not running."""
	from camas.main.dispatch import budget_summary_lines

	gen = Task("cargo clippy", name="gen")
	sarif = Task("clippy-sarif", name="sarif")
	timings = {
		CacheKey("gen", 0): TaskTiming(9.0, 5),
		CacheKey("sarif", 0): TaskTiming(0.1, 5),
	}
	lines = budget_summary_lines(plan_under(Pipe(gen, sarif), 1.0, timings))
	assert "running 0 leaf(s) (0 unmeasured), excluded 1 over budget" in lines[0]
	assert "A mid-pipe cut would rewire the pipeline — nothing to run." in lines[-1]


def test_budget_summary_names_the_all_exceed_case() -> None:
	from camas.main.dispatch import budget_summary_lines

	slow = Task("slow", name="slow")
	lines = budget_summary_lines(plan_under(slow, 1.0, {CacheKey("slow", 0): TaskTiming(9.0, 5)}))
	assert "All leaves exceed the budget — nothing to run." in lines[-1]


def test_drop_unjustified_running_drops_a_stage_whose_untimed_sibling_was_pruned() -> None:
	"""The keep-whole decision is re-validated after scoping: a running-over-budget stage
	whose untimed sibling did not survive scoping is dropped — the over-budget stage blows
	the budget with nothing measured."""
	from camas.core import timings
	from camas.core.budget import drop_unjustified_running

	gen = Task("cargo clippy {paths}", name="gen", paths="rust")
	sarif = Task("clippy-sarif {paths}", name="sarif", paths="docs")
	plan = plan_under(Pipe(gen, sarif), 1.0, {CacheKey("gen", 0): TaskTiming(9.0, 5)})
	pruned_sibling = timings.observed(None, plan.node, ("rust/lib.rs",))
	assert drop_unjustified_running(pruned_sibling.node, plan, pruned_sibling.pairs) is None
	kept_sibling = timings.observed(None, plan.node, ("rust/lib.rs", "docs/readme.md"))
	assert (
		drop_unjustified_running(kept_sibling.node, plan, kept_sibling.pairs) == kept_sibling.node
	)
	assert drop_unjustified_running(None, plan) is None


def test_drop_unjustified_running_applies_the_cut_semantics() -> None:
	"""A running stage at the pipe's head drops the whole pipe (mid-pipe cut); a running
	suffix drops to the surviving prefix."""
	from camas.core import timings
	from camas.core.budget import drop_unjustified_running

	head = Pipe(
		Task("run", name="run"),
		Task("mid-fit", name="midfit"),
		Task("tail {paths}", name="tail", paths="docs"),
	)
	plan = plan_under(
		head,
		1.0,
		{CacheKey("run", 0): TaskTiming(9.0, 5), CacheKey("midfit", 0): TaskTiming(0.1, 5)},
	)
	scoped = timings.observed(None, plan.node, ("src/x.rs",))
	assert drop_unjustified_running(scoped.node, plan, scoped.pairs) is None

	suffix = Pipe(
		Task("gen", name="gen"),
		Task("run", name="run"),
		Task("tail {paths}", name="tail", paths="docs"),
	)
	plan = plan_under(
		suffix,
		1.0,
		{CacheKey("gen", 0): TaskTiming(0.1, 5), CacheKey("run", 0): TaskTiming(9.0, 5)},
	)
	scoped = timings.observed(None, plan.node, ("src/x.rs",))
	assert drop_unjustified_running(scoped.node, plan, scoped.pairs) == Pipe(
		Task("gen", name="gen")
	)


def test_drop_unjustified_running_drops_an_all_dropped_group() -> None:
	from camas.core.budget import drop_unjustified_running

	a_pipe = Pipe(Task("a", name="a"), Task("au {paths}", name="au", paths="docs"))
	b_pipe = Pipe(Task("b", name="b"), Task("bu {paths}", name="bu", paths="docs"))
	plan = plan_under(
		Parallel(a_pipe, b_pipe),
		1.0,
		{CacheKey("a", 0): TaskTiming(9.0, 5), CacheKey("b", 0): TaskTiming(9.0, 5)},
	)
	# plan_under works on expand_matrix's clones — take the stages from the plan's own tree.
	dropped = Parallel(Pipe(plan.node.tasks[0].tasks[0]), Pipe(plan.node.tasks[1].tasks[0]))
	assert drop_unjustified_running(dropped, plan) is None


def test_scoped_tree_keeps_a_suffix_pruned_pipe_prefix() -> None:
	"""A mid-pipe prune would rewire the pipeline, but a suffix-only prune keeps the prefix."""
	from camas.core.scope import scoped_tree

	a, b, c = Task("a"), Task("b"), Task("c")
	pipe = Pipe(a, b, c)
	assert scoped_tree(pipe, {id(a): a, id(b): b}) == Pipe(a, b)
	assert scoped_tree(pipe, {id(a): a, id(c): c}) is None
	assert scoped_tree(pipe, {}) is None


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


def test_pipe_spawn_failure_reports_a_finished_earlier_stage_as_stopped(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
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
	from datetime import datetime

	from camas.core.execution import Interrupts, RunContext, run_pipe
	from camas.v0.leaf_state import Running
	from camas.v0.task_event import CompletedEvent, StartedEvent

	a = Task(("python", "-c", "import time; time.sleep(60)"))
	b = Task("b")
	states: list[LeafState] = [Running(a, datetime.now(), b""), Waiting(b)]
	events: list[TaskEvent] = []
	interrupts = Interrupts(procs={})

	async def dispatch(leaf_idx: int, event: TaskEvent) -> None:
		if isinstance(event, StartedEvent):
			interrupts.count = 1
		events.append(event)

	async def scenario() -> tuple[TaskResult, ...]:
		leaves = (a, b)
		index_map = {id(a): 0, id(b): 1}
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


def test_pipe_interrupt_unwind_reports_an_unowned_finished_stage_finished(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""The unwind labels each spawned stage by its state, like wait_and_complete does: a stage
	the interrupt never owned — it finished naturally, its registration predating the press —
	reads Finished with its own exit code; the owned stage reads Stopped."""
	import time
	from contextlib import nullcontext
	from datetime import datetime

	from camas.core import execution as execution_module
	from camas.core.execution import Interrupts, RunContext, run_pipe
	from camas.v0.leaf_state import Running
	from camas.v0.task_event import CompletedEvent, StartedEvent

	original_spawn = execution_module._spawn_stage  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # the monkeypatched seam, kept for pass-through
	a_proc: list[asyncio.subprocess.Process] = []

	async def delayed_spawn(
		task: Task,
		*,
		stdin: int | None,
		stdout: int,
		stderr: int,
		base: Path | None,
		leaf_color: bool,
	) -> asyncio.subprocess.Process:
		proc = await original_spawn(
			task, stdin=stdin, stdout=stdout, stderr=stderr, base=base, leaf_color=leaf_color
		)
		if task.cmd == ("python", "-c", "pass"):
			a_proc.append(proc)
		elif task.cmd == ("python", "-c", "import time; time.sleep(60)"):
			# Wait for stage a's observed reap before the interrupt lands, so its natural
			# exit code — not the unwind's kill — is the one observed.
			deadline = time.monotonic() + 5
			while a_proc and a_proc[0].returncode is None:
				if (
					time.monotonic() > deadline
				):  # pragma: no cover  # only a stuck stage reaches the deadline
					pytest.fail("stage a never exited")
				await asyncio.sleep(0.01)
		return proc

	monkeypatch.setattr(execution_module, "_spawn_stage", delayed_spawn)
	a = Task(("python", "-c", "pass"))
	b = Task(("python", "-c", "import time; time.sleep(60)"))
	c = Task("c")
	states: list[LeafState] = [
		Running(a, datetime.now(), b""),
		Running(b, datetime.now(), b""),
		Waiting(c),
	]
	events: list[TaskEvent] = []
	interrupts = Interrupts(procs={})

	async def dispatch(leaf_idx: int, event: TaskEvent) -> None:
		if isinstance(event, StartedEvent) and event.leaf_index == 1:
			interrupts.count = 1
		events.append(event)

	async def scenario() -> tuple[TaskResult, ...]:
		leaves = (a, b, c)
		index_map = {id(a): 0, id(b): 1, id(c): 2}
		ctx = RunContext(
			dispatch, leaves, index_map, nullcontext(), interrupts, states, None, None, True, None
		)
		return await run_pipe((a, b, c), ctx)

	results = asyncio.run(scenario())
	assert isinstance(results[0].completion, Finished)
	assert results[0].completion.returncode == 0
	assert isinstance(results[1].completion, Stopped)
	assert results[2].completion.returncode == INTERRUPT_RC
	completions = [e for e in events if isinstance(e, CompletedEvent)]
	assert [c.leaf_index for c in completions] == [0, 1, 2]


def test_pipe_interrupt_after_a_spawn_failure_unwinds_with_the_failure_semantics() -> None:
	"""An interrupt landing after a mid-pipe spawn failure still reports every stage exactly
	once: the spawned one Stopped with its own code, the failed one Errored, the rest
	Skipped — the failure's semantics, not a blanket interrupt."""
	from contextlib import nullcontext
	from datetime import datetime

	from camas.core.execution import Interrupts, RunContext, run_pipe
	from camas.core.leaf_state import to_interrupting
	from camas.v0.leaf_state import Running
	from camas.v0.task_event import CompletedEvent, StartedEvent

	a = Task(("python", "-c", "import time; time.sleep(60)"))
	bad = Task("no-such-cmd-xyz")
	c = Task("c")
	states: list[LeafState] = [Running(a, datetime.now(), b""), Waiting(bad), Waiting(c)]
	events: list[TaskEvent] = []
	interrupts = Interrupts(procs={})

	async def dispatch(leaf_idx: int, event: TaskEvent) -> None:
		if isinstance(event, StartedEvent) and event.leaf_index == 1:
			interrupts.count = 1
			states[0] = to_interrupting(states[0], 1)
		events.append(event)

	async def scenario() -> tuple[TaskResult, ...]:
		leaves = (a, bad, c)
		index_map = {id(a): 0, id(bad): 1, id(c): 2}
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


def test_pipe_cancel_inside_spawn_closes_the_fresh_pipe_fds(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A cancel landing after os.pipe() but inside the spawn await closes the fresh pair —
	nothing leaks toward EMFILE in a long-lived server."""
	from camas.core import execution as execution_module

	async def slow_spawn(*args: object, **kwargs: object) -> object:
		await asyncio.sleep(60)
		raise AssertionError("unreachable")

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


def test_reap_cancelled_spawn_kills_even_when_the_reaper_is_cancelled() -> None:
	"""A cancel propagated into the reaper while it awaits the spawn task is uncancelled and
	retried — the kill still runs."""
	from camas.core.execution import (
		_reap_cancelled_spawn,  # pyright: ignore[reportPrivateUsage]
	)

	child_holder: list[asyncio.subprocess.Process] = []

	async def spawn_child() -> asyncio.subprocess.Process:
		proc = await asyncio.create_subprocess_exec("python", "-c", "import time; time.sleep(60)")
		child_holder.append(proc)
		try:
			await asyncio.sleep(0.2)
		except asyncio.CancelledError:
			return proc
		raise AssertionError("unreachable")

	async def scenario() -> None:
		spawn_task = asyncio.create_task(spawn_child())
		reaper = asyncio.create_task(_reap_cancelled_spawn(spawn_task))
		await asyncio.sleep(0.05)
		reaper.cancel()
		await reaper
		assert child_holder[0].returncode is not None

	asyncio.run(scenario())


def test_reap_cancelled_spawn_stops_when_the_spawn_itself_cancels() -> None:
	from camas.core.execution import (
		_reap_cancelled_spawn,  # pyright: ignore[reportPrivateUsage]
	)

	async def plain_spawn() -> asyncio.subprocess.Process:
		await asyncio.sleep(60)
		raise AssertionError("unreachable")

	async def scenario() -> None:
		spawn_task = asyncio.create_task(plain_spawn())
		reaper = asyncio.create_task(_reap_cancelled_spawn(spawn_task))
		await asyncio.sleep(0.05)
		reaper.cancel()
		await reaper
		assert spawn_task.cancelled()

	asyncio.run(scenario())


def test_pipe_cancel_during_spawn_kills_a_child_the_spawn_task_still_returns(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A cancel landing inside the spawn await can still hand back the Process — the unwind
	obtains it from the finished spawn task and kills it, so no child is orphaned."""
	import sys
	from os import kill

	from camas.core import execution as execution_module

	original_spawn = execution_module._spawn_stage  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # the monkeypatched seam, kept for pass-through
	spawned: list[asyncio.subprocess.Process] = []

	async def surviving_spawn(
		task: Task,
		*,
		stdin: int | None,
		stdout: int,
		stderr: int,
		base: Path | None,
		leaf_color: bool,
	) -> asyncio.subprocess.Process:
		proc = await original_spawn(
			task, stdin=stdin, stdout=stdout, stderr=stderr, base=base, leaf_color=leaf_color
		)
		spawned.append(proc)
		try:
			await asyncio.sleep(60)
		except asyncio.CancelledError:
			return proc
		raise AssertionError("unreachable")

	monkeypatch.setattr(execution_module, "_spawn_stage", surviving_spawn)
	pipe = Pipe(
		Task(("python", "-c", "import time; time.sleep(60)")),
		Task(("python", "-c", "pass")),
	)

	async def scenario() -> None:
		main_task = asyncio.ensure_future(run(pipe, jobs=1))
		await asyncio.sleep(0.2)
		main_task.cancel()
		with pytest.raises(asyncio.CancelledError):
			await main_task
		# kill_all awaited the reaper before re-raising — the reaped returncode is the
		# cross-platform proof; the pid probe adds the POSIX liveness check (on Windows it
		# would read the transport's still-open handle).
		assert spawned[0].returncode is not None
		if (
			sys.platform != "win32"
		):  # pragma: no cover  # the win32 arm never runs on a POSIX CI runner
			with pytest.raises(ProcessLookupError):
				kill(spawned[0].pid, 0)

	asyncio.run(scenario())


def test_render_shows_a_pipe_with_the_pipe_separator() -> None:
	from camas.core.render import GroupHeader, flatten_rows, render_tree_lines

	assert render_tree_lines(Pipe("a", "b")) == ["a | b", "├─ a", "└─ b"]
	assert render_tree_lines(Pipe("a", "b", name="fmt")) == ["fmt |", "├─ a", "└─ b"]
	header = flatten_rows(Pipe("a", "b"))[0]
	assert isinstance(header, GroupHeader)
	assert header.label == "a | b"
