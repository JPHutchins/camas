# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Pipe (#276): fd-wired stages, pipefail, and the agent/human runner split."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

import pytest

from camas import Parallel, Pipe, Sequential, Task
from camas.core.budget import Fits, OverBudget, Untimed, plan_under
from camas.core.execution import Interrupts, RunContext, run
from camas.core.gate import strip_agent_only_pipes, with_agent_format
from camas.core.matrix import expand_matrix
from camas.core.timings import CacheKey, TaskTiming
from camas.v0.completion import INTERRUPT_RC, Errored, Finished, Skipped, Stopped
from camas.v0.leaf_state import Waiting
from camas.v0.task import AgentFormat

if TYPE_CHECKING:
	import subprocess
	from collections.abc import Awaitable, Callable, Sequence
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


def test_an_unresolved_ref_stage_dies_at_the_engine_boundary_with_the_stage_message() -> None:
	"""A hand-written Ref in a pipe constructs (the expression surface resolves and
	re-validates) but the engine rejects it with the stage message, not an assert_never."""
	from typing import cast

	from camas.core.matrix import expand_matrix
	from camas.v0.ref import Ref

	with pytest.raises(ValueError, match="stages must be Tasks"):
		expand_matrix(Pipe(Task("a"), cast("Task", Ref("b"))))


def test_a_ref_outside_a_pipe_dies_at_the_engine_boundary_with_the_resolution_message() -> None:
	"""A hand-written Ref outside a pipe: the engine names the real rule — an unresolved
	reference — instead of diagnosing an invalid pipe stage."""
	from typing import cast

	from camas.core.matrix import expand_matrix
	from camas.v0.ref import Ref

	with pytest.raises(ValueError, match="unresolved Ref reached the engine"):
		expand_matrix(cast("TaskNode", Ref("b")))


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


def test_gt_operator_extends_the_left_pipe_keeping_its_fields_and_agent_only() -> None:
	extended = Pipe("a", env={"K": "v"}, agent_only=True) > "b"
	assert extended.env == {"K": "v"}
	assert extended.agent_only is True
	assert (Pipe("a", agent_only=True) > Pipe("b")).agent_only is True


def test_gt_operator_rejects_a_scoped_right_pipe() -> None:
	"""A stage can't nest, so a right pipe's fields or ``agent_only`` would be dropped by a
	splice; ``.extend(right.tasks)`` runs its stages in the left pipe's scope instead."""
	scoped = Pipe("b", env={"K": "v"})
	with pytest.raises(ValueError, match=r"pipe its \.tasks instead: x > y\.tasks"):
		_ = Pipe("a") > scoped
	with pytest.raises(ValueError, match=r"pipe its \.tasks instead: x > y\.tasks"):
		_ = Task("a") > Pipe("b", agent_only=True)
	assert (Task("a") > scoped.tasks) == Pipe("a", "b")
	assert (Pipe("a") > scoped.tasks) == Pipe("a", "b")
	assert Pipe("a").extend(scoped.tasks) == Pipe("a", "b")


def test_gt_operator_rejects_a_right_pipe_subclass_naming_its_type() -> None:
	class Staged(Pipe):  # pyrefly: ignore[bad-class-definition]
		__slots__ = ()

	with pytest.raises(ValueError, match=r"cannot splice Staged .* subclass type"):
		_ = Pipe("a") > Staged("b")


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
	from camas.core.budget import outcome_lines, resolve_budget

	gen = Task("cargo clippy", name="gen")
	sarif = Task("clippy-sarif", name="sarif")
	plan = plan_under(Pipe(gen, sarif), 1.0, {CacheKey("gen", 0): TaskTiming(9.0, 5)})
	lines = outcome_lines(resolve_budget(plan, None, ()))
	assert "running 2 leaf(s) (1 unmeasured), excluded 0 over budget" in lines[0]
	assert "running anyway to measure untimed pipe siblings: gen ~9.00s" in lines[1]
	assert all("  over budget:" not in line for line in lines)


def test_budget_summary_counts_nothing_running_when_the_pipe_drops() -> None:
	"""The running count derives from the runnable schedule — a fits leaf of a dropped pipe
	is not running."""
	from camas.core.budget import outcome_lines, resolve_budget

	gen = Task("cargo clippy", name="gen")
	sarif = Task("clippy-sarif", name="sarif")
	timings = {
		CacheKey("gen", 0): TaskTiming(9.0, 5),
		CacheKey("sarif", 0): TaskTiming(0.1, 5),
	}
	lines = outcome_lines(resolve_budget(plan_under(Pipe(gen, sarif), 1.0, timings), None, ()))
	assert "running 0 leaf(s) (0 unmeasured), excluded 1 over budget" in lines[0]
	assert "A mid-pipe cut would rewire the pipeline — nothing to run." in lines[-1]


def test_budget_summary_names_the_all_exceed_case() -> None:
	from camas.core.budget import outcome_lines, resolve_budget

	slow = Task("slow", name="slow")
	plan = plan_under(slow, 1.0, {CacheKey("slow", 0): TaskTiming(9.0, 5)})
	lines = outcome_lines(resolve_budget(plan, None, ()))
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
	assert plan.node is not None
	pruned_sibling = timings.observed(None, plan.node, ("rust/lib.rs",))
	assert pruned_sibling.node is not None
	assert drop_unjustified_running(pruned_sibling.node, plan, pruned_sibling.pairs).node is None
	kept_sibling = timings.observed(None, plan.node, ("rust/lib.rs", "docs/readme.md"))
	assert kept_sibling.node is not None
	assert (
		drop_unjustified_running(kept_sibling.node, plan, kept_sibling.pairs).node
		== kept_sibling.node
	)
	plain_plan = plan_under(
		Pipe(Task("gen", name="gen"), Task("sarif", name="sarif")),
		1.0,
		{CacheKey("gen", 0): TaskTiming(0.1, 5), CacheKey("sarif", 0): TaskTiming(0.1, 5)},
	)
	assert plain_plan.node is not None
	assert drop_unjustified_running(plain_plan.node, plain_plan).node is plain_plan.node


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
	assert plan.node is not None
	scoped = timings.observed(None, plan.node, ("src/x.rs",))
	assert scoped.node is not None
	head_cut = drop_unjustified_running(scoped.node, plan, scoped.pairs)
	assert head_cut.node is None
	assert [t.name for t in head_cut.pipe_cut] == ["midfit"]

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
	assert plan.node is not None
	scoped = timings.observed(None, plan.node, ("src/x.rs",))
	assert scoped.node is not None
	suffix_cut = drop_unjustified_running(scoped.node, plan, scoped.pairs)
	assert suffix_cut.node == Pipe(Task("gen", name="gen"))
	assert suffix_cut.pipe_cut == ()


def test_scoped_leaves_and_scoped_tree_agree_on_a_mid_pipe_prune() -> None:
	"""The pairs follow the same pipe cut semantics as the tree — a mid-pipe prune drops the
	pipe's pairs too, so the identities stay parallel to the tree. The surviving leaf carries
	``{paths}`` so the pair's members differ, pinning the original-first orientation the
	consumers rely on."""
	from camas.core import timings
	from camas.core.execution import run
	from camas.core.scope import Pruned, prune_pipes, scoped_leaves

	a = Task("a {paths}", name="a", paths="docs")
	b = Task("b", name="b")
	c = Task("c {paths}", name="c", paths=".")
	scoped_c = Task("c src/x.rs", name="c", paths=".")
	node = Parallel(Pipe(a, b), c)
	keying = timings.observed(None, node, ("src/x.rs",))
	assert keying.node == Parallel(scoped_c)
	assert scoped_leaves(node, ("src/x.rs",)) == ((c, scoped_c),)
	resolved = {id(b): b, id(c): scoped_c}
	assert prune_pipes(node, lambda t: resolved.get(id(t))) == Pruned(Parallel(scoped_c), (b,))
	assert keying.node is not None
	asyncio.run(run(keying.node, identities=keying.identities))


def test_dropped_prefix_runs_with_matching_identities() -> None:
	"""drop_unjustified_running can shrink the scoped tree — the identities re-derived from
	the dropped tree are parallel to its leaves, so run() accepts them."""
	from camas.core import timings
	from camas.core.budget import drop_unjustified_running
	from camas.core.execution import run

	a = Task(("python", "-c", "pass"), name="a")
	b = Task(("python", "-c", "pass"), name="b")
	c = Task("c {paths}", name="c", paths="docs")
	plan = plan_under(
		Pipe(a, b, c),
		1.0,
		{CacheKey("a", 0): TaskTiming(0.1, 5), CacheKey("b", 0): TaskTiming(9.0, 5)},
	)
	assert plan.node is not None
	keying = timings.observed(None, plan.node, ("src/x.rs",))
	assert keying.node is not None
	dropped = drop_unjustified_running(keying.node, plan, keying.pairs)
	assert dropped.node == Pipe(a)
	final = timings.narrowed(keying, dropped)
	assert final.node is not None
	asyncio.run(run(final.node, identities=final.identities))


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
	assert isinstance(plan.node, Parallel)
	pipe_a, pipe_b = plan.node.tasks
	assert isinstance(pipe_a, Pipe)
	assert isinstance(pipe_b, Pipe)
	dropped = Parallel(Pipe(pipe_a.tasks[0]), Pipe(pipe_b.tasks[0]))
	assert drop_unjustified_running(dropped, plan).node is None


def test_prune_pipes_keeps_a_suffix_pruned_prefix_and_records_a_mid_pipe_cut() -> None:
	"""A mid-pipe prune would rewire the pipeline, but a suffix-only prune keeps the prefix;
	the stages a mid-pipe cut takes down with the pipe are recorded, the pruned ones are not."""
	from camas.core.scope import Pruned, prune_pipes

	a, b, c = Task("a"), Task("b"), Task("c")
	pipe = Pipe(a, b, c)

	def keeping(*kept: Task) -> Callable[[Task], Task | None]:
		return lambda t: t if any(t is k for k in kept) else None

	assert prune_pipes(pipe, keeping(a, b)) == Pruned(Pipe(a, b), ())
	assert prune_pipes(pipe, keeping(a, c)) == Pruned(None, (a, c))
	assert prune_pipes(pipe, keeping()) == Pruned(None, ())


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
		if task.cmd == ("python", "-c", "import time; time.sleep(60)"):
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
	assert isinstance(results[2].completion, Stopped)
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
	nothing leaks toward EMFILE in a long-lived server. The fake spawn is held on an event the
	scenario releases, so the unwind does not pay a minute for a fake's sleep."""
	import os

	from camas.core import execution as execution_module

	fds: list[int] = []
	fds_recorded = asyncio.Event()
	release = asyncio.Event()

	async def slow_spawn(
		task: Task,
		*,
		stdin: int | None,
		stdout: int,
		stderr: int,
		base: Path | None,
		leaf_color: bool,
	) -> asyncio.subprocess.Process:
		fds.extend(fd for fd in (stdin, stdout, stderr) if isinstance(fd, int) and fd >= 0)
		fds_recorded.set()
		await release.wait()
		raise AssertionError("unreachable")

	monkeypatch.setattr(execution_module, "_spawn_stage", slow_spawn)
	pipe = Pipe(
		Task(("python", "-c", "pass")),
		Task(("python", "-c", "pass")),
	)

	async def scenario() -> None:
		main_task = asyncio.ensure_future(run(pipe, jobs=1))
		await asyncio.wait_for(fds_recorded.wait(), 5)
		main_task.cancel()
		release.set()
		with pytest.raises(asyncio.CancelledError):
			await main_task
		for fd in fds:
			with pytest.raises(OSError):  # noqa: PT011  # EBADF's errno-specific form differs per platform
				os.fstat(fd)

	asyncio.run(scenario())


def _pipe_ctx(stages: tuple[Task, ...], interrupts: Interrupts | None = None) -> RunContext:
	"""A ``RunContext`` for driving ``run_pipe`` directly; with ``interrupts``, every event
	lands the interrupt."""
	from contextlib import nullcontext

	landing = interrupts if interrupts is not None else Interrupts(procs={})

	async def dispatch(leaf_idx: int, event: TaskEvent) -> None:
		if interrupts is not None:
			landing.count = 1

	return RunContext(
		dispatch=dispatch,
		leaves=stages,
		index_map={id(stage): i for i, stage in enumerate(stages)},
		limiter=nullcontext(),
		interrupts=landing,
		states=[Waiting(stage) for stage in stages],
		base=None,
		child_stdin=None,
		leaf_color=True,
		identities=None,
	)


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


@pytest.mark.skipif(
	sys.platform == "win32", reason="the forked seam is the unix subprocess transport"
)
@pytest.mark.usefixtures("cancel_inside_spawn")
async def test_a_cancel_inside_a_real_stage_spawn_leaves_no_child(
	forked: list[subprocess.Popen[bytes]],
) -> None:
	"""asyncio's own transport kills and reaps a stage whose spawn is cancelled, so the pipe
	needs no shield or detached reaper — pinned against the real spawn, not a fake suspension."""
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
	from camas.core.execution import run_pipe

	_cancelled_unwind(monkeypatch)
	stages = (
		Task(("python", "-c", "import time; time.sleep(60)")),
		Task(("camas-no-such-executable-on-path",)),
	)
	with pytest.raises(asyncio.CancelledError):
		await run_pipe(stages, _pipe_ctx(stages))


async def test_an_absorbed_cancel_on_the_interrupt_path_closes_each_fd_once(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A cancel absorbed by the interrupt path's unwind must not re-enter a second teardown that
	closes the already-closed pipe end again — that number may by then belong to another file."""
	import os
	from types import SimpleNamespace

	from camas.core import execution as execution_module
	from camas.core.execution import run_pipe

	closed: list[int] = []

	def recording_close(fd: int) -> None:
		closed.append(fd)
		os.close(fd)

	monkeypatch.setattr(
		execution_module, "os", SimpleNamespace(**{**vars(os), "close": recording_close})
	)
	_cancelled_unwind(monkeypatch)
	sleeper = ("python", "-c", "import time; time.sleep(60)")
	stages = (Task(sleeper), Task(sleeper))
	with pytest.raises(asyncio.CancelledError):
		await run_pipe(stages, _pipe_ctx(stages, Interrupts(procs={})))
	assert closed
	assert len(closed) == len(set(closed))


async def test_a_timeout_firing_during_the_spawn_failure_unwind_still_times_out(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A caller's ``wait_for`` deadline expiring while the spawn-failure path unwinds must
	surface as ``TimeoutError``, not as the path's normal return — through 3.10's legacy
	``wait_for`` and 3.12+'s ``timeout()``-based one alike."""
	from typing import Any

	from camas.core import execution as execution_module
	from camas.core.execution import run_pipe
	from camas.core.unwind import unwind as real_unwind

	async def slow_unwind(children: Any, tasks: Any, timeout_s: float = 5.0) -> Any:
		lingering = asyncio.ensure_future(asyncio.sleep(0.3))
		return await real_unwind(children, (*tasks, lingering), timeout_s)

	monkeypatch.setattr(execution_module, "unwind", slow_unwind)
	stages = (
		Task(("python", "-c", "import time; time.sleep(60)")),
		Task(("camas-no-such-executable-on-path",)),
	)
	with pytest.raises(asyncio.TimeoutError):
		await asyncio.wait_for(run_pipe(stages, _pipe_ctx(stages)), 0.1)


async def test_the_interrupt_path_rethrows_a_cancel_its_unwind_absorbed(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""The landed-interrupt path returns results normally after its unwind — a cancel absorbed
	there must still surface."""
	from camas.core.execution import run_pipe

	_cancelled_unwind(monkeypatch)
	sleeper = ("python", "-c", "import time; time.sleep(60)")
	stages = (Task(sleeper), Task(sleeper))
	with pytest.raises(asyncio.CancelledError):
		await run_pipe(stages, _pipe_ctx(stages, Interrupts(procs={})))


@pytest.mark.skipif(sys.platform == "win32", reason="the grandchild handoff is POSIX-only")
async def test_a_grandchild_holding_a_stage_pipe_cannot_wedge_the_unwind(
	tmp_path: Path,
	capsys: pytest.CaptureFixture[str],
	forked: list[subprocess.Popen[bytes]],
	wait_until: Callable[[Callable[[], bool], float], Awaitable[None]],
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
		await wait_until(pid_file.exists, 10)
		started = time.monotonic()
		assert running.cancel()
		with pytest.raises(asyncio.CancelledError):
			await running
		assert time.monotonic() - started < UNWIND_TIMEOUT_S + 2
		assert "1 output reader(s) still open" in capsys.readouterr().err
	finally:
		if pid_file.exists():  # pragma: no branch — the pid is announced before the cancel
			with suppress(ProcessLookupError):
				os.kill(int(pid_file.read_text()), signal.SIGTERM)
		if forked:  # pragma: no branch — empty only when the run never spawned
			# The stage's transport closes only once the dead grandchild's end of the pipe
			# EOFs; let it, or the transport is collected after the loop closes.
			stage_stderr = forked[0].stderr
			assert stage_stderr is not None
			await wait_until(lambda: stage_stderr.closed, 10)
			await asyncio.sleep(0)


async def test_jobs_bounds_pipes_like_leaves(tmp_path: Path) -> None:
	"""Under ``jobs=1`` no two pipes overlap (#336): each head holds an exclusive lock file for
	its lifetime, so an overlapping pipe fails to create it and fails the run."""
	lock = str(tmp_path / "lock")
	head = (
		"python",
		"-c",
		"import os, sys, time; fd = os.open(sys.argv[1], os.O_CREAT | os.O_EXCL); "
		"time.sleep(0.2); os.close(fd); os.remove(sys.argv[1])",
		lock,
	)
	drain = ("python", "-c", "import sys; sys.stdin.read()")
	pipes = Parallel(
		*(Pipe(Task(head, name=f"head{i}"), Task(drain, name=f"drain{i}")) for i in range(3))
	)
	assert (await run(pipes, jobs=1)).returncode == 0


def test_render_shows_a_pipe_with_the_pipe_separator() -> None:
	from camas.core.render import GroupHeader, flatten_rows, render_tree_lines

	assert render_tree_lines(Pipe("a", "b")) == ["a | b", "├─ a", "└─ b"]
	assert render_tree_lines(Pipe("a", "b", name="fmt")) == ["fmt |", "├─ a", "└─ b"]
	header = flatten_rows(Pipe("a", "b"))[0]
	assert isinstance(header, GroupHeader)
	assert header.label == "a | b"
