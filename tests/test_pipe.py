# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Pipe (#276): fd-wired stages, pipefail, and the agent/human runner split."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from camas import Parallel, Pipe, Sequential, Task
from camas.core.budget import Fits, plan_under
from camas.core.execution import run
from camas.core.gate import strip_agent_only_pipes, with_agent_format
from camas.core.matrix import expand_matrix
from camas.core.timings import CacheKey, TaskTiming
from camas.v0.completion import Finished
from camas.v0.task import AgentFormat

if TYPE_CHECKING:
	from collections.abc import Sequence
	from pathlib import Path

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


def test_pipe_equality_includes_agent_only() -> None:
	assert Pipe("a", "b") != Pipe("a", "b", agent_only=True)
	assert Pipe("a", "b", agent_only=True) == Pipe("a", "b", agent_only=True)


def test_pipe_operators_compose_as_an_opaque_group() -> None:
	pipe = Pipe("a", "b")
	assert (pipe | "c").tasks == (pipe, Task("c"))
	assert (pipe + "c").tasks == (pipe, Task("c"))


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
		async def setup(self, task: TaskNode) -> None:
			return None

		async def on_event(self, event: TaskEvent, states: Sequence[LeafState], ctx: None) -> None:
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


def test_render_shows_a_pipe_with_the_pipe_separator() -> None:
	from camas.core.render import GroupHeader, flatten_rows, render_tree_lines

	assert render_tree_lines(Pipe("a", "b")) == ["a | b", "├─ a", "└─ b"]
	assert render_tree_lines(Pipe("a", "b", name="fmt")) == ["fmt |", "├─ a", "└─ b"]
	header = flatten_rows(Pipe("a", "b"))[0]
	assert isinstance(header, GroupHeader)
	assert header.label == "a | b"
