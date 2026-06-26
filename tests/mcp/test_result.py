# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING

from camas import AgentFormat, Parallel, Sequential, Task
from camas.core.completion import RunResult, TaskResult
from camas.core.execution import run
from camas.mcp import wire
from camas.mcp.result import agent_envelopes, to_agent_envelope, to_run_response
from camas.v0.completion import Finished, Skipped, Stopped

if TYPE_CHECKING:
	from camas.mcp.result import Verbosity
	from camas.v0.task import TaskNode


async def _resp(
	node: TaskNode, *, verbosity: Verbosity = "failures", tail: int = 50
) -> wire.RunResponse:
	return to_run_response(node, await run(node, interactive=False), verbosity=verbosity, tail=tail)


async def test_counts_and_blocked_by() -> None:
	node = Sequential(
		Task(("python", "-c", "raise SystemExit(1)"), name="fail"),
		Task(("python", "-c", "pass"), name="after"),
	)
	resp = await _resp(node)
	assert (resp.passed, resp.failed, resp.skipped) == (0, 1, 1)
	after = next(leaf for leaf in resp.leaves if leaf.name == "after").completion
	assert isinstance(after, wire.Skipped)
	assert after.blocked_by == "fail"


async def test_failures_verbosity_shows_only_failed_output() -> None:
	node = Parallel(
		Task(("python", "-c", "print('ok')"), name="good"),
		Task(("python", "-c", "import sys; print('boom'); sys.exit(1)"), name="bad"),
	)
	resp = await _resp(node, verbosity="failures")
	good = next(leaf for leaf in resp.leaves if leaf.name == "good").completion
	bad = next(leaf for leaf in resp.leaves if leaf.name == "bad").completion
	assert isinstance(good, wire.Finished)
	assert good.output == []
	assert isinstance(bad, wire.Finished)
	assert any("boom" in line for line in bad.output)


async def test_summary_verbosity_omits_all_output() -> None:
	node = Task(("python", "-c", "import sys; print('x'); sys.exit(1)"), name="t")
	resp = await _resp(node, verbosity="summary")
	comp = resp.leaves[0].completion
	assert isinstance(comp, wire.Finished)
	assert comp.output == []


async def test_full_verbosity_includes_passing_output_with_truncation() -> None:
	node = Task(("python", "-c", "for i in range(10): print(i)"), name="many")
	resp = await _resp(node, verbosity="full", tail=3)
	comp = resp.leaves[0].completion
	assert isinstance(comp, wire.Finished)
	assert comp.output == ["7", "8", "9"]
	assert resp.truncated is True
	assert resp.leaves[0].command.startswith("python -c")


async def test_str_command_preserved() -> None:
	resp = await _resp(Task("python --version", name="ver"), verbosity="summary")
	assert resp.leaves[0].command == "python --version"


def test_stopped_leaf_maps_to_wire_stopped() -> None:
	node = Task(("python", "-c", "pass"), name="t")
	result = RunResult(
		returncode=130,
		results=(TaskResult("t", Stopped(130, 0.5, (b"bye\n",))),),
		elapsed=0.5,
		interrupt_count=1,
	)
	resp = to_run_response(node, result, verbosity="full")
	comp = resp.leaves[0].completion
	assert isinstance(comp, wire.Stopped)
	assert comp.output == ["bye"]
	assert (resp.passed, resp.failed, resp.skipped, resp.interrupt_count) == (0, 1, 0, 1)


def test_agent_envelope_finished_carries_verbatim_payload() -> None:
	task = Task(
		"ruff check .", name="lint", agent_format=AgentFormat("--output-format sarif", "sarif")
	)
	env = to_agent_envelope(task, TaskResult("lint", Finished(1, 0.1, (b"E1\n", b"E2\n"))))
	assert (env.name, env.exit_code, env.output_kind) == ("lint", 1, "sarif")
	assert env.payload == "E1\nE2"
	assert env.truncated is False


def test_agent_envelope_skipped_has_empty_payload() -> None:
	env = to_agent_envelope(Task("x", name="x"), TaskResult("x", Skipped(1, "dep")))
	assert env.exit_code == 1
	assert env.payload == ""


def test_agent_envelope_stopped_carries_partial_output() -> None:
	env = to_agent_envelope(
		Task("x", name="x"), TaskResult("x", Stopped(130, 0.1, (b"partial\n",)))
	)
	assert env.exit_code == 130
	assert env.payload == "partial"


async def test_agent_envelopes_are_failures_only() -> None:
	node = Parallel(
		Task(("python", "-c", "pass"), name="ok"),
		Task(("python", "-c", "raise SystemExit(1)"), name="bad"),
	)
	envelopes = agent_envelopes(node, await run(node, interactive=False))
	assert tuple(e.name for e in envelopes) == ("bad",)
