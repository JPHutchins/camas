# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from camas import AgentFormat, Parallel, Sequential, Task
from camas.core.completion import RunResult, TaskResult
from camas.core.execution import run
from camas.main.state import LoadOk
from camas.mcp import wire
from camas.mcp.result import (
	agent_envelopes,
	to_agent_envelope,
	to_check_response,
	to_run_response,
)
from camas.v0.completion import Errored, Finished, Skipped, Stopped

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


def test_errored_leaf_maps_to_wire_errored() -> None:
	node = Task(("does-not-exist-xyz",), name="ghost")
	result = RunResult(
		returncode=1,
		results=(
			TaskResult("ghost", Errored(127, "no such file or directory: does-not-exist-xyz")),
		),
		elapsed=0.01,
	)
	resp = to_run_response(node, result, verbosity="full")
	comp = resp.leaves[0].completion
	assert isinstance(comp, wire.Errored)
	assert comp.returncode == 127
	assert comp.message == "no such file or directory: does-not-exist-xyz"
	assert (resp.passed, resp.failed, resp.skipped) == (0, 1, 0)


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


def test_agent_envelope_errored_carries_message_as_payload() -> None:
	env = to_agent_envelope(
		Task("x", name="x"), TaskResult("x", Errored(127, "no such file or directory: x"))
	)
	assert env.exit_code == 127
	assert env.payload == "no such file or directory: x"


async def test_agent_envelopes_are_failures_only() -> None:
	node = Parallel(
		Task(("python", "-c", "pass"), name="ok"),
		Task(("python", "-c", "raise SystemExit(1)"), name="bad"),
	)
	envelopes = agent_envelopes(node, await run(node, interactive=False))
	assert tuple(e.name for e in envelopes) == ("bad",)


def test_agent_envelope_under_limit_passes_through() -> None:
	task = Task("lint", name="lint", agent_format=AgentFormat("--out sarif", "sarif", limit=100))
	env = to_agent_envelope(task, TaskResult("lint", Finished(1, 0.1, (b"short\n",))))
	assert env.payload == "short"
	assert env.truncated is False
	assert env.log is None


def test_agent_envelope_over_limit_is_replaced_with_pointer_not_tailed() -> None:
	task = Task("lint", name="lint", agent_format=AgentFormat("--out sarif", "sarif", limit=10))
	env = to_agent_envelope(task, TaskResult("lint", Finished(1, 0.1, (b"way too long\n",))))
	assert "way too long" not in env.payload
	assert "limit" in env.payload
	assert env.truncated is True


def test_agent_envelope_raw_kind_ignores_limit_and_tails_instead() -> None:
	task = Task("lint", name="lint")
	lines = tuple(f"{i}\n".encode() for i in range(60))
	env = to_agent_envelope(task, TaskResult("lint", Finished(1, 0.1, lines)), tail=5)
	assert env.output_kind == "raw"
	assert env.payload == "\n".join(str(i) for i in range(55, 60))
	assert env.truncated is True


def test_agent_envelope_path_mode_reads_report_file_not_stdout(tmp_path: Path) -> None:
	report_path = tmp_path / "report.xml"
	report_path.write_text("<xml/>")
	task = Task("pytest", name="t", agent_format=AgentFormat("--junitxml {report}", "junit"))
	env = to_agent_envelope(
		task,
		TaskResult("t", Finished(1, 0.1, (b"ignored stdout\n",))),
		report_path=report_path,
	)
	assert env.payload == "<xml/>"
	assert env.log == str(report_path)
	assert env.truncated is False


def test_agent_envelope_path_mode_over_limit_points_at_report_file(tmp_path: Path) -> None:
	report_path = tmp_path / "report.xml"
	report_path.write_text("way too long for the limit")
	task = Task(
		"pytest",
		name="t",
		agent_format=AgentFormat("--junitxml {report}", "junit", limit=10),
	)
	env = to_agent_envelope(task, TaskResult("t", Finished(1, 0.1, ())), report_path=report_path)
	assert str(report_path) in env.payload
	assert env.log == str(report_path)
	assert env.truncated is True


_PYPROJECT = Path("/proj/pyproject.toml")


def test_to_check_response_inert_paths_task_carries_warning() -> None:
	state = LoadOk(
		tasks={"cargo": Task("cargo build", name="cargo", paths=".")},
		source=_PYPROJECT,
		scope_effects={},
	)
	resp = to_check_response(state)
	assert resp.status == "ok"
	assert len(resp.warnings) == 1
	assert "cargo" in resp.warnings[0]
	assert "paths=" in resp.warnings[0]


def test_to_check_response_clean_tree_has_no_warnings() -> None:
	state = LoadOk(
		tasks={"cargo": Task("cargo build", name="cargo")}, source=_PYPROJECT, scope_effects={}
	)
	assert to_check_response(state).warnings == ()


def test_to_check_response_dedupes_shared_node() -> None:
	shared = Task("cargo build", name="cargo", paths=".")
	state = LoadOk(tasks={"a": shared, "b": shared}, source=_PYPROJECT, scope_effects={})
	assert len(to_check_response(state).warnings) == 1
