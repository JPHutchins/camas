# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING

from camas import AgentFormat, Parallel, Sequential, Task
from camas.core.completion import RunResult
from camas.core.gate import GateOutcome, decision_of, run_gate, with_agent_format
from camas.core.task import task_label
from camas.core.timings import TaskTiming

if TYPE_CHECKING:
	from pathlib import Path

	import pytest

	from camas.v0.task import TaskNode

CHECK_PASS = Task(("python", "-c", "pass"), name="check")
CHECK_FAIL = Task(("python", "-c", "raise SystemExit(1)"), name="check")


def test_decision_of() -> None:
	assert decision_of("green") == "continue"
	assert decision_of("needs_reasoning") == "block"


async def test_gate_pass_is_green() -> None:
	out = await run_gate(Parallel(CHECK_PASS), ())
	assert out.residual_class == "green"
	assert out.node is not None
	assert out.result is not None
	assert out.result.returncode == 0
	assert decision_of(out.residual_class) == "continue"


async def test_gate_failing_check_needs_reasoning() -> None:
	out = await run_gate(Parallel(CHECK_FAIL), ())
	assert out.residual_class == "needs_reasoning"
	assert out.result is not None
	assert out.result.returncode != 0
	assert decision_of(out.residual_class) == "block"


async def test_gate_scoped_to_nothing_is_green_noop() -> None:
	node = Task(("cargo", "check", "{paths}"), name="rust", paths="rust")
	assert await run_gate(node, ("src/app.py",)) == GateOutcome("green", None, None, None)


async def test_gate_threads_base_into_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
	captured: dict[str, Path | None] = {}

	async def fake_run(
		node: TaskNode,
		*,
		jobs: int | None = None,
		base: Path | None = None,
		interactive: bool = True,
	) -> RunResult:
		captured["base"] = base
		return RunResult(0, (), 0.0)

	monkeypatch.setattr("camas.core.gate.run", fake_run)
	await run_gate(Parallel(CHECK_PASS), (), base=tmp_path)
	assert captured["base"] == tmp_path


async def test_gate_runs_untimed_check_under_budget() -> None:
	out = await run_gate(Parallel(CHECK_PASS), (), under=1.0, timings={})
	assert out.residual_class == "green"
	assert out.result is not None
	assert out.budget is not None


async def test_gate_budget_runs_fitting_check() -> None:
	out = await run_gate(
		Parallel(CHECK_PASS), (), under=1.0, timings={task_label(CHECK_PASS): TaskTiming(0.01, 1)}
	)
	assert out.residual_class == "green"
	assert out.result is not None


async def test_gate_budget_skips_over_budget_check() -> None:
	out = await run_gate(
		Parallel(CHECK_PASS), (), under=0.001, timings={task_label(CHECK_PASS): TaskTiming(99.0, 1)}
	)
	assert out.residual_class == "green"
	assert out.result is None
	assert out.budget is not None
	assert out.budget.node is None


def test_with_agent_format_appends_to_tuple_command() -> None:
	t = Task(("ruff", "check", "."), agent_format=AgentFormat("--output-format sarif", "sarif"))
	result = with_agent_format(t)
	assert isinstance(result, Task)
	assert result.cmd == ("ruff", "check", ".", "--output-format", "sarif")


def test_with_agent_format_recurses_groups_and_leaves_plain_untouched() -> None:
	fmt = Task("ruff check .", name="lint", agent_format=AgentFormat("--out sarif", "sarif"))
	plain = Task("mypy .", name="types")
	out = with_agent_format(Sequential(fmt, plain))
	assert isinstance(out, Sequential)
	first, second = out.tasks
	assert isinstance(first, Task)
	assert isinstance(second, Task)
	assert first.cmd == "ruff check . --out sarif"
	assert second.cmd == "mypy ."


async def test_gate_tags_residual_with_agent_format_kind() -> None:
	chk = Task(
		("python", "-c", "import sys; sys.exit(1)"),
		name="lint",
		agent_format=AgentFormat("--quiet", "sarif"),
	)
	out = await run_gate(Parallel(chk), ())
	assert out.residual_class == "needs_reasoning"
