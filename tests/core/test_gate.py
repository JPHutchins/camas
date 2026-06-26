# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING

from camas import AgentFormat, Parallel, Sequential, Task
from camas.core.gate import (
	GateOutcome,
	decision_of,
	filter_by_mutates,
	run_gate,
	with_agent_format,
)
from camas.core.task import task_label
from camas.core.timings import TaskTiming

if TYPE_CHECKING:
	from camas.v0.task import TaskNode

CHECK_PASS = Task(("python", "-c", "pass"), name="check")
CHECK_FAIL = Task(("python", "-c", "raise SystemExit(1)"), name="check")
FIX = Task(("python", "-c", "pass"), name="fmt", mutates=True)
FIX_FAIL = Task(("python", "-c", "raise SystemExit(1)"), name="fmt", mutates=True)


def test_filter_by_mutates_keeps_matching_group() -> None:
	assert filter_by_mutates(Sequential(FIX, CHECK_PASS), mutates=True) == Sequential(FIX)


def test_filter_by_mutates_prunes_emptied_group() -> None:
	assert filter_by_mutates(Parallel(CHECK_PASS), mutates=True) is None


async def test_gate_all_pass_is_autofixed() -> None:
	out = await run_gate(Parallel(FIX, CHECK_PASS), ())
	assert out.residual_class == "autofixed"
	assert out.residual_result is None
	assert decision_of(out.residual_class) == "continue"


async def test_gate_failing_check_needs_reasoning() -> None:
	out = await run_gate(Parallel(FIX, CHECK_FAIL), ())
	assert out.residual_class == "needs_reasoning"
	assert out.residual_result is not None
	assert out.residual_result.returncode != 0
	assert decision_of(out.residual_class) == "block"


async def test_gate_failing_autofix_needs_reasoning() -> None:
	out = await run_gate(Parallel(FIX_FAIL, CHECK_PASS), ())
	assert out.residual_class == "needs_reasoning"
	assert out.residual_node is not None


async def test_gate_no_mutating_leaves_runs_checks() -> None:
	out = await run_gate(Parallel(CHECK_PASS), ())
	assert out.residual_class == "autofixed"


async def test_gate_only_mutating_leaves_autofixed() -> None:
	out = await run_gate(Parallel(FIX), ())
	assert out.residual_class == "autofixed"
	assert out.residual_result is None


async def test_gate_scoped_to_nothing_is_noop() -> None:
	node: TaskNode = Task(("cargo", "check", "{paths}"), name="rust", paths="rust")
	assert await run_gate(node, ("src/app.py",)) == GateOutcome("autofixed", None, None, None)


async def test_gate_budget_excludes_untimed_checks() -> None:
	out = await run_gate(Parallel(CHECK_PASS), (), under=1.0, timings={})
	assert out.residual_class == "autofixed"
	assert out.budget is not None


async def test_gate_budget_runs_fitting_check() -> None:
	out = await run_gate(
		Parallel(CHECK_PASS), (), under=1.0, timings={task_label(CHECK_PASS): TaskTiming(0.01, 1)}
	)
	assert out.residual_class == "autofixed"
	assert out.budget is not None


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
