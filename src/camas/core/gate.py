# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The SA-delegation gate: scope the check node to the changed paths, run the checks, and
classify the residual ``green`` vs ``needs_reasoning``. The gate
never mutates — the deterministic fixers run separately on ``PostToolBatch`` (``camas mcp fix``).
"""

from __future__ import annotations

import dataclasses
import shlex
import sys
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeAlias

from ..v0.task import Group, Task
from .budget import plan_under
from .execution import run
from .matrix import expand_matrix
from .scope import scope_to_changed

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..v0.task import TaskNode
	from .budget import BudgetPlan
	from .completion import RunResult
	from .timings import TaskLabel, TaskTiming


ResidualClass: TypeAlias = Literal["green", "needs_reasoning"]
Decision: TypeAlias = Literal["continue", "block"]


class GateOutcome(NamedTuple):
	"""A gate run's verdict and the check run that produced it."""

	residual_class: ResidualClass
	node: TaskNode | None
	"""The check node that ran and its result, paired — both set when a run happened (whether
	``green`` or ``needs_reasoning``), both ``None`` when nothing ran. Diagnostics derive from
	them only on ``needs_reasoning``.
	"""
	result: RunResult | None
	budget: BudgetPlan | None


def decision_of(residual_class: ResidualClass) -> Decision:
	"""The routing for a residual class: a surviving residual blocks, else continue.

	>>> decision_of("green"), decision_of("needs_reasoning")
	('continue', 'block')
	"""
	match residual_class:
		case "green":
			return "continue"
		case "needs_reasoning":
			return "block"
		case _:
			assert_never(residual_class)


def with_agent_format(node: TaskNode) -> TaskNode:
	"""Append each leaf's ``agent_format.args`` to its command — the agent-only structured-output
	variant the gate runs; a human run never applies this, so ``cmd`` is otherwise untouched.

	>>> from camas.v0.task import AgentFormat
	>>> with_agent_format(Task("ruff check .", agent_format=AgentFormat("--output-format sarif", "sarif"))).cmd
	'ruff check . --output-format sarif'
	>>> with_agent_format(Task("ruff check .")).cmd
	'ruff check .'
	"""
	match node:
		case Task():
			if node.agent_format is None:
				return node
			args = node.agent_format.args
			cmd = (
				f"{node.cmd} {args}"
				if isinstance(node.cmd, str)
				else (*node.cmd, *shlex.split(args))
			)
			return dataclasses.replace(node, cmd=cmd)
		case Group() as group:
			return type(group)(
				*(with_agent_format(c) for c in group.tasks),
				name=group.name,
				matrix=group.matrix,
				env=group.env,
				cwd=group.cwd,
				help=group.help,
				paths=group.paths,
			)
		case _:
			assert_never(node)


async def run_gate(
	node: TaskNode,
	changed: tuple[str, ...],
	*,
	under: float | None = None,
	jobs: int | None = None,
	timings: Mapping[TaskLabel, TaskTiming] | None = None,
) -> GateOutcome:
	"""Run the check ``node`` over the ``changed`` paths and classify the residual.

	The check node is expanded, time-boxed (``under``), scoped to ``changed``, and run; the gate
	never mutates. Untimed leaves are run (and thereby measured); only leaves measured to exceed
	``under`` are skipped. ``green`` means the checks passed — or the change touched nothing the
	checks cover, or every leaf was measured too slow for ``under``; ``needs_reasoning`` means a
	check still fails. Budgeting precedes scoping so each leaf's estimate reuses its unscoped
	record (a scoped run is no slower than the whole).
	"""
	expanded = expand_matrix(node)
	plan = plan_under(expanded, under, timings or {}) if under is not None else None
	budgeted = plan.node if plan is not None else expanded
	if budgeted is None:
		return GateOutcome("green", None, None, plan)
	scoped = scope_to_changed(budgeted, changed)
	if scoped is None:
		return GateOutcome("green", None, None, plan)
	checks_node = with_agent_format(scoped)
	checks = await run(checks_node, jobs=jobs, interactive=False)
	residual: ResidualClass = "needs_reasoning" if checks.returncode != 0 else "green"
	return GateOutcome(residual, checks_node, checks, plan)
