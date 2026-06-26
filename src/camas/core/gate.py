# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The SA-delegation gate: scope a task to the changed paths, autofix, run the remaining checks,
and classify the residual ``autofixed`` vs ``needs_reasoning`` for a ``PostToolBatch`` hook.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeAlias

from ..v0.task import Parallel, Sequential, Task
from .budget import plan_under
from .execution import run
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


ResidualClass: TypeAlias = Literal["autofixed", "needs_reasoning"]
Decision: TypeAlias = Literal["continue", "block"]


class GateOutcome(NamedTuple):
	"""A gate run's verdict and the residual that produced it."""

	residual_class: ResidualClass
	residual_node: TaskNode | None
	"""The failing run's node and result, paired — both set on ``needs_reasoning``, both ``None`` otherwise."""
	residual_result: RunResult | None
	budget: BudgetPlan | None


def decision_of(residual_class: ResidualClass) -> Decision:
	"""The ``PostToolBatch`` routing for a class: a surviving residual blocks, else continue.

	>>> decision_of("autofixed"), decision_of("needs_reasoning")
	('continue', 'block')
	"""
	match residual_class:
		case "autofixed":
			return "continue"
		case "needs_reasoning":
			return "block"
		case _:
			assert_never(residual_class)


def filter_by_mutates(node: TaskNode, *, mutates: bool) -> TaskNode | None:
	"""``node`` with only the leaves whose ``mutates`` equals ``mutates``, emptied groups pruned.

	>>> chk = Task("ruff check .", name="lint")
	>>> filter_by_mutates(chk, mutates=True) is None
	True
	>>> filter_by_mutates(chk, mutates=False) is chk
	True
	"""
	match node:
		case Task():
			return node if node.mutates == mutates else None
		case Sequential(tasks=children, name=name, matrix=matrix, env=env, cwd=cwd, help=help):
			kept = tuple(
				k
				for k in (filter_by_mutates(c, mutates=mutates) for c in children)
				if k is not None
			)
			return (
				Sequential(*kept, name=name, matrix=matrix, env=env, cwd=cwd, help=help)
				if kept
				else None
			)
		case Parallel(tasks=children, name=name, matrix=matrix, env=env, cwd=cwd, help=help):
			kept = tuple(
				k
				for k in (filter_by_mutates(c, mutates=mutates) for c in children)
				if k is not None
			)
			return (
				Parallel(*kept, name=name, matrix=matrix, env=env, cwd=cwd, help=help)
				if kept
				else None
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
	"""Scope ``node`` to ``changed``, autofix, run the remaining checks, and classify the residual."""
	scoped = scope_to_changed(node, changed)
	if scoped is None:
		return GateOutcome("autofixed", None, None, None)
	mutating = filter_by_mutates(scoped, mutates=True)
	if (
		mutating is not None
		and (autofix := await run(mutating, jobs=jobs, interactive=False)).returncode != 0
	):
		return GateOutcome("needs_reasoning", mutating, autofix, None)
	readonly = filter_by_mutates(scoped, mutates=False)
	if readonly is None:
		return GateOutcome("autofixed", None, None, None)
	plan = plan_under(readonly, under, timings or {}) if under is not None else None
	checks_node = plan.node if plan is not None else readonly
	if checks_node is None:
		return GateOutcome("autofixed", None, None, plan)
	if (checks := await run(checks_node, jobs=jobs, interactive=False)).returncode != 0:
		return GateOutcome("needs_reasoning", checks_node, checks, plan)
	return GateOutcome("autofixed", None, None, plan)
