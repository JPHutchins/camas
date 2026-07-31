# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Time-budgeted scheduling: select the leaves of a task that fit a wall-clock budget."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, TypeAlias

from ..v0.task import Parallel, Sequential, Task
from .matrix import expand_matrix
from .timings import estimate
from .traversal import flatten_leaves

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..v0.task import TaskNode
	from .timings import CacheKey, TaskTiming


class Fits(NamedTuple):
	"""A leaf whose estimate is within budget — selected to run."""

	task: Task
	estimated_s: float


class OverBudget(NamedTuple):
	"""A leaf whose estimate exceeds the budget — excluded."""

	task: Task
	estimated_s: float


class Untimed(NamedTuple):
	"""A leaf with no recorded estimate — run anyway (and thereby measured), since skipping it
	would keep it forever unmeasured.
	"""

	task: Task


Disposition: TypeAlias = Fits | OverBudget | Untimed


class BudgetPlan(NamedTuple):
	"""A budget's partition of a task's leaves, with the runnable schedule of those that fit."""

	budget_s: float
	node: TaskNode | None
	fits: tuple[Fits, ...]
	over_budget: tuple[OverBudget, ...]
	untimed: tuple[Untimed, ...]


def classify(
	task: Task, budget_s: float, timings: Mapping[CacheKey, TaskTiming], scope: int = 0
) -> Disposition:
	"""A leaf's disposition under ``budget_s``, read from its observed estimate at ``scope``.

	>>> from camas.core.timings import CacheKey, TaskTiming
	>>> classify(Task("a"), 1.0, {CacheKey("a", 0): TaskTiming(0.5, 1)})
	Fits(task=Task(cmd='a', name=None, env={}, cwd=None), estimated_s=0.5)
	>>> classify(Task("a"), 1.0, {CacheKey("a", 0): TaskTiming(2.0, 1)})
	OverBudget(task=Task(cmd='a', name=None, env={}, cwd=None), estimated_s=2.0)
	>>> classify(Task("a"), 1.0, {})
	Untimed(task=Task(cmd='a', name=None, env={}, cwd=None))
	>>> scoped = Task("a {paths}", paths=".")
	>>> classify(scoped, 1.0, {CacheKey("a .", 0): TaskTiming(0.5, 1)}, scope=1).task.name is None
	True
	>>> isinstance(classify(scoped, 1.0, {CacheKey("a .", 0): TaskTiming(0.5, 1)}, scope=1), Untimed)
	True
	"""
	est = estimate(task, timings, scope)
	if est is None:
		return Untimed(task)
	if est.elapsed_s <= budget_s:
		return Fits(task, est.elapsed_s)
	return OverBudget(task, est.elapsed_s)


def plan_under(
	node: TaskNode, budget_s: float, timings: Mapping[CacheKey, TaskTiming], scope: int = 0
) -> BudgetPlan:
	"""Partition ``node``'s expanded, de-duplicated leaves by ``budget_s``. Only leaves measured
	to exceed the budget are excluded; untimed leaves are run (and thereby measured), since a
	budget that skipped them would keep them forever unmeasured. ``scope`` selects which
	observations count as measurements of this run — see :func:`camas.core.timings.estimate`.
	"""
	leaves = tuple(dict.fromkeys(info.task for info in flatten_leaves(expand_matrix(node))))
	dispositions = tuple(classify(leaf, budget_s, timings, scope) for leaf in leaves)
	fits = tuple(d for d in dispositions if isinstance(d, Fits))
	over_budget = tuple(d for d in dispositions if isinstance(d, OverBudget))
	untimed = tuple(d for d in dispositions if isinstance(d, Untimed))
	runnable = tuple(d.task for d in dispositions if not isinstance(d, OverBudget))
	return BudgetPlan(budget_s, schedule(runnable), fits, over_budget, untimed)


def schedule(fitting: tuple[Task, ...]) -> TaskNode | None:
	"""Mutating leaves run before the read-only group, so a formatter never races a
	checker over the same files.

	>>> schedule(()) is None
	True
	>>> schedule((Task("ruff check ."),))
	Parallel(tasks=(Task(cmd='ruff check .', name=None, env={}, cwd=None),), name=None, matrix=None, env={}, cwd=None)
	>>> schedule((Task("ruff format .", mutates=True),))
	Sequential(tasks=(Task(cmd='ruff format .', name=None, env={}, cwd=None, mutates=True),), name=None, matrix=None, env={}, cwd=None)
	"""
	mutating = tuple(t for t in fitting if t.mutates)
	readonly = tuple(t for t in fitting if not t.mutates)
	if not mutating and not readonly:
		return None
	if not mutating:
		return Parallel(*readonly)
	if not readonly:
		return Sequential(*mutating)
	return Sequential(*mutating, Parallel(*readonly))
