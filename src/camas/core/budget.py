# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Time-budgeted scheduling: select the leaves of a task that fit a wall-clock budget."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from typing import TYPE_CHECKING, Final, NamedTuple, TypeAlias

from ..v0.task import Parallel, Sequential, Task, rebuilt
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
	"""Partition ``node``'s expanded leaves by ``budget_s``, preserving the tree's structure:
	a ``Sequential``'s ordering survives, and the mutating-first reordering applies only
	across a ``Parallel``'s siblings (#306). Only leaves measured to exceed the budget are
	excluded; untimed leaves are run (and thereby measured), since a budget that skipped
	them would keep them forever unmeasured. ``scope`` selects which observations count as
	measurements of this run — see :func:`camas.core.timings.estimate`.
	"""
	runnable, dispositions = _plan_under(expand_matrix(node), budget_s, timings, scope)
	fits = tuple(d for d in dispositions if isinstance(d, Fits))
	over_budget = tuple(d for d in dispositions if isinstance(d, OverBudget))
	untimed = tuple(d for d in dispositions if isinstance(d, Untimed))
	return BudgetPlan(budget_s, runnable, fits, over_budget, untimed)


def _plan_under(
	node: TaskNode, budget_s: float, timings: Mapping[CacheKey, TaskTiming], scope: int
) -> tuple[TaskNode | None, tuple[Disposition, ...]]:
	"""Recurse the tree: classify leaves, keep ``Sequential`` order, split a ``Parallel``'s
	children mutating-first — the shape :func:`schedule` flattens to, but without discarding
	the structure the ordering lives in.
	"""
	match node:
		case Task():
			disposition: Final = classify(node, budget_s, timings, scope)
			return (None if isinstance(disposition, OverBudget) else node), (disposition,)
		case Sequential(tasks=children):
			planned = tuple(_plan_under(child, budget_s, timings, scope) for child in children)
			kept = tuple(child_node for child_node, _ in planned if child_node is not None)
			return (None if not kept else rebuilt(node, *kept)), _collect(planned)
		case Parallel(tasks=children):
			planned = tuple(_plan_under(child, budget_s, timings, scope) for child in children)
			kept = tuple(child_node for child_node, _ in planned if child_node is not None)
			mutating = tuple(child for child in kept if _has_mutating(child))
			readonly = tuple(child for child in kept if not _has_mutating(child))
			return (
				None
				if not kept
				else (
					rebuilt(node, *(mutating + readonly))
					if not mutating or not readonly
					else Sequential(
						*mutating, Parallel(*readonly), name=node.name, env=node.env, cwd=node.cwd
					)
				),
				_collect(planned),
			)
		case _:
			assert_never(node)


def _collect(
	planned: tuple[tuple[TaskNode | None, tuple[Disposition, ...]], ...],
) -> tuple[Disposition, ...]:
	"""The planned children's dispositions, in DFS order."""
	return tuple(disposition for _, dispositions in planned for disposition in dispositions)


def _has_mutating(node: TaskNode) -> bool:
	"""Whether any leaf in the subtree writes the workspace."""
	return any(info.task.mutates for info in flatten_leaves(node))


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
