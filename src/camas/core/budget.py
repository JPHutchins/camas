# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Time-budgeted scheduling: select the leaves of a task that fit a wall-clock budget."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from typing import TYPE_CHECKING, Any, Final, NamedTuple, TypeAlias, overload

from ..v0.task import GROUP_FIELDS, Parallel, Sequential, Task, rebuilt
from .matrix import expand_matrix
from .timings import estimate

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..v0.task import Group, TaskNode
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


class _Planned(NamedTuple):
	"""A subtree's planning result: the kept node, its dispositions in DFS order, and whether
	any kept leaf mutates.
	"""

	node: TaskNode | None
	dispositions: tuple[Disposition, ...]
	has_mutating: bool


def plan_under(
	node: TaskNode, budget_s: float, timings: Mapping[CacheKey, TaskTiming], scope: int = 0
) -> BudgetPlan:
	"""Partition ``node``'s expanded leaves by ``budget_s``, preserving the tree's structure:
	a ``Sequential``'s ordering survives, and across a ``Parallel``'s siblings the mutating
	subtrees are serialized ahead of the pure read-only ones, which keep their concurrency
	(#306). A mutating subtree is kept whole, so a nested mixed shape loses cross-subtree
	parallelism rather than ordering. A repeated leaf keeps every occurrence, and each
	counts against the budget. Only leaves measured to exceed the budget are excluded;
	untimed leaves are run (and thereby measured), since a budget that skipped them would
	keep them forever unmeasured. ``scope`` selects which observations count as
	measurements of this run — see :func:`camas.core.timings.estimate`.
	"""
	planned = _plan_under(expand_matrix(node), budget_s, timings, scope)
	fits = tuple(d for d in planned.dispositions if isinstance(d, Fits))
	over_budget = tuple(d for d in planned.dispositions if isinstance(d, OverBudget))
	untimed = tuple(d for d in planned.dispositions if isinstance(d, Untimed))
	return BudgetPlan(budget_s, planned.node, fits, over_budget, untimed)


def _plan_under(
	node: TaskNode, budget_s: float, timings: Mapping[CacheKey, TaskTiming], scope: int
) -> _Planned:
	"""The recursive pass: classify leaves, keep ``Sequential`` order, and reorder ``Parallel``
	siblings mutating-first.
	"""
	match node:
		case Task():
			disposition: Final = classify(node, budget_s, timings, scope)
			kept: Final = not isinstance(disposition, OverBudget)
			return _Planned(None if not kept else node, (disposition,), kept and node.mutates)
		case Sequential(tasks=children):
			planned = tuple(_plan_under(child, budget_s, timings, scope) for child in children)
			kept_children = tuple(
				child_node for child_node, _, _ in planned if child_node is not None
			)
			return _Planned(
				None if not kept_children else _collapse(rebuilt(node, *kept_children)),
				_collect(planned),
				any(child.has_mutating for child in planned),
			)
		case Parallel(tasks=children):
			planned = tuple(_plan_under(child, budget_s, timings, scope) for child in children)
			mutating = tuple(
				child_node
				for child_node, _, has_mutating in planned
				if child_node is not None and has_mutating
			)
			readonly = tuple(
				child_node
				for child_node, _, has_mutating in planned
				if child_node is not None and not has_mutating
			)
			runnable: TaskNode | None
			if not mutating and not readonly:
				runnable = None
			elif not readonly:
				runnable = _collapse(Sequential(*mutating, **_fields_of(node)))
			elif not mutating:
				runnable = _collapse(rebuilt(node, *readonly))
			else:
				runnable = Sequential(
					*mutating,
					Parallel(*readonly, name=node.name, env=node.env, cwd=node.cwd),
					**_fields_of(node),
				)
			return _Planned(
				runnable, _collect(planned), any(child.has_mutating for child in planned)
			)
		case _:
			assert_never(node)


def _collect(planned: tuple[_Planned, ...]) -> tuple[Disposition, ...]:
	"""The planned children's dispositions, in DFS order."""
	return tuple(disposition for child in planned for disposition in child.dispositions)


def _fields_of(node: Group) -> dict[str, Any]:
	""":func:`rebuilt`'s GROUP_FIELDS carry, for a manufactured wrapper of a different kind
	than ``node``.
	"""
	return {field: getattr(node, field) for field in GROUP_FIELDS}


@overload
def _collapse(node: Sequential) -> Sequential: ...
@overload
def _collapse(node: Parallel) -> Parallel: ...
def _collapse(node: Group) -> Group:
	"""A single-child wrapper whose child is the same kind is dropped, matching the ``|``/``+``
	flattening.
	"""
	children: Final = node.tasks
	if len(children) != 1:
		return node
	match node:
		case Sequential() if isinstance(children[0], Sequential):
			return children[0]
		case Parallel() if isinstance(children[0], Parallel):
			return children[0]
		case _:
			return node
