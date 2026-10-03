# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Time-budgeted scheduling: select the leaves of a task that fit a wall-clock budget."""

from __future__ import annotations

import sys
from itertools import chain

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from typing import TYPE_CHECKING, Any, Final, NamedTuple, TypeAlias, cast, overload

from ..v0.task import GROUP_FIELDS, Group, Parallel, Pipe, Sequential, Task, fieldless, rebuilt
from .matrix import expand_matrix
from .scope import Pruned, coverage_message, prune_pipes
from .task import task_label
from .timings import Observed, estimate, narrowed, observed, pairs_index
from .traversal import flatten_leaves

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping, Sequence
	from pathlib import Path

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
	running_over_budget: tuple[OverBudget, ...]
	"""Over-budget leaves that run anyway — a pipe kept whole so an untimed sibling gets its
	first measurement. Counted in ``node``, not in ``over_budget``.
	"""
	runnable: tuple[Task, ...]
	"""The schedule's leaves in DFS order — what the plan actually runs; the dispositions are
	a census, not a runnable set.
	"""
	pipe_cut: tuple[Task, ...]
	"""The fitting stages the plan's own mid-pipe cut dropped."""


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
	"""A subtree's planning result: the kept node, its dispositions in DFS order, whether any
	kept leaf mutates, and the stages its pipe cuts dropped.
	"""

	node: TaskNode | None
	dispositions: Iterable[Disposition]
	has_mutating: bool
	pipe_cut: Iterable[Task]


def plan_under(
	node: TaskNode, budget_s: float, timings: Mapping[CacheKey, TaskTiming], scope: int = 0
) -> BudgetPlan:
	"""Partition ``node``'s expanded leaves by ``budget_s``, preserving the tree's structure:
	a ``Sequential``'s ordering survives, and across a ``Parallel``'s siblings the mutating
	subtrees are serialized ahead of the pure read-only ones, which keep their concurrency
	(#306). A mutating subtree is kept whole, so a nested mixed shape loses cross-subtree
	parallelism rather than ordering. A repeated leaf keeps every occurrence: each is
	classified against the budget individually, so a serialized repeat can consume its slot
	twice and record one sample per occurrence. Only leaves measured to exceed the budget
	are excluded — except a pipe kept whole for its untimed siblings, which runs its
	over-budget stages too (``running_over_budget``);
	untimed leaves are run (and thereby measured), since a budget that skipped them would
	keep them forever unmeasured. ``scope`` selects which observations count as
	measurements of this run — see :func:`camas.core.timings.estimate`.
	"""
	planned = _plan_under(expand_matrix(node), budget_s, timings, scope)
	dispositions: Final = tuple(planned.dispositions)
	runnable: Final = (
		tuple(info.task for info in flatten_leaves(planned.node))
		if planned.node is not None
		else ()
	)
	runnable_ids: Final = {id(t) for t in runnable}
	running_over_budget: Final = tuple(
		d for d in dispositions if isinstance(d, OverBudget) and id(d.task) in runnable_ids
	)
	fits = tuple(d for d in dispositions if isinstance(d, Fits))
	over_budget = tuple(
		d for d in dispositions if isinstance(d, OverBudget) and id(d.task) not in runnable_ids
	)
	untimed = tuple(d for d in dispositions if isinstance(d, Untimed))
	return BudgetPlan(
		budget_s,
		planned.node,
		fits,
		over_budget,
		untimed,
		running_over_budget,
		runnable,
		tuple(planned.pipe_cut),
	)


def _empty_plan_message(plan: BudgetPlan) -> str:
	"""Why ``plan.node`` is ``None``."""
	return (
		"All leaves exceed the budget — nothing to run."
		if not plan.fits
		else "A mid-pipe cut would rewire the pipeline — nothing to run."
	)


_BUDGET_DROPPED_MESSAGE: Final = (
	"The budget dropped the last runnable leaf for the changed paths — nothing to run."
)


class CensusLeaf(NamedTuple):
	"""A planned leaf the run will not execute, with the estimate it had when it fits."""

	task: Task
	estimated_s: float | None


class BudgetSummary(NamedTuple):
	"""A plan's census as a surface reports it: the plan's own schedule, the final post-drop
	tree when the keying is given, or the empty census of a run nothing executes.
	"""

	budget_s: float
	runnable: tuple[Task, ...]
	unmeasured: tuple[Untimed, ...]
	running_anyway: tuple[OverBudget, ...]
	excluded: tuple[OverBudget, ...]
	dropped: tuple[CensusLeaf, ...]
	"""The planned leaves a pipe cut removed — the plan's own cut, the scoping's mid-pipe cut,
	or the post-plan drop's re-cut."""
	not_covered: tuple[CensusLeaf, ...]
	"""The planned leaves the changed paths pruned themselves — scope-pruned fits and untimed
	leaves, which otherwise vanish from the census."""


def summarize(plan: BudgetPlan, keying: Observed | None = None) -> BudgetSummary:
	"""The census a surface reports — from ``plan`` alone, or narrowed to the tree the run
	actually executes when the final (post-drop) keying is given; a keying whose tree is
	``None`` yields the empty census (nothing runs, every excluded disposition carries over).
	A planned leaf that will not run is ``dropped`` when a recorded pipe cut removed it and
	``not_covered`` otherwise — its own paths missed the change set.
	"""
	if keying is None:
		runnable = plan.runnable
		unmeasured = plan.untimed
		running_anyway = plan.running_over_budget
		excluded = plan.over_budget
	elif keying.node is None:
		runnable = ()
		unmeasured = ()
		running_anyway = ()
		excluded = plan.over_budget + plan.running_over_budget
	else:
		original_for: Final = keying.original_for()
		leaves: Final = flatten_leaves(keying.node)
		surviving: Final = {id(original_for.get(id(info.task), info.task)) for info in leaves}
		runnable = tuple(original_for.get(id(info.task), info.task) for info in leaves)
		unmeasured = tuple(u for u in plan.untimed if id(u.task) in surviving)
		running_anyway = tuple(o for o in plan.running_over_budget if id(o.task) in surviving)
		excluded = plan.over_budget + tuple(
			o for o in plan.running_over_budget if id(o.task) not in surviving
		)
	runnable_ids: Final = frozenset(id(t) for t in runnable)
	cut_ids: Final = frozenset(
		id(t) for t in chain(plan.pipe_cut, keying.pipe_cut if keying is not None else ())
	)
	absent: Final = tuple(
		CensusLeaf(f.task, f.estimated_s) for f in plan.fits if id(f.task) not in runnable_ids
	) + tuple(CensusLeaf(u.task, None) for u in plan.untimed if id(u.task) not in runnable_ids)
	return BudgetSummary(
		budget_s=plan.budget_s,
		runnable=runnable,
		unmeasured=unmeasured,
		running_anyway=running_anyway,
		excluded=excluded,
		dropped=tuple(leaf for leaf in absent if id(leaf.task) in cut_ids),
		not_covered=tuple(leaf for leaf in absent if id(leaf.task) not in cut_ids),
	)


class BudgetRun(NamedTuple):
	"""A budget that leaves something to run: the final keying and its census."""

	keying: Observed
	summary: BudgetSummary


class NothingToRun(NamedTuple):
	"""A budget that runs nothing: the census, and why."""

	summary: BudgetSummary
	cause: str


BudgetOutcome: TypeAlias = BudgetRun | NothingToRun


def resolve_budget(
	plan: BudgetPlan, camas_dir: Path | None, changed: Sequence[str]
) -> BudgetOutcome:
	"""``plan`` scoped to ``changed``, its keep-whole pipes re-validated."""
	if plan.node is None:
		return NothingToRun(summarize(plan), _empty_plan_message(plan))
	keying: Final = observed(camas_dir, plan.node, changed)
	if keying.node is None:
		return NothingToRun(summarize(plan, keying), coverage_message(changed))
	final: Final = narrowed(keying, drop_unjustified_running(keying.node, plan, keying.pairs))
	if final.node is None:
		return NothingToRun(summarize(plan, final), _BUDGET_DROPPED_MESSAGE)
	return BudgetRun(final, summarize(plan, final))


class BudgetCensus(NamedTuple):
	"""The display facts of a budget census — the one shape :func:`summary_lines` renders,
	projected from a :class:`BudgetSummary` or from a wire report.
	"""

	budget_s: float
	running: int
	unmeasured: tuple[str, ...]
	running_anyway: tuple[tuple[str, float | None], ...]
	dropped: tuple[tuple[str, float | None], ...]
	not_covered: tuple[tuple[str, float | None], ...]
	excluded: tuple[tuple[str, float | None], ...]


def census_of(summary: BudgetSummary) -> BudgetCensus:
	"""The display projection of a census — labels instead of dispositions."""
	return BudgetCensus(
		budget_s=summary.budget_s,
		running=len(summary.runnable),
		unmeasured=tuple(task_label(u.task) for u in summary.unmeasured),
		running_anyway=tuple((task_label(o.task), o.estimated_s) for o in summary.running_anyway),
		dropped=tuple((task_label(f.task), f.estimated_s) for f in summary.dropped),
		not_covered=tuple(
			(task_label(leaf.task), leaf.estimated_s) for leaf in summary.not_covered
		),
		excluded=tuple((task_label(o.task), o.estimated_s) for o in summary.excluded),
	)


def outcome_lines(outcome: BudgetOutcome) -> tuple[str, ...]:
	"""The census lines of ``outcome``, then why nothing runs when nothing does."""
	match outcome:
		case BudgetRun(summary=summary):
			return summary_lines(census_of(summary))
		case NothingToRun(summary=summary, cause=cause):
			return (*summary_lines(census_of(summary)), cause)
		case _:
			assert_never(outcome)


def summary_lines(census: BudgetCensus) -> tuple[str, ...]:
	"""The one budget formatter: the headline plus a line per non-empty disposition."""
	lines = [
		f"Time budget {census.budget_s:.2f}s — running {census.running} leaf(s) "
		f"({len(census.unmeasured)} unmeasured), excluded {len(census.excluded)} over budget."
	]
	if census.running_anyway:
		lines.append(
			"  running anyway to measure untimed pipe siblings: "
			+ ", ".join(_note(leaf) for leaf in census.running_anyway)
		)
	if census.dropped:
		lines.append("  dropped by pipe cut: " + ", ".join(_note(leaf) for leaf in census.dropped))
	if census.not_covered:
		lines.append(
			"  not covered by the changed paths: "
			+ ", ".join(_note(leaf) for leaf in census.not_covered)
		)
	if census.excluded:
		lines.append("  over budget: " + ", ".join(_note(leaf) for leaf in census.excluded))
	if census.unmeasured:
		lines.append(
			"  unmeasured (running to record an estimate): " + ", ".join(census.unmeasured)
		)
	return tuple(lines)


def _note(leaf: tuple[str, float | None]) -> str:
	"""A display leaf's label with its estimate when it has one."""
	name, estimated_s = leaf
	return f"{name} ~{estimated_s:.2f}s" if estimated_s is not None else name


def drop_unjustified_running(
	node: TaskNode,
	plan: BudgetPlan,
	pairs: Iterable[tuple[Task, Task]] = (),
) -> Pruned:
	"""``node`` after scope pruning: a running-over-budget stage whose untimed sibling did not
	survive scoping has lost its justification — drop it with the cut semantics of
	:func:`camas.core.scope.prune_pipes`. ``pairs`` is the scoping's original-to-scoped
	pairing, so a rebuilt stage still matches the plan disposition it came from, and the
	stages a cut drops come back as their originals; a stage missing from the pairs is
	dropped — the guard fails closed rather than letting an unjustified stage run.
	"""
	if not plan.running_over_budget:
		return Pruned(node, ())
	original_of: Final = pairs_index(pairs)
	running_ids: Final = {id(o.task) for o in plan.running_over_budget}
	untimed_ids: Final = {id(u.task) for u in plan.untimed}

	def keep_stage(stage: Task) -> Task | None:
		original = original_of.get(id(stage))
		return None if original is None or id(original) in running_ids else stage

	def keep_whole_pipe(stages: tuple[Task, ...]) -> bool:
		return any(
			(original := original_of.get(id(t))) is not None and id(original) in untimed_ids
			for t in stages
		)

	pruned: Final = prune_pipes(node, keep_stage, keep_whole_pipe)
	return Pruned(pruned.node, tuple(original_of[id(t)] for t in pruned.pipe_cut))


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
			return _Planned(None if not kept else node, (disposition,), kept and node.mutates, ())
		case Sequential(tasks=children):
			planned = tuple(_plan_under(child, budget_s, timings, scope) for child in children)
			kept_children = tuple(child.node for child in planned if child.node is not None)
			return _Planned(
				None if not kept_children else _collapse(rebuilt(node, *kept_children)),
				_collect(planned),
				any(child.has_mutating for child in planned),
				_collect_cuts(planned),
			)
		case Pipe(tasks=children):
			stages: Final = cast("tuple[Task, ...]", children)
			dispositions = tuple(classify(t, budget_s, timings, scope) for t in stages)
			over_budget_ids = frozenset(
				id(d.task) for d in dispositions if isinstance(d, OverBudget)
			)
			has_untimed = any(isinstance(d, Untimed) for d in dispositions)
			# A mid-pipe cut would rewire the pipeline (the survivor before the cut feeding the
			# one after it) — a suffix-only cut keeps the surviving prefix, and an untimed
			# sibling keeps the pipe whole for the first run that measures it.
			pruned = prune_pipes(
				node, lambda t: None if id(t) in over_budget_ids else t, lambda _: has_untimed
			)
			return _Planned(
				pruned.node,
				dispositions,
				pruned.node is not None
				and any(info.task.mutates for info in flatten_leaves(pruned.node)),
				pruned.pipe_cut,
			)
		case Parallel(tasks=children):
			planned = tuple(_plan_under(child, budget_s, timings, scope) for child in children)
			mutating = tuple(
				child.node for child in planned if child.node is not None and child.has_mutating
			)
			readonly = tuple(
				child.node for child in planned if child.node is not None and not child.has_mutating
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
					_collapse(Parallel(*readonly, env=node.env, cwd=node.cwd)),
					**_fields_of(node),
				)
			return _Planned(
				runnable,
				_collect(planned),
				any(child.has_mutating for child in planned),
				_collect_cuts(planned),
			)
		case _:
			assert_never(node)


def _collect(planned: tuple[_Planned, ...]) -> Iterable[Disposition]:
	"""The planned children's dispositions, in DFS order."""
	return chain.from_iterable(child.dispositions for child in planned)


def _collect_cuts(planned: tuple[_Planned, ...]) -> Iterable[Task]:
	"""The stages the planned children's pipe cuts dropped, in DFS order."""
	return chain.from_iterable(child.pipe_cut for child in planned)


def _fields_of(node: Group) -> dict[str, Any]:
	""":func:`rebuilt`'s GROUP_FIELDS carry, for a manufactured wrapper of a different kind
	than ``node``.
	"""
	return {field: getattr(node, field) for field in GROUP_FIELDS}


@overload
def _collapse(node: Sequential) -> Sequential: ...
@overload
def _collapse(node: Parallel) -> Parallel: ...
def _collapse(node: Sequential | Parallel) -> Sequential | Parallel:
	"""A fieldless single-child wrapper whose child is the same kind is dropped, matching the
	``|``/``+`` flattening; a wrapper carrying annotations keeps them.
	"""
	children: Final = node.tasks
	if len(children) != 1 or not fieldless(node):
		return node
	match node:
		case Sequential() if isinstance(children[0], Sequential):
			return children[0]
		case Parallel() if isinstance(children[0], Parallel):
			return children[0]
		case _:
			return node
