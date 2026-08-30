# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from camas import Parallel, Sequential, Task
from camas.core.budget import Fits, OverBudget, Untimed, classify, plan_under, schedule
from camas.core.timings import CacheKey, TaskTiming


def test_classify_fits_over_and_untimed() -> None:
	timings = {CacheKey("fast", 0): TaskTiming(0.5, 3), CacheKey("slow", 0): TaskTiming(9.0, 2)}
	assert classify(Task("x", name="fast"), 1.0, timings) == Fits(Task("x", name="fast"), 0.5)
	assert classify(Task("x", name="slow"), 1.0, timings) == OverBudget(Task("x", name="slow"), 9.0)
	assert classify(Task("x", name="new"), 1.0, timings) == Untimed(Task("x", name="new"))


def test_classify_boundary_is_inclusive() -> None:
	assert isinstance(
		classify(Task("x", name="a"), 1.0, {CacheKey("a", 0): TaskTiming(1.0, 1)}), Fits
	)


def test_schedule_orders_mutating_first_then_parallel() -> None:
	fmt = Task("fmt", mutates=True)
	assert schedule((Task("lint"), fmt, Task("mypy"))) == Sequential(
		fmt, Parallel(Task("lint"), Task("mypy"))
	)


def test_schedule_only_readonly_is_parallel() -> None:
	assert schedule((Task("a"), Task("b"))) == Parallel(Task("a"), Task("b"))


def test_schedule_only_mutating_is_sequential() -> None:
	assert schedule((Task("a", mutates=True),)) == Sequential(Task("a", mutates=True))


def test_schedule_empty_is_none() -> None:
	assert schedule(()) is None


def test_plan_under_partitions_and_schedules() -> None:
	fmt = Task("ruff format", name="fmt", mutates=True)
	lint = Task("ruff check", name="lint")
	test = Task("pytest", name="test")
	source = Sequential(fmt, Parallel(lint, test))
	timings = {
		CacheKey("fmt", 0): TaskTiming(0.2, 5),
		CacheKey("lint", 0): TaskTiming(0.4, 5),
		CacheKey("test", 0): TaskTiming(9.0, 5),
	}
	plan = plan_under(source, 1.0, timings)
	assert plan.node == Sequential(fmt, Parallel(lint))
	assert [f.task for f in plan.fits] == [fmt, lint]
	assert [o.task for o in plan.over_budget] == [test]
	assert plan.untimed == ()


def test_plan_under_runs_untimed() -> None:
	a, b = Task("a", name="a"), Task("b", name="b")
	plan = plan_under(Parallel(a, b), 5.0, {CacheKey("a", 0): TaskTiming(0.1, 1)})
	assert plan.node == Parallel(a, b)
	assert [u.task for u in plan.untimed] == [b]


def test_plan_under_preserves_structure_including_repeats() -> None:
	"""#306: the planner walks the tree instead of flattening it — a repeated leaf keeps its
	place instead of being silently deduplicated away."""
	a = Task("ruff", name="lint")
	plan = plan_under(Parallel(a, Sequential(a)), 5.0, {CacheKey("lint", 0): TaskTiming(0.1, 1)})
	assert plan.node == Parallel(a, Sequential(a))
	assert len(plan.fits) == 2


def test_plan_under_preserves_sequential_ordering() -> None:
	"""#306: a Sequential's ordering is its contract — the mutating-first heuristic reorders
	only across Parallel siblings."""
	clean_before = Task("git check", name="clean-before")
	gen = Task("make gen", name="gen", mutates=True)
	clean_after = Task("git check", name="clean-after")
	timings = {
		CacheKey("clean-before", 0): TaskTiming(0.1, 5),
		CacheKey("gen", 0): TaskTiming(0.2, 5),
		CacheKey("clean-after", 0): TaskTiming(0.1, 5),
	}
	plan = plan_under(Sequential(clean_before, gen, clean_after), 1.0, timings)
	assert plan.node == Sequential(clean_before, gen, clean_after)
	assert len(plan.fits) == 3


def test_plan_under_drops_an_over_budget_leaf_without_reordering() -> None:
	"""#306: an excluded leaf leaves the surviving order intact — the gate's checks keep
	their positions even when one of them is over budget."""
	clean_before = Task("git check", name="clean-before")
	gen = Task("make gen", name="gen", mutates=True)
	clean_after = Task("git check", name="clean-after")
	timings = {
		CacheKey("gen", 0): TaskTiming(0.2, 5),
		CacheKey("clean-after", 0): TaskTiming(9.0, 5),
	}
	plan = plan_under(Sequential(clean_before, gen, clean_after), 1.0, timings)
	assert plan.node == Sequential(clean_before, gen)
	assert [o.task.name for o in plan.over_budget] == ["clean-after"]


def test_plan_under_nothing_fits_is_none() -> None:
	plan = plan_under(Task("a", name="a"), 0.1, {CacheKey("a", 0): TaskTiming(9.0, 1)})
	assert plan.node is None
	assert plan.fits == ()
	assert [o.task for o in plan.over_budget] == [Task("a", name="a")]
