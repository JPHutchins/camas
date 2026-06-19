# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from camas import Parallel, Sequential, Task
from camas.core.budget import Fits, OverBudget, Untimed, classify, plan_under, schedule
from camas.core.timings import TaskTiming


def test_classify_fits_over_and_untimed() -> None:
	timings = {"fast": TaskTiming(0.5, 3), "slow": TaskTiming(9.0, 2)}
	assert classify(Task("x", name="fast"), 1.0, timings) == Fits(Task("x", name="fast"), 0.5)
	assert classify(Task("x", name="slow"), 1.0, timings) == OverBudget(Task("x", name="slow"), 9.0)
	assert classify(Task("x", name="new"), 1.0, timings) == Untimed(Task("x", name="new"))


def test_classify_boundary_is_inclusive() -> None:
	assert isinstance(classify(Task("x", name="a"), 1.0, {"a": TaskTiming(1.0, 1)}), Fits)


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
	timings = {"fmt": TaskTiming(0.2, 5), "lint": TaskTiming(0.4, 5), "test": TaskTiming(9.0, 5)}
	plan = plan_under(source, 1.0, timings)
	assert plan.node == Sequential(fmt, Parallel(lint))
	assert [f.task for f in plan.fits] == [fmt, lint]
	assert [o.task for o in plan.over_budget] == [test]
	assert plan.untimed == ()


def test_plan_under_excludes_untimed() -> None:
	a, b = Task("a", name="a"), Task("b", name="b")
	plan = plan_under(Parallel(a, b), 5.0, {"a": TaskTiming(0.1, 1)})
	assert plan.node == Parallel(a)
	assert [u.task for u in plan.untimed] == [b]


def test_plan_under_dedups_repeated_leaves() -> None:
	a = Task("ruff", name="lint")
	plan = plan_under(Parallel(a, Sequential(a)), 5.0, {"lint": TaskTiming(0.1, 1)})
	assert plan.node == Parallel(a)
	assert len(plan.fits) == 1


def test_plan_under_nothing_fits_is_none() -> None:
	plan = plan_under(Task("a", name="a"), 0.1, {"a": TaskTiming(9.0, 1)})
	assert plan.node is None
	assert plan.fits == ()
	assert [o.task for o in plan.over_budget] == [Task("a", name="a")]
