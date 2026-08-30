# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from pathlib import Path

from camas import Clean, Parallel, Sequential, Task
from camas.core.budget import Fits, OverBudget, Untimed, classify, plan_under
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
	assert plan.fits == (Fits(a, 0.1), Fits(a, 0.1))
	assert plan.over_budget == ()
	assert plan.untimed == ()


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
	assert plan.fits == (Fits(clean_before, 0.1), Fits(gen, 0.2), Fits(clean_after, 0.1))
	assert plan.over_budget == ()
	assert plan.untimed == ()


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
	assert plan.fits == (Fits(gen, 0.2),)
	assert plan.over_budget == (OverBudget(clean_after, 9.0),)
	assert plan.untimed == (Untimed(clean_before),)


def test_plan_under_nothing_fits_is_none() -> None:
	plan = plan_under(Task("a", name="a"), 0.1, {CacheKey("a", 0): TaskTiming(9.0, 1)})
	assert plan.node is None
	assert plan.fits == ()
	assert [o.task for o in plan.over_budget] == [Task("a", name="a")]


def test_plan_under_serializes_all_mutating_parallel_siblings() -> None:
	"""#306: mutating siblings never run concurrently — the planner serializes them."""
	fmt1 = Task("fmt1", name="fmt1", mutates=True)
	fmt2 = Task("fmt2", name="fmt2", mutates=True)
	timings = {
		CacheKey("fmt1", 0): TaskTiming(0.1, 5),
		CacheKey("fmt2", 0): TaskTiming(0.1, 5),
	}
	plan = plan_under(Parallel(fmt1, fmt2), 1.0, timings)
	assert plan.node == Sequential(fmt1, fmt2)
	assert plan.fits == (Fits(fmt1, 0.1), Fits(fmt2, 0.1))
	assert plan.over_budget == ()
	assert plan.untimed == ()


def test_plan_under_serializes_a_repeated_mutating_leaf() -> None:
	"""#306: a repeated mutating leaf keeps both occurrences, serialized instead of racing
	itself."""
	fmt = Task("fmt", name="fmt", mutates=True)
	plan = plan_under(Parallel(fmt, fmt), 1.0, {CacheKey("fmt", 0): TaskTiming(0.1, 5)})
	assert plan.node == Sequential(fmt, fmt)
	assert plan.fits == (Fits(fmt, 0.1), Fits(fmt, 0.1))
	assert plan.over_budget == ()
	assert plan.untimed == ()


def test_plan_under_mixed_parallel_serializes_mutating_first() -> None:
	"""#306: across Parallel siblings, the mutating leaf runs before the read-only group."""
	fmt = Task("fmt", name="fmt", mutates=True)
	lint = Task("lint", name="lint")
	test = Task("test", name="test")
	timings = {
		CacheKey("fmt", 0): TaskTiming(0.1, 5),
		CacheKey("lint", 0): TaskTiming(0.1, 5),
		CacheKey("test", 0): TaskTiming(0.1, 5),
	}
	plan = plan_under(Parallel(fmt, lint, test), 1.0, timings)
	assert plan.node == Sequential(fmt, Parallel(lint, test))
	assert plan.fits == (Fits(fmt, 0.1), Fits(lint, 0.1), Fits(test, 0.1))
	assert plan.over_budget == ()
	assert plan.untimed == ()


def test_plan_under_keeps_a_mutating_subtree_whole() -> None:
	"""#306: a sibling with a mutator anywhere is serialized whole — its own read-only
	children run in its slot, ahead of the trailing read-only group."""
	gen = Task("gen", name="gen", mutates=True)
	check = Task("check", name="check")
	inner = Sequential(gen, check)
	lint = Task("lint", name="lint")
	timings = {
		CacheKey("gen", 0): TaskTiming(0.2, 5),
		CacheKey("check", 0): TaskTiming(0.1, 5),
		CacheKey("lint", 0): TaskTiming(0.1, 5),
	}
	plan = plan_under(Parallel(inner, lint), 1.0, timings)
	assert plan.node == Sequential(inner, Parallel(lint))
	assert plan.fits == (Fits(gen, 0.2), Fits(check, 0.1), Fits(lint, 0.1))
	assert plan.over_budget == ()
	assert plan.untimed == ()


def test_plan_under_serializes_mutating_subtrees_whole() -> None:
	"""#306: a Clean gate beside a second formatter keeps its checks out of the other
	writer's run."""
	clean = Clean(Task("make gen", name="gen", mutates=True))
	fmt2 = Task("fmt2", name="fmt2", mutates=True)
	timings = {
		CacheKey("gen-before", 0): TaskTiming(0.1, 5),
		CacheKey("gen", 0): TaskTiming(0.2, 5),
		CacheKey("gen-after", 0): TaskTiming(0.1, 5),
		CacheKey("fmt2", 0): TaskTiming(0.1, 5),
	}
	plan = plan_under(Parallel(clean, fmt2), 1.0, timings)
	assert plan.node == Sequential(clean, fmt2)
	assert [f.task.name for f in plan.fits] == ["gen-before", "gen", "gen-after", "fmt2"]
	assert plan.over_budget == ()
	assert plan.untimed == ()


def test_plan_under_drops_an_over_budget_clean_mutator() -> None:
	"""#306: a Clean mutator measured over budget drops like any leaf — the gate degenerates
	to its checks around an un-run generator."""
	mutator = Task("make gen", name="gen", mutates=True)
	clean = Clean(mutator)
	timings = {
		CacheKey("gen-before", 0): TaskTiming(0.1, 5),
		CacheKey("gen", 0): TaskTiming(9.0, 5),
		CacheKey("gen-after", 0): TaskTiming(0.1, 5),
	}
	plan = plan_under(clean, 1.0, timings)
	assert plan.node == Sequential(clean.tasks[0], clean.tasks[2])
	assert [f.task.name for f in plan.fits] == ["gen-before", "gen-after"]
	assert plan.over_budget == (OverBudget(mutator, 9.0),)
	assert plan.untimed == ()


def test_plan_under_drops_an_over_budget_mutating_leaf_from_a_parallel() -> None:
	fmt = Task("fmt", name="fmt", mutates=True)
	lint = Task("lint", name="lint")
	timings = {
		CacheKey("fmt", 0): TaskTiming(9.0, 5),
		CacheKey("lint", 0): TaskTiming(0.1, 5),
	}
	plan = plan_under(Parallel(fmt, lint), 1.0, timings)
	assert plan.node == Parallel(lint)
	assert plan.fits == (Fits(lint, 0.1),)
	assert plan.over_budget == (OverBudget(fmt, 9.0),)
	assert plan.untimed == ()


def test_plan_under_collapses_a_rebuilt_single_child_parallel() -> None:
	over = Task("slow", name="slow")
	a, b = Task("a", name="a"), Task("b", name="b")
	inner = Parallel(a, b)
	timings = {
		CacheKey("slow", 0): TaskTiming(9.0, 5),
		CacheKey("a", 0): TaskTiming(0.1, 5),
		CacheKey("b", 0): TaskTiming(0.1, 5),
	}
	plan = plan_under(Parallel(over, inner), 1.0, timings)
	assert plan.node == inner
	assert plan.fits == (Fits(a, 0.1), Fits(b, 0.1))
	assert plan.over_budget == (OverBudget(over, 9.0),)
	assert plan.untimed == ()


def test_plan_under_collapses_a_rebuilt_single_child_sequential() -> None:
	gen = Task("gen", name="gen", mutates=True)
	inner = Sequential(gen)
	plan = plan_under(Parallel(inner), 1.0, {CacheKey("gen", 0): TaskTiming(0.1, 5)})
	assert plan.node == inner
	assert plan.fits == (Fits(gen, 0.1),)
	assert plan.over_budget == ()
	assert plan.untimed == ()


def test_plan_under_keeps_a_named_single_child_wrapper() -> None:
	"""#306: collapsing never strips annotations — a named wrapper survives with its fields."""
	a, b = Task("a", name="a"), Task("b", name="b")
	inner = Parallel(a, b)
	source = Parallel(inner, name="outer", help="drift gate")
	timings = {
		CacheKey("a", 0): TaskTiming(0.1, 5),
		CacheKey("b", 0): TaskTiming(0.1, 5),
	}
	plan = plan_under(source, 1.0, timings)
	assert plan.node == Parallel(inner, name="outer", help="drift gate")
	assert plan.fits == (Fits(a, 0.1), Fits(b, 0.1))
	assert plan.over_budget == ()
	assert plan.untimed == ()


def test_plan_under_collapses_a_fieldless_inner_readonly_wrapper() -> None:
	"""#306: the mixed branch's inner read-only group collapses when fieldless — no
	Parallel(Parallel(...)) nesting."""
	gen = Task("gen", name="gen", mutates=True)
	a, b = Task("a", name="a"), Task("b", name="b")
	timings = {
		CacheKey("gen", 0): TaskTiming(0.1, 5),
		CacheKey("a", 0): TaskTiming(0.1, 5),
		CacheKey("b", 0): TaskTiming(0.1, 5),
	}
	plan = plan_under(Parallel(gen, Parallel(a, b)), 1.0, timings)
	assert plan.node == Sequential(gen, Parallel(a, b))
	assert plan.fits == (Fits(gen, 0.1), Fits(a, 0.1), Fits(b, 0.1))
	assert plan.over_budget == ()
	assert plan.untimed == ()


def test_plan_under_carries_group_fields_onto_the_reordered_wrapper() -> None:
	"""The manufactured Sequential carries every GROUP_FIELDS value; the inner read-only
	Parallel keeps the group's env/cwd, with the identity left on the outer wrapper."""
	fmt = Task("fmt", name="fmt", mutates=True)
	lint = Task("lint", name="lint")
	source = Parallel(
		fmt, lint, name="gate", env={"A": "1"}, cwd=Path(), help="drift gate", paths="."
	)
	timings = {
		CacheKey("fmt", 0): TaskTiming(0.1, 5),
		CacheKey("lint", 0): TaskTiming(0.1, 5),
	}
	plan = plan_under(source, 1.0, timings)
	assert plan.node == Sequential(
		Task("fmt", name="fmt", mutates=True, env={"A": "1"}, cwd=Path(), paths=".", when="."),
		Parallel(
			Task("lint", name="lint", env={"A": "1"}, cwd=Path(), paths=".", when="."),
			env={"A": "1"},
			cwd=Path(),
		),
		name="gate",
		env={"A": "1"},
		cwd=Path(),
		help="drift gate",
		paths=".",
	)
	assert [f.task.name for f in plan.fits] == ["fmt", "lint"]
	assert plan.over_budget == ()
	assert plan.untimed == ()
