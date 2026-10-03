# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from pathlib import Path

from camas import Clean, Parallel, Pipe, Sequential, Task
from camas.core.budget import Fits, OverBudget, Untimed, classify, plan_under, summarize
from camas.core.task import task_label
from camas.core.timings import CacheKey, Observed, TaskTiming


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


def test_summarize_an_empty_observed_tree_excludes_everything() -> None:
	"""An observed whose tree scoped away: nothing runs, a running-over-budget stage joins the
	excluded rather than the running-anyway set, no planned fit is counted dropped, and the
	scope-pruned untimed leaf lands in the not-covered census."""
	gen = Task("gen", name="gen")
	sarif = Task("sarif", name="sarif")
	plan = plan_under(Pipe(gen, sarif), 1.0, {CacheKey("sarif", 0): TaskTiming(9.0, 1)})
	summary = summarize(
		plan, Observed(camas_dir=None, scope=1, identities=None, node=None, pairs=())
	)
	assert summary.budget_s == 1.0
	assert summary.runnable == ()
	assert summary.unmeasured == ()
	assert summary.running_anyway == ()
	assert [o.task.name for o in summary.excluded] == ["sarif"]
	assert summary.dropped == ()
	assert [leaf.task.name for leaf in summary.not_covered] == ["gen"]


def test_summarize_counts_a_fit_the_scope_cut_removed_as_dropped() -> None:
	"""A fits leaf the plan selected but a mid-pipe scope cut removed (while a Parallel sibling
	survived) must appear in the dropped census — a machine consumer cannot find it in any
	other field."""
	from camas.core import timings
	from camas.core.budget import drop_unjustified_running

	f = Task("f", name="f")
	u = Task("u {paths}", name="u", paths="src")
	o = Task("o", name="o")
	g = Task("g {paths}", paths="docs")
	plan = plan_under(
		Parallel(Pipe(f, u, o), g),
		1.0,
		{
			CacheKey("f", 0): TaskTiming(0.1, 1),
			CacheKey("o", 0): TaskTiming(9.0, 1),
			CacheKey("g {paths}", 0): TaskTiming(0.1, 1),
		},
	)
	assert plan.node is not None
	keying = timings.observed(None, plan.node, ("docs/x.md",))
	assert keying.node is not None
	dropped = drop_unjustified_running(keying.node, plan, keying.pairs)
	summary = summarize(plan, timings.narrowed(keying, dropped))
	assert [task_label(t) for t in summary.runnable] == ["g {paths}"]
	assert [d.task.name for d in summary.dropped] == ["f"]
	assert [o.task.name for o in summary.excluded] == ["o"]


def test_summarize_reports_a_leaf_missing_from_the_pairs_by_its_own_label() -> None:
	"""A hand-built Observed whose pairs miss a leaf: the fallback reports the leaf itself, so
	a paired tree and an unpaired leaf never mix label spaces."""
	a = Task("a {paths}", paths=".")
	b = Task("b")
	scoped_a = Task("a x.py", paths=".")
	plan = plan_under(
		Parallel(a, b),
		60.0,
		{CacheKey("a {paths}", 0): TaskTiming(0.1, 1), CacheKey("b", 0): TaskTiming(0.1, 1)},
	)
	observed = Observed(
		camas_dir=None, scope=1, identities=None, node=Parallel(scoped_a, b), pairs=((a, scoped_a),)
	)
	assert [task_label(t) for t in summarize(plan, observed).runnable] == ["a {paths}", "b"]


def test_summarize_labels_a_scope_pruned_fit_as_not_covered_not_pipe_cut() -> None:
	"""A fitting leaf whose own paths missed the change set (no pipe anywhere near it) is
	reported not-covered — the pipe_cut reason is reserved for pipe cuts."""
	from camas.core import timings
	from camas.core.budget import drop_unjustified_running

	check = Task("check {paths}", name="check", paths="src")
	lint = Task("lint {paths}", name="lint", paths="docs")
	plan = plan_under(
		Parallel(check, lint),
		60.0,
		{CacheKey("check", 0): TaskTiming(0.1, 1), CacheKey("lint", 0): TaskTiming(0.1, 1)},
	)
	assert plan.node is not None
	keying = timings.observed(None, plan.node, ("docs/x.md",))
	assert keying.node is not None
	summary = summarize(
		plan, timings.narrowed(keying, drop_unjustified_running(keying.node, plan, keying.pairs))
	)
	assert [task_label(t) for t in summary.runnable] == ["lint"]
	assert summary.dropped == ()
	assert [(leaf.task.name, leaf.estimated_s) for leaf in summary.not_covered] == [("check", 0.1)]


def test_summarize_reports_the_plans_own_mid_pipe_cut_as_dropped() -> None:
	"""The plan's own cut inside a surviving tree, on a full run with no path filtering: the
	pipe's fitting stages were cut, not missed by their paths."""
	a = Task("a", name="a")
	b = Task("b", name="b")
	c = Task("c", name="c")
	d = Task("d", name="d")
	plan = plan_under(
		Parallel(Pipe(a, b, c), d),
		1.0,
		{
			CacheKey("a", 0): TaskTiming(0.1, 1),
			CacheKey("b", 0): TaskTiming(9.0, 1),
			CacheKey("c", 0): TaskTiming(0.1, 1),
			CacheKey("d", 0): TaskTiming(0.1, 1),
		},
	)
	assert plan.node == Parallel(d)
	summary = summarize(plan)
	assert [leaf.task.name for leaf in summary.dropped] == ["a", "c"]
	assert summary.not_covered == ()


def test_resolve_budget_reports_a_coverage_emptied_pipe_as_not_covered() -> None:
	"""Changed paths no stage covers prune each stage by its own paths — no pipe was cut, so
	the census and the cause both blame the paths."""
	from camas.core.budget import NothingToRun, resolve_budget
	from camas.core.scope import coverage_message

	f = Task("f {paths}", name="f", paths="docs")
	g = Task("g {paths}", name="g", paths="docs")
	h = Task("h {paths}", name="h", paths="src")
	plan = plan_under(
		Parallel(Pipe(f, g), h),
		60.0,
		{CacheKey(name, 0): TaskTiming(0.1, 1) for name in ("f", "g", "h")},
	)
	outcome = resolve_budget(plan, None, ("zzz/x.txt",))
	assert isinstance(outcome, NothingToRun)
	assert outcome.summary.dropped == ()
	assert [leaf.task.name for leaf in outcome.summary.not_covered] == ["f", "g", "h"]
	assert outcome.cause == coverage_message(("zzz/x.txt",))


def test_resolve_budget_reports_an_untimed_stage_a_scope_cut_removed_as_dropped() -> None:
	"""A mid-pipe scope cut takes a covered untimed stage down with its pipe — dropped, with
	no estimate, while the stage whose paths missed is not covered."""
	from camas.core.budget import BudgetRun, resolve_budget

	head = Task("head {paths}", name="head", paths="docs")
	untimed = Task("untimed", name="untimed")
	other = Task("other {paths}", name="other", paths="src")
	plan = plan_under(
		Parallel(Pipe(head, untimed), other),
		60.0,
		{CacheKey("head", 0): TaskTiming(0.1, 1), CacheKey("other", 0): TaskTiming(0.1, 1)},
	)
	outcome = resolve_budget(plan, None, ("src/x.py",))
	assert isinstance(outcome, BudgetRun)
	assert [(leaf.task.name, leaf.estimated_s) for leaf in outcome.summary.dropped] == [
		("untimed", None)
	]
	assert [leaf.task.name for leaf in outcome.summary.not_covered] == ["head"]


def test_resolve_budget_reports_the_drops_own_cut_when_it_empties_the_run() -> None:
	"""The post-scoping drop removes the running stage and, by the cut rule, its fitting
	sibling — the census carries that cut even though nothing is left to run."""
	from camas.core.budget import NothingToRun, resolve_budget

	plan = plan_under(
		Pipe(
			Task("run", name="run"),
			Task("mid-fit", name="midfit"),
			Task("tail {paths}", name="tail", paths="docs"),
		),
		1.0,
		{CacheKey("run", 0): TaskTiming(9.0, 5), CacheKey("midfit", 0): TaskTiming(0.1, 5)},
	)
	outcome = resolve_budget(plan, None, ("src/x.rs",))
	assert isinstance(outcome, NothingToRun)
	assert outcome.cause.startswith("The budget dropped the last runnable leaf")
	assert [o.task.name for o in outcome.summary.excluded] == ["run"]
	assert [(leaf.task.name, leaf.estimated_s) for leaf in outcome.summary.dropped] == [
		("midfit", 0.1)
	]
	assert [leaf.task.name for leaf in outcome.summary.not_covered] == ["tail"]


def test_narrowed_returns_the_keying_itself_when_the_drop_changed_nothing() -> None:
	from camas.core import timings
	from camas.core.scope import Pruned

	a = Task("a", name="a")
	keying = timings.observed(None, Parallel(a, Task("b", name="b")), ())
	assert keying.node is not None
	assert timings.narrowed(keying, Pruned(keying.node, ())) is keying
	assert timings.narrowed(keying, Pruned(Parallel(a), ())).identities == (CacheKey("a", 0),)


def test_drop_unjustified_running_fails_closed_on_a_stage_missing_from_the_pairs() -> None:
	"""A stage the pairs do not cover is dropped — the guard fails closed rather than letting
	an unjustified running-over-budget stage run."""
	from camas.core.budget import drop_unjustified_running

	a = Task("a", name="a")
	u = Task("u", name="u")
	plan = plan_under(Pipe(a, u), 1.0, {CacheKey("a", 0): TaskTiming(9.0, 1)})
	assert plan.node is not None
	assert drop_unjustified_running(plan.node, plan).node is None
