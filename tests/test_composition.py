# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from typing_extensions import assert_type

from camas import Parallel, Pipe, Project, Sequential, Task
from camas.core.matrix import expand_matrix
from camas.core.traversal import flatten_leaves
from camas.v0.task import GROUP_FIELDS, Group, fieldless

if TYPE_CHECKING:
	from pathlib import Path

	from camas.v0.task import Nodes, TaskNode

a = Task("a")
b = Task("b")
c = Task("c")
d = Task("d")
e = Task("e")
f = Task("f")


def executed(node: TaskNode) -> tuple[tuple[str | tuple[str, ...], Path | None, str], ...]:
	return tuple(
		(leaf.task.cmd, leaf.task.cwd, repr(sorted(leaf.task.env.items())))
		for leaf in flatten_leaves(expand_matrix(node))
	)


def test_parallel_2() -> None:
	x = a | b
	assert_type(x, Parallel)
	assert Parallel(a, b) == x


def test_parallel_3() -> None:
	x = a | b | c
	assert_type(x, Parallel)
	assert Parallel(a, b, c) == x


def test_parallel_merge() -> None:
	x = a | b
	y = x | c
	assert_type(y, Parallel)
	assert Parallel(a, b, c) == y


def test_parallel_merge_is_ordered_not_commutative() -> None:
	x = a | b
	y = b | a
	assert_type(y, Parallel)
	assert Parallel(b, a) == y
	assert x != y


def test_parallel_merge_associative() -> None:
	x = (a | b) | c
	y = a | (b | c)
	assert_type(x, Parallel)
	assert_type(y, Parallel)
	assert Parallel(a, b, c) == x
	assert Parallel(a, Parallel(b, c)) == y
	assert executed(x) == executed(y)


def test_parallel_merge_with_sequential() -> None:
	x = a | b
	y = x | (c + d)
	assert_type(y, Parallel)
	assert Parallel(a, b, Sequential(c, d)) == y


def test_parallel_merge_of_parallels() -> None:
	x = a | b
	y = c | d
	z = x | y
	merged = x | y.tasks
	assert_type(z, Parallel)
	assert_type(merged, Parallel)
	assert Parallel(a, b, y) == z
	assert a | b | c | d == merged
	assert x.extend(y.tasks) == merged
	assert executed(merged) == executed(z)


def test_parallel_merge_keeps_right_groups_whole() -> None:
	x = (a + b) | (c | d)
	y = (a | b) | (c > d) | (e | f)
	assert_type(x, Parallel)
	assert_type(y, Parallel)
	assert Parallel(Sequential(a, b), Parallel(c, d)) == x
	assert Parallel(a, b, Pipe(c, d), Parallel(e, f)) == y


def test_parallel_merge_is_one_level_deep() -> None:
	x = Parallel(Parallel(a, b), c) | d
	assert_type(x, Parallel)
	assert Parallel(Parallel(a, b), c, d) == x


def test_sequential_2() -> None:
	x = a + b
	assert_type(x, Sequential)
	assert Sequential(a, b) == x


def test_sequential_3() -> None:
	x = a + b + c
	assert_type(x, Sequential)
	assert Sequential(a, b, c) == x


def test_sequential_merge() -> None:
	x = a + b
	y = x + c
	assert_type(y, Sequential)
	assert Sequential(a, b, c) == y


def test_sequential_merge_is_ordered_not_commutative() -> None:
	x = a + b
	y = b + a
	assert_type(y, Sequential)
	assert Sequential(b, a) == y
	assert x != y


def test_sequential_merge_associative() -> None:
	x = (a + b) + c
	y = a + (b + c)
	assert_type(x, Sequential)
	assert_type(y, Sequential)
	assert Sequential(a, b, c) == x
	assert Sequential(a, Sequential(b, c)) == y
	assert executed(x) == executed(y)


def test_sequential_merge_with_parallel() -> None:
	x = a + b
	y = x + (c | d)
	assert_type(y, Sequential)
	assert Sequential(a, b, Parallel(c, d)) == y


def test_sequential_merge_of_sequentials() -> None:
	x = a + b
	y = c + d
	z = x + y
	merged = x + y.tasks
	assert_type(z, Sequential)
	assert_type(merged, Sequential)
	assert Sequential(a, b, y) == z
	assert a + b + c + d == merged
	assert x.extend(y.tasks) == merged
	assert executed(merged) == executed(z)


def test_sequential_merge_keeps_right_groups_whole() -> None:
	x = (a | b) + (c + d)
	y = (a + b) + (c > d) + (e + f)
	z = (a | b) + (c | d)
	assert_type(x, Sequential)
	assert_type(y, Sequential)
	assert_type(z, Sequential)
	assert Sequential(Parallel(a, b), Sequential(c, d)) == x
	assert Sequential(a, b, Pipe(c, d), Sequential(e, f)) == y
	assert Sequential(Parallel(a, b), Parallel(c, d)) == z


def test_sequential_merge_is_one_level_deep() -> None:
	x = Sequential(Sequential(a, b), c) + d
	assert_type(x, Sequential)
	assert Sequential(Sequential(a, b), c, d) == x


def test_add_binds_tighter_than_or() -> None:
	x = a | b + c
	y = a + b | c + d
	z = (a | b) + c
	assert_type(x, Parallel)
	assert_type(y, Parallel)
	assert_type(z, Sequential)
	assert Parallel(a, Sequential(b, c)) == x
	assert Parallel(Sequential(a, b), Sequential(c, d)) == y
	assert Sequential(Parallel(a, b), c) == z


def test_check_keeps_typecheck_whole_and_gate_swaps_test_for_coverage() -> None:
	format_check = Task("ruff format --check")
	lint = Task("ruff check")
	actionlint = Task("actionlint")
	typecheck = Task("mypy") | Task("pyright")
	test = Task("pytest")
	coverage = Task("pytest --cov")
	check = format_check | lint | actionlint | typecheck | test
	gate = (check - test) | coverage
	assert_type(check, Parallel)
	assert_type(gate, Parallel)
	assert Parallel(format_check, lint, actionlint, typecheck, test) == check
	assert Parallel(format_check, lint, actionlint, typecheck, coverage) == gate


def test_extend_adds_a_group_whole_and_its_tasks_as_children() -> None:
	x = Parallel(a, b)
	y = Parallel(c, d)
	assert_type(x.extend(y), Parallel)
	assert Parallel(a, b, y) == x.extend(y)
	assert Parallel(a, b, c, d) == x.extend(y.tasks)
	assert Parallel(a, b, c) == x.extend("c")
	assert Parallel(a, b) == x.extend(())


def test_or_nests_a_scoped_left_and_extend_joins_its_scope() -> None:
	scoped = Parallel(a, cwd="w")
	assert Parallel(scoped, b) == scoped | b
	assert Parallel(a, b, cwd="w") == scoped.extend(b)
	assert executed(Parallel(Task("a", cwd="w"), b)) == executed(scoped | b)
	assert executed(Parallel(Task("a", cwd="w"), Task("b", cwd="w"))) == executed(scoped.extend(b))


def test_remove_drops_every_equal_direct_child_and_keeps_fields() -> None:
	assert_type(Parallel(a, b).remove(b), Parallel)
	assert_type(Parallel(a, b) - b, Parallel)
	assert Parallel(a, c, cwd="w") == Parallel(a, b, c, b, cwd="w").remove(b)
	assert Sequential(c) == Sequential(a, b, c).remove((a, b))
	assert Parallel(a) == Parallel(a, b) - b
	assert Sequential(a, c) == (a + b + c) - b


def test_remove_rejects_a_node_that_is_not_a_direct_child() -> None:
	nested = Parallel(a, Parallel(b))
	with pytest.raises(ValueError, match="not a direct child"):
		_ = nested - b
	with pytest.raises(ValueError, match="not a direct child"):
		nested.remove((a, c))


def test_a_str_inside_a_sequence_is_rejected_not_split() -> None:
	"""``("python", "-c", "...")`` is a tuple command, never three tasks."""
	with pytest.raises(TypeError, match=r"tuple command — pass Task\(\(\.\.\.\)\) for one command"):
		_ = a | cast("tuple[TaskNode, ...]", ("python", "-c", "print(1)"))
	with pytest.raises(TypeError, match="tuple command"):
		Parallel(a).remove(cast("tuple[TaskNode, ...]", ("a",)))


def tree(
	kind: type[Parallel | Sequential],
	children: list[TaskNode],
	cwd: str | None,
	env: dict[str, str],
) -> Parallel | Sequential:
	return kind(*children, cwd=cwd, env=env)


def trees(base: st.SearchStrategy[TaskNode]) -> st.SearchStrategy[TaskNode]:
	return st.recursive(
		base,
		lambda children: st.builds(
			tree,
			st.sampled_from((Parallel, Sequential)),
			st.lists(children, max_size=3),
			st.sampled_from((None, "front", "back")),
			st.sampled_from(({}, {"K": "1"}, {"K": "2"})),
		),
		max_leaves=8,
	)


def operand(drawn: TaskNode | str | list[TaskNode]) -> tuple[Nodes, tuple[TaskNode, ...]]:
	match drawn:
		case str():
			return drawn, (Task(drawn),)
		case list():
			return tuple(drawn), tuple(drawn)
		case _:
			return drawn, (drawn,)


def operands(
	nodes: st.SearchStrategy[TaskNode],
) -> st.SearchStrategy[tuple[Nodes, tuple[TaskNode, ...]]]:
	return st.one_of(nodes, st.sampled_from(("x", "y")), st.lists(nodes, max_size=3)).map(operand)


LEAVES: Final = st.builds(Task, st.sampled_from(("a", "b", "c", "d")))
PIPES: Final = st.lists(LEAVES, min_size=1, max_size=3).map(lambda stages: Pipe(*stages))
NODES: Final = trees(st.one_of(LEAVES, PIPES, st.sampled_from(("libs", "api")).map(Project)))
RUNNABLE_NODES: Final = trees(st.one_of(LEAVES, PIPES))
GROUPS: Final = st.builds(
	tree,
	st.sampled_from((Parallel, Sequential)),
	st.lists(RUNNABLE_NODES, max_size=4),
	st.sampled_from((None, "w")),
	st.sampled_from(({}, {"K": "1"})),
)


def kept(kind: type[Group], left: TaskNode) -> tuple[TaskNode, ...]:
	return left.tasks if isinstance(left, kind) and fieldless(left) else (left,)


@given(NODES, operands(NODES))
def test_or_extends_a_plain_parallel_and_adds_the_right_operand_as_given(
	left: TaskNode, operand: tuple[Nodes, tuple[TaskNode, ...]]
) -> None:
	right, children = operand
	composed = left | right
	assert_type(composed, Parallel)
	assert Parallel(*kept(Parallel, left), *children) == composed


@given(NODES, operands(NODES))
def test_add_extends_a_plain_sequential_and_adds_the_right_operand_as_given(
	left: TaskNode, operand: tuple[Nodes, tuple[TaskNode, ...]]
) -> None:
	right, children = operand
	composed = left + right
	assert_type(composed, Sequential)
	assert Sequential(*kept(Sequential, left), *children) == composed


@given(RUNNABLE_NODES, operands(RUNNABLE_NODES))
def test_or_runs_every_leaf_as_the_nested_parallel_would(
	left: TaskNode, operand: tuple[Nodes, tuple[TaskNode, ...]]
) -> None:
	right, children = operand
	composed = left | right
	assert_type(composed, Parallel)
	assert executed(Parallel(left, *children)) == executed(composed)


@given(RUNNABLE_NODES, operands(RUNNABLE_NODES))
def test_add_runs_every_leaf_as_the_nested_sequential_would(
	left: TaskNode, operand: tuple[Nodes, tuple[TaskNode, ...]]
) -> None:
	right, children = operand
	composed = left + right
	assert_type(composed, Sequential)
	assert executed(Sequential(left, *children)) == executed(composed)


@given(RUNNABLE_NODES, RUNNABLE_NODES, operands(RUNNABLE_NODES))
def test_or_is_associative_in_execution(
	x: TaskNode, y: TaskNode, operand: tuple[Nodes, tuple[TaskNode, ...]]
) -> None:
	z, _ = operand
	left_first = (x | y) | z
	right_first = x | (y | z)
	assert_type(left_first, Parallel)
	assert_type(right_first, Parallel)
	assert executed(left_first) == executed(right_first)


@given(RUNNABLE_NODES, RUNNABLE_NODES, operands(RUNNABLE_NODES))
def test_add_is_associative_in_execution(
	x: TaskNode, y: TaskNode, operand: tuple[Nodes, tuple[TaskNode, ...]]
) -> None:
	z, _ = operand
	left_first = (x + y) + z
	right_first = x + (y + z)
	assert_type(left_first, Sequential)
	assert_type(right_first, Sequential)
	assert executed(left_first) == executed(right_first)


@given(GROUPS, st.data())
def test_sub_removes_every_equal_direct_child_and_keeps_fields(
	group: Parallel | Sequential, data: st.DataObject
) -> None:
	assume(group.tasks)
	target = data.draw(st.sampled_from(group.tasks))
	removed = group - target
	assert group.remove(target) == removed
	assert tuple(child for child in group.tasks if child != target) == removed.tasks
	assert all(getattr(removed, field) == getattr(group, field) for field in GROUP_FIELDS)


@given(GROUPS, RUNNABLE_NODES)
def test_remove_undoes_extend(group: Parallel | Sequential, node: TaskNode) -> None:
	assume(node not in group.tasks)
	assert group == group.extend(node).remove(node)


def test_a_fresh_group_is_plain() -> None:
	"""A freshly constructed group has every field at its default, so ``|``/``+`` extend it — a
	new Group field whose stored default is not ``None`` (or, for ``env``, not empty) nests every
	fresh left operand instead and breaks this assert."""
	assert Parallel("a") | "b" == Parallel("a", "b")
	assert Sequential("a") + "b" == Sequential("a", "b")
	assert Parallel("a", env=cast("dict[str, str]", MappingProxyType({}))) | "b" == Parallel(
		"a", "b"
	)


def test_or_appends_to_a_parallel() -> None:
	assert Parallel("format", "lint") | "integration" == Parallel("format", "lint", "integration")


def test_gt_rejects_a_project_reference_as_a_stage() -> None:
	"""A referenced project is a group, not a leaf stage — ``>`` raises the same ValueError the
	Pipe constructor raises for a nested stage."""
	with pytest.raises(ValueError, match="stages must be Tasks"):
		_ = Project("libs") > "lint"


def test_operators_assert_their_declared_types() -> None:
	"""#298's acceptance contract: each operator's static return type — ``assert_type`` is a
	runtime no-op enforced by every checker in the CI battery, so this test fails at analysis
	time if an operator's declared type drifts."""
	check = Parallel("format", "lint", "types", "tests")
	assert_type(check | "integration", Parallel)
	assert_type(check + "integration", Sequential)
	assert_type(Task("format") | Task("lint"), Parallel)
	assert_type(Task("build") + Task("test"), Sequential)
	assert_type(Sequential("build") | "lint", Parallel)
	assert_type(Project("libs") | "lint", Parallel)
	assert_type(Project("libs") + "lint", Sequential)
	assert_type(Task("gen") > "sarif", Pipe)
	assert_type(Task("a") > Pipe("b"), Pipe)
	assert_type(Pipe("a") > "b", Pipe)
	assert_type(Pipe("a") > Pipe("b", "c"), Pipe)
	assert_type(check | check.tasks, Parallel)
	assert_type(check - "tests", Parallel)
	assert_type(Sequential("a") - "a", Sequential)
	assert_type(Pipe("a", "b") - "b", Pipe)
	assert_type(check.extend(check.tasks), Parallel)
	assert_type(check.remove(check.tasks), Parallel)


def test_or_nests_a_sequential_as_one_child() -> None:
	seq = Sequential("build", "test")
	assert seq | "lint" == Parallel(seq, Task("lint"))


def test_add_appends_to_a_sequential() -> None:
	assert Sequential("build") + "test" == Sequential("build", "test")


def test_add_coerces_a_parallel_to_a_sequential() -> None:
	"""``Parallel(...) + integration`` runs the whole group first, then the new node —
	the coercion #298 asks for."""
	check = Parallel("format", "lint")
	assert check + "integration" == Sequential(check, Task("integration"))


def test_composition_is_associative_in_execution() -> None:
	assert executed((a | b) | c) == executed(a | (b | c))
	assert executed((a + b) + c) == executed(a + (b + c))
	left_named = Parallel("a", name="n")
	assert executed((left_named | Parallel("b")) | Parallel("c")) == executed(
		left_named | (Parallel("b") | Parallel("c"))
	)
	assert executed((a | b) | Parallel("c", name="n")) == executed(
		a | (b | Parallel("c", name="n"))
	)
	assert executed((a + b) + Sequential("c", name="n")) == executed(
		a + (b + Sequential("c", name="n"))
	)

	class Named(Parallel):  # pyrefly: ignore[bad-class-definition]
		__slots__ = ()

	class Staged(Sequential):  # pyrefly: ignore[bad-class-definition]
		__slots__ = ()

	assert executed((a | b) | Named("c")) == executed(a | (b | Named("c")))
	assert executed((a + b) + Staged("c")) == executed(a + (b + Staged("c")))


def test_operators_leave_their_operands_unchanged() -> None:
	check = Parallel("format")
	_ = check | "lint"
	_ = check + "integration"
	_ = check - "format"
	_ = check.extend("lint")
	assert check == Parallel("format")


def test_a_scoped_subclass_left_nests_and_extend_carries_its_fields_and_type() -> None:
	class Named(Parallel):  # pyrefly: ignore[bad-class-definition]
		__slots__ = ()

	check = Named("format", name="check", paths=".")
	assert Parallel(check, Task("lint")) == check | "lint"
	extended = check.extend("lint")
	assert type(extended) is Named
	assert Named("format", "lint", name="check", paths=".") == extended


def test_named_parallels_nest_whole() -> None:
	left = Parallel("a", name="left")
	right = Parallel("b", name="right")
	assert Parallel(left, right) == left | right


def test_a_right_parallel_is_one_child_of_a_leaf() -> None:
	assert Task("a") | Parallel("b", name="check") == Parallel(
		Task("a"), Parallel("b", name="check")
	)


def test_a_right_subclass_sequential_is_one_child() -> None:
	class Staged(Sequential):  # pyrefly: ignore[bad-class-definition]
		__slots__ = ()

	staged = Staged("b", name="stage")
	assert Parallel("a") + staged == Sequential(Parallel("a"), staged)
	assert type(staged.extend("c")) is Staged


def test_composition_rejects_non_nodes_loudly() -> None:
	with pytest.raises(TypeError, match="None"):
		_ = Task("a") | cast("TaskNode", None)


def test_composition_reguards_a_group_operands_children() -> None:
	"""A group built through the lenient constructor can hold a non-node child; composition
	re-checks the children it touches — an extended group's, and a merged ``.tasks`` — instead
	of carrying the broken tree along."""
	with pytest.raises(TypeError, match="None"):
		_ = Parallel(cast("TaskNode", None)) | "a"
	with pytest.raises(TypeError, match="None"):
		_ = Parallel("a") | Parallel(cast("TaskNode", None)).tasks


def test_or_composes_a_project_reference() -> None:
	"""A :class:`Project` reference is a node for composition — the loader resolves it."""
	assert Task("a") | Project("libs") == Parallel(Task("a"), Project("libs"))


def test_project_references_compose_on_the_left() -> None:
	assert Project("libs") | Task("lint") == Parallel(Project("libs"), Task("lint"))
	assert Project("libs") + "lint" == Sequential(Project("libs"), "lint")
