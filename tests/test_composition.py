# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st
from typing_extensions import assert_type

from camas import Parallel, Pipe, Project, Sequential, Task
from camas.core.matrix import expand_matrix
from camas.core.traversal import flatten_leaves

if TYPE_CHECKING:
	from pathlib import Path

	from camas.v0.task import Group, TaskNode

a = Task("a")
b = Task("b")
c = Task("c")
d = Task("d")
e = Task("e")
f = Task("f")


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
	assert x == y
	assert Parallel(a, b, c) == y


def test_parallel_merge_with_sequential() -> None:
	x = a | b
	y = x | (c + d)
	assert_type(y, Parallel)
	assert Parallel(a, b, Sequential(c, d)) == y


def test_parallel_merge_of_parallels() -> None:
	x = (a | b) | (c | d)
	y = (a | b) | (c | d) | (e | f)
	assert_type(x, Parallel)
	assert_type(y, Parallel)
	assert Parallel(a, b, c, d) == x
	assert Parallel(a, b, c, d, e, f) == y


def test_parallel_merge_keeps_sequentials_and_pipes_whole() -> None:
	x = (a + b) | (c | d)
	y = (a | b) | (c > d) | (e | f)
	assert_type(x, Parallel)
	assert_type(y, Parallel)
	assert Parallel(Sequential(a, b), c, d) == x
	assert Parallel(a, b, Pipe(c, d), e, f) == y


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
	assert x == y
	assert Sequential(a, b, c) == y


def test_sequential_merge_with_parallel() -> None:
	x = a + b
	y = x + (c | d)
	assert_type(y, Sequential)
	assert Sequential(a, b, Parallel(c, d)) == y


def test_sequential_merge_of_sequentials() -> None:
	x = (a + b) + (c + d)
	y = (a + b) + (c + d) + (e + f)
	assert_type(x, Sequential)
	assert_type(y, Sequential)
	assert Sequential(a, b, c, d) == x
	assert Sequential(a, b, c, d, e, f) == y


def test_sequential_merge_keeps_parallels_and_pipes_whole() -> None:
	x = (a | b) + (c + d)
	y = (a + b) + (c > d) + (e + f)
	z = (a | b) + (c | d)
	assert_type(x, Sequential)
	assert_type(y, Sequential)
	assert_type(z, Sequential)
	assert Sequential(Parallel(a, b), c, d) == x
	assert Sequential(a, b, Pipe(c, d), e, f) == y
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


def group(
	kind: type[Parallel | Sequential],
	children: list[TaskNode],
	cwd: str | None = None,
	env: dict[str, str] | None = None,
) -> Parallel | Sequential:
	return kind(*children, cwd=cwd, env=env)


LEAVES: Final = st.builds(Task, st.sampled_from(("a", "b", "c", "d")))
PIPES: Final = st.lists(LEAVES, min_size=1, max_size=3).map(lambda stages: Pipe(*stages))
PROJECTS: Final = st.sampled_from(("libs", "api")).map(Project)
GROUP_KINDS: Final = st.sampled_from((Parallel, Sequential))

PLAIN_NODES: Final[st.SearchStrategy[TaskNode]] = st.recursive(
	st.one_of(LEAVES, PIPES, PROJECTS),
	lambda children: st.builds(group, GROUP_KINDS, st.lists(children, max_size=3)),
	max_leaves=8,
)
OPERANDS: Final[st.SearchStrategy[TaskNode | str]] = st.one_of(
	PLAIN_NODES, st.sampled_from(("x", "y"))
)
SCOPED_NODES: Final[st.SearchStrategy[TaskNode]] = st.recursive(
	st.one_of(LEAVES, PIPES),
	lambda children: st.builds(
		group,
		GROUP_KINDS,
		st.lists(children, max_size=3),
		st.sampled_from((None, "front", "back")),
		st.sampled_from(({}, {"K": "1"}, {"K": "2"})),
	),
	max_leaves=8,
)


def contributed(kind: type[Group], operand: TaskNode | str) -> tuple[TaskNode, ...]:
	if isinstance(operand, str):
		return (Task(operand),)
	return operand.tasks if isinstance(operand, kind) else (operand,)


def executed(node: TaskNode) -> tuple[tuple[str | tuple[str, ...], Path | None, str], ...]:
	return tuple(
		(leaf.task.cmd, leaf.task.cwd, repr(sorted(leaf.task.env.items())))
		for leaf in flatten_leaves(expand_matrix(node))
	)


@given(PLAIN_NODES, OPERANDS)
def test_or_contributes_a_parallels_children_and_any_other_operand_whole(
	left: TaskNode, right: TaskNode | str
) -> None:
	composed = left | right
	assert_type(composed, Parallel)
	assert Parallel(*contributed(Parallel, left), *contributed(Parallel, right)) == composed


@given(PLAIN_NODES, OPERANDS)
def test_add_contributes_a_sequentials_children_and_any_other_operand_whole(
	left: TaskNode, right: TaskNode | str
) -> None:
	composed = left + right
	assert_type(composed, Sequential)
	assert Sequential(*contributed(Sequential, left), *contributed(Sequential, right)) == composed


@given(PLAIN_NODES, PLAIN_NODES, OPERANDS)
def test_or_is_associative(x: TaskNode, y: TaskNode, z: TaskNode | str) -> None:
	left_first = (x | y) | z
	right_first = x | (y | z)
	assert_type(left_first, Parallel)
	assert_type(right_first, Parallel)
	assert left_first == right_first


@given(PLAIN_NODES, PLAIN_NODES, OPERANDS)
def test_add_is_associative(x: TaskNode, y: TaskNode, z: TaskNode | str) -> None:
	left_first = (x + y) + z
	right_first = x + (y + z)
	assert_type(left_first, Sequential)
	assert_type(right_first, Sequential)
	assert left_first == right_first


@pytest.mark.xfail(strict=True, reason="flattening moves leaves into or out of a group's cwd/env")
@given(SCOPED_NODES, SCOPED_NODES)
def test_or_runs_every_leaf_as_the_nested_parallel_would(left: TaskNode, right: TaskNode) -> None:
	composed = left | right
	assert_type(composed, Parallel)
	assert executed(Parallel(left, right)) == executed(composed)


@pytest.mark.xfail(strict=True, reason="flattening moves leaves into or out of a group's cwd/env")
@given(SCOPED_NODES, SCOPED_NODES)
def test_add_runs_every_leaf_as_the_nested_sequential_would(
	left: TaskNode, right: TaskNode
) -> None:
	composed = left + right
	assert_type(composed, Sequential)
	assert executed(Sequential(left, right)) == executed(composed)


def test_a_fresh_plain_left_operand_adopts_the_right_fields() -> None:
	"""The tie-break's default table, pinned through public behavior: a freshly constructed
	plain group counts as fieldless, so the right operand's fields adopt — a new Group field
	whose stored default is not ``None`` (or, for ``env``, not empty) makes the fresh left
	fieldful and breaks this assert."""
	assert Parallel("a") | Parallel("b", name="n") == Parallel("a", "b", name="n")
	assert Sequential("a") + Sequential("b", name="n") == Sequential("a", "b", name="n")
	assert Parallel("a", env=cast("dict[str, str]", MappingProxyType({}))) | Parallel(
		"b", name="n"
	) == Parallel("a", "b", name="n")


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


def test_composition_is_associative() -> None:
	assert (a | b) | c == a | (b | c)
	assert (a + b) + c == a + (b + c)
	left_named = Parallel("a", name="n")
	assert (left_named | Parallel("b")) | Parallel("c") == left_named | (
		Parallel("b") | Parallel("c")
	)
	assert (a | b) | Parallel("c", name="n") == a | (b | Parallel("c", name="n"))
	assert (a + b) + Sequential("c", name="n") == a + (b + Sequential("c", name="n"))
	assert (a | b) | Parallel("c", matrix={}) == a | (b | Parallel("c", matrix={}))
	assert (a + b) + Sequential("c", matrix={}) == a + (b + Sequential("c", matrix={}))

	class Named(Parallel):  # pyrefly: ignore[bad-class-definition]
		__slots__ = ()

	class Staged(Sequential):  # pyrefly: ignore[bad-class-definition]
		__slots__ = ()

	assert (a | b) | Named("c") == a | (b | Named("c"))
	assert (a + b) + Staged("c") == a + (b + Staged("c"))


def test_operators_leave_their_operands_unchanged() -> None:
	check = Parallel("format")
	_ = check | "lint"
	_ = check + "integration"
	assert check == Parallel("format")


def test_parallel_operand_carries_its_fields_and_type() -> None:
	class Named(Parallel):  # pyrefly: ignore[bad-class-definition]
		__slots__ = ()

	check = Named("format", name="check", paths=".")
	ci = check | "lint"
	assert type(ci) is Named
	assert ci == Named("format", "lint", name="check", paths=".")


def test_left_parallel_wins_when_both_sides_are_parallels() -> None:
	"""Both sides flatten; the fields of the right-side Parallel have no home, so the
	left's carry — the documented tie-break."""
	assert Parallel("a", name="left") | Parallel("b", name="right") == Parallel(
		"a", "b", name="left"
	)


def test_right_parallel_carries_when_the_left_is_a_leaf() -> None:
	assert Task("a") | Parallel("b", name="check") == Parallel("a", "b", name="check")


def test_sequential_operand_carries_its_fields_and_type() -> None:
	class Staged(Sequential):  # pyrefly: ignore[bad-class-definition]
		__slots__ = ()

	staged = Staged("b", name="stage")
	assert Parallel("a") + staged == Staged(Parallel("a"), "b", name="stage")


def test_composition_rejects_non_nodes_loudly() -> None:
	with pytest.raises(TypeError, match="None"):
		_ = Task("a") | cast("TaskNode", None)


def test_composition_reguards_a_group_operands_children() -> None:
	"""A group built through the lenient constructor can hold a non-node child; composition
	re-checks its children instead of carrying the broken tree along."""
	with pytest.raises(TypeError, match="None"):
		_ = Parallel("a") | Parallel(cast("TaskNode", None))


def test_or_composes_a_project_reference() -> None:
	"""A :class:`Project` reference is a node for composition — the loader resolves it."""
	assert Task("a") | Project("libs") == Parallel(Task("a"), Project("libs"))


def test_project_references_compose_on_the_left() -> None:
	assert Project("libs") | Task("lint") == Parallel(Project("libs"), Task("lint"))
	assert Project("libs") + "lint" == Sequential(Project("libs"), "lint")
