# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Coverage for the variadic ``*tasks`` constructor API, str coercion, and
hashability of ``Task``/``Sequential``/``Parallel``."""

from __future__ import annotations

import pytest

from camas import Parallel, Sequential, Task


def test_sequential_variadic_with_task_nodes() -> None:
	assert Sequential(Task("a"), Task("b")).tasks == (Task("a"), Task("b"))


def test_parallel_variadic_with_task_nodes() -> None:
	assert Parallel(Task("a"), Task("b")).tasks == (Task("a"), Task("b"))


def test_sequential_coerces_strings_to_tasks() -> None:
	assert Sequential("a", "b").tasks == (Task("a"), Task("b"))


def test_parallel_coerces_strings_to_tasks() -> None:
	assert Parallel("a", "b").tasks == (Task("a"), Task("b"))


def test_mixed_str_and_task_in_sequential() -> None:
	assert Sequential("a", Task("b", name="explicit")).tasks == (
		Task("a"),
		Task("b", name="explicit"),
	)


def test_nested_sequential_in_parallel_via_variadic() -> None:
	tree = Parallel("a", Sequential("b", "c"))
	assert isinstance(tree, Parallel)
	assert isinstance(tree.tasks[1], Sequential)


def test_variadic_with_keyword_args() -> None:
	tree = Sequential("a", "b", name="ci", env={"K": "v"})
	assert tree.name == "ci"
	assert tree.env == {"K": "v"}
	assert tree.tasks == (Task("a"), Task("b"))


def test_sequential_pattern_matching_after_variadic() -> None:
	tree = Sequential("a", "b", name="ci")
	match tree:
		case Sequential(tasks=tasks, name=name):
			assert name == "ci"
			assert tasks == (Task("a"), Task("b"))
		case _:
			pytest.fail("did not match")


def test_parallel_pattern_matching_after_variadic() -> None:
	tree = Parallel("a", "b")
	match tree:
		case Parallel(tasks=tasks):
			assert tasks == (Task("a"), Task("b"))
		case _:
			pytest.fail("did not match")


def test_isinstance_still_works() -> None:
	assert isinstance(Sequential("a"), Sequential)
	assert isinstance(Parallel("a"), Parallel)


def test_equality() -> None:
	assert Sequential("a", "b") == Sequential("a", "b")
	assert Sequential("a", "b") != Sequential("a", "c")
	assert Parallel("a", "b") == Parallel("a", "b")


def test_empty_sequential() -> None:
	assert Sequential().tasks == ()


def test_empty_parallel() -> None:
	assert Parallel().tasks == ()


def test_immutable() -> None:
	tree = Sequential("a")
	with pytest.raises((AttributeError, TypeError)):
		setattr(tree, "name", "new")  # noqa: B010


def test_variadic_with_unpacking_iterable() -> None:
	cmds = ["a", "b", "c"]
	tree = Sequential(*cmds)
	assert tree.tasks == (Task("a"), Task("b"), Task("c"))


def test_repr_uses_field_names() -> None:
	r = repr(Sequential("a"))
	assert "tasks=" in r
	assert "name=" in r


def test_task_is_hashable() -> None:
	assert hash(Task("a")) == hash(Task("a"))
	assert hash(Task("a", env={"X": "1", "Y": "2"})) == hash(Task("a", env={"Y": "2", "X": "1"}))


def test_task_set_membership() -> None:
	assert {Task("a"), Task("b"), Task("a")} == {Task("a"), Task("b")}


def test_sequential_is_hashable() -> None:
	assert hash(Sequential("a", "b")) == hash(Sequential("a", "b"))


def test_parallel_is_hashable() -> None:
	assert hash(Parallel("a", "b")) == hash(Parallel("a", "b"))


def test_groups_in_set() -> None:
	assert len({Sequential("a"), Parallel("a"), Task("a")}) == 3
