from __future__ import annotations

import pytest

from camas import Parallel, Sequential, Task, flatten_leaves


def test_single_cmd() -> None:
	leaves = flatten_leaves(Task("echo hi"))
	assert len(leaves) == 1
	assert leaves[0].name == "echo hi"


def test_parallel() -> None:
	task = Parallel(tasks=(Task("a"), Task("b"), Task("c")))
	leaves = flatten_leaves(task)
	assert [leaf.name for leaf in leaves] == ["a", "b", "c"]


def test_sequential() -> None:
	task = Sequential(tasks=(Task("a"), Task("b")))
	leaves = flatten_leaves(task)
	assert [leaf.name for leaf in leaves] == ["a", "b"]


def test_nested_parallel_in_sequential() -> None:
	task = Sequential(
		tasks=(
			Parallel(tasks=(Task("a"), Task("b"))),
			Task("c"),
		)
	)
	leaves = flatten_leaves(task)
	assert [leaf.name for leaf in leaves] == ["a", "b", "c"]


def test_deeply_nested() -> None:
	task = Parallel(
		tasks=(
			Sequential(
				tasks=(
					Parallel(tasks=(Task("a"), Task("b"))),
					Task("c"),
				)
			),
			Task("d"),
		)
	)
	leaves = flatten_leaves(task)
	assert [leaf.name for leaf in leaves] == ["a", "b", "c", "d"]


@pytest.mark.parametrize(
	("cmd", "expected_name"),
	[
		(Task("echo hello world"), "echo hello world"),
		(Task(("echo", "hello")), "echo hello"),
		(Task("echo hi", name="greet"), "greet"),
	],
)
def test_task_display_name(cmd: Task, expected_name: str) -> None:
	leaves = flatten_leaves(cmd)
	assert leaves[0].name == expected_name
