# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import pytest

from camas import ChainLink, Parallel, Sequential, Task, flatten_leaves
from camas.effect.termtree import print_tree


def test_single_cmd() -> None:
	leaves = flatten_leaves(Task("echo hi"))
	assert len(leaves) == 1
	assert leaves[0].task.cmd == "echo hi"
	assert leaves[0].depth == 0
	assert leaves[0].is_last_chain == ()


def test_parallel() -> None:
	task = Parallel(tasks=(Task("a"), Task("b"), Task("c")))
	leaves = flatten_leaves(task)
	assert [leaf.task.cmd for leaf in leaves] == ["a", "b", "c"]
	assert [leaf.depth for leaf in leaves] == [1, 1, 1]
	assert [leaf.is_last_chain for leaf in leaves] == [
		(ChainLink(False, True),),
		(ChainLink(False, True),),
		(ChainLink(True, True),),
	]


def test_sequential() -> None:
	task = Sequential(tasks=(Task("a"), Task("b")))
	leaves = flatten_leaves(task)
	assert [leaf.task.cmd for leaf in leaves] == ["a", "b"]
	assert [leaf.is_last_chain for leaf in leaves] == [
		(ChainLink(False, False),),
		(ChainLink(True, False),),
	]


def test_nested_parallel_in_sequential() -> None:
	task = Sequential(
		tasks=(
			Parallel(tasks=(Task("a"), Task("b"))),
			Task("c"),
		)
	)
	leaves = flatten_leaves(task)
	assert [leaf.task.cmd for leaf in leaves] == ["a", "b", "c"]
	assert [leaf.depth for leaf in leaves] == [2, 2, 1]
	assert [leaf.is_last_chain for leaf in leaves] == [
		(ChainLink(False, False), ChainLink(False, True)),
		(ChainLink(False, False), ChainLink(True, True)),
		(ChainLink(True, False),),
	]


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
	assert [leaf.task.cmd for leaf in leaves] == ["a", "b", "c", "d"]
	assert [leaf.depth for leaf in leaves] == [3, 3, 2, 1]


@pytest.mark.parametrize(
	("cmd", "expected_cmd"),
	[
		(Task("echo hello world"), "echo hello world"),
		(Task(("echo", "hello")), ("echo", "hello")),
		(Task("echo hi", name="greet"), "echo hi"),
	],
)
def test_leaf_task_reference(cmd: Task, expected_cmd: str | tuple[str, ...]) -> None:
	leaves = flatten_leaves(cmd)
	assert leaves[0].task.cmd == expected_cmd


def test_print_tree(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(tasks=(Task("a", name="alpha"), Task("b", name="beta")))
	print_tree(task)
	captured = capsys.readouterr()
	assert "alpha" in captured.out
	assert "beta" in captured.out
	assert "┃ alpha" in captured.out
	assert "┃ beta" in captured.out


def test_print_tree_deeply_nested(capsys: pytest.CaptureFixture[str]) -> None:
	task = Sequential(
		tasks=(
			Parallel(tasks=(Task("a"), Task("b"))),
			Task("c"),
		)
	)
	print_tree(task)
	captured = capsys.readouterr()
	assert "│ ┃ a" in captured.out
	assert "│ ┃ b" in captured.out
	assert "└─ c" in captured.out
