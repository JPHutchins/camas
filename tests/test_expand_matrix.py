# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import pytest

from camas import Parallel, Sequential, Task, expand_matrix


def test_no_matrix_passthrough() -> None:
	task = Parallel(tasks=(Task("echo hi"),))
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=(Task(cmd="echo hi"),)):
			pass
		case _:
			raise AssertionError(f"unexpected: {result}")


@pytest.mark.parametrize(
	("matrix", "expected_count"),
	[
		({"PY": ("3.12", "3.13")}, 2),
		({"PY": ("3.12", "3.13", "3.14")}, 3),
	],
)
def test_single_dim_parallel(matrix: dict[str, tuple[str, ...]], expected_count: int) -> None:
	task = Parallel(tasks=(Task("test"),), matrix=matrix)
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=children):
			assert len(children) == expected_count
		case _:
			raise AssertionError(f"unexpected: {result}")


@pytest.mark.parametrize(
	("matrix", "expected_count"),
	[
		({"OS": ("linux", "mac"), "PY": ("3.12", "3.13")}, 4),
		({"A": ("1", "2"), "B": ("x", "y"), "C": ("a",)}, 4),
		({"A": ("1", "2", "3"), "B": ("x", "y")}, 6),
	],
)
def test_multi_dim_cartesian_product(
	matrix: dict[str, tuple[str, ...]], expected_count: int
) -> None:
	task = Parallel(tasks=(Task("build"),), matrix=matrix)
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=children):
			assert len(children) == expected_count
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_substitution_in_str_cmd() -> None:
	task = Parallel(
		tasks=(Task("test --python {PY}"),),
		matrix={"PY": ("3.12", "3.13")},
	)
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=(Task(cmd=cmd1), Task(cmd=cmd2))):
			assert isinstance(cmd1, str)
			assert isinstance(cmd2, str)
			assert "3.12" in cmd1
			assert "3.13" in cmd2
			assert "{PY}" not in cmd1
			assert "{PY}" not in cmd2
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_substitution_in_tuple_cmd() -> None:
	task = Parallel(
		tasks=(Task(("test", "--python", "{PY}")),),
		matrix={"PY": ("3.12",)},
	)
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=(Task(cmd=cmd),)):
			assert isinstance(cmd, tuple)
			assert cmd == ("test", "--python", "3.12")
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_env_merge_preserves_existing() -> None:
	task = Parallel(
		tasks=(Task("test", env={"EXISTING": "val"}),),
		matrix={"PY": ("3.12",)},
	)
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=(Task(env=env),)):
			assert env["EXISTING"] == "val"
			assert env["PY"] == "3.12"
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_sequential_clones_to_parallel_of_sequentials() -> None:
	task = Sequential(
		tasks=(Task("build"), Task("test")),
		name="ci",
		matrix={"PY": ("3.12", "3.13")},
	)
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=children):
			assert len(children) == 2
			for child in children:
				match child:
					case Sequential(tasks=seq_tasks):
						assert len(seq_tasks) == 2
					case _:
						raise AssertionError(f"expected Sequential, got {child}")
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_nested_group_children() -> None:
	inner = Parallel(tasks=(Task("lint"), Task("test")))
	task = Parallel(
		tasks=(inner,),
		matrix={"PY": ("3.12", "3.13")},
	)
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=children):
			assert len(children) == 2
			for child in children:
				match child:
					case Parallel(tasks=inner_tasks):
						assert len(inner_tasks) == 2
					case _:
						raise AssertionError(f"expected Parallel, got {child}")
		case _:
			raise AssertionError(f"unexpected: {result}")


@pytest.mark.parametrize(
	"name_suffix",
	["[PY=3.12]", "[PY=3.13]"],
)
def test_name_suffix_applied(name_suffix: str) -> None:
	task = Parallel(
		tasks=(Task("test"),),
		matrix={"PY": ("3.12", "3.13")},
	)
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=children):
			names = [c.name for c in children if isinstance(c, Task)]
			assert any(name_suffix in (n or "") for n in names)
		case _:
			raise AssertionError(f"unexpected: {result}")
