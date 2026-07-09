# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from pathlib import Path

import pytest

from camas import Parallel, Sequential, Task
from camas.core.matrix import expand_matrix
from camas.core.traversal import flatten_leaves


def _leaf(node: object) -> Task:
	assert isinstance(node, Task)
	return node


def test_no_matrix_passthrough() -> None:
	task = Parallel(Task("echo hi"))
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
	task = Parallel(Task("test"), matrix=matrix)
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
	task = Parallel(Task("build"), matrix=matrix)
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=children):
			assert len(children) == expected_count
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_substitution_in_str_cmd() -> None:
	task = Parallel(Task("test --python {PY}"), matrix={"PY": ("3.12", "3.13")})
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
	task = Parallel(Task(("test", "--python", "{PY}")), matrix={"PY": ("3.12",)})
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=(Task(cmd=cmd),)):
			assert isinstance(cmd, tuple)
			assert cmd == ("test", "--python", "3.12")
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_env_merge_preserves_existing() -> None:
	task = Parallel(Task("test", env={"EXISTING": "val"}), matrix={"PY": ("3.12",)})
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=(Task(env=env),)):
			assert env["EXISTING"] == "val"
			assert env["PY"] == "3.12"
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_sequential_clones_to_parallel_of_sequentials() -> None:
	task = Sequential(Task("build"), Task("test"), name="ci", matrix={"PY": ("3.12", "3.13")})
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
	inner = Parallel(Task("lint"), Task("test"))
	task = Parallel(inner, matrix={"PY": ("3.12", "3.13")})
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


def test_nested_group_env_threads_and_survives_specialization() -> None:
	"""https://github.com/JPHutchins/camas/issues/86 — nested-group env under an ancestor matrix.

	The group's own env threads into its leaves (execution) and is preserved on the specialized
	group node (specialize_node keeps env, substituted per binding — not just name/cwd/help/paths).
	"""
	task = Parallel(
		Sequential(Task("run {PY}"), env={"NESTED": "v-{PY}"}, name="inner"),
		matrix={"PY": ("3.12", "3.13")},
	)
	result = expand_matrix(task)
	assert isinstance(result, Parallel)
	assert [group.env for group in result.tasks] == [{"NESTED": "v-3.12"}, {"NESTED": "v-3.13"}]
	assert [leaf.task.env for leaf in flatten_leaves(result)] == [
		{"NESTED": "v-3.12", "PY": "3.12"},
		{"NESTED": "v-3.13", "PY": "3.13"},
	]


def test_nested_group_matrix_expands_under_ancestor_matrix() -> None:
	"""https://github.com/JPHutchins/camas/issues/86 — a nested group's own matrix is fully expanded
	before specialize_node runs, so it never sees a group carrying a matrix; the axes compose into
	the leaves."""
	task = Parallel(
		Sequential(Task("t {OS} {PY}"), matrix={"OS": ("lin", "mac")}, name="inner"),
		matrix={"PY": ("3.12",)},
	)
	result = expand_matrix(task)
	cmds = sorted(leaf.task.cmd for leaf in flatten_leaves(result))
	assert cmds == ["t lin 3.12", "t mac 3.12"]


def test_container_env_propagates_to_leaves() -> None:
	task = Parallel(Task("a"), Task("b"), env={"K": "v"})
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=(Task(env=ea), Task(env=eb))):
			assert ea == {"K": "v"}
			assert eb == {"K": "v"}
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_container_env_template_substituted_via_matrix() -> None:
	task = Sequential(Parallel(Task("x"), env={"VENV": ".venv-{PY}"}), matrix={"PY": ("3.10",)})
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=(Sequential(tasks=(Parallel(tasks=(Task(env=env),)),)),)):
			assert env == {"VENV": ".venv-3.10", "PY": "3.10"}
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_task_env_overrides_container_env() -> None:
	task = Parallel(Task("x", env={"K": "task"}), env={"K": "container"})
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=(Task(env=env),)):
			assert env["K"] == "task"
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_binding_overrides_both_container_and_task_env() -> None:
	task = Parallel(
		Task("x", env={"PY": "task"}), env={"PY": "container"}, matrix={"PY": ("3.10",)}
	)
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=(Task(env=env),)):
			assert env["PY"] == "3.10"
		case _:
			raise AssertionError(f"unexpected: {result}")


@pytest.mark.parametrize(
	"name_suffix",
	["[PY=3.12]", "[PY=3.13]"],
)
def test_name_suffix_applied(name_suffix: str) -> None:
	task = Parallel(Task("test"), matrix={"PY": ("3.12", "3.13")})
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=children):
			names = [c.name for c in children if isinstance(c, Task)]
			assert any(name_suffix in (n or "") for n in names)
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_help_preserved_without_matrix() -> None:
	task = Sequential(Task("echo hi", help="say hi"), help="greeting")
	result = expand_matrix(task)
	match result:
		case Sequential(tasks=(Task(help="say hi"),), help="greeting"):
			pass
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_help_interpolated_on_specialized_leaves() -> None:
	task = Parallel(
		Task("test", help="test on {PY}"), matrix={"PY": ("3.12", "3.13")}, help="fan out"
	)
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=(Task(help="test on 3.12"), Task(help="test on 3.13")), help="fan out"):
			pass
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_help_interpolated_on_cloned_sequentials() -> None:
	task = Sequential(
		Parallel(Task("a"), help="inner {PY}"),
		name="ci",
		matrix={"PY": ("3.12",)},
		help="ci on {PY}",
	)
	result = expand_matrix(task)
	match result:
		case Parallel(
			tasks=(Sequential(tasks=(Parallel(help="inner 3.12"),), help="ci on 3.12"),),
			help="ci on {PY}",
		):
			pass
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_container_when_propagates_to_leaves() -> None:
	task = Parallel(Task("cargo build", name="cargo"), when="src")
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=(Task(when=when),)):
			assert when == "src"
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_task_own_when_overrides_container_when() -> None:
	task = Parallel(Task("cargo build", name="cargo", when="src"), when="docs")
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=(Task(when=when),)):
			assert when == "src"
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_when_str_substituted_per_binding() -> None:
	task = Parallel(Task("build {PY}", when="pkg-{PY}"), matrix={"PY": ("3.12", "3.13")})
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=children):
			whens = sorted(
				c.when for c in children if isinstance(c, Task) and isinstance(c.when, str)
			)
			assert whens == ["pkg-3.12", "pkg-3.13"]
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_when_tuple_substituted_per_binding() -> None:
	task = Parallel(Task("build {PY}", when=("pkg-{PY}", "shared")), matrix={"PY": ("3.12",)})
	result = expand_matrix(task)
	match result:
		case Parallel(tasks=(Task(when=when),)):
			assert when == ("pkg-3.12", "shared")
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_callable_when_survives_specialization() -> None:
	def predicate(changed: tuple[str, ...]) -> bool:
		return True

	task = Parallel(Task("build {PY}", when=predicate), matrix={"PY": ("3.12",)})
	result = expand_matrix(task)
	assert predicate(()) is True
	match result:
		case Parallel(tasks=(Task(when=when),)):
			assert when is predicate
		case _:
			raise AssertionError(f"unexpected: {result}")


def test_relative_cwd_becomes_when_fallback() -> None:
	assert _leaf(expand_matrix(Task("cargo build", cwd="code-gen"))).when == "code-gen"


def test_container_cwd_becomes_leaf_when_fallback() -> None:
	result = expand_matrix(Parallel(Task("cargo build", name="cargo"), cwd="code-gen"))
	assert isinstance(result, Parallel)
	assert _leaf(result.tasks[0]).when == "code-gen"


def test_explicit_when_wins_over_cwd_fallback() -> None:
	assert _leaf(expand_matrix(Task("cargo build", cwd="code-gen", when="only"))).when == "only"


def test_when_dot_opts_out_of_cwd_fallback() -> None:
	assert _leaf(expand_matrix(Task("cargo build", cwd="code-gen", when="."))).when == "."


def test_absolute_cwd_has_no_when_fallback() -> None:
	assert _leaf(expand_matrix(Task("cargo build", cwd=Path.cwd()))).when is None
