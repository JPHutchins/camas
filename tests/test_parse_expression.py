# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import pytest

from camas import Parallel, Sequential, Task
from camas.main import parse_expression


@pytest.mark.parametrize(
	("expr", "expected"),
	[
		('Task("echo hi")', Task("echo hi")),
		('Task("echo hi", name="greet")', Task("echo hi", name="greet")),
		('Task(("echo", "hi"))', Task(("echo", "hi"))),
		('Task("test", env={"X": "1"})', Task("test", env={"X": "1"})),
		(
			'Parallel(Task("a"),Task("b"))',
			Parallel(Task("a"), Task("b")),
		),
		(
			'Sequential(Task("a"),Task("b"))',
			Sequential(Task("a"), Task("b")),
		),
		(
			'Sequential(Parallel(Task("a"),Task("b")),Task("c"))',
			Sequential(Parallel(Task("a"), Task("b")), Task("c")),
		),
		(
			'Parallel(Sequential(Task("a"),Task("b")),Task("c"))',
			Parallel(Sequential(Task("a"), Task("b")), Task("c")),
		),
		(
			"""Sequential(Parallel(Sequential(Task("a"),Task("b")),Task("c")),Task("d"))""",
			Sequential(Parallel(Sequential(Task("a"), Task("b")), Task("c")), Task("d")),
		),
	],
)
def test_parse_expression(expr: str, expected: Task | Parallel | Sequential) -> None:
	assert parse_expression(expr) == expected


@pytest.mark.parametrize(
	("expr", "expected"),
	[
		(
			'Parallel(Task("test"),matrix={"PY": ("3.12", "3.13")})',
			Parallel(Task("test"), matrix={"PY": ("3.12", "3.13")}),
		),
		(
			'Parallel(Task("build"),matrix={"OS": ("linux", "mac"), "PY": ("3.12", "3.13", "3.14")})',
			Parallel(
				Task("build"), matrix={"OS": ("linux", "mac"), "PY": ("3.12", "3.13", "3.14")}
			),
		),
		(
			'Sequential(Task("a"),matrix={"X": ("1",)})',
			Sequential(Task("a"), matrix={"X": ("1",)}),
		),
		(
			'Parallel(Task("pytest --python {PY}"),matrix={"PY": ("3.12", "3.13", "3.14")})',
			Parallel(Task("pytest --python {PY}"), matrix={"PY": ("3.12", "3.13", "3.14")}),
		),
		(
			'Sequential(Task("build {OS}"),Task("test {OS}"),matrix={"OS": ("linux", "mac")})',
			Sequential(Task("build {OS}"), Task("test {OS}"), matrix={"OS": ("linux", "mac")}),
		),
		(
			'Parallel(Task(("uv", "run", "--python", "{PY}", "pytest")),matrix={"PY": ("3.12",)})',
			Parallel(Task(("uv", "run", "--python", "{PY}", "pytest")), matrix={"PY": ("3.12",)}),
		),
	],
)
def test_parse_matrix(expr: str, expected: Parallel | Sequential) -> None:
	assert parse_expression(expr) == expected


@pytest.mark.parametrize(
	"expr",
	[
		"not valid python +++",
		"1 + 2",
		"import os",
	],
)
def test_parse_invalid_syntax(expr: str) -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression(expr)


@pytest.mark.parametrize(
	"expr",
	[
		'os.system("rm -rf /")',
		'__import__("os")',
		"open('file')",
	],
)
def test_parse_rejects_unsafe(expr: str) -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression(expr)


def test_parse_rejects_unknown_type() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression('Foo(tasks=(Task("a"),))')


def test_parse_bare_string_coerces_to_task() -> None:
	assert parse_expression('"just a string"') == Task("just a string")


def test_parse_task_requires_cmd() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression("Task()")


def test_parse_ref_requires_name() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression("Parallel(Ref())", tasks={})


def test_parse_explicit_none_name() -> None:
	assert parse_expression('Task("hi", name=None)') == Task("hi", name=None)


# Parser-side fluent syntax: strings inside literals and the README one-liner.


def test_parse_top_level_tuple_with_bare_strings() -> None:
	assert parse_expression('("a", "b", "c")') == Sequential(Task("a"), Task("b"), Task("c"))


def test_parse_top_level_set_with_bare_strings() -> None:
	result = parse_expression('{"a", "b", "c"}')
	assert isinstance(result, Parallel)
	assert {t.cmd for t in result.tasks if isinstance(t, Task)} == {"a", "b", "c"}


def test_parse_readme_oneliner_mixed_str_tuple_set() -> None:
	"""The exact CLI example from README/EXAMPLES — outer tuple, inner set, bare strings."""
	result = parse_expression('("ruff format . --check", {"mypy .", "pyright ."}, "pytest")')
	assert isinstance(result, Sequential)
	assert result.tasks[0] == Task("ruff format . --check")
	assert isinstance(result.tasks[1], Parallel)
	assert {t.cmd for t in result.tasks[1].tasks if isinstance(t, Task)} == {
		"mypy .",
		"pyright .",
	}
	assert result.tasks[2] == Task("pytest")


def test_parse_deeply_nested_fluent() -> None:
	"""Tuple containing a set containing a tuple."""
	result = parse_expression('("a", {("b", "c"), "d"})')
	assert isinstance(result, Sequential)
	assert result.tasks[0] == Task("a")
	assert isinstance(result.tasks[1], Parallel)


def test_parse_bare_string_in_explicit_call() -> None:
	"""Inside a Sequential/Parallel call, a bare string still coerces to Task."""
	assert parse_expression('Sequential("a", "b")') == Sequential(Task("a"), Task("b"))


def test_parse_fluent_with_explicit_constructors_mixed() -> None:
	"""Mixing fluent and explicit forms is fine."""
	result = parse_expression('Sequential(Task("setup"), {"x", "y"})')
	assert isinstance(result, Sequential)
	assert isinstance(result.tasks[1], Parallel)
