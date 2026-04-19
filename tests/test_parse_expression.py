# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import pytest

from camas import Parallel, Sequential, Task
from camas.__main__ import parse_expression


@pytest.mark.parametrize(
	("expr", "expected"),
	[
		('Task("echo hi")', Task("echo hi")),
		('Task("echo hi", name="greet")', Task("echo hi", name="greet")),
		('Task(("echo", "hi"))', Task(("echo", "hi"))),
		('Task("test", env={"X": "1"})', Task("test", env={"X": "1"})),
		(
			'Parallel(tasks=(Task("a"), Task("b")))',
			Parallel(tasks=(Task("a"), Task("b"))),
		),
		(
			'Sequential(tasks=(Task("a"), Task("b")))',
			Sequential(tasks=(Task("a"), Task("b"))),
		),
		(
			'Sequential(tasks=(Parallel(tasks=(Task("a"), Task("b"))), Task("c")))',
			Sequential(tasks=(Parallel(tasks=(Task("a"), Task("b"))), Task("c"))),
		),
		(
			'Parallel(tasks=(Sequential(tasks=(Task("a"), Task("b"))), Task("c")))',
			Parallel(tasks=(Sequential(tasks=(Task("a"), Task("b"))), Task("c"))),
		),
		(
			"""Sequential(tasks=(
				Parallel(tasks=(
					Sequential(tasks=(Task("a"), Task("b"))),
					Task("c"),
				)),
				Task("d"),
			))""",
			Sequential(
				tasks=(
					Parallel(
						tasks=(
							Sequential(tasks=(Task("a"), Task("b"))),
							Task("c"),
						)
					),
					Task("d"),
				)
			),
		),
	],
)
def test_parse_expression(expr: str, expected: Task | Parallel | Sequential) -> None:
	assert parse_expression(expr) == expected


@pytest.mark.parametrize(
	("expr", "expected"),
	[
		(
			'Parallel(tasks=(Task("test"),), matrix={"PY": ("3.12", "3.13")})',
			Parallel(tasks=(Task("test"),), matrix={"PY": ("3.12", "3.13")}),
		),
		(
			'Parallel(tasks=(Task("build"),), matrix={"OS": ("linux", "mac"), "PY": ("3.12", "3.13", "3.14")})',
			Parallel(
				tasks=(Task("build"),),
				matrix={"OS": ("linux", "mac"), "PY": ("3.12", "3.13", "3.14")},
			),
		),
		(
			'Sequential(tasks=(Task("a"),), matrix={"X": ("1",)})',
			Sequential(tasks=(Task("a"),), matrix={"X": ("1",)}),
		),
		(
			'Parallel(tasks=(Task("pytest --python {PY}"),), matrix={"PY": ("3.12", "3.13", "3.14")})',
			Parallel(
				tasks=(Task("pytest --python {PY}"),),
				matrix={"PY": ("3.12", "3.13", "3.14")},
			),
		),
		(
			'Sequential(tasks=(Task("build {OS}"), Task("test {OS}")), matrix={"OS": ("linux", "mac")})',
			Sequential(
				tasks=(Task("build {OS}"), Task("test {OS}")),
				matrix={"OS": ("linux", "mac")},
			),
		),
		(
			'Parallel(tasks=(Task(("uv", "run", "--python", "{PY}", "pytest")),), matrix={"PY": ("3.12",)})',
			Parallel(
				tasks=(Task(("uv", "run", "--python", "{PY}", "pytest")),),
				matrix={"PY": ("3.12",)},
			),
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


def test_parse_bare_string_not_a_task() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression('"just a string"')
