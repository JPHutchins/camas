# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from camas import AgentFormat, Parallel, Sequential, Task
from camas.main.expression import parse_expression, to_expression
from camas.v0.task import Group


def test_to_expression_marks_callable_paths() -> None:
	"""A callable scope has no source, so it renders as a non-parseable marker (preview only)."""
	rendered = to_expression(Task("ruff {paths}", paths=lambda c: c))
	assert rendered == 'Task("ruff {paths}", paths=<callable>)'


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


def test_parse_expression_threads_cwd_and_help() -> None:
	"""``cwd`` / ``help`` reach the constructors from the expression mini-language,
	at parity with ``tasks.py`` — not silently dropped (#25)."""
	assert parse_expression('Task("cargo build", cwd="src-tauri", help="Build")') == Task(
		"cargo build", cwd="src-tauri", help="Build"
	)
	assert parse_expression('Parallel(Task("a"), cwd="work", help="grp")') == Parallel(
		Task("a"), cwd="work", help="grp"
	)


def test_eval_node_threads_every_public_constructor_kwarg() -> None:
	"""Drift guard: a new public kwarg on Task/Sequential/Parallel fails here until
	it's both wired through ``eval_node`` and given a sentinel below — so a silently
	dropped kwarg (the #25 bug) can't reappear unnoticed."""
	# fragment + (attribute, expected value) for each constructor kwarg.
	samples = {
		"name": ("name='n'", "name", "n"),
		"env": ("env={'K': 'v'}", "env", {"K": "v"}),
		"cwd": ("cwd='d'", "cwd", Path("d")),
		"help": ("help='h'", "help", "h"),
		"mutates": ("mutates=True", "mutates", True),
		"matrix": ("matrix={'X': ('1',)}", "matrix", {"X": ("1",)}),
		"paths": ("paths='.'", "paths", "."),
		"agent_format": (
			"agent_format=AgentFormat('--x', 'sarif')",
			"agent_format",
			AgentFormat("--x", "sarif"),
		),
	}
	for cls, prefix, variadic in (
		(Task, "Task('c', ", "cmd"),
		(Group, "Parallel(Task('a'), ", "tasks"),
	):
		for kw in set(inspect.signature(cls).parameters) - {variadic}:
			frag, attr, expected = samples[kw]  # KeyError flags an unsampled new kwarg
			node = parse_expression(prefix + frag + ")")
			assert getattr(node, attr) == expected, (cls.__name__, kw)


# Parser-side fluent syntax: strings inside literals and the README one-liner.


def test_eval_node_rejects_unknown_agent_format_kind() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression("Task('c', agent_format=AgentFormat('--x', 'bogus'))")


def test_eval_node_rejects_incomplete_agent_format() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression("Task('c', agent_format=AgentFormat('--x'))")


def test_eval_node_rejects_non_agent_format_value() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression("Task('c', agent_format='sarif')")


def test_eval_node_accepts_agent_format_tuple_shorthand() -> None:
	node = parse_expression("Task('c', agent_format=('--x', 'sarif'))")
	assert isinstance(node, Task)
	assert node.agent_format == AgentFormat("--x", "sarif")


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


def test_invalid_expression() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression("not valid +++")


def test_unknown_type() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression('Foo(tasks=(Task("a"),))')


def test_bare_string_coerces_to_task() -> None:
	assert parse_expression('"just a string"') == Task("just a string")


@pytest.mark.parametrize(
	"expr",
	[
		'os.system("ls")',
		'__import__("os")',
	],
)
def test_rejects_unsafe(expr: str) -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression(expr)


def test_dict_with_non_str_key() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression('Task("x", env={1: "val"})')


def test_eval_env_rejects_unpacking(capsys: pytest.CaptureFixture[str]) -> None:
	"""``Task("x", env={**other, "K": "v"})`` would silently drop ``**other`` —
	reject explicitly so users see a deterministic error."""
	with pytest.raises(SystemExit, match="2"):
		parse_expression('Task("x", env={**other, "K": "v"})')
	assert "** unpacking" in capsys.readouterr().err


def test_eval_matrix_rejects_unpacking(capsys: pytest.CaptureFixture[str]) -> None:
	"""Same guard for ``matrix={**other, "PY": ("3.13",)}``."""
	with pytest.raises(SystemExit, match="2"):
		parse_expression('Parallel("x", matrix={**other, "PY": ("3.13",)})')
	assert "** unpacking" in capsys.readouterr().err
