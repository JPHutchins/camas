# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from camas import Parallel, Sequential, Task
from camas.main import (
	build_parser,
	main,
	parse_effects,
	parse_expression,
	print_task_summary_listing,
	task_summary,
)


def test_help(tmp_path: Path) -> None:
	result = subprocess.run(
		[sys.executable, "-m", "camas", "--help"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
	)
	assert result.returncode == 0
	assert "expression" in result.stdout


def test_version(tmp_path: Path) -> None:
	result = subprocess.run(
		[sys.executable, "-m", "camas", "--version"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
	)
	assert result.returncode == 0
	assert "camas" in result.stdout


def test_no_args_with_no_tasks_errors(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="2"):
		with patch("sys.argv", ["camas"]):
			main()
	assert "expression is required" in capsys.readouterr().err


def test_no_args_with_tasks_lists_them(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	tasks_py = tmp_path / "tasks.py"
	tasks_py.write_text(
		"from camas import Parallel, Sequential, Task\n"
		'lint = Task("ruff check .")\n'
		'test = Task("pytest")\n'
		"ci = Sequential(lint, test)\n"
	)
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="2"):
		with patch("sys.argv", ["camas"]):
			main()
	captured = capsys.readouterr()
	assert "Available tasks from" in captured.out
	assert "tasks.py" in captured.out
	assert "ci" in captured.out
	assert "lint, test" in captured.out
	assert "lint" in captured.out
	assert "ruff check ." in captured.out
	assert "task or expression is required" in captured.err


def test_list_flag_with_tasks(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	tasks_py = tmp_path / "tasks.py"
	tasks_py.write_text(
		"from camas import Parallel, Task\n"
		'a = Task("echo a")\n'
		'b = Task("echo b")\n'
		"both = Parallel(a, b)\n"
	)
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "--list"]):
			main()
	out = capsys.readouterr().out
	assert "both" in out
	assert "a | b" in out


def test_tree_flag_prints_expanded_trees(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	tasks_py = tmp_path / "tasks.py"
	tasks_py.write_text(
		"from camas import Parallel, Task\n"
		'a = Task("echo a")\n'
		'b = Task("echo b")\n'
		"both = Parallel(a, b)\n"
	)
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "--tree"]):
			main()
	out = capsys.readouterr().out
	assert "Available tasks from" in out
	assert "echo a" in out
	assert "echo b" in out


def test_parse_effects_rejects_invalid_syntax() -> None:
	with pytest.raises(ValueError, match="invalid syntax"):
		parse_effects("Summary(unbalanced(")


def test_parse_effects_rejects_non_tuple() -> None:
	with pytest.raises(ValueError, match="must be a tuple"):
		parse_effects("Summary()")


def test_parse_effects_rejects_unknown_effect() -> None:
	with pytest.raises(ValueError, match="unsupported syntax"):
		parse_effects("(Bogus(),)")


def test_parse_effects_rejects_non_effect_value() -> None:
	with pytest.raises(ValueError, match="expected an Effect"):
		parse_effects("(SummaryOptions(),)")


def test_dispatch_rejects_bad_effects(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="2"):
		with patch("sys.argv", ["camas", "--effects=nope(", 'Task("echo hi")']):
			main()
	assert "--effects" in capsys.readouterr().err


def test_parser_has_expression_arg() -> None:
	parser = build_parser()
	args = parser.parse_args(['Task("echo hi")'])
	assert args.expression == 'Task("echo hi")'
	assert args.dry_run is False


def test_parser_dry_run_flag() -> None:
	parser = build_parser()
	args = parser.parse_args(["--dry-run", 'Task("echo hi")'])
	assert args.dry_run is True


def test_dry_run_prints_tree(capsys: pytest.CaptureFixture[str]) -> None:
	with pytest.raises(SystemExit, match="0"):
		with patch(
			"sys.argv",
			["camas", "--dry-run", 'Parallel(Task("echo a"),Task("echo b"))'],
		):
			main()
	captured = capsys.readouterr()
	assert "echo a" in captured.out
	assert "echo b" in captured.out


def test_dry_run_matrix(capsys: pytest.CaptureFixture[str]) -> None:
	with pytest.raises(SystemExit, match="0"):
		with patch(
			"sys.argv",
			[
				"camas",
				"--dry-run",
				'Parallel(Task("test {PY}"),matrix={"PY": ("3.12", "3.13")})',
			],
		):
			main()
	captured = capsys.readouterr()
	assert "3.12" in captured.out
	assert "3.13" in captured.out


def test_successful_execution() -> None:
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", 'Task(("python", "-c", "pass"), name="ok")']):
			main()


def test_failed_execution() -> None:
	with pytest.raises(SystemExit, match="1"):
		with patch(
			"sys.argv", ["camas", 'Task(("python", "-c", "raise SystemExit(1)"), name="fail")']
		):
			main()


def test_invalid_expression() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression("not valid +++")


def test_unknown_type() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression('Foo(tasks=(Task("a"),))')


def test_bare_string_coerces_to_task() -> None:
	from camas import Task

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


def test_task_summary_leaf_str_cmd() -> None:
	assert task_summary(Task("ruff check ."), frozenset()) == "ruff check ."


def test_task_summary_leaf_tuple_cmd() -> None:
	assert task_summary(Task(("python", "-c", "pass")), frozenset()) == "python -c pass"


def test_task_summary_sequential() -> None:
	assert task_summary(Sequential(Task("a"), Task("b")), frozenset()) == "a, b"


def test_task_summary_parallel() -> None:
	assert task_summary(Parallel(Task("a"), Task("b")), frozenset()) == "a | b"


def test_task_summary_parallel_inside_sequential_no_parens() -> None:
	assert (
		task_summary(Sequential(Task("a"), Parallel(Task("b"), Task("c"))), frozenset())
		== "a, b | c"
	)


def test_task_summary_sequential_inside_parallel_parens() -> None:
	assert (
		task_summary(Parallel(Sequential(Task("a"), Task("b")), Task("c")), frozenset())
		== "(a, b) | c"
	)


def test_task_summary_named_ref_used_for_children() -> None:
	lint = Task("ruff check .", name="lint")
	test = Task("pytest", name="test")
	tree = Sequential(lint, test, name="ci")
	assert task_summary(tree, frozenset({"lint", "test", "ci"})) == "lint, test"


def test_task_summary_named_ref_in_parallel_no_parens() -> None:
	"""A named Sequential reference inside a Parallel renders as a bare name (no parens)."""
	fix = Sequential(Task("a", name="lint_fix"), Task("b", name="format"), name="fix")
	tree = Parallel(fix, Task("c", name="test"))
	assert (
		task_summary(tree, frozenset({"fix", "lint_fix", "format", "test"})) == "fix | test"
	)


def test_task_summary_root_name_does_not_self_reference() -> None:
	"""The root being rendered is not collapsed to its own name."""
	assert task_summary(Task("echo hi", name="greet"), frozenset({"greet"})) == "echo hi"


def test_print_listing_empty(capsys: pytest.CaptureFixture[str]) -> None:
	print_task_summary_listing({}, None)
	out = capsys.readouterr().out
	assert "No tasks file found" in out


def test_print_listing_with_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
	from camas import TaskNode

	source = tmp_path / "tasks.py"
	source.touch()
	lint = Task("ruff check .", name="lint")
	test = Task("pytest", name="test")
	tasks: dict[str, TaskNode] = {
		"lint": lint,
		"test": test,
		"ci": Sequential(lint, test, name="ci"),
	}
	print_task_summary_listing(tasks, source)
	out = capsys.readouterr().out
	assert f"Available tasks from {source}:" in out
	assert "ci" in out
	assert "lint, test" in out
	assert "lint" in out
	assert "ruff check ." in out


def test_print_listing_matrix_annotation(capsys: pytest.CaptureFixture[str]) -> None:
	matrix = Sequential(Task("uv sync"), name="matrix", matrix={"PY": ("3.12", "3.13")})
	print_task_summary_listing({"matrix": matrix}, None)
	out = capsys.readouterr().out
	assert "[matrix: PY]" in out
