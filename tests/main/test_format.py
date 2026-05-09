# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from camas import Parallel, Sequential, Task
from camas.core.task import TaskNode
from camas.main.format import (
	first_line_doc,
	format_annotation,
	format_available_effects,
	format_default,
	format_signature,
	format_task_summary_listing,
	format_try_hint,
	print_available_effects,
	print_task_summary_listing,
	print_task_trees,
	task_summary,
)
from camas.main.mypyc import MISSING


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
	assert task_summary(tree, frozenset({"fix", "lint_fix", "format", "test"})) == "fix | test"


def test_task_summary_root_name_does_not_self_reference() -> None:
	"""The root being rendered is not collapsed to its own name."""
	assert task_summary(Task("echo hi", name="greet"), frozenset({"greet"})) == "echo hi"


def test_print_listing_empty(capsys: pytest.CaptureFixture[str]) -> None:
	print_task_summary_listing({}, None)
	out = capsys.readouterr().out
	assert "No tasks file found" in out


def test_print_listing_with_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
	assert "[matrix: PY×2 (3.12..3.13)]" in out


def test_print_listing_matrix_single_value_annotation(capsys: pytest.CaptureFixture[str]) -> None:
	matrix = Sequential(Task("uv sync"), name="matrix", matrix={"PY": ("3.13",)})
	print_task_summary_listing({"matrix": matrix}, None)
	out = capsys.readouterr().out
	assert "[matrix: PY=3.13]" in out


def test_print_listing_matrix_multi_axis_annotation(capsys: pytest.CaptureFixture[str]) -> None:
	t = Sequential(
		Task("x"),
		name="ci",
		matrix={"DB": ("sqlite", "postgres"), "OPT": ("debug",)},
	)
	print_task_summary_listing({"ci": t}, None)
	out = capsys.readouterr().out
	assert "[matrix: DB×2 (sqlite..postgres) OPT=debug]" in out


def test_print_listing_help_replaces_command_body(capsys: pytest.CaptureFixture[str]) -> None:
	t = Task(
		"uv run pytest --doctest-modules -m 'not slow' --cov --cov-report=term",
		help="Run the full test suite with coverage",
	)
	print_task_summary_listing({"coverage": t}, None)
	out = capsys.readouterr().out
	assert "Run the full test suite with coverage" in out
	assert "uv run pytest --doctest-modules" not in out


def test_print_listing_help_on_group(capsys: pytest.CaptureFixture[str]) -> None:
	t = Parallel(Task("uv run mypy ."), Task("uv run pyright ."), help="Type-check in parallel")
	print_task_summary_listing({"typecheck": t}, None)
	out = capsys.readouterr().out
	assert "Type-check in parallel" in out
	assert "uv run mypy" not in out


def test_print_listing_no_help_falls_back_to_command(capsys: pytest.CaptureFixture[str]) -> None:
	t = Task("ruff check .", name="lint")
	print_task_summary_listing({"lint": t}, None)
	out = capsys.readouterr().out
	assert "ruff check ." in out


def test_format_task_summary_listing_no_tasks_with_source(tmp_path: Path) -> None:
	source = tmp_path / "tasks.py"
	out = format_task_summary_listing({}, source, color=False)
	assert "No tasks defined in" in out
	assert str(source) in out


def test_format_task_summary_listing_no_tasks_no_source() -> None:
	out = format_task_summary_listing({}, None, color=False)
	assert "No tasks file found" in out
	assert "tasks.py" in out


def test_format_task_summary_listing_no_source_header() -> None:
	out = format_task_summary_listing({"a": Task("x", name="a")}, None, color=False)
	assert out.startswith("Tasks:")


def test_format_task_summary_listing_with_color() -> None:
	out = format_task_summary_listing(
		{"a": Sequential(Task("x"), Task("y"), name="a")},
		None,
		color=True,
	)
	assert "\033[" in out


def test_print_task_trees_empty(capsys: pytest.CaptureFixture[str]) -> None:
	print_task_trees({}, None)
	assert "No tasks file found" in capsys.readouterr().out


def test_print_task_trees_with_tasks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
	source = tmp_path / "tasks.py"
	source.touch()
	print_task_trees({"a": Task("echo a", name="a")}, source)
	assert "echo a" in capsys.readouterr().out


def test_format_try_hint_plain() -> None:
	out = format_try_hint(color=False)
	assert "Try:" in out
	assert "( ) → Sequential" in out
	assert "{ } → Parallel" in out
	assert "Summary" in out


def test_format_try_hint_colored() -> None:
	assert "\033[" in format_try_hint(color=True)


def test_format_available_effects_default_color() -> None:
	out = format_available_effects()
	assert "Summary" in out
	assert "Termtree" in out


def test_format_available_effects_no_color() -> None:
	out = format_available_effects(color=False)
	assert "\033[" not in out
	assert "Summary" in out


def test_format_available_effects_empty(monkeypatch: pytest.MonkeyPatch) -> None:
	from collections.abc import Mapping

	from camas.main import format as format_mod

	def empty_available(
		_scope: Mapping[str, Any] = {},
	) -> tuple[Mapping[str, Any], tuple[tuple[str, Any], ...]]:
		return {}, ()

	monkeypatch.setattr(format_mod, "available_effects", empty_available)
	assert format_available_effects() == ""


def test_print_available_effects(capsys: pytest.CaptureFixture[str]) -> None:
	print_available_effects()
	assert "Summary" in capsys.readouterr().out


def test_first_line_doc_with_doc() -> None:
	class C:
		"""First line.

		Second line.
		"""

	assert first_line_doc(C) == "First line."


def test_first_line_doc_empty() -> None:
	class C: ...

	assert first_line_doc(C) == ""


def test_first_line_doc_blank_then_text() -> None:
	class C:
		__doc__ = "\n\n  real line\n"

	assert first_line_doc(C) == "real line"


def test_format_annotation_any() -> None:
	from typing import Any as _Any

	assert format_annotation(_Any, color=False) == "Any"


def test_format_annotation_concrete_type() -> None:
	assert format_annotation(int, color=False) == "int"


def test_format_annotation_string_repr() -> None:
	assert format_annotation("camas.foo.Bar | None", color=False) == "Bar | None"


def test_format_annotation_colored() -> None:
	assert "\033[" in format_annotation(int, color=True)


def test_format_default_missing() -> None:
	assert format_default(MISSING, color=False) == ""


def test_format_default_string() -> None:
	assert format_default("hi", color=False) == " = 'hi'"


def test_format_default_other() -> None:
	assert format_default(3, color=False) == " = 3"


def test_format_default_colored() -> None:
	assert "\033[" in format_default(None, color=True)


def test_format_signature_no_fields() -> None:
	class Empty: ...

	out = format_signature(Empty, indent="  ", color=False)
	assert out == ["  Empty()"]


def test_format_signature_with_fields_and_nested() -> None:
	from camas.effect.summary import Summary

	out = "\n".join(format_signature(Summary, indent="", color=False))
	assert "Summary(" in out
	assert "SummaryOptions(" in out
	assert "Auto()" in out
