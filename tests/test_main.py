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
	MISSING,
	apply_passthrough,
	build_parser,
	discover_effects,
	dispatch,
	first_line_doc,
	format_annotation,
	format_available_effects,
	format_default,
	format_signature,
	format_task_summary_listing,
	format_try_hint,
	main,
	parse_effects,
	parse_expression,
	print_available_effects,
	print_task_summary_listing,
	print_task_trees,
	signature_fields,
	split_passthrough,
	task_summary,
)


def test_help(tmp_path: Path) -> None:
	result = subprocess.run(
		[sys.executable, "-m", "camas", "--help"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		encoding="utf-8",
	)
	assert result.returncode == 0
	assert "expression" in result.stdout


def test_version(tmp_path: Path) -> None:
	result = subprocess.run(
		[sys.executable, "-m", "camas", "--version"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		encoding="utf-8",
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


def test_print_task_help_with_axes(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("echo {PY}"), matrix={"PY": ("3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch({"check": task}, ["check", "--help"])
	out = capsys.readouterr().out
	assert "Matrix axes" in out
	assert "--PY" in out


def test_dispatch_per_axis_flag_overrides(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("echo {PY}"), matrix={"PY": ("3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch({"check": task}, ["--dry-run", "check", "--PY", "3.13"])
	out = capsys.readouterr().out
	assert "[PY=3.13]" in out
	assert "[PY=3.12]" not in out


def test_dispatch_matrix_flag_overrides(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("echo {PY}"), matrix={"PY": ("3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch({"check": task}, ["--dry-run", "check", "--matrix", "PY=3.13"])
	out = capsys.readouterr().out
	assert "[PY=3.13]" in out
	assert "[PY=3.12]" not in out


def test_dispatch_matrix_flag_bad_syntax_errors(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("echo {PY}"), matrix={"PY": ("3.12",)})
	with pytest.raises(SystemExit, match="2"):
		dispatch({"check": task}, ["check", "--matrix", "noequals"])
	assert "--matrix expects KEY=VAL" in capsys.readouterr().err


def test_dispatch_per_axis_flag_empty_value_errors(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("echo {PY}"), matrix={"PY": ("3.12",)})
	with pytest.raises(SystemExit, match="2"):
		dispatch({"check": task}, ["check", "--PY", ""])
	assert "--PY" in capsys.readouterr().err


def test_dispatch_matrix_unknown_axis_errors(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("echo {PY}"), matrix={"PY": ("3.12",)})
	with pytest.raises(SystemExit, match="2"):
		dispatch({"check": task}, ["check", "--matrix", "XX=1"])
	assert "unknown matrix axis" in capsys.readouterr().err


def test_dispatch_skips_reserved_axis_name_for_auto_flag(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = Parallel(Task("echo {matrix}"), matrix={"matrix": ("a", "b")})
	with pytest.raises(SystemExit, match="0"):
		dispatch({"check": task}, ["--dry-run", "check", "--matrix", "matrix=b"])
	out = capsys.readouterr().out
	assert "[matrix=b]" in out
	assert "[matrix=a]" not in out


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
	assert task_summary(tree, frozenset({"fix", "lint_fix", "format", "test"})) == "fix | test"


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
	assert "[matrix: PY×2 (3.12..3.13)]" in out


def test_print_listing_matrix_single_value_annotation(
	capsys: pytest.CaptureFixture[str],
) -> None:
	matrix = Sequential(Task("uv sync"), name="matrix", matrix={"PY": ("3.13",)})
	print_task_summary_listing({"matrix": matrix}, None)
	out = capsys.readouterr().out
	assert "[matrix: PY=3.13]" in out


def test_print_listing_matrix_multi_axis_annotation(
	capsys: pytest.CaptureFixture[str],
) -> None:
	t = Sequential(
		Task("x"),
		name="ci",
		matrix={"DB": ("sqlite", "postgres"), "OPT": ("debug",)},
	)
	print_task_summary_listing({"ci": t}, None)
	out = capsys.readouterr().out
	assert "[matrix: DB×2 (sqlite..postgres) OPT=debug]" in out


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


def test_discover_effects_returns_summary_termtree() -> None:
	constructors, effects = discover_effects()
	names = {n for n, _ in effects}
	assert names == {"Summary", "Termtree"}
	assert "SummaryOptions" in constructors
	assert "Auto" in constructors
	assert "Fixed" in constructors


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
	from typing import Any

	from camas import main as main_mod

	def empty_discover() -> tuple[Mapping[str, Any], tuple[tuple[str, Any], ...]]:
		return {}, ()

	monkeypatch.setattr(main_mod, "discover_effects", empty_discover)
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


def test_signature_fields_namedtuple_with_defaults() -> None:
	from camas.effect.summary import SummaryOptions

	fields = signature_fields(SummaryOptions)
	names = [n for n, _, _, _ in fields]
	assert names == ["term_width", "show_passing"]


def test_signature_fields_namedtuple_without_defaults() -> None:
	from camas.effect.summary import Fixed

	fields = signature_fields(Fixed)
	assert [(n, d is MISSING) for n, _, _, d in fields] == [("columns", True)]


def test_signature_fields_no_init() -> None:
	"""builtins.object has no inspectable params for our purposes."""

	class Empty: ...

	# Empty has no __init__ params except self which inspect.signature elides.
	assert signature_fields(Empty) == []


def test_signature_fields_via_inspect_signature() -> None:
	"""Ordinary class (not NamedTuple, not dataclass) → inspect.signature path."""

	class C:
		def __init__(self, a: int, b: str = "x") -> None: ...

	fields = signature_fields(C)
	names = [n for n, _, _, _ in fields]
	assert names == ["a", "b"]


def test_signature_fields_var_args() -> None:
	"""``*args``/``**kwargs`` are reported with their kinds."""

	class Tricky:
		def __init__(self, *args: int, **kwargs: int) -> None: ...

	fields = signature_fields(Tricky)
	assert any(name in {"args", "kwargs"} for name, _, _, _ in fields)


def test_signature_fields_dataclass() -> None:
	from dataclasses import dataclass, field

	@dataclass
	class D:
		x: int
		y: str = "hi"
		z: tuple[int, ...] = field(default=(), init=False)

	fields = signature_fields(D)
	names = [n for n, _, _, _ in fields]
	assert names == ["x", "y"]  # init=False excluded


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


def test_eval_value_constant_string() -> None:
	import ast as _ast

	from camas.main import eval_value

	node = _ast.parse('"hello"', mode="eval").body
	assert eval_value(node, {}) == "hello"


def test_eval_value_bare_name() -> None:
	import ast as _ast

	from camas.effect.summary import Summary
	from camas.main import eval_value

	node = _ast.parse("Summary", mode="eval").body
	assert eval_value(node, {"Summary": Summary}) is Summary


def test_dispatch_effects_no_value_lists(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "--effects"]):
			main()
	assert "Available Effects:" in capsys.readouterr().out


def test_help_includes_tasks_and_effects(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	(tmp_path / "tasks.py").write_text('from camas import Task\nlint = Task("ruff check .")\n')
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "--help"]):
			main()
	out = capsys.readouterr().out
	assert "Available tasks from" in out
	assert "lint" in out
	assert "Available Effects:" in out
	assert "Try:" in out


def test_isinstance_effect_branch_in_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Cover the plugin path: a module-scope Effect *instance* (not class)."""
	import pkgutil as _pkgutil
	import sys as _sys
	import types
	from collections.abc import Iterator
	from pkgutil import ModuleInfo

	import camas.effect as effect_pkg
	from camas.effect.summary import Summary
	from camas.main import discover_effects as _discover

	plugin = types.ModuleType("camas.effect._test_plugin")
	setattr(plugin, "my_effect", Summary())
	monkeypatch.setitem(_sys.modules, "camas.effect._test_plugin", plugin)

	real_iter = _pkgutil.iter_modules

	finders = [info.module_finder for info in real_iter(list(effect_pkg.__path__))]

	def fake_iter(path: list[str]) -> Iterator[ModuleInfo]:
		yield from real_iter(path)
		yield ModuleInfo(finders[0], "_test_plugin", False)

	monkeypatch.setattr(_pkgutil, "iter_modules", fake_iter)
	_discover.cache_clear()
	try:
		_, effects = _discover()
	finally:
		_discover.cache_clear()
	names = {n for n, _ in effects}
	assert "my_effect" in names


def test_reachable_classes_seen_short_circuit() -> None:
	from camas.effect.summary import Summary
	from camas.main import reachable_classes

	seen = {Summary}
	out = reachable_classes(Summary, seen)
	assert out is seen  # already seen → returns immediately


def test_build_parser_format_help_no_tasks_no_effects(monkeypatch: pytest.MonkeyPatch) -> None:
	from collections.abc import Mapping
	from typing import Any

	from camas import main as main_mod

	def empty_discover() -> tuple[Mapping[str, Any], tuple[tuple[str, Any], ...]]:
		return {}, ()

	monkeypatch.setattr(main_mod, "discover_effects", empty_discover)
	parser = build_parser()
	out = parser.format_help()
	assert "Available tasks" not in out
	assert "Available Effects" not in out
	assert "Try:" in out


def test_signature_fields_from_source_namedtuple() -> None:
	from camas.effect.summary import SummaryOptions
	from camas.main import signature_fields_from_source

	out = signature_fields_from_source(SummaryOptions)
	assert out is not None
	names = [n for n, _, _, _ in out]
	assert names == ["term_width", "show_passing"]


def test_signature_fields_from_source_regular_class() -> None:
	from camas.effect.summary import Summary, SummaryOptions
	from camas.main import signature_fields_from_source

	out = signature_fields_from_source(Summary)
	assert out is not None
	assert len(out) == 1
	name, _kind, annot, default = out[0]
	assert name == "options"
	# Resolved against module namespace, so the union resolves to a real type.
	assert "SummaryOptions" in str(annot)
	assert SummaryOptions.__name__ in str(annot)
	assert default is None


def test_signature_fields_from_source_module_not_loaded() -> None:
	from camas.main import signature_fields_from_source

	class Stranded: ...

	Stranded.__module__ = "no_such_module_xyz"
	assert signature_fields_from_source(Stranded) is None


def test_signature_fields_from_source_no_file(monkeypatch: pytest.MonkeyPatch) -> None:
	import sys as _sys
	import types

	from camas.main import signature_fields_from_source

	mod = types.ModuleType("fake_no_file")
	# No __file__ attribute set on the module
	monkeypatch.setitem(_sys.modules, "fake_no_file", mod)

	class C: ...

	C.__module__ = "fake_no_file"
	assert signature_fields_from_source(C) is None


def test_signature_fields_from_source_missing_py(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import sys as _sys
	import types

	from camas.main import signature_fields_from_source

	# Module __file__ points at a fake .so; the sibling .py doesn't exist.
	fake_so = tmp_path / "ghost.cpython-314-x86_64-linux-gnu.so"
	fake_so.write_bytes(b"")
	mod = types.ModuleType("ghost")
	mod.__file__ = str(fake_so)
	monkeypatch.setitem(_sys.modules, "ghost", mod)

	class C: ...

	C.__module__ = "ghost"
	assert signature_fields_from_source(C) is None


def test_signature_fields_from_source_class_not_in_file(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import sys as _sys
	import types

	from camas.main import signature_fields_from_source

	source = tmp_path / "tinymod.py"
	source.write_text("class Other: pass\n")
	mod = types.ModuleType("tinymod")
	mod.__file__ = str(source)
	monkeypatch.setitem(_sys.modules, "tinymod", mod)

	class Missing: ...

	Missing.__module__ = "tinymod"
	assert signature_fields_from_source(Missing) is None


def test_signature_fields_from_source_no_init(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import sys as _sys
	import types

	from camas.main import signature_fields_from_source

	source = tmp_path / "noinit.py"
	source.write_text("class C:\n    pass\n")
	mod = types.ModuleType("noinit")
	mod.__file__ = str(source)
	monkeypatch.setitem(_sys.modules, "noinit", mod)

	class C: ...

	C.__module__ = "noinit"
	assert signature_fields_from_source(C) is None


def test_signature_fields_from_source_parse_error(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import sys as _sys
	import types

	from camas.main import signature_fields_from_source

	source = tmp_path / "broken.py"
	source.write_text("def x(:\n")  # syntax error
	mod = types.ModuleType("broken")
	mod.__file__ = str(source)
	monkeypatch.setitem(_sys.modules, "broken", mod)

	class C: ...

	C.__module__ = "broken"
	assert signature_fields_from_source(C) is None


def test_signature_fields_from_source_with_required_arg(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Required (non-defaulted) ``__init__`` params return ``MISSING``."""
	import sys as _sys
	import types

	from camas.main import MISSING as _MISSING_LOCAL
	from camas.main import signature_fields_from_source

	source = tmp_path / "required.py"
	source.write_text(
		"class C:\n"
		"    def __init__(self, must_have: int, optional: str = 'x') -> None:\n"
		"        ...\n"
	)
	mod = types.ModuleType("required")
	mod.__file__ = str(source)
	monkeypatch.setitem(_sys.modules, "required", mod)

	class C: ...

	C.__module__ = "required"
	out = signature_fields_from_source(C)
	assert out is not None
	(must_have, optional) = out
	assert must_have[0] == "must_have" and must_have[3] is _MISSING_LOCAL
	assert optional[0] == "optional" and optional[3] == "x"


def test_signature_fields_falls_back_for_compiled_namedtuple(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""When ``get_type_hints`` returns an ``Any``/``type`` annotation (the mypyc
	fingerprint for NamedTuples), ``signature_fields`` falls through to the
	source-parse path."""
	import typing as _typing

	from camas.effect.summary import SummaryOptions
	from camas.main import signature_fields

	def fake_hints(_obj: object) -> dict[str, object]:
		# Mimic mypyc's mangled output: union annotations become bare ``type``.
		return {"term_width": type, "show_passing": bool}

	monkeypatch.setattr(_typing, "get_type_hints", fake_hints)
	out = signature_fields(SummaryOptions)
	# Source-parse should kick in and return the real annotations.
	annot_term = next(annot for n, _, annot, _ in out if n == "term_width")
	assert "Auto" in str(annot_term) or "Fixed" in str(annot_term)


def test_signature_fields_falls_back_for_compiled_class(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""When ``inspect.signature`` returns all-``Any`` annotations (the mypyc
	fingerprint for regular classes), ``signature_fields`` falls through to
	source-parse."""
	import inspect as _inspect

	from camas.effect.summary import Summary
	from camas.main import signature_fields

	def fake_signature(_obj: object) -> _inspect.Signature:
		# Single param with no annotation, default None — what mypyc surfaces.
		return _inspect.Signature(
			parameters=[
				_inspect.Parameter(
					"options", _inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
				)
			]
		)

	monkeypatch.setattr(_inspect, "signature", fake_signature)
	out = signature_fields(Summary)
	annot = next(annot for n, _, annot, _ in out if n == "options")
	assert "SummaryOptions" in str(annot)


def test_signature_fields_namedtuple_fallback_returns_none(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""When the NamedTuple inspect path returns ``Any``/``type`` and the
	source-parse fallback also fails (class not loaded as module), keep the
	original inspected fields rather than crashing."""
	import typing as _typing
	from typing import NamedTuple as _NT

	from camas.main import signature_fields

	class Stranded(_NT):
		x: int = 1

	def fake_hints(_obj: object) -> dict[str, object]:
		return {"x": type}

	# Force the Any-fingerprint and force source-parse to give up.
	monkeypatch.setattr(_typing, "get_type_hints", fake_hints)
	Stranded.__module__ = "no_such_module_xyz"
	out = signature_fields(Stranded)
	assert [n for n, _, _, _ in out] == ["x"]


def test_signature_fields_class_fallback_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Same defensive path for ordinary classes."""
	import inspect as _inspect

	from camas.main import signature_fields

	class Stranded:
		def __init__(self, x: int = 1) -> None: ...

	def fake_signature(_o: object) -> _inspect.Signature:
		return _inspect.Signature(
			parameters=[
				_inspect.Parameter("x", _inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1)
			]
		)

	monkeypatch.setattr(_inspect, "signature", fake_signature)
	Stranded.__module__ = "no_such_module_xyz"
	out = signature_fields(Stranded)
	assert [n for n, _, _, _ in out] == ["x"]


def test_resolve_in_namespace_evaluates() -> None:
	import ast as _ast

	from camas.main import resolve_in_namespace

	node = _ast.parse("1 + 2", mode="eval").body
	assert resolve_in_namespace(node, {}) == 3


def test_resolve_in_namespace_falls_back_to_string() -> None:
	import ast as _ast

	from camas.main import resolve_in_namespace

	node = _ast.parse("undefined_name_xyz", mode="eval").body
	assert resolve_in_namespace(node, {}) == "undefined_name_xyz"


def test_signature_fields_namedtuple_unresolvable_hints(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	from camas import main as main_mod
	from camas.effect.summary import Auto

	def boom(_obj: object) -> dict[str, object]:
		raise NameError("forward ref")

	monkeypatch.setattr(main_mod, "signature_fields", signature_fields)
	# Patch get_type_hints inside signature_fields' import scope by monkeypatching typing.
	import typing as _typing

	import camas.main as _m

	monkeypatch.setattr(_typing, "get_type_hints", boom)
	fields = _m.signature_fields(Auto)
	# Auto has no fields anyway, but the get_type_hints exception path should be taken.
	assert fields == []


def test_split_passthrough_no_separator() -> None:
	split = split_passthrough(["a", "b"])
	assert split.head == ("a", "b")
	assert split.passthrough == ()


def test_split_passthrough_with_separator() -> None:
	split = split_passthrough(["mytask", "--", "-v", "-k", "x"])
	assert split.head == ("mytask",)
	assert split.passthrough == ("-v", "-k", "x")


def test_split_passthrough_empty_passthrough() -> None:
	split = split_passthrough(["mytask", "--"])
	assert split.head == ("mytask",)
	assert split.passthrough == ()


def test_split_passthrough_only_first_separator() -> None:
	split = split_passthrough(["mytask", "--", "--", "x"])
	assert split.passthrough == ("--", "x")


def test_apply_passthrough_str_cmd_stays_string() -> None:
	assert apply_passthrough(Task("pytest"), ("-v",)) == Task("pytest -v")


def test_apply_passthrough_str_cmd_preserves_quoting() -> None:
	"""String cmds keep their original quoting; passthrough args are shell-joined."""
	assert apply_passthrough(Task("git commit -m 'big msg'"), ("--no-verify",)) == Task(
		"git commit -m 'big msg' --no-verify"
	)


def test_apply_passthrough_str_cmd_quotes_passthrough_with_spaces() -> None:
	assert apply_passthrough(Task("pytest"), ("-k", "a b")) == Task("pytest -k 'a b'")


def test_apply_passthrough_tuple_cmd_preserves_name_env_cwd() -> None:
	original = Task(("pytest",), name="test", env={"X": "1"}, cwd=Path("/tmp"))
	assert apply_passthrough(original, ("-v",)) == Task(
		("pytest", "-v"), name="test", env={"X": "1"}, cwd=Path("/tmp")
	)


def test_apply_passthrough_str_cmd_preserves_name_env_cwd() -> None:
	original = Task("pytest", name="test", env={"X": "1"}, cwd=Path("/tmp"))
	assert apply_passthrough(original, ("-v",)) == Task(
		"pytest -v", name="test", env={"X": "1"}, cwd=Path("/tmp")
	)


def test_apply_passthrough_rejects_sequential() -> None:
	with pytest.raises(ValueError, match="only apply to Task, got Sequential"):
		apply_passthrough(Sequential(Task("a"), Task("b")), ("-v",))


def test_apply_passthrough_rejects_parallel() -> None:
	with pytest.raises(ValueError, match="only apply to Task, got Parallel"):
		apply_passthrough(Parallel(Task("a"), Task("b")), ("-v",))


def test_passthrough_appended_to_inline_task(capsys: pytest.CaptureFixture[str]) -> None:
	with pytest.raises(SystemExit, match="0"):
		with patch(
			"sys.argv",
			["camas", "--dry-run", 'Task(("python", "-c", "print(1)"))', "--", "extra"],
		):
			main()
	assert "extra" in capsys.readouterr().out


def test_passthrough_to_named_task(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	(tmp_path / "tasks.py").write_text(
		"from camas import Task\ntest = Task('pytest')\n",
	)
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "--dry-run", "test", "--", "-v", "-k", "x"]):
			main()
	out = capsys.readouterr().out
	assert "pytest" in out and "-v" in out and "-k" in out and "x" in out


def test_passthrough_rejects_sequential_task(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	(tmp_path / "tasks.py").write_text(
		"from camas import Sequential, Task\nci = Sequential(Task('a'), Task('b'))\n",
	)
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="2"):
		with patch("sys.argv", ["camas", "ci", "--", "-v"]):
			main()
	assert "only apply to Task" in capsys.readouterr().err


def test_passthrough_help_after_separator_not_consumed(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""``camas test -- --help`` should treat ``--help`` as pass-through, not show task help."""
	(tmp_path / "tasks.py").write_text(
		"from camas import Task\ntest = Task(('python', '-c', 'pass'))\n",
	)
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "--dry-run", "test", "--", "--help"]):
			main()
	out = capsys.readouterr().out
	assert "--help" in out
	assert "usage:" not in out


def test_signature_fields_signature_raises(monkeypatch: pytest.MonkeyPatch) -> None:
	"""inspect.signature raises ValueError for some C-implemented callables."""
	import inspect as _inspect

	class Plain: ...

	def boom(_obj: object) -> object:
		raise ValueError("no signature")

	monkeypatch.setattr(_inspect, "signature", boom)
	assert signature_fields(Plain) == []
