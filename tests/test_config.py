# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from camas import Parallel, Sequential, Task, TaskNode
from camas.main import (
	Ref,
	dispatch_arg,
	find_pyproject,
	load_tasks,
	main,
	parse_expression,
	parse_task_value,
	resolve_refs,
	run_cli,
)


def _par(
	tasks: tuple[TaskNode | Ref, ...],
	name: str | None = None,
	matrix: dict[str, tuple[str, ...]] | None = None,
) -> Parallel:
	return Parallel(tasks=cast(tuple[TaskNode, ...], tasks), name=name, matrix=matrix)


def _seq(
	tasks: tuple[TaskNode | Ref, ...],
	name: str | None = None,
	matrix: dict[str, tuple[str, ...]] | None = None,
) -> Sequential:
	return Sequential(tasks=cast(tuple[TaskNode, ...], tasks), name=name, matrix=matrix)


def test_parse_task_value_bare_string() -> None:
	assert parse_task_value("ruff check .") == Task("ruff check .")


def test_parse_task_value_with_spaces_and_flags() -> None:
	assert parse_task_value("pytest -x -q tests/") == Task("pytest -x -q tests/")


def test_parse_task_value_explicit_task() -> None:
	assert parse_task_value('Task("pytest", name="test")') == Task("pytest", name="test")


def test_parse_task_value_parallel() -> None:
	assert parse_task_value('Parallel(tasks=(Task("a"), Task("b")))') == Parallel(
		tasks=(Task("a"), Task("b"))
	)


def test_parse_task_value_sequential() -> None:
	assert parse_task_value('Sequential(tasks=(Task("a"), Task("b")))') == Sequential(
		tasks=(Task("a"), Task("b"))
	)


def test_parse_task_value_bare_ref_inside_expr() -> None:
	assert parse_task_value("Parallel(tasks=(a, b))") == _par((Ref("a"), Ref("b")))


def test_parse_task_value_explicit_ref() -> None:
	assert parse_task_value('Sequential(tasks=(Ref("lint"), Ref("test")))') == _seq(
		(Ref("lint"), Ref("test"))
	)


def test_parse_task_value_matrix_preserved() -> None:
	assert parse_task_value(
		'Parallel(tasks=(Task("t {PY}"),), matrix={"PY": ("3.12", "3.13")})'
	) == Parallel(tasks=(Task("t {PY}"),), matrix={"PY": ("3.12", "3.13")})


def test_parse_task_value_invalid_syntax() -> None:
	with pytest.raises(ValueError, match="invalid expression"):
		parse_task_value("Task(unbalanced(")


def test_parse_task_value_rejects_tuple_top_level() -> None:
	with pytest.raises(ValueError, match="expected Task"):
		parse_task_value('Task("a"), Task("b")')


def test_resolve_refs_passthrough_task() -> None:
	t = Task("x")
	assert resolve_refs(t, {}, frozenset()) is t


def test_resolve_refs_simple() -> None:
	assert resolve_refs(Ref("a"), {"a": Task("hi")}, frozenset()) == Task("hi")


def test_resolve_refs_nested_in_parallel() -> None:
	pre: dict[str, TaskNode | Ref] = {
		"lint": Task("ruff ."),
		"test": Task("pytest"),
	}
	tree = _par((Ref("lint"), Ref("test")))
	assert resolve_refs(tree, pre, frozenset()) == Parallel(tasks=(Task("ruff ."), Task("pytest")))


def test_resolve_refs_nested_in_sequential_preserves_matrix() -> None:
	pre: dict[str, TaskNode | Ref] = {"build": Task("make")}
	tree = _seq((Ref("build"),), matrix={"X": ("1", "2")})
	result = resolve_refs(tree, pre, frozenset())
	assert result == Sequential(tasks=(Task("make"),), matrix={"X": ("1", "2")})


def test_resolve_refs_unknown() -> None:
	with pytest.raises(ValueError, match="unknown task ref 'missing'"):
		resolve_refs(Ref("missing"), {"a": Task("x")}, frozenset())


def test_resolve_refs_unknown_empty_defs() -> None:
	with pytest.raises(ValueError, match="known: none"):
		resolve_refs(Ref("x"), {}, frozenset())


def test_resolve_refs_self_cycle() -> None:
	pre: dict[str, TaskNode | Ref] = {"a": Ref("a")}
	with pytest.raises(ValueError, match="cycle"):
		resolve_refs(Ref("a"), pre, frozenset())


def test_resolve_refs_mutual_cycle() -> None:
	pre: dict[str, TaskNode | Ref] = {"a": Ref("b"), "b": Ref("a")}
	with pytest.raises(ValueError, match="cycle"):
		resolve_refs(Ref("a"), pre, frozenset())


def test_resolve_refs_transitive_cycle() -> None:
	pre: dict[str, TaskNode | Ref] = {
		"a": _par((Ref("b"),)),
		"b": _seq((Ref("c"),)),
		"c": Ref("a"),
	}
	with pytest.raises(ValueError, match="cycle"):
		resolve_refs(Ref("a"), pre, frozenset())


def test_find_pyproject_in_cwd(tmp_path: Path) -> None:
	(tmp_path / "pyproject.toml").write_text("")
	assert find_pyproject(tmp_path) == tmp_path / "pyproject.toml"


def test_find_pyproject_walks_up(tmp_path: Path) -> None:
	(tmp_path / "pyproject.toml").write_text("")
	sub = tmp_path / "a" / "b" / "c"
	sub.mkdir(parents=True)
	assert find_pyproject(sub) == tmp_path / "pyproject.toml"


def test_find_pyproject_none(tmp_path: Path) -> None:
	sub = tmp_path / "nested"
	sub.mkdir()
	assert find_pyproject(sub) is None


def test_run_cli_collects_tasks_from_scope(capsys: pytest.CaptureFixture[str]) -> None:
	scope: dict[str, object] = {
		"lint": Task("ruff ."),
		"_private": Task("secret"),
		"other": 42,
	}
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["check.py", "--list"]):
			run_cli(scope)
	out = capsys.readouterr().out
	assert "lint" in out
	assert "_private" not in out
	assert "other" not in out


def test_load_tasks_full(tmp_path: Path) -> None:
	pyproject = tmp_path / "pyproject.toml"
	pyproject.write_text(
		"""
[tool.camas.tasks]
lint = "ruff check ."
mypy = "mypy ."
pyright = "pyright ."
typecheck = "Parallel(tasks=(mypy, pyright))"
test = "pytest"
ci = 'Sequential(tasks=(Ref("lint"), Ref("typecheck"), Ref("test")))'
"""
	)
	tasks = load_tasks(pyproject)
	assert tasks["lint"] == Task("ruff check .")
	assert tasks["typecheck"] == Parallel(tasks=(Task("mypy ."), Task("pyright .")))
	assert tasks["ci"] == Sequential(
		tasks=(
			Task("ruff check ."),
			Parallel(tasks=(Task("mypy ."), Task("pyright ."))),
			Task("pytest"),
		)
	)


def test_load_tasks_matrix(tmp_path: Path) -> None:
	pyproject = tmp_path / "pyproject.toml"
	pyproject.write_text(
		"""
[tool.camas.tasks]
tests = 'Parallel(tasks=(Task("pytest --python {PY}"),), matrix={"PY": ("3.12","3.13")})'
"""
	)
	tasks = load_tasks(pyproject)
	assert tasks["tests"] == Parallel(
		tasks=(Task("pytest --python {PY}"),),
		matrix={"PY": ("3.12", "3.13")},
	)


def test_load_tasks_empty(tmp_path: Path) -> None:
	pyproject = tmp_path / "pyproject.toml"
	pyproject.write_text('[project]\nname = "x"\nversion = "0"\n')
	assert load_tasks(pyproject) == {}


def test_load_tasks_rejects_non_table(tmp_path: Path) -> None:
	pyproject = tmp_path / "pyproject.toml"
	pyproject.write_text("[tool.camas]\ntasks = 'oops'\n")
	with pytest.raises(ValueError, match="must be a table"):
		load_tasks(pyproject)


def test_load_tasks_rejects_non_string_value(tmp_path: Path) -> None:
	pyproject = tmp_path / "pyproject.toml"
	pyproject.write_text("[tool.camas.tasks]\nbad = 42\n")
	with pytest.raises(ValueError, match="must be a string"):
		load_tasks(pyproject)


def test_load_tasks_surfaces_cycle(tmp_path: Path) -> None:
	pyproject = tmp_path / "pyproject.toml"
	pyproject.write_text(
		"""
[tool.camas.tasks]
a = "Ref(\\"b\\")"
b = "Ref(\\"a\\")"
"""
	)
	with pytest.raises(ValueError, match="cycle"):
		load_tasks(pyproject)


def test_dispatch_arg_bare_name() -> None:
	tasks = {"lint": Task("ruff .")}
	assert dispatch_arg("lint", tasks) == Task("ruff .")


def test_dispatch_arg_hyphenated_name() -> None:
	tasks = {"matrix-ci": Task("uv run camas ci")}
	assert dispatch_arg("matrix-ci", tasks) == Task("uv run camas ci")


def test_dispatch_arg_unknown_name() -> None:
	with pytest.raises(SystemExit, match="2"):
		dispatch_arg("nope", {"a": Task("x")})


def test_dispatch_arg_inline_expression() -> None:
	assert dispatch_arg('Task("echo hi")', {}) == Task("echo hi")


def test_dispatch_arg_inline_with_refs() -> None:
	tasks = {"lint": Task("ruff .")}
	assert dispatch_arg("Parallel(tasks=(lint,))", tasks) == Parallel(tasks=(Task("ruff ."),))


def test_dispatch_arg_inline_invalid_syntax() -> None:
	with pytest.raises(SystemExit, match="2"):
		dispatch_arg("not valid +++", {})


def test_parse_expression_non_task_with_tasks() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression('"just a string"', tasks={})


def test_parse_expression_with_undefined_ref() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression("Parallel(tasks=(undefined,))", tasks={})


def test_parse_expression_top_level_ref_without_tasks() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression('Ref("x")')


def test_parse_expression_rejects_positional_on_parallel() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression('Parallel((Task("a"),))')


def test_list_flag_no_tasks(
	tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0"\n')
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "--list"]):
			main()
	assert "(no tasks defined)" in capsys.readouterr().out


def test_list_flag_with_tasks(
	tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "pyproject.toml").write_text(
		'[tool.camas.tasks]\nlint = "ruff ."\ntest = "pytest"\n'
	)
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "--list"]):
			main()
	out = capsys.readouterr().out
	assert "lint:" in out
	assert "test:" in out
	assert "ruff ." in out
	assert "pytest" in out


def test_named_task_dry_run(
	tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "pyproject.toml").write_text(
		'[tool.camas.tasks]\nci = \'Parallel(tasks=(Task("echo a"), Task("echo b")))\'\n'
	)
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "--dry-run", "ci"]):
			main()
	out = capsys.readouterr().out
	assert "echo a" in out
	assert "echo b" in out


def test_named_task_from_subdirectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	(tmp_path / "pyproject.toml").write_text(
		'[tool.camas.tasks]\nhi = \'Task(("python", "-c", "pass"))\'\n'
	)
	sub = tmp_path / "a" / "b"
	sub.mkdir(parents=True)
	monkeypatch.chdir(sub)
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "hi"]):
			main()


def test_broken_pyproject_exits(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "pyproject.toml").write_text("[tool.camas.tasks]\nbad = 42\n")
	with pytest.raises(SystemExit, match="2"):
		with patch("sys.argv", ["camas", 'Task("x")']):
			main()
	assert "must be a string" in capsys.readouterr().err


def test_missing_expression_and_no_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="2"):
		with patch("sys.argv", ["camas"]):
			main()


def test_named_task_end_to_end(tmp_path: Path) -> None:
	(tmp_path / "pyproject.toml").write_text(
		'[tool.camas.tasks]\nok = \'Task(("python", "-c", "pass"))\'\n'
	)
	result = subprocess.run(
		[sys.executable, "-m", "camas", "ok"],
		cwd=tmp_path,
		capture_output=True,
	)
	assert result.returncode == 0
