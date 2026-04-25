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
	_assign_key_name,  # pyright: ignore[reportPrivateUsage]
	dispatch_arg,
	find_pyproject,
	find_tasks_py,
	load_tasks,
	load_tasks_from_py,
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
	cwd: Path | None = None,
) -> Parallel:
	return Parallel(*cast(tuple[TaskNode, ...], tasks), name=name, matrix=matrix, cwd=cwd)


def _seq(
	tasks: tuple[TaskNode | Ref, ...],
	name: str | None = None,
	matrix: dict[str, tuple[str, ...]] | None = None,
	cwd: Path | None = None,
) -> Sequential:
	return Sequential(*cast(tuple[TaskNode, ...], tasks), name=name, matrix=matrix, cwd=cwd)


def test_parse_task_value_bare_string() -> None:
	assert parse_task_value("ruff check .") == Task("ruff check .")


def test_parse_task_value_with_spaces_and_flags() -> None:
	assert parse_task_value("pytest -x -q tests/") == Task("pytest -x -q tests/")


def test_parse_task_value_explicit_task() -> None:
	assert parse_task_value('Task("pytest", name="test")') == Task("pytest", name="test")


def test_parse_task_value_parallel() -> None:
	assert parse_task_value('Parallel(Task("a"),Task("b"))') == Parallel(Task("a"), Task("b"))


def test_parse_task_value_sequential() -> None:
	assert parse_task_value('Sequential(Task("a"),Task("b"))') == Sequential(Task("a"), Task("b"))


def test_parse_task_value_bare_ref_inside_expr() -> None:
	assert parse_task_value("Parallel(a,b)") == _par((Ref("a"), Ref("b")))


def test_parse_task_value_explicit_ref() -> None:
	assert parse_task_value('Sequential(Ref("lint"),Ref("test"))') == _seq(
		(Ref("lint"), Ref("test"))
	)


def test_parse_task_value_matrix_preserved() -> None:
	assert parse_task_value('Parallel(Task("t {PY}"),matrix={"PY": ("3.12", "3.13")})') == Parallel(
		Task("t {PY}"), matrix={"PY": ("3.12", "3.13")}
	)


def test_parse_task_value_invalid_syntax() -> None:
	with pytest.raises(ValueError, match="invalid syntax"):
		parse_task_value("Task(unbalanced(")


def test_parse_task_value_top_level_tuple_coerces_to_sequential() -> None:
	assert parse_task_value('Task("a"), Task("b")') == Sequential(Task("a"), Task("b"))


def test_parse_task_value_top_level_set_coerces_to_parallel() -> None:
	result = parse_task_value('{Task("a"), Task("b")}')
	assert isinstance(result, Parallel)


def test_parse_task_value_top_level_paren_tuple_coerces_to_sequential() -> None:
	assert parse_task_value('(Task("a"), Task("b"))') == Sequential(Task("a"), Task("b"))


def test_parse_task_value_fluent_with_refs() -> None:
	assert parse_task_value("(lint, test)") == _seq((Ref("lint"), Ref("test")))


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
	assert resolve_refs(tree, pre, frozenset()) == Parallel(Task("ruff ."), Task("pytest"))


def test_resolve_refs_nested_in_sequential_preserves_matrix() -> None:
	pre: dict[str, TaskNode | Ref] = {"build": Task("make")}
	tree = _seq((Ref("build"),), matrix={"X": ("1", "2")})
	result = resolve_refs(tree, pre, frozenset())
	assert result == Sequential(Task("make"), matrix={"X": ("1", "2")})


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
		with patch("sys.argv", ["tasks.py", "--list"]):
			run_cli(scope)
	out = capsys.readouterr().out
	assert "lint: ruff ." in out
	assert "_private" not in out
	assert "secret" not in out
	assert "other" not in out


def test_load_tasks_full(tmp_path: Path) -> None:
	pyproject = tmp_path / "pyproject.toml"
	pyproject.write_text(
		"""
[tool.camas.tasks]
lint = "ruff check ."
mypy = "mypy ."
pyright = "pyright ."
typecheck = "Parallel(mypy, pyright)"
test = "pytest"
ci = 'Sequential(Ref("lint"), Ref("typecheck"), Ref("test"))'
"""
	)
	tasks = load_tasks(pyproject)
	assert tasks["lint"] == Task("ruff check .", name="lint")
	assert tasks["typecheck"] == Parallel(
		Task("mypy .", name="mypy"), Task("pyright .", name="pyright"), name="typecheck"
	)
	assert tasks["ci"] == Sequential(
		Task("ruff check .", name="lint"),
		Parallel(Task("mypy .", name="mypy"), Task("pyright .", name="pyright"), name="typecheck"),
		Task("pytest", name="test"),
		name="ci",
	)


def test_load_tasks_matrix(tmp_path: Path) -> None:
	pyproject = tmp_path / "pyproject.toml"
	pyproject.write_text(
		"""
[tool.camas.tasks]
tests = 'Parallel(Task("pytest --python {PY}"), matrix={"PY": ("3.12","3.13")})'
"""
	)
	tasks = load_tasks(pyproject)
	assert tasks["tests"] == Parallel(
		Task("pytest --python {PY}"), matrix={"PY": ("3.12", "3.13")}, name="tests"
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


def test_load_tasks_surfaces_syntax_error_with_task_name(tmp_path: Path) -> None:
	pyproject = tmp_path / "pyproject.toml"
	pyproject.write_text(
		"[tool.camas.tasks]\n"
		"matrix = '''\n"
		"Parallel(\n"
		'\tTask("t"),\n'
		'\tmatrix={"PY"=["3.12"]},\n'
		")\n"
		"'''\n"
	)
	with pytest.raises(ValueError) as exc:
		load_tasks(pyproject)
	msg = str(exc.value)
	assert "task 'matrix'" in msg
	assert "invalid syntax (line 3" in msg
	assert '"PY"=["3.12"]' in msg
	assert "^" in msg
	assert "\\n" not in msg


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
	assert dispatch_arg("Parallel(lint)", tasks) == Parallel(Task("ruff ."))


def test_dispatch_arg_inline_invalid_syntax() -> None:
	with pytest.raises(SystemExit, match="2"):
		dispatch_arg("not valid +++", {})


def test_parse_expression_bare_string_coerces_to_task() -> None:
	assert parse_expression('"just a string"', tasks={}) == Task("just a string")


def test_parse_expression_with_undefined_ref() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression("Parallel(undefined)", tasks={})


def test_parse_expression_top_level_ref_without_tasks() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression('Ref("x")')


def test_parse_expression_top_level_tuple_coerces_to_sequential() -> None:
	assert parse_expression('(Task("a"), Task("b"))') == Sequential(Task("a"), Task("b"))


def test_parse_expression_top_level_set_coerces_to_parallel() -> None:
	result = parse_expression('{Task("a"), Task("b")}')
	assert isinstance(result, Parallel)
	assert {t.cmd for t in result.tasks if isinstance(t, Task)} == {"a", "b"}


def test_parse_expression_nested_tuple_coerces_to_sequential() -> None:
	# Inside Parallel, a tuple literal becomes a nested Sequential.
	assert parse_expression('Parallel((Task("a"), Task("b")))') == Parallel(
		Sequential(Task("a"), Task("b"))
	)


def test_parse_expression_nested_set_coerces_to_parallel() -> None:
	# Inside Sequential, a set literal becomes a nested Parallel.
	result = parse_expression('Sequential({Task("a"), Task("b")})')
	assert isinstance(result, Sequential)
	assert len(result.tasks) == 1
	inner = result.tasks[0]
	assert isinstance(inner, Parallel)


def test_parse_expression_bare_string_in_sequential() -> None:
	assert parse_expression('Sequential("a", "b")') == Sequential(Task("a"), Task("b"))


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
	assert "lint: ruff ." in out
	assert "test: pytest" in out


def test_named_task_dry_run(
	tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "pyproject.toml").write_text(
		'[tool.camas.tasks]\nci = \'Parallel(Task("echo a"), Task("echo b"))\'\n'
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


def test_find_tasks_py_in_cwd(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text("")
	assert find_tasks_py(tmp_path) == tmp_path / "tasks.py"


def test_find_tasks_py_walks_up(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text("")
	sub = tmp_path / "a" / "b"
	sub.mkdir(parents=True)
	assert find_tasks_py(sub) == tmp_path / "tasks.py"


def test_find_tasks_py_none(tmp_path: Path) -> None:
	assert find_tasks_py(tmp_path) is None


def test_load_tasks_from_py_propagates_names(tmp_path: Path) -> None:
	tasks_py = tmp_path / "tasks.py"
	tasks_py.write_text(
		"from camas import Parallel, Task\n"
		"mypy = Task('mypy .')\n"
		"pyright = Task('pyright .')\n"
		"typecheck = Parallel(mypy, pyright)\n"
		"_private = Task('nope')\n"
	)
	tasks = load_tasks_from_py(tasks_py)
	assert tasks["mypy"] == Task("mypy .", name="mypy")
	assert tasks["typecheck"] == Parallel(
		Task("mypy .", name="mypy"), Task("pyright .", name="pyright"), name="typecheck"
	)
	assert "_private" not in tasks


def test_tasks_py_preferred_over_pyproject(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(
		"from camas import Task\nhi = Task(('python', '-c', 'pass'))\n"
	)
	(tmp_path / "pyproject.toml").write_text('[tool.camas.tasks]\nhi = "python -c \\"fail\\""\n')
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "--dry-run", "hi"]):
			main()
	captured = capsys.readouterr()
	assert "python -c pass" in captured.out
	assert 'python -c "fail"' not in captured.out


def test_explicit_py_file_arg(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	other = tmp_path / "other_tasks.py"
	other.write_text("from camas import Task\ngreet = Task('echo other')\n")
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", str(other), "--dry-run", "greet"]):
			main()
	assert "echo other" in capsys.readouterr().out


def test_explicit_py_file_missing(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="2"):
		with patch("sys.argv", ["camas", "nope.py", "x"]):
			main()
	assert "no such file" in capsys.readouterr().err


def test_assign_key_name_preserves_cwd() -> None:
	assert _assign_key_name(Task("x", cwd=Path("rust")), "lint") == Task(
		"x", name="lint", cwd=Path("rust")
	)
	assert _assign_key_name(Parallel(Task("x"), cwd=Path("rust")), "grp") == Parallel(
		Task("x"), name="grp", cwd=Path("rust")
	)
	assert _assign_key_name(Sequential(Task("x"), cwd=Path("rust")), "grp") == Sequential(
		Task("x"), name="grp", cwd=Path("rust")
	)


def test_load_tasks_from_py_preserves_cwd(tmp_path: Path) -> None:
	tasks_py = tmp_path / "tasks.py"
	tasks_py.write_text(
		"from pathlib import Path\n"
		"from camas import Parallel, Task\n"
		"leaf = Task('cargo test', cwd=Path('src-tauri'))\n"
		"group = Parallel(Task('cargo check'), cwd=Path('src-tauri'))\n"
	)
	tasks = load_tasks_from_py(tasks_py)
	assert tasks["leaf"].cwd == Path("src-tauri")
	assert tasks["group"].cwd == Path("src-tauri")


def test_resolve_refs_preserves_group_cwd() -> None:
	defs: dict[str, TaskNode | Ref] = {"a": Task("x")}
	seq = _seq((Ref("a"),), cwd=Path("work"))
	par = _par((Ref("a"),), cwd=Path("work"))
	assert resolve_refs(seq, defs, frozenset()).cwd == Path("work")
	assert resolve_refs(par, defs, frozenset()).cwd == Path("work")


def test_subcommand_help_shows_tree(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(
		"from camas import Parallel, Task\n"
		"a = Task('do-a')\n"
		"b = Task('do-b')\n"
		"both = Parallel(a, b)\n"
	)
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "both", "--help"]):
			main()
	out = capsys.readouterr().out
	assert "runs the 'both' task" in out
	assert "do-a" in out
	assert "do-b" in out


def test_explicit_py_file_load_error(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	broken = tmp_path / "broken.py"
	broken.write_text("raise RuntimeError('boom from tasks file')\n")
	with pytest.raises(SystemExit, match="2"):
		with patch("sys.argv", ["camas", str(broken), "--list"]):
			main()
	err = capsys.readouterr().err
	assert str(broken) in err
	assert "boom from tasks file" in err


def test_autodiscover_tasks_py_load_error(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text("raise RuntimeError('boom autodiscover')\n")
	with pytest.raises(SystemExit, match="2"):
		with patch("sys.argv", ["camas", "--list"]):
			main()
	err = capsys.readouterr().err
	assert "tasks.py" in err
	assert "boom autodiscover" in err


def test_nearer_pyproject_wins_over_farther_tasks_py(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	(tmp_path / "tasks.py").write_text("from camas import Task\nhi = Task('from-far-tasks-py')\n")
	sub = tmp_path / "sub"
	sub.mkdir()
	(sub / "pyproject.toml").write_text('[tool.camas.tasks]\nhi = "from-near-pyproject"\n')
	monkeypatch.chdir(sub)
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "--dry-run", "hi"]):
			main()
	out = capsys.readouterr().out
	assert "from-near-pyproject" in out
	assert "from-far-tasks-py" not in out


def test_walk_skips_pyproject_without_camas_tasks(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	(tmp_path / "tasks.py").write_text(
		"from camas import Task\nhi = Task('from-ancestor-tasks-py')\n"
	)
	sub = tmp_path / "sub"
	sub.mkdir()
	(sub / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0"\n')
	monkeypatch.chdir(sub)
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "--dry-run", "hi"]):
			main()
	assert "from-ancestor-tasks-py" in capsys.readouterr().out


def test_autodiscover_skips_warning_when_pyproject_tasks_invalid(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text("from camas import Task\nhi = Task('echo hi')\n")
	(tmp_path / "pyproject.toml").write_text("[tool.camas]\ntasks = 'oops'\n")
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", "--list"]):
			main()
	captured = capsys.readouterr()
	assert "both define tasks" not in captured.err
	assert "hi" in captured.out
