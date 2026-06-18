# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from camas.main import main


def test_help(tmp_path: Path) -> None:
	result = subprocess.run(
		[sys.executable, "-m", "camas", "--help"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		encoding="utf-8",
		check=False,
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
		check=False,
	)
	assert result.returncode == 0
	assert "camas" in result.stdout


def test_top_level_help_source_is_local_filesystem_path(tmp_path: Path) -> None:
	"""Agents reading --help need a filesystem path they can read directly
	without network access. Source is local; examples is remote (the wheel
	doesn't ship the examples folder)."""
	import camas as camas_pkg

	result = subprocess.run(
		[sys.executable, "-m", "camas", "--help"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		encoding="utf-8",
		check=False,
	)
	assert result.returncode == 0
	assert "Reference:" in result.stdout
	expected_source = str(Path(camas_pkg.__file__).parent)
	assert expected_source in result.stdout, result.stdout
	assert "https://github.com/JPHutchins/camas/tree/main/examples" in result.stdout
	assert "https://pypi.org/project/camas/" in result.stdout


def test_no_args_with_no_tasks_errors(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="2"), patch("sys.argv", ["camas"]):
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
	with pytest.raises(SystemExit, match="2"), patch("sys.argv", ["camas"]):
		main()
	captured = capsys.readouterr()
	assert "Available tasks from" in captured.out
	assert "tasks.py" in captured.out
	assert "ci" in captured.out
	assert "lint, test" in captured.out
	assert "lint" in captured.out
	assert "ruff check ." in captured.out
	assert "task or expression is required" in captured.err


def test_no_args_with_tasks_includes_reference_block(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	import camas as camas_pkg

	tasks_py = tmp_path / "tasks.py"
	tasks_py.write_text("from camas import Task\nlint = Task('ruff check .')\n")
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="2"), patch("sys.argv", ["camas"]):
		main()
	out = capsys.readouterr().out
	assert "Reference:" in out
	assert str(Path(camas_pkg.__file__).parent) in out, out


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
	with pytest.raises(SystemExit, match="0"), patch("sys.argv", ["camas", "--list"]):
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
	with pytest.raises(SystemExit, match="0"), patch("sys.argv", ["camas", "--tree"]):
		main()
	out = capsys.readouterr().out
	assert "Available tasks from" in out
	assert "echo a" in out
	assert "echo b" in out


def test_dispatch_rejects_bad_effects(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	with (
		pytest.raises(SystemExit, match="2"),
		patch("sys.argv", ["camas", "--effects=nope(", 'Task("echo hi")']),
	):
		main()
	assert "--effects" in capsys.readouterr().err


def test_dry_run_prints_tree(capsys: pytest.CaptureFixture[str]) -> None:
	with (
		pytest.raises(SystemExit, match="0"),
		patch(
			"sys.argv",
			["camas", "--dry-run", 'Parallel(Task("echo a"),Task("echo b"))'],
		),
	):
		main()
	captured = capsys.readouterr()
	assert "echo a" in captured.out
	assert "echo b" in captured.out


def test_dry_run_matrix(capsys: pytest.CaptureFixture[str]) -> None:
	with (
		pytest.raises(SystemExit, match="0"),
		patch(
			"sys.argv",
			[
				"camas",
				"--dry-run",
				'Parallel(Task("test {PY}"),matrix={"PY": ("3.12", "3.13")})',
			],
		),
	):
		main()
	captured = capsys.readouterr()
	assert "3.12" in captured.out
	assert "3.13" in captured.out


def test_dispatch_effects_no_value_lists(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="0"), patch("sys.argv", ["camas", "--effects"]):
		main()
	assert "Available Effects:" in capsys.readouterr().out


def test_help_includes_tasks_and_effects(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	(tmp_path / "tasks.py").write_text('from camas import Task\nlint = Task("ruff check .")\n')
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="0"), patch("sys.argv", ["camas", "--help"]):
		main()
	out = capsys.readouterr().out
	assert "Available tasks from" in out
	assert "lint" in out
	assert "Available Effects:" in out
	assert "Try:" in out


def test_successful_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	with (
		pytest.raises(SystemExit, match="0"),
		patch("sys.argv", ["camas", 'Task(("python", "-c", "pass"), name="ok")']),
	):
		main()


def test_failed_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	with (
		pytest.raises(SystemExit, match="1"),
		patch("sys.argv", ["camas", 'Task(("python", "-c", "raise SystemExit(1)"), name="fail")']),
	):
		main()


def test_passthrough_appended_to_inline_task(capsys: pytest.CaptureFixture[str]) -> None:
	with (
		pytest.raises(SystemExit, match="0"),
		patch(
			"sys.argv",
			["camas", "--dry-run", 'Task(("python", "-c", "print(1)"))', "--", "extra"],
		),
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
	with (
		pytest.raises(SystemExit, match="0"),
		patch("sys.argv", ["camas", "--dry-run", "test", "--", "-v", "-k", "x"]),
	):
		main()
	out = capsys.readouterr().out
	assert "pytest" in out
	assert "-v" in out
	assert "-k" in out
	assert "x" in out


def test_passthrough_rejects_sequential_task(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	(tmp_path / "tasks.py").write_text(
		"from camas import Sequential, Task\nci = Sequential(Task('a'), Task('b'))\n",
	)
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="2"), patch("sys.argv", ["camas", "ci", "--", "-v"]):
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
	with (
		pytest.raises(SystemExit, match="0"),
		patch("sys.argv", ["camas", "--dry-run", "test", "--", "--help"]),
	):
		main()
	out = capsys.readouterr().out
	assert "--help" in out
	assert "usage:" not in out
