# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from camas import Sequential
from camas.main.dispatch import dispatch
from camas.main.init import starter_text, write_starter_tasks_py
from camas.main.state import EMPTY_STATE, LoadErr
from camas.main.tasks import load_tasks_from_py

if TYPE_CHECKING:
	from pathlib import Path


def _camas(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
	env = {**os.environ, "NO_COLOR": "1"}
	env.pop("GITHUB_ACTIONS", None)
	return subprocess.run(
		[sys.executable, "-m", "camas", *args],
		cwd=cwd,
		capture_output=True,
		text=True,
		encoding="utf-8",
		env=env,
		check=False,
	)


def test_write_starter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
	assert write_starter_tasks_py(tmp_path) == 0
	assert (tmp_path / "tasks.py").read_text(encoding="utf-8") == starter_text()
	assert (tmp_path / ".camas" / ".gitignore").read_text(encoding="utf-8") == "*\n"
	out = capsys.readouterr().out
	assert "Wrote" in out
	assert ".camas" in out


def test_write_starter_refuses_existing_leaves_no_camas_dir(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text("untouched = 1\n")
	assert write_starter_tasks_py(tmp_path) == 2
	assert not (tmp_path / ".camas").exists()


def test_write_starter_refuses_existing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
	(tmp_path / "tasks.py").write_text("untouched = 1\n")
	assert write_starter_tasks_py(tmp_path) == 2
	assert (tmp_path / "tasks.py").read_text() == "untouched = 1\n"
	assert "exists" in capsys.readouterr().err


def test_dispatch_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="0"):
		dispatch(EMPTY_STATE, ["--init"])
	assert (tmp_path / "tasks.py").exists()


def test_dispatch_init_under_load_err(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	"""``--init`` still works when a tasks file elsewhere failed to load."""
	monkeypatch.chdir(tmp_path)
	state = LoadErr(source=tmp_path / "parent" / "tasks.py", exception=ValueError("boom"))
	with pytest.raises(SystemExit, match="0"):
		dispatch(state, ["--init"])
	assert (tmp_path / "tasks.py").exists()


def test_starter_loads_with_config_default(tmp_path: Path) -> None:
	write_starter_tasks_py(tmp_path)
	loaded = load_tasks_from_py(tmp_path / "tasks.py")
	assert set(loaded.tasks) == {"hello", "greet", "ci", "fix"}
	assert loaded.scope_effects == {}
	assert loaded.config is not None
	assert loaded.config.default_task == loaded.tasks["ci"]
	assert isinstance(loaded.config.default_task, Sequential)
	assert loaded.config.agent is not None
	assert loaded.config.agent.fix == loaded.tasks["fix"]


def test_starter_runs_to_completion(tmp_path: Path) -> None:
	"""Bare ``camas`` in a freshly scaffolded directory runs the whole default
	tree green — the placeholder tasks must be infallible cross-platform."""
	write_starter_tasks_py(tmp_path)
	result = _camas("--effects=(Summary(show_passing=True),)", cwd=tmp_path)
	assert result.returncode == 0, result.stderr
	assert "hello from camas" in result.stdout
	assert "hello, Ada!" in result.stdout
	assert "hello, Grace!" in result.stdout
	assert "Python" in result.stdout


def test_starter_cli_init_then_list(tmp_path: Path) -> None:
	created = _camas("--init", cwd=tmp_path)
	assert created.returncode == 0, created.stderr
	again = _camas("--init", cwd=tmp_path)
	assert again.returncode == 2
	assert "exists" in again.stderr
	listed = _camas("--list", cwd=tmp_path)
	assert listed.returncode == 0, listed.stderr
	assert "say hello to everyone at once" in listed.stdout
	assert all(name in listed.stdout for name in ("hello", "greet", "ci"))


def test_starter_runs_standalone(tmp_path: Path) -> None:
	"""``python tasks.py --list`` dispatches through the scaffold's
	``run_cli(globals())`` block."""
	write_starter_tasks_py(tmp_path)
	env = {**os.environ, "NO_COLOR": "1"}
	result = subprocess.run(
		[sys.executable, str(tmp_path / "tasks.py"), "--list"],
		capture_output=True,
		text=True,
		encoding="utf-8",
		env=env,
		check=False,
	)
	assert result.returncode == 0, result.stderr
	assert all(name in result.stdout for name in ("hello", "greet", "ci"))
