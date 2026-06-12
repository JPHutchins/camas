# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from camas import Parallel, Task
from camas.main.dispatch import dispatch
from camas.main.state import LoadOk

if TYPE_CHECKING:
	from collections.abc import Mapping

	from camas.v0.effect import Effect
	from camas.v0.task import TaskNode


def _state(tasks: Mapping[str, TaskNode]) -> LoadOk:
	loaded: dict[str, TaskNode] = dict(tasks)
	effects: dict[str, type[Effect[Any]]] = {}
	return LoadOk(tasks=loaded, source=None, scope_effects=effects)


def test_print_task_help_with_axes(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("echo {PY}"), matrix={"PY": ("3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"check": task}), ["check", "--help"])
	out = capsys.readouterr().out
	assert "Matrix axes" in out
	assert "--PY" in out


def test_print_task_help_shows_help_text(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("a"), Task("b"), help="Run two things side-by-side")
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"both": task}), ["both", "--help"])
	out = capsys.readouterr().out
	assert "Run two things side-by-side" in out
	idx_help = out.index("Run two things side-by-side")
	idx_runs = out.index("runs the 'both' task")
	assert idx_help < idx_runs


def test_dispatch_per_axis_flag_overrides(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("echo {PY}"), matrix={"PY": ("3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"check": task}), ["--dry-run", "check", "--PY", "3.13"])
	out = capsys.readouterr().out
	assert "[PY=3.13]" in out
	assert "[PY=3.12]" not in out


def test_dispatch_matrix_flag_overrides(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("echo {PY}"), matrix={"PY": ("3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"check": task}), ["--dry-run", "check", "--matrix", "PY=3.13"])
	out = capsys.readouterr().out
	assert "[PY=3.13]" in out
	assert "[PY=3.12]" not in out


def test_dispatch_matrix_flag_bad_syntax_errors(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("echo {PY}"), matrix={"PY": ("3.12",)})
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"check": task}), ["check", "--matrix", "noequals"])
	assert "--matrix expects KEY=VAL" in capsys.readouterr().err


def test_dispatch_per_axis_flag_empty_value_errors(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("echo {PY}"), matrix={"PY": ("3.12",)})
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"check": task}), ["check", "--PY", ""])
	assert "--PY" in capsys.readouterr().err


def test_dispatch_matrix_unknown_axis_errors(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("echo {PY}"), matrix={"PY": ("3.12",)})
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"check": task}), ["check", "--matrix", "XX=1"])
	assert "unknown matrix axis" in capsys.readouterr().err


def test_dispatch_skips_reserved_axis_name_for_auto_flag(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = Parallel(Task("echo {matrix}"), matrix={"matrix": ("a", "b")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"check": task}), ["--dry-run", "check", "--matrix", "matrix=b"])
	out = capsys.readouterr().out
	assert "[matrix=b]" in out
	assert "[matrix=a]" not in out
