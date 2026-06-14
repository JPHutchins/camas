# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from camas import Config, Parallel, Task
from camas.main.dispatch import dispatch, print_interrupt_banner
from camas.main.state import LoadOk

if TYPE_CHECKING:
	from collections.abc import Mapping

	from camas.v0.effect import Effect
	from camas.v0.task import TaskNode


def _state(tasks: Mapping[str, TaskNode], config: Config | None = None) -> LoadOk:
	loaded: dict[str, TaskNode] = dict(tasks)
	effects: dict[str, type[Effect[Any]]] = {}
	return LoadOk(tasks=loaded, source=None, scope_effects=effects, config=config)


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


def test_dispatch_jobs_runs_to_completion() -> None:
	task = Task(("python", "-c", "print('hi')"))
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"go": task}), ["go", "--jobs", "1", "--effects", "()"])


def test_dispatch_bad_camas_jobs_errors(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setenv("CAMAS_JOBS", "nope")
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"go": Task(("python", "-c", "pass"))}), ["go", "--effects", "()"])
	assert "CAMAS_JOBS" in capsys.readouterr().err


def test_bare_runs_default_task_locally(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
	config = Config(default_task=Task("echo DEFAULT", name="default"))
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({}, config), ["--dry-run"])
	assert "echo DEFAULT" in capsys.readouterr().out


def test_bare_runs_github_task_under_actions(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setenv("GITHUB_ACTIONS", "true")
	config = Config(default_task=Task("echo DEFAULT"), github_task=Task("echo GH"))
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({}, config), ["--dry-run"])
	out = capsys.readouterr().out
	assert "echo GH" in out
	assert "echo DEFAULT" not in out


def test_bare_github_falls_back_to_default(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setenv("GITHUB_ACTIONS", "true")
	config = Config(default_task=Task("echo DEFAULT"))
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({}, config), ["--dry-run"])
	assert "echo DEFAULT" in capsys.readouterr().out


def test_bare_no_config_prints_help_and_errors(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"lint": Task("ruff check .")}), [])
	captured = capsys.readouterr()
	assert "task or expression is required" in captured.err
	assert "Reference:" in captured.out


def test_bare_default_runs_to_completion(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
	config = Config(default_task=Task(("python", "-c", "pass")))
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({}, config), ["--effects", "()"])


def test_bare_default_task_matrix_override(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
	config = Config(default_task=Parallel(Task("echo {PY}"), matrix={"PY": ("3.12", "3.13")}))
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({}, config), ["--dry-run", "--matrix", "PY=3.13"])
	out = capsys.readouterr().out
	assert "[PY=3.13]" in out
	assert "[PY=3.12]" not in out


def test_bare_default_task_per_axis_flag_override(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
	config = Config(default_task=Parallel(Task("echo {PY}"), matrix={"PY": ("3.12", "3.13")}))
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({}, config), ["--dry-run", "--PY", "3.13"])
	out = capsys.readouterr().out
	assert "[PY=3.13]" in out
	assert "[PY=3.12]" not in out


def test_interrupt_banner_silent_when_uninterrupted(capsys: pytest.CaptureFixture[str]) -> None:
	print_interrupt_banner(0)
	assert capsys.readouterr().out == ""


def test_interrupt_banner_colored(
	capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr("camas.core.render.color_on", lambda: True)
	print_interrupt_banner(4)
	out = capsys.readouterr().out
	assert "Ctrl-C (4) received - exiting" in out
	assert "\x1b[97m" in out  # WHITE


def test_interrupt_banner_plain_without_color(
	capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr("camas.core.render.color_on", lambda: False)
	print_interrupt_banner(2)
	assert capsys.readouterr().out.strip() == "Ctrl-C (2) received - exiting"
