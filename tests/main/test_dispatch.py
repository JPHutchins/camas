# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from camas import Claude, Config, Parallel, Sequential, Task
from camas.core import timings
from camas.core.color import WHITE
from camas.core.completion import RunResult
from camas.main.dispatch import dispatch, print_interrupt_banner, run_under
from camas.main.state import LoadOk

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

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


def test_interrupt_banner_colored(
	capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr("camas.core.render.color_on", lambda: True)
	print_interrupt_banner(4)
	out = capsys.readouterr().out
	assert "Ctrl-C (4) received - exiting" in out
	assert WHITE in out


def test_interrupted_run_prints_banner_and_exits_130(
	capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	async def fake_run(task: TaskNode, effects: object = (), jobs: object = None) -> RunResult:
		return RunResult(returncode=130, results=(), elapsed=0.0, interrupt_count=3)

	monkeypatch.setattr("camas.main.dispatch.run", fake_run)
	with pytest.raises(SystemExit, match="130"):
		dispatch(_state({}, Config(default_task=Task("true"))), [])
	assert "Ctrl-C (3) received - exiting" in capsys.readouterr().out


def test_dispatch_under_no_timings_runs_untimed(capsys: pytest.CaptureFixture[str]) -> None:
	noop = ("python", "-c", "pass")
	with pytest.raises(SystemExit, match="0"):
		dispatch(
			_state({"check": Parallel(Task(noop, name="a"), Task(noop, name="b"))}),
			["check", "--under", "1s"],
		)
	out = capsys.readouterr().out
	assert "2 unmeasured" in out


def test_dispatch_under_rejects_passthrough(capsys: pytest.CaptureFixture[str]) -> None:
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"t": Task("pytest", name="t")}), ["t", "--under", "1s", "--", "-v"])
	assert "passthrough args cannot be combined with --under" in capsys.readouterr().err


def test_dispatch_under_bad_jobs_env_errors(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setenv("CAMAS_JOBS", "nope")
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"t": Task("echo", name="t")}), ["t", "--under", "1s"])
	assert "CAMAS_JOBS" in capsys.readouterr().err


def _camas_with_timings(tmp_path: Path, leaves: list[tuple[str, float]]) -> Path:
	camas = tmp_path / ".camas"
	camas.mkdir()
	timings.record(camas, leaves)
	return camas


def test_run_under_dry_run_shows_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
	camas = _camas_with_timings(tmp_path, [("fmt", 0.1), ("lint", 0.2), ("slow", 9.0)])
	source = Sequential(
		Task("echo fmt", name="fmt", mutates=True),
		Parallel(Task("echo lint", name="lint"), Task("echo slow", name="slow")),
	)
	code = run_under(
		source, 1.0, camas_dir=camas, effects=(), jobs=None, dry_run=True, passthrough=()
	)
	assert code == 0
	out = capsys.readouterr().out
	assert "running 2 leaf(s)" in out
	assert "over budget: slow ~9.00s" in out


def test_run_under_executes_selected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
	camas = _camas_with_timings(tmp_path, [("a", 0.1)])
	source = Parallel(
		Task(("python", "-c", "print('a')"), name="a"),
		Task(("python", "-c", "print('b')"), name="b"),
	)
	code = run_under(
		source, 1.0, camas_dir=camas, effects=(), jobs=None, dry_run=False, passthrough=()
	)
	assert code == 0
	assert "unmeasured (running to record an estimate): b" in capsys.readouterr().out


def test_run_under_all_over_budget_runs_nothing(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	camas = _camas_with_timings(tmp_path, [("slow", 9.0)])
	code = run_under(
		Parallel(Task("echo slow", name="slow")),
		0.5,
		camas_dir=camas,
		effects=(),
		jobs=None,
		dry_run=False,
		passthrough=(),
	)
	assert code == 0
	assert "All leaves exceed the budget — nothing to run." in capsys.readouterr().out


def test_dispatch_fix_resolves_agent_fix() -> None:
	fix = Task(("python", "-c", "pass"), name="fix", mutates=True)
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({}, Config(agent=Claude(fix=fix))), ["fix", "--effects", "()"])


def test_dispatch_paths_scopes_to_changed(capsys: pytest.CaptureFixture[str]) -> None:
	task = Task("ruff check {paths}", name="lint", paths=".")
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"lint": task}), ["--dry-run", "lint", "--paths", "src/app.py"])
	assert "ruff check src/app.py" in capsys.readouterr().out


def test_dispatch_paths_no_match_skips(capsys: pytest.CaptureFixture[str]) -> None:
	task = Task("cargo check {paths}", name="rust", paths="rust")
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"rust": task}), ["rust", "--paths", "src/app.py"])
	out = capsys.readouterr().out
	assert "src/app.py" in out
	assert "nothing to run" in out


def test_dispatch_paths_empty_skips(capsys: pytest.CaptureFixture[str]) -> None:
	task = Task("ruff check {paths}", name="lint", paths=".")
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"lint": task}), ["lint", "--paths", ""])
	assert "(no paths given)" in capsys.readouterr().out
