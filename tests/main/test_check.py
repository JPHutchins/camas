# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final
from unittest.mock import patch

import pytest

from camas.main import check as check_mod
from camas.main.check import (
	CHECKER_PRIORITY,
	INSTALL_HINT,
	CheckerErr,
	CheckerNotFound,
	CheckerOk,
	EvalErr,
	EvalOk,
	FoundChecker,
	checker_argv,
	deepest_user_frame,
	describe_check_help,
	find_typechecker,
	format_minimal_trace,
	report_eval_error,
	run_eval,
	run_typecheck,
	run_typecheck_only,
)

FIXTURES: Final = Path(__file__).parents[1] / "fixtures" / "check"


_FOUND: Final = find_typechecker()
CHECKER_NAME: Final = _FOUND.name if _FOUND is not None else None
"""Name of the checker actually available in this test env (``ty`` or ``mypy``).
Tests that exercise checker output match against this so they stay valid
regardless of which one happens to be installed."""

requires_checker = pytest.mark.skipif(
	CHECKER_NAME is None,
	reason="real-checker test requires ty or mypy on PATH (install camas[check])",
)


def _stub_ok(_p: Path) -> CheckerOk:
	return CheckerOk(name="ty")


def _stub_err(_p: Path) -> CheckerErr:
	return CheckerErr(name="ty", output="TYPE_FAIL_MARKER")


def _stub_err_marker(_p: Path) -> CheckerErr:
	return CheckerErr(name="ty", output="MARKER")


def _stub_not_found(_p: Path) -> CheckerNotFound:
	return CheckerNotFound()


def _stub_no_typechecker() -> None:
	return None


def _stub_found_ty() -> FoundChecker:
	return FoundChecker(name="ty", path=Path("/usr/bin/ty"))


def test_checker_priority_is_ty_then_mypy() -> None:
	assert CHECKER_PRIORITY == ("ty", "mypy")


EXE_SUFFIX: Final = ".exe" if sys.platform == "win32" else ""


def test_checker_argv_ty() -> None:
	assert checker_argv(FoundChecker("ty", Path("ty")), Path("tasks.py")) == [
		"ty",
		"check",
		"tasks.py",
	]


def test_checker_argv_mypy() -> None:
	assert checker_argv(FoundChecker("mypy", Path("mypy")), Path("tasks.py")) == [
		"mypy",
		"tasks.py",
	]


def test_find_typechecker_internal_ty_preferred(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	(tmp_path / f"python{EXE_SUFFIX}").write_text("")
	(tmp_path / f"ty{EXE_SUFFIX}").write_text("")
	monkeypatch.setattr(sys, "executable", str(tmp_path / f"python{EXE_SUFFIX}"))
	monkeypatch.setenv("PATH", "")
	assert find_typechecker() == FoundChecker(name="ty", path=tmp_path / f"ty{EXE_SUFFIX}")


def test_find_typechecker_falls_back_to_path_ty(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	(tmp_path / f"python{EXE_SUFFIX}").write_text("")
	path_dir = tmp_path / "bin"
	path_dir.mkdir()
	(path_dir / f"ty{EXE_SUFFIX}").write_text("")
	(path_dir / f"ty{EXE_SUFFIX}").chmod(0o755)
	monkeypatch.setattr(sys, "executable", str(tmp_path / f"python{EXE_SUFFIX}"))
	monkeypatch.setenv("PATH", str(path_dir))
	result = find_typechecker()
	assert result is not None
	assert result.name == "ty"
	assert result.path == path_dir / f"ty{EXE_SUFFIX}"


def test_find_typechecker_internal_mypy_when_no_ty(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	(tmp_path / f"python{EXE_SUFFIX}").write_text("")
	(tmp_path / f"mypy{EXE_SUFFIX}").write_text("")
	monkeypatch.setattr(sys, "executable", str(tmp_path / f"python{EXE_SUFFIX}"))
	monkeypatch.setenv("PATH", "")
	assert find_typechecker() == FoundChecker(name="mypy", path=tmp_path / f"mypy{EXE_SUFFIX}")


def test_find_typechecker_falls_back_to_path_mypy(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	(tmp_path / f"python{EXE_SUFFIX}").write_text("")
	path_dir = tmp_path / "bin"
	path_dir.mkdir()
	(path_dir / f"mypy{EXE_SUFFIX}").write_text("")
	(path_dir / f"mypy{EXE_SUFFIX}").chmod(0o755)
	monkeypatch.setattr(sys, "executable", str(tmp_path / f"python{EXE_SUFFIX}"))
	monkeypatch.setenv("PATH", str(path_dir))
	result = find_typechecker()
	assert result is not None
	assert result.name == "mypy"


def test_find_typechecker_returns_none_when_neither(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	(tmp_path / f"python{EXE_SUFFIX}").write_text("")
	empty = tmp_path / "empty"
	empty.mkdir()
	monkeypatch.setattr(sys, "executable", str(tmp_path / f"python{EXE_SUFFIX}"))
	monkeypatch.setenv("PATH", str(empty))
	assert find_typechecker() is None


@pytest.mark.skipif(sys.platform == "win32", reason="exercises the non-Windows suffix path")
def test_find_typechecker_win32_uses_exe_suffix(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	(tmp_path / "python.exe").write_text("")
	(tmp_path / "ty.exe").write_text("")
	monkeypatch.setattr(sys, "platform", "win32")
	monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
	monkeypatch.setenv("PATH", "")
	assert find_typechecker() == FoundChecker(name="ty", path=tmp_path / "ty.exe")


def test_run_eval_pass(tmp_path: Path) -> None:
	p = tmp_path / "ok.py"
	p.write_text("x = 1\n")
	assert isinstance(run_eval(p), EvalOk)


def test_run_eval_captures_exception(tmp_path: Path) -> None:
	p = tmp_path / "bad.py"
	p.write_text("raise ValueError('boom')\n")
	result = run_eval(p)
	assert isinstance(result, EvalErr)
	assert isinstance(result.exception, ValueError)
	assert str(result.exception) == "boom"


@requires_checker
def test_run_typecheck_passes(tmp_path: Path) -> None:
	p = tmp_path / "clean.py"
	p.write_text("x: int = 1\n")
	result = run_typecheck(p)
	assert isinstance(result, CheckerOk)
	assert result.name == CHECKER_NAME


@requires_checker
def test_run_typecheck_fails(tmp_path: Path) -> None:
	p = tmp_path / "broken.py"
	p.write_text('x: int = "wrong"\n')
	result = run_typecheck(p)
	assert isinstance(result, CheckerErr)
	# Both ty and mypy reference the declared type in their error message;
	# the exact wording differs, so we only assert the shared anchor.
	assert "int" in result.output


def test_run_typecheck_no_checker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(check_mod, "find_typechecker", _stub_no_typechecker)
	assert isinstance(run_typecheck(tmp_path / "x.py"), CheckerNotFound)


def test_deepest_user_frame_finds_tasks_py(tmp_path: Path) -> None:
	p = tmp_path / "tasks.py"
	p.write_text("raise RuntimeError('boom')\n")
	r = run_eval(p)
	assert isinstance(r, EvalErr)
	frame = deepest_user_frame(r.exception, p)
	assert frame is not None
	assert Path(frame.filename).resolve() == p.resolve()
	assert frame.lineno == 1


def test_deepest_user_frame_none_when_no_match(tmp_path: Path) -> None:
	other = tmp_path / "other.py"
	exc = RuntimeError("not from tasks.py")
	assert deepest_user_frame(exc, other) is None


def test_format_minimal_trace_simple_error(tmp_path: Path) -> None:
	p = tmp_path / "tasks.py"
	p.write_text("raise RuntimeError('boom')\n")
	r = run_eval(p)
	assert isinstance(r, EvalErr)
	out = format_minimal_trace(r.exception, p)
	assert f"error: {p}:1" in out
	assert "raise RuntimeError('boom')" in out
	assert "RuntimeError: boom" in out


@pytest.mark.skipif(
	sys.version_info < (3, 11), reason="PEP 657 column-precise tracebacks require Python 3.11+"
)
def test_format_minimal_trace_includes_caret(tmp_path: Path) -> None:
	p = tmp_path / "tasks.py"
	p.write_text("x = undefined_thing\n")
	r = run_eval(p)
	assert isinstance(r, EvalErr)
	out = format_minimal_trace(r.exception, p)
	assert "^^^^^^^^^^^^^^^" in out
	assert "NameError" in out
	assert "undefined_thing" in out


def test_format_minimal_trace_falls_back_when_no_user_frame(tmp_path: Path) -> None:
	exc = RuntimeError("from elsewhere")
	out = format_minimal_trace(exc, tmp_path / "tasks.py")
	assert "RuntimeError" in out
	assert "from elsewhere" in out


def test_format_minimal_trace_without_col_info(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Frames missing PEP 657 col info (3.10, synthetic frames) skip the caret."""
	import traceback as tb_mod

	p = tmp_path / "tasks.py"
	p.write_text("raise RuntimeError('boom')\n")
	frame = tb_mod.FrameSummary(filename=str(p), lineno=1, name="<module>", line=None)

	def _stub_frame(_exc: Exception, _path: Path) -> tb_mod.FrameSummary:
		return frame

	monkeypatch.setattr(check_mod, "deepest_user_frame", _stub_frame)
	out = format_minimal_trace(RuntimeError("boom"), p)
	assert "^" not in out
	assert "raise RuntimeError('boom')" in out


def test_caret_line_negative_offset_returns_none() -> None:
	"""Defensive guard: col offset inside the line's leading whitespace yields no caret."""
	from camas.main.check import caret_line

	assert caret_line(colno=2, end_colno=6, raw_source="        indented_code()") is None


def test_format_minimal_trace_blank_source_line(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""When the source line is blank the source/caret rows are elided."""
	import traceback as tb_mod

	p = tmp_path / "tasks.py"
	p.write_text("\n")
	frame = tb_mod.FrameSummary(filename=str(p), lineno=1, name="<module>", line=None)

	def _stub_frame(_exc: Exception, _path: Path) -> tb_mod.FrameSummary:
		return frame

	monkeypatch.setattr(check_mod, "deepest_user_frame", _stub_frame)
	out = format_minimal_trace(RuntimeError("boom"), p)
	assert "^" not in out
	assert out.splitlines()[0] == f"error: {p}:1"
	assert out.splitlines()[1] == "RuntimeError: boom"


def test_report_eval_error_prints_trace_and_silent_checker_ok(
	tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr(check_mod, "run_typecheck", _stub_ok)
	p = tmp_path / "tasks.py"
	p.write_text("raise RuntimeError('boom')\n")
	r = run_eval(p)
	assert isinstance(r, EvalErr)
	assert report_eval_error(p, r.exception) == 1
	err = capsys.readouterr().err
	assert "RuntimeError: boom" in err
	assert "ty:" not in err


def test_report_eval_error_appends_checker_output_with_header(
	tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr(check_mod, "run_typecheck", _stub_err)
	p = tmp_path / "tasks.py"
	p.write_text("raise RuntimeError('boom')\n")
	r = run_eval(p)
	assert isinstance(r, EvalErr)
	assert report_eval_error(p, r.exception) == 1
	err = capsys.readouterr().err
	assert "RuntimeError: boom" in err
	assert "\nty:\nTYPE_FAIL_MARKER" in err
	assert err.index("RuntimeError") < err.index("TYPE_FAIL_MARKER")


def test_report_eval_error_silent_on_no_checker(
	tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr(check_mod, "run_typecheck", _stub_not_found)
	p = tmp_path / "tasks.py"
	p.write_text("raise RuntimeError('boom')\n")
	r = run_eval(p)
	assert isinstance(r, EvalErr)
	assert report_eval_error(p, r.exception) == 1
	err = capsys.readouterr().err
	assert "RuntimeError: boom" in err
	assert INSTALL_HINT not in err


def test_describe_check_help_when_checker_available(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(check_mod, "find_typechecker", _stub_found_ty)
	assert "(ty)" in describe_check_help()


def test_describe_check_help_when_no_checker(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(check_mod, "find_typechecker", _stub_no_typechecker)
	out = describe_check_help()
	assert "add ty or mypy" in out
	assert "camas[check]" in out


def test_run_typecheck_only_none_source() -> None:
	assert run_typecheck_only(None) == 0


def test_run_typecheck_only_toml_source(tmp_path: Path) -> None:
	assert run_typecheck_only(tmp_path / "pyproject.toml") == 0


def test_run_typecheck_only_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(check_mod, "run_typecheck", _stub_ok)
	p = tmp_path / "tasks.py"
	p.write_text("")
	assert run_typecheck_only(p) == 0


def test_run_typecheck_only_checker_err(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setattr(check_mod, "run_typecheck", _stub_err_marker)
	p = tmp_path / "tasks.py"
	p.write_text("")
	assert run_typecheck_only(p) == 1
	err = capsys.readouterr().err
	assert "ty:\nMARKER" in err


def test_run_typecheck_only_no_checker(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setattr(check_mod, "run_typecheck", _stub_not_found)
	p = tmp_path / "tasks.py"
	p.write_text("")
	assert run_typecheck_only(p) == 1
	assert INSTALL_HINT in capsys.readouterr().err


def _camas(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		[sys.executable, "-m", "camas", *args],
		cwd=cwd,
		capture_output=True,
		text=True,
		encoding="utf-8",
		env={**os.environ, "NO_COLOR": "1"},
		check=False,
	)


@requires_checker
def test_camas_check_pass_fixture() -> None:
	r = _camas("--check", cwd=FIXTURES / "pass")
	assert r.returncode == 0, r.stderr
	assert r.stdout == ""
	assert r.stderr == ""


def test_camas_check_eval_fail_fixture() -> None:
	r = _camas("--check", cwd=FIXTURES / "eval-fail")
	assert r.returncode == 1
	assert "FileNotFoundError" in r.stderr
	assert ".python-version" in r.stderr


@requires_checker
def test_camas_check_type_fail_fixture() -> None:
	r = _camas("--check", cwd=FIXTURES / "type-fail")
	assert r.returncode == 1
	assert f"{CHECKER_NAME}:" in r.stderr


@requires_checker
def test_camas_check_both_fail_fixture() -> None:
	r = _camas("--check", cwd=FIXTURES / "both-fail")
	assert r.returncode == 1
	assert "FileNotFoundError" in r.stderr
	assert f"{CHECKER_NAME}:" in r.stderr
	assert r.stderr.index("FileNotFoundError") < r.stderr.index(f"{CHECKER_NAME}:")


@requires_checker
def test_camas_run_task_eval_fail_shows_minimal_trace_and_typecheck(
	tmp_path: Path,
) -> None:
	(tmp_path / "tasks.py").write_text(
		"from camas import Parallel, Task\n"
		"\n"
		"lint = Task('ruff .')\n"
		"check = Parallel(lint, undefined_ref)\n"
	)
	r = _camas("check", cwd=tmp_path)
	assert r.returncode == 1
	assert f"error: {tmp_path / 'tasks.py'}:4" in r.stderr
	assert "NameError" in r.stderr
	assert "undefined_ref" in r.stderr
	assert f"{CHECKER_NAME}:" in r.stderr


def test_camas_help_works_with_broken_tasks_py(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text("raise RuntimeError('cannot load')\n")
	r = _camas("--help", cwd=tmp_path)
	assert r.returncode == 0
	assert "usage: camas" in r.stdout
	assert "Tasks unavailable" in r.stdout
	assert "camas --check" in r.stdout


def test_camas_list_with_broken_tasks_py(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text("raise RuntimeError('cannot load')\n")
	r = _camas("--list", cwd=tmp_path)
	assert r.returncode == 0
	assert "Tasks unavailable" in r.stdout
	assert "camas --check" in r.stdout


def test_camas_tree_with_broken_tasks_py(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text("raise RuntimeError('cannot load')\n")
	r = _camas("--tree", cwd=tmp_path)
	assert r.returncode == 0
	assert "Tasks unavailable" in r.stdout


def test_camas_check_via_dispatch_run_cli_path(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	from camas.main.dispatch import run_cli

	(tmp_path / "tasks.py").write_text("from camas import Task\nhi = Task('echo hi')\n")
	scope: dict[str, object] = {"__file__": str(tmp_path / "tasks.py"), "hi": object()}
	monkeypatch.setattr(check_mod, "run_typecheck", _stub_ok)
	with pytest.raises(SystemExit, match="0"), patch("sys.argv", ["tasks.py", "--check"]):
		run_cli(scope)


def test_dispatch_check_with_load_error_runs_report_eval_error(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	from camas.main.dispatch import dispatch
	from camas.main.state import LoadErr

	p = tmp_path / "tasks.py"
	p.write_text("raise RuntimeError('boom from dispatch test')\n")
	monkeypatch.setattr(check_mod, "run_typecheck", _stub_err)
	with pytest.raises(SystemExit, match="1"):
		dispatch(LoadErr(source=p, exception=RuntimeError("boom from dispatch test")), ["--check"])
	err = capsys.readouterr().err
	assert "RuntimeError: boom from dispatch test" in err
	assert "TYPE_FAIL_MARKER" in err


def test_dispatch_task_with_load_error_runs_report_eval_error(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	from camas.main.dispatch import dispatch
	from camas.main.state import LoadErr

	p = tmp_path / "tasks.py"
	p.write_text("raise RuntimeError('boom in dispatch task')\n")
	monkeypatch.setattr(check_mod, "run_typecheck", _stub_ok)
	with pytest.raises(SystemExit, match="1"):
		dispatch(LoadErr(source=p, exception=RuntimeError("boom in dispatch task")), ["all"])
	err = capsys.readouterr().err
	assert "RuntimeError: boom in dispatch task" in err


def test_dispatch_effects_listing_works_under_load_err(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	"""``camas --effects`` (with no value) lists available Effects even when
	the tasks file failed to evaluate."""
	from camas.main.dispatch import dispatch
	from camas.main.state import LoadErr

	p = tmp_path / "tasks.py"
	p.write_text("raise RuntimeError('broken')\n")
	with pytest.raises(SystemExit, match="0"):
		dispatch(LoadErr(source=p, exception=RuntimeError("broken")), ["--effects"])
	assert "Available Effects" in capsys.readouterr().out


def test_help_output_for_load_err_includes_hint(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	"""``parser.format_help()`` substitutes the load-error hint for the tasks listing."""
	from camas.main.parser import build_parser
	from camas.main.state import LoadErr

	p = tmp_path / "tasks.py"
	out = build_parser(LoadErr(source=p, exception=RuntimeError("cannot load"))).format_help()
	assert "Tasks unavailable" in out
	assert "camas --check" in out


def test_dispatch_load_err_rejects_unknown_flag(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	"""Even when ``tasks.py`` failed to load, unknown CLI flags must surface as
	argparse errors — not silently fall through to ``exit_for_load_err``."""
	from camas.main.dispatch import dispatch
	from camas.main.state import LoadErr

	p = tmp_path / "tasks.py"
	with pytest.raises(SystemExit, match="2"):
		dispatch(
			LoadErr(source=p, exception=RuntimeError("boom")),
			["--no-such-flag-typo"],
		)
	assert "unrecognized arguments" in capsys.readouterr().err


def test_run_cli_accepts_path_for_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	"""``run_cli`` should propagate ``__file__`` whether it's a ``str`` or a ``Path``."""
	from camas.main.dispatch import run_cli

	(tmp_path / "tasks.py").write_text("from camas import Task\nhi = Task('echo hi')\n")
	scope: dict[str, object] = {"__file__": tmp_path / "tasks.py", "hi": object()}
	monkeypatch.setattr(check_mod, "run_typecheck", _stub_ok)
	with pytest.raises(SystemExit, match="0"), patch("sys.argv", ["tasks.py", "--check"]):
		run_cli(scope)


def test_empty_state_is_immutable() -> None:
	"""``EMPTY_STATE`` is reused across calls, so its dicts must be read-only.

	Cast through ``dict`` only to bypass the static guarantee — the test
	intentionally exercises the runtime ``TypeError`` raised by
	``MappingProxyType.__setitem__``.
	"""
	from typing import cast

	from camas.main.state import EMPTY_STATE

	with pytest.raises(TypeError):
		cast("dict[str, object]", EMPTY_STATE.tasks)["x"] = object()
	with pytest.raises(TypeError):
		cast("dict[str, object]", EMPTY_STATE.scope_effects)["y"] = object()
