# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The headless ``camas mcp gate`` CLI verb: arg parsing, the PostToolBatch-event stdin
delivery, the verdict JSON + exit code, and the residual-to-stderr block path."""

from __future__ import annotations

import io
import json
import os
import time
from typing import TYPE_CHECKING

from camas.core import timings
from camas.core.hook_event import HookEvent, changed_from_stdin
from camas.mcp import serve, wire

if TYPE_CHECKING:
	from pathlib import Path

	import pytest

_TASKS = (
	"from camas import Config, Task\n\ncheck = Task(\n"
	'\t("python", "-c", "import pathlib, sys; sys.exit({marker!r} in pathlib.Path(\'sample.py\').read_text())"),\n'
	'\tname="check",\n)\n_ = Config(default_task=check)\n'
)


def _chdir_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: str) -> None:
	monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(_TASKS.format(marker=marker))


def test_parse_gate_args() -> None:
	args = serve.parse_gate_args(
		["check", "--paths", "a.py", "--paths", "b.py", "--under", "5", "--jobs", "2"]
	)
	assert args == serve.GateArgs(task="check", paths=("a.py", "b.py"), under=5.0, jobs=2)


def test_parse_gate_args_defaults() -> None:
	assert serve.parse_gate_args([]) == serve.GateArgs(task=None, paths=(), under=None, jobs=None)


def test_parse_gate_args_under_accepts_duration_suffix() -> None:
	"""``--under`` takes a suffixed duration, not just a bare number of seconds — the top-level
	``camas --under`` gotcha (argparse ``type=float`` rejecting ``5s``) fixed for ``camas mcp
	gate`` too."""
	assert serve.parse_gate_args(["--under", "5s"]).under == 5.0
	assert serve.parse_gate_args(["--under", "500ms"]).under == 0.5


def test_parse_gate_args_nudge_flag() -> None:
	assert serve.parse_gate_args(["--nudge"]).nudge is True
	assert serve.parse_gate_args([]).nudge is False


def test_rerun_command() -> None:
	rerun = wire.GateRerun(task="check", paths=("a.py", "b.py"), under=5.0)
	assert (
		serve.rerun_command(rerun) == "camas mcp gate check --paths a.py --paths b.py --under 5.0"
	)
	assert serve.rerun_command(wire.GateRerun()) == "camas mcp gate"


def test_gate_cli_green(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	_chdir_project(tmp_path, monkeypatch, "FIXME")
	(tmp_path / "sample.py").write_text("ok = 1\n")
	assert serve.gate_cli(["--paths", "sample.py"]) == 0
	out = json.loads(capsys.readouterr().out)
	assert out["decision"] == "continue"
	assert out["residual_class"] == "green"
	assert out["rerun"]["paths"] == ["sample.py"]


def test_gate_cli_anchors_leaf_cwd_to_tasks_dir_from_subdir(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""Invoked from a child cwd, the leaf spawns in the resolved tasks.py directory (the rebasing
	frame), not the process cwd — matching the MCP server's ``base_for``.
	"""
	monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
	root = tmp_path.resolve()
	script = (
		"import pathlib, sys; "
		f"sys.exit(0 if pathlib.Path.cwd().resolve() == pathlib.Path({str(root)!r}).resolve() else 1)"
	)
	(tmp_path / "tasks.py").write_text(
		"from camas import Config, Task\n"
		f"check = Task(('python', '-c', {script!r}), name='check')\n"
		"_ = Config(default_task=check)\n"
	)
	sub = tmp_path / "sub"
	sub.mkdir()
	monkeypatch.chdir(sub)
	assert serve.gate_cli(["--paths", "x.py"]) == 0
	assert json.loads(capsys.readouterr().out)["decision"] == "continue"


def test_gate_cli_block_prints_residual_to_stderr(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	_chdir_project(tmp_path, monkeypatch, "FIXME")
	(tmp_path / "sample.py").write_text("FIXME\n")
	assert serve.gate_cli(["--paths", "sample.py"]) == 2
	captured = capsys.readouterr()
	assert json.loads(captured.out)["decision"] == "block"
	assert "Re-gate this scope: camas mcp gate" in captured.err


def test_gate_cli_unfilled_required_axis_blocks(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""A check node with an empty-value matrix axis would expand to zero leaves and false-green;
	the gate errors to stderr and blocks (exit 2) rather than reporting a green continue."""
	monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(
		"from camas import Config, Parallel, Task\n"
		'check = Parallel(Task("echo {version}"), matrix={"version": ()}, name="check")\n'
		"_ = Config(default_task=check)\n"
	)
	assert serve.gate_cli(["--paths", "sample.py"]) == 2
	assert "required but unset" in capsys.readouterr().err


def test_gate_cli_nudge_green_is_silent(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	_chdir_project(tmp_path, monkeypatch, "FIXME")
	(tmp_path / "sample.py").write_text("ok = 1\n")
	assert serve.gate_cli(["--paths", "sample.py", "--nudge"]) == 0
	captured = capsys.readouterr()
	assert captured.out == ""
	assert captured.err == ""


def test_gate_cli_nudge_block_prints_fixer_ladder_nudge_to_stderr(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	_chdir_project(tmp_path, monkeypatch, "FIXME")
	(tmp_path / "sample.py").write_text("FIXME\n")
	assert serve.gate_cli(["--paths", "sample.py", "--nudge"]) == 2
	captured = capsys.readouterr()
	assert captured.out == ""
	assert "camas-lint-fixer-haiku" in captured.err
	assert "camas-test-fixer" in captured.err
	assert "not green" in captured.err


def test_nudge_text_quotes_diagnostics() -> None:
	resp = wire.GateResponse(
		decision="block",
		residual_class="needs_reasoning",
		diagnostics=(
			wire.AgentEnvelope(
				name="lint", exit_code=1, output_kind="raw", payload="x", truncated=True
			),
		),
		rerun=wire.GateRerun(),
	)
	text = serve.nudge_text(resp)
	assert "lint (exit 1, raw)" in text
	assert "… earlier output truncated" in text
	assert "camas-lint-fixer-sonnet" in text


_FIX_ONLY_TASKS = (
	"from camas import Claude, Config, Task\n"
	'tidy = Task(("python", "-c", "pass"), name="tidy", mutates=True, paths=".")\n'
	"_ = Config(agent=Claude(fix=tidy))\n"
)


def _stop_event(prompt_id: str, **extra: object) -> str:
	return json.dumps(
		{"session_id": "s-1", "prompt_id": prompt_id, "hook_event_name": "Stop", **extra}
	)


def test_gate_cli_nudge_fix_only_project_exits_zero(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""A fix-only project has nothing to gate — the async Stop hook must not rewake the agent
	over a configuration state (the harness rewake-loop regression).
	"""
	monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(_FIX_ONLY_TASKS)
	assert serve.gate_cli(["--nudge"]) == 0
	assert "no check node" in capsys.readouterr().err


def test_gate_cli_nudge_load_error_exits_zero(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text("raise ValueError('boom')\n")
	assert serve.gate_cli(["--nudge"]) == 0
	assert "cannot load" in capsys.readouterr().err


def test_gate_cli_nudge_wakes_at_most_once_per_prompt(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	_chdir_project(tmp_path, monkeypatch, "FIXME")
	(tmp_path / "sample.py").write_text("FIXME\n")
	marker_dir = tmp_path / "markers"
	marker_dir.mkdir()
	monkeypatch.setattr("camas.mcp.serve.tempfile.gettempdir", lambda: str(marker_dir))
	monkeypatch.setattr("sys.stdin", io.StringIO(_stop_event("p-1")))
	assert serve.gate_cli(["--nudge"]) == 2
	assert "camas-lint-fixer-haiku" in capsys.readouterr().err
	monkeypatch.setattr("sys.stdin", io.StringIO(_stop_event("p-1")))
	assert serve.gate_cli(["--nudge"]) == 0
	assert capsys.readouterr().err == ""
	monkeypatch.setattr("sys.stdin", io.StringIO(_stop_event("p-2")))
	assert serve.gate_cli(["--nudge"]) == 2
	assert "camas-lint-fixer-haiku" in capsys.readouterr().err


def test_gate_cli_nudge_honors_stop_hook_active(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	_chdir_project(tmp_path, monkeypatch, "FIXME")
	(tmp_path / "sample.py").write_text("FIXME\n")
	marker_dir = tmp_path / "markers"
	marker_dir.mkdir()
	monkeypatch.setattr("camas.mcp.serve.tempfile.gettempdir", lambda: str(marker_dir))
	monkeypatch.setattr("sys.stdin", io.StringIO(_stop_event("p-1", stop_hook_active=True)))
	assert serve.gate_cli(["--nudge"]) == 0
	assert capsys.readouterr().err == ""


def test_gate_cli_nudge_invalid_utf8_marker_allows_nudge(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""A marker with invalid UTF-8 bytes (UnicodeDecodeError, a ValueError) must not crash — it
	reads as unreadable, so one nudge is still allowed."""
	_chdir_project(tmp_path, monkeypatch, "FIXME")
	(tmp_path / "sample.py").write_text("FIXME\n")
	marker_dir = tmp_path / "markers"
	marker_dir.mkdir()
	monkeypatch.setattr("camas.mcp.serve.tempfile.gettempdir", lambda: str(marker_dir))
	monkeypatch.setattr("sys.stdin", io.StringIO(_stop_event("p-1")))
	assert serve.gate_cli(["--nudge"]) == 2
	capsys.readouterr()
	(marker,) = tuple(marker_dir.iterdir())
	marker.write_bytes(b"\xff\xfe")
	monkeypatch.setattr("sys.stdin", io.StringIO(_stop_event("p-1")))
	assert serve.gate_cli(["--nudge"]) == 2
	assert "camas-lint-fixer-haiku" in capsys.readouterr().err


def test_gate_cli_nudge_empty_session_id_writes_no_shared_marker(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""An empty session_id is treated like None: the nudge fires but no marker is written, so
	sessions never share one."""
	_chdir_project(tmp_path, monkeypatch, "FIXME")
	(tmp_path / "sample.py").write_text("FIXME\n")
	marker_dir = tmp_path / "markers"
	marker_dir.mkdir()
	monkeypatch.setattr("camas.mcp.serve.tempfile.gettempdir", lambda: str(marker_dir))
	monkeypatch.setattr(
		"sys.stdin",
		io.StringIO(json.dumps({"session_id": "", "prompt_id": "p-1", "hook_event_name": "Stop"})),
	)
	assert serve.gate_cli(["--nudge"]) == 2
	assert "camas-lint-fixer-haiku" in capsys.readouterr().err
	assert list(marker_dir.iterdir()) == []


def test_prune_stale_nudge_markers_removes_only_stale(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr("camas.mcp.serve.tempfile.gettempdir", lambda: str(tmp_path))
	stale = tmp_path / f"{serve.NUDGE_MARKER_PREFIX}stale"
	fresh = tmp_path / f"{serve.NUDGE_MARKER_PREFIX}fresh"
	stale.write_text("p-old")
	fresh.write_text("p-new")
	old = time.time() - 7200.0
	os.utime(stale, (old, old))
	serve.prune_stale_nudge_markers()
	assert not stale.exists()
	assert fresh.exists()


def test_record_nudge_prunes_stale_markers_while_writing(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr("camas.mcp.serve.tempfile.gettempdir", lambda: str(tmp_path))
	stale = tmp_path / f"{serve.NUDGE_MARKER_PREFIX}stale"
	stale.write_text("p-old")
	old = time.time() - 7200.0
	os.utime(stale, (old, old))
	event = HookEvent(changed=None, session_id="s-9", prompt_id="p-1", stop_hook_active=False)
	serve.record_nudge(event)
	assert not stale.exists()
	(fresh,) = tuple(tmp_path.glob(f"{serve.NUDGE_MARKER_PREFIX}*"))
	assert fresh.read_text(encoding="utf-8") == "p-1"


def test_nudge_text_without_diagnostics() -> None:
	resp = wire.GateResponse(
		decision="block",
		residual_class="needs_reasoning",
		rerun=wire.GateRerun(),
	)
	text = serve.nudge_text(resp)
	assert "not green" in text
	assert "camas-lint-fixer-haiku" in text
	assert "camas-test-fixer" in text
	assert "Residual (failing checks):" not in text


def test_gate_cli_reads_stdin_event(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	_chdir_project(tmp_path, monkeypatch, "FIXME")
	(tmp_path / "sample.py").write_text("ok\n")
	event = json.dumps({"tool_calls": [{"tool_input": {"file_path": str(tmp_path / "sample.py")}}]})
	monkeypatch.setattr("sys.stdin", io.StringIO(event))
	assert serve.gate_cli([]) == 0
	assert json.loads(capsys.readouterr().out)["rerun"]["paths"] == ["sample.py"]


def test_gate_cli_no_check_node_errors(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
	monkeypatch.chdir(tmp_path)
	assert serve.gate_cli([]) == 2
	assert "no check node" in capsys.readouterr().err


def test_gate_cli_load_error(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text("raise ValueError('boom')\n")
	assert serve.gate_cli([]) == 2
	assert "cannot load" in capsys.readouterr().err


def test_gate_cli_load_error_skew_hints_cli_fallback(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(
		"# /// script\n# dependencies = [\"camas==999.0.0\"]\n# ///\nraise ValueError('boom')\n"
	)
	assert serve.gate_cli([]) == 2
	err = capsys.readouterr().err
	assert "cannot load" in err
	assert "does not satisfy tasks.py pin camas==999.0.0" in err
	assert "uv run tasks.py" in err


def test_gate_cli_unknown_task_errors(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	_chdir_project(tmp_path, monkeypatch, "FIXME")
	assert serve.gate_cli(["nope"]) == 2
	assert "no task named 'nope'" in capsys.readouterr().err


def test_gate_text_marks_truncated_diagnostic() -> None:
	resp = wire.GateResponse(
		decision="block",
		residual_class="needs_reasoning",
		diagnostics=(
			wire.AgentEnvelope(
				name="lint", exit_code=1, output_kind="raw", payload="x", truncated=True
			),
		),
		rerun=wire.GateRerun(),
	)
	text = serve.gate_text(resp)
	assert "… earlier output truncated" in text
	assert "Re-gate this scope" in text


def test_changed_from_stdin_extracts_edited_files(monkeypatch: pytest.MonkeyPatch) -> None:
	event = json.dumps(
		{
			"tool_calls": [
				{"tool_input": {"file_path": "a.py"}},
				{"tool_input": {"file_path": "a.py"}},
				{"tool_input": {"path": "b.py"}},
				{"tool_input": {}},
				"not-a-dict",
			]
		}
	)
	monkeypatch.setattr("sys.stdin", io.StringIO(event))
	assert changed_from_stdin() == ("a.py", "b.py")


def test_changed_from_stdin_tty_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
	class _Tty(io.StringIO):
		def isatty(self) -> bool:
			return True

	monkeypatch.setattr("sys.stdin", _Tty("x"))
	assert changed_from_stdin() == ()


def test_changed_from_stdin_empty(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("sys.stdin", io.StringIO("  "))
	assert changed_from_stdin() == ()


def test_changed_from_stdin_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
	assert changed_from_stdin() == ()


def test_changed_from_stdin_dict_without_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"foo": "bar"})))
	assert changed_from_stdin() == ()


def test_changed_from_stdin_non_dict_event(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps([1, 2])))
	assert changed_from_stdin() == ()


def test_gate_cli_dry_run_shows_plan(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	_chdir_project(tmp_path, monkeypatch, "FIXME")
	(tmp_path / "sample.py").write_text("ok = 1\n")
	assert serve.gate_cli(["--paths", "sample.py", "--dry-run"]) == 0
	out = capsys.readouterr().out
	assert "Dry run" in out
	assert "check" in out


def test_gate_cli_dry_run_no_match(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(
		"from camas import Config, Task\n"
		'check = Task("echo scope-miss {paths}", name="check", paths="src")\n'
		"_ = Config(default_task=check)\n"
	)
	assert serve.gate_cli(["--paths", "unrelated.txt", "--dry-run"]) == 0
	out = capsys.readouterr().out
	assert "No leaves cover" in out
	assert "nothing would run" in out


_BUDGET_TASKS = (
	"from camas import Config, Parallel, Task\n"
	'fast = Task("python --version", name="fast")\n'
	'slow = Task("python --version", name="slow")\n'
	'check = Parallel(fast, slow, name="check")\n'
	"_ = Config(default_task=check)\n"
)


def test_gate_cli_dry_run_under_excludes_over_budget_leaf(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(_BUDGET_TASKS)
	camas_dir = tmp_path / ".camas"
	camas_dir.mkdir()
	timings.record(camas_dir, [("fast", 0.5), ("slow", 99.0)])
	assert serve.gate_cli(["--paths", "sample.py", "--under", "5", "--dry-run"]) == 0
	preview, _, headline = capsys.readouterr().out.partition("Time budget")
	assert "Dry run" in preview
	assert "fast" in preview
	assert "slow" not in preview
	assert "excluded 1 over budget" in headline
	assert "slow" in headline


def test_gate_cli_dry_run_under_all_over_budget(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(
		"from camas import Config, Task\n"
		'check = Task("python --version", name="only")\n'
		"_ = Config(default_task=check)\n"
	)
	camas_dir = tmp_path / ".camas"
	camas_dir.mkdir()
	timings.record(camas_dir, [("only", 99.0)])
	assert serve.gate_cli(["--paths", "sample.py", "--under", "5", "--dry-run"]) == 0
	out = capsys.readouterr().out
	assert "nothing would run" in out
	assert "excluded 1 over budget" in out
	assert "only" in out
