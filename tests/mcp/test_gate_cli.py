# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The headless ``camas mcp gate`` CLI verb: arg parsing, the PostToolBatch-event stdin
delivery, the verdict JSON + exit code, and the residual-to-stderr block path."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

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


def test_gate_cli_block_prints_residual_to_stderr(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	_chdir_project(tmp_path, monkeypatch, "FIXME")
	(tmp_path / "sample.py").write_text("FIXME\n")
	assert serve.gate_cli(["--paths", "sample.py"]) == 2
	captured = capsys.readouterr()
	assert json.loads(captured.out)["decision"] == "block"
	assert "Re-gate this scope: camas mcp gate" in captured.err


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
	assert serve.changed_from_stdin() == ("a.py", "b.py")


def test_changed_from_stdin_tty_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
	class _Tty(io.StringIO):
		def isatty(self) -> bool:
			return True

	monkeypatch.setattr("sys.stdin", _Tty("x"))
	assert serve.changed_from_stdin() == ()


def test_changed_from_stdin_empty(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("sys.stdin", io.StringIO("  "))
	assert serve.changed_from_stdin() == ()


def test_changed_from_stdin_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
	assert serve.changed_from_stdin() == ()


def test_changed_from_stdin_dict_without_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"foo": "bar"})))
	assert serve.changed_from_stdin() == ()


def test_changed_from_stdin_non_dict_event(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps([1, 2])))
	assert serve.changed_from_stdin() == ()
