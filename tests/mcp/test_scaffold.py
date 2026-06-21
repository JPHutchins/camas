# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from camas.mcp.scaffold import write_mcp_json

if TYPE_CHECKING:
	from collections.abc import Callable
	from pathlib import Path


def _which(*found: str) -> Callable[[str], str | None]:
	return lambda name: f"/usr/bin/{name}" if name in found else None


def test_uv_project_emits_portable_uv_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "uv.lock").write_text("")
	monkeypatch.setattr("shutil.which", _which("uv", "camas"))
	assert write_mcp_json([]) == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert entry["type"] == "stdio"
	assert (entry["command"], entry["args"]) == ("uv", ["run", "camas", "mcp"])


def test_uv_present_without_lock_falls_through(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("uv", "camas"))
	assert write_mcp_json([]) == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert entry["command"] == "camas"


def test_camas_on_path_appends_rich(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	assert write_mcp_json(["--rich"]) == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert (entry["command"], entry["args"]) == ("camas", ["mcp", "--rich"])


def test_errors_when_no_portable_launcher(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which())
	assert write_mcp_json([]) == 2
	err = capsys.readouterr().err
	assert "not on PATH" in err
	assert "uv add camas" in err
	assert not (tmp_path / ".mcp.json").exists()


def test_merges_preserving_other_servers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
	assert write_mcp_json([]) == 0
	servers = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]
	assert servers["other"] == {"command": "x"}
	assert servers["camas"]["command"] == "camas"


def test_malformed_json_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / ".mcp.json").write_text("{not json")
	assert write_mcp_json([]) == 2


def test_non_object_json_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / ".mcp.json").write_text("[]")
	assert write_mcp_json([]) == 2


def test_non_object_mcpservers_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": "nope"}))
	assert write_mcp_json([]) == 2


def test_entrypoint_mcp_init_routes_to_scaffold(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	from camas.main import main

	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	monkeypatch.setattr("sys.argv", ["camas", "mcp", "init"])
	with pytest.raises(SystemExit) as exc:
		main()
	assert exc.value.code == 0
	assert (tmp_path / ".mcp.json").exists()


def test_mcp_cli_help_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
	from camas.mcp.cli import main

	main(["--help"])
	assert "camas mcp" in capsys.readouterr().out


def test_mcp_dash_init_errors_with_hint(capsys: pytest.CaptureFixture[str]) -> None:
	from camas.mcp.cli import main

	with pytest.raises(SystemExit) as exc:
		main(["--init"])
	assert exc.value.code == 2
	err = capsys.readouterr().err
	assert "--init" in err
	assert "camas mcp init" in err


def test_mcp_unexpected_arg_errors(capsys: pytest.CaptureFixture[str]) -> None:
	from camas.mcp.cli import main

	with pytest.raises(SystemExit) as exc:
		main(["serve"])
	assert exc.value.code == 2
	assert "unexpected argument" in capsys.readouterr().err
