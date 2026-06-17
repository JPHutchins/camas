# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from camas.mcp.scaffold import write_mcp_json

if TYPE_CHECKING:
	from pathlib import Path


def test_creates_mcp_json_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	assert write_mcp_json([]) == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert entry["type"] == "stdio"
	assert entry["command"] == sys.executable
	assert entry["args"] == ["-m", "camas", "mcp"]


def test_merges_preserving_other_servers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
	assert write_mcp_json([]) == 0
	servers = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]
	assert servers["other"] == {"command": "x"}
	assert servers["camas"]["command"] == sys.executable


def test_rich_flag_appends_rich_arg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	assert write_mcp_json(["--rich"]) == 0
	args = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]["args"]
	assert args == ["-m", "camas", "mcp", "--rich"]


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
	monkeypatch.setattr("sys.argv", ["camas", "mcp", "init"])
	with pytest.raises(SystemExit) as exc:
		main()
	assert exc.value.code == 0
	assert (tmp_path / ".mcp.json").exists()


def test_mcp_cli_help_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
	from camas.mcp.cli import main

	main(["--help"])
	assert "camas mcp" in capsys.readouterr().out
