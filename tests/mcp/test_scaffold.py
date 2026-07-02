# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from camas.mcp.scaffold import launch_command_str, write_hooks, write_mcp_json

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


def test_uv_present_without_lock_resolves_uvx(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("uv", "uvx"))
	assert write_mcp_json([]) == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert entry["command"] == "uvx"
	assert entry["args"] == ["camas[mcp]", "mcp"]


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
	assert not (tmp_path / ".camas").exists()


def test_creates_camas_dir_with_gitignore(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	assert write_mcp_json([]) == 0
	assert (tmp_path / ".camas" / ".gitignore").read_text(encoding="utf-8") == "*\n"
	assert "created" in capsys.readouterr().out.lower()


def test_existing_camas_dir_left_intact(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".camas").mkdir()
	(tmp_path / ".camas" / "timings.txt").write_text("0\n", encoding="utf-8")
	assert write_mcp_json([]) == 0
	assert (tmp_path / ".camas" / "timings.txt").read_text(encoding="utf-8") == "0\n"
	assert "created" not in capsys.readouterr().out.lower()


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


def test_launch_command_str_uv_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "uv.lock").write_text("")
	monkeypatch.setattr("shutil.which", _which("uv", "camas"))
	assert launch_command_str(rich=False) == "uv run camas mcp"


def test_launch_command_str_camas_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which("camas"))
	assert launch_command_str(rich=False) == "camas mcp"


def test_launch_command_str_rich(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which("camas"))
	assert launch_command_str(rich=True) == "camas mcp --rich"


def test_launch_command_str_uvx_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which("uvx"))
	assert launch_command_str(rich=False) == "uvx 'camas[mcp]' mcp"


def test_launch_command_str_uvx_with_rich(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which("uvx"))
	assert launch_command_str(rich=True) == "uvx 'camas[mcp]' mcp --rich"


def test_launch_command_str_none_when_no_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which())
	assert launch_command_str(rich=False) is None


def test_write_hooks_writes_settings_json(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	assert write_hooks([]) == 0
	settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
	file_changed = settings["hooks"]["FileChanged"][0]["hooks"][0]
	assert file_changed["type"] == "command"
	assert file_changed["command"] == "camas mcp fix --paths ${file_path}"
	assert "PostToolBatch" not in settings["hooks"]
	assert "FileChanged autofix hook" in capsys.readouterr().out


def test_write_hooks_errors_when_no_launcher(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which())
	assert write_hooks([]) == 2
	err = capsys.readouterr().err
	assert "not on PATH" in err
	assert "uv add camas" in err
	assert not (tmp_path / ".claude" / "settings.json").exists()


def test_write_hooks_errors_on_malformed_settings(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_bytes(b"\x00not json")
	assert write_hooks([]) == 2


def test_write_hooks_errors_on_non_object_root(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_text("[]")
	assert write_hooks([]) == 2


def test_write_hooks_errors_on_non_object_hooks(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_text(json.dumps({"hooks": "not an object"}))
	assert write_hooks([]) == 2


def test_write_hooks_merges_existing_settings(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_text(
		json.dumps({"other_key": "value", "hooks": {}})
	)
	assert write_hooks([]) == 0
	settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
	assert settings["other_key"] == "value"
	assert "FileChanged" in settings["hooks"]


def test_write_hooks_tolerates_matcher_null(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_text(
		json.dumps(
			{
				"hooks": {
					"FileChanged": [
						{
							"hooks": [{"type": "command", "command": "echo hi"}],
							"matcher": None,
						}
					]
				}
			}
		)
	)
	assert write_hooks([]) == 0
	settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
	assert "FileChanged" in settings["hooks"]


def test_mcp_cli_init_hooks_routes_to_write_hooks(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	from camas.mcp.cli import main

	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	with pytest.raises(SystemExit) as exc:
		main(["init", "--hooks"])
	assert exc.value.code == 0
	assert (tmp_path / ".claude" / "settings.json").exists()
