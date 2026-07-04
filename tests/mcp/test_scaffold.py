# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from camas.mcp.scaffold import (
	launch_command,
	launch_command_str,
	resolve_pin,
	tasks_py_path,
	write_agent_skill_templates,
	write_claude,
	write_hooks,
	write_mcp_json,
)

if TYPE_CHECKING:
	from collections.abc import Callable
	from pathlib import Path


def _which(*found: str) -> Callable[[str], str | None]:
	return lambda name: f"/usr/bin/{name}" if name in found else None


def test_launch_command_uv_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "uv.lock").write_text("")
	monkeypatch.setattr("shutil.which", _which("uv", "camas"))
	assert launch_command() == ("uv", ["run", "camas", "mcp"])


def test_launch_command_uvx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("uvx"))
	assert launch_command() == ("uvx", ["camas[mcp]", "mcp"])


def test_launch_command_camas(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which("camas"))
	assert launch_command() == ("camas", ["mcp"])


def test_launch_command_none(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which())
	assert launch_command() is None


def test_launch_command_uvx_with_pin(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which("uvx"))
	assert launch_command(pin="camas[mcp]>=0.1.8") == ("uvx", ["camas[mcp]>=0.1.8", "mcp"])


def test_launch_command_uv_with_lock_ignores_pin(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "uv.lock").write_text("")
	monkeypatch.setattr("shutil.which", _which("uv", "camas"))
	assert launch_command(pin="camas[mcp]>=0.1.8") == ("uv", ["run", "camas", "mcp"])


def test_launch_command_camas_ignores_pin(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which("camas"))
	assert launch_command(pin="camas[mcp]>=0.1.8") == ("camas", ["mcp"])


def test_launch_command_str_camas(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which("camas"))
	assert launch_command_str() == "camas mcp"


def test_launch_command_str_uv_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "uv.lock").write_text("")
	monkeypatch.setattr("shutil.which", _which("uv", "camas"))
	assert launch_command_str() == "uv run camas mcp"


def test_launch_command_str_uvx(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which("uvx"))
	assert launch_command_str() == "uvx 'camas[mcp]' mcp"


def test_launch_command_str_uvx_with_pin(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which("uvx"))
	assert launch_command_str(pin="camas[mcp]>=0.1.8") == "uvx 'camas[mcp]>=0.1.8' mcp"


def test_launch_command_str_none(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which())
	assert launch_command_str() is None


def test_launch_command_uv_with_pep723_tasks_py(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(
		'# /// script\n# dependencies = ["camas>=0.1.8"]\n# ///\nfrom camas import Task\n'
	)
	monkeypatch.setattr("shutil.which", _which("uv"))
	assert launch_command() == ("uv", ["run", "tasks.py", "mcp"])
	assert launch_command(pin="camas[mcp]>=0.1.8") == ("uv", ["run", "tasks.py", "mcp"])


def test_launch_command_uv_lock_wins_over_pep723_tasks_py(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "uv.lock").write_text("")
	(tmp_path / "tasks.py").write_text(
		'# /// script\n# dependencies = ["camas>=0.1.8"]\n# ///\nfrom camas import Task\n'
	)
	monkeypatch.setattr("shutil.which", _which("uv", "camas"))
	assert launch_command() == ("uv", ["run", "camas", "mcp"])


def test_launch_command_pep723_tasks_py_must_be_in_cwd(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	sub = tmp_path / "sub"
	sub.mkdir()
	(sub / "tasks.py").write_text(
		'# /// script\n# dependencies = ["camas>=0.1.8"]\n# ///\nfrom camas import Task\n'
	)
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("uv"))
	assert launch_command() is None


def test_launch_command_tasks_py_without_pep723_header_falls_through(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text("from camas import Task\nlint = Task('echo')\n")
	monkeypatch.setattr("shutil.which", _which("uv"))
	assert launch_command() is None


def test_write_mcp_json_uses_uv_run_tasks_py_when_pep723(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(
		'# /// script\n# dependencies = ["camas>=0.1.8"]\n# ///\nfrom camas import Task\n'
	)
	monkeypatch.setattr("shutil.which", _which("uv"))
	assert write_mcp_json([]) == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert (entry["command"], entry["args"]) == ("uv", ["run", "tasks.py", "mcp"])


def test_write_hooks_uses_uv_run_tasks_py_when_pep723(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(
		'# /// script\n# dependencies = ["camas>=0.1.8"]\n# ///\nfrom camas import Task\n'
	)
	monkeypatch.setattr("shutil.which", _which("uv"))
	assert write_hooks([]) == 0
	out = capsys.readouterr().out
	assert "uv run tasks.py mcp fix" in out


def test_tasks_py_path_finds_tasks_py(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	(tmp_path / "tasks.py").write_text("")
	monkeypatch.chdir(tmp_path)
	assert tasks_py_path() == tmp_path / "tasks.py"


def test_tasks_py_path_returns_none_when_absent(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	assert tasks_py_path() is None


def test_tasks_py_path_walks_ancestors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	(tmp_path / "tasks.py").write_text("")
	sub = tmp_path / "sub"
	sub.mkdir()
	monkeypatch.chdir(sub)
	assert tasks_py_path() == tmp_path / "tasks.py"


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


def test_uv_without_lock_or_uvx_falls_through_to_camas(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("uv", "camas"))
	assert write_mcp_json([]) == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert entry["command"] == "camas"
	assert entry["args"] == ["mcp"]


def test_rich_no_longer_appended(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	"""``--rich`` is the server default; the launcher no longer emits it."""
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	assert write_mcp_json(["--rich"]) == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert (entry["command"], entry["args"]) == ("camas", ["mcp"])


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


def test_write_mcp_json_writes_file_despite_camas_dir_error(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))

	def _raise_oserror(_dir: object) -> None:
		raise OSError("disk full")

	monkeypatch.setattr("camas.mcp.scaffold.ensure_camas_dir", _raise_oserror)
	assert write_mcp_json([]) == 0
	assert (tmp_path / ".mcp.json").exists()
	assert "warning" in capsys.readouterr().err


def test_write_hooks_writes_settings_json(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	assert write_hooks([]) == 0
	settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
	post_tool_batch = settings["hooks"]["PostToolBatch"][0]["hooks"][0]
	assert post_tool_batch["type"] == "command"
	assert post_tool_batch["command"] == "camas mcp fix"
	assert "FileChanged" not in settings["hooks"]
	assert "PostToolBatch autofix hook" in capsys.readouterr().out


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
	assert "PostToolBatch" in settings["hooks"]


def test_write_hooks_sweeps_stale_hooks_from_all_events(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""https://github.com/JPHutchins/camas/issues/157 — a camas autofix hook left under a non-current
	event by an older camas (the pre-PostToolBatch ``FileChanged`` hook) is swept out on the next
	init --claude: an event holding only the stale camas hook is dropped, a non-camas hook in another
	event is preserved, and the current hook lands under PostToolBatch."""
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_text(
		json.dumps(
			{
				"hooks": {
					"FileChanged": [
						{
							"hooks": [
								{"type": "command", "command": "camas mcp fix --paths ${file_path}"}
							]
						}
					],
					"PreToolUse": [
						{
							"hooks": [
								{
									"type": "command",
									"command": "camas mcp fix --paths ${file_path}",
								},
								{"type": "command", "command": "echo keep-me"},
							]
						}
					],
				}
			}
		)
	)
	assert write_hooks([]) == 0
	hooks = json.loads((tmp_path / ".claude" / "settings.json").read_text())["hooks"]
	assert "FileChanged" not in hooks
	assert [h["command"] for g in hooks["PreToolUse"] for h in g["hooks"]] == ["echo keep-me"]
	assert "mcp fix" in hooks["PostToolBatch"][-1]["hooks"][0]["command"]


def test_write_hooks_preserves_non_camas_hook_mentioning_mcp_fix(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""A non-camas hook whose command merely contains the substring ``mcp fix`` is preserved —
	the sweep requires the ``camas`` token too, so it doesn't false-positive on unrelated hooks."""
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_text(
		json.dumps(
			{
				"hooks": {
					"PostToolBatch": [
						{"hooks": [{"type": "command", "command": "./scripts/mcp fixer.sh"}]}
					]
				}
			}
		)
	)
	assert write_hooks([]) == 0
	hooks = json.loads((tmp_path / ".claude" / "settings.json").read_text())["hooks"]
	commands = [h["command"] for g in hooks["PostToolBatch"] for h in g["hooks"]]
	assert "./scripts/mcp fixer.sh" in commands


def test_write_hooks_rejects_matcher_null(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_text(
		json.dumps(
			{
				"hooks": {
					"PostToolBatch": [
						{
							"hooks": [{"type": "command", "command": "echo hi"}],
							"matcher": None,
						}
					]
				}
			}
		)
	)
	assert write_hooks([]) == 2


def test_write_hooks_preserves_matcher_empty_string(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_text(
		json.dumps(
			{
				"hooks": {
					"PostToolBatch": [
						{
							"hooks": [{"type": "command", "command": "echo hi"}],
							"matcher": "",
						}
					]
				}
			}
		)
	)
	assert write_hooks([]) == 0
	settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
	ptb = settings["hooks"]["PostToolBatch"]
	assert len(ptb) == 2
	echo_group = next(g for g in ptb if any("echo hi" in h["command"] for h in g["hooks"]))
	assert echo_group.get("matcher") == ""
	assert any(h["command"] == "camas mcp fix" for g in ptb for h in g["hooks"])


def test_write_hooks_preserves_extra_hook_command_fields(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_text(
		json.dumps(
			{
				"hooks": {
					"PreToolUse": [
						{"hooks": [{"type": "command", "command": "guard", "timeout": 30}]}
					],
					"PostToolBatch": [
						{
							"hooks": [
								{
									"type": "command",
									"command": "echo hi",
									"statusMessage": "linting",
								}
							]
						}
					],
				}
			}
		)
	)
	assert write_hooks([]) == 0
	settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
	assert settings["hooks"]["PreToolUse"][0]["hooks"][0]["timeout"] == 30
	echo = next(
		h
		for g in settings["hooks"]["PostToolBatch"]
		for h in g["hooks"]
		if h["command"] == "echo hi"
	)
	assert echo["statusMessage"] == "linting"
	assert any(
		h["command"] == "camas mcp fix"
		for g in settings["hooks"]["PostToolBatch"]
		for h in g["hooks"]
	)


def test_write_hooks_preserves_key_order_and_omits_matcher(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_text(
		json.dumps(
			{
				"$schema": "https://example.com/schema.json",
				"permissions": {"allow": ["Read"]},
				"hooks": {
					"PreToolUse": [{"hooks": [{"type": "command", "command": "guard"}]}],
				},
			}
		)
	)
	assert write_hooks([]) == 0
	settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
	assert list(settings.keys()) == ["$schema", "permissions", "hooks"]
	assert list(settings["hooks"].keys()) == ["PreToolUse", "PostToolBatch"]
	assert "matcher" not in settings["hooks"]["PreToolUse"][0]
	assert "matcher" not in settings["hooks"]["PostToolBatch"][0]
	assert settings["hooks"]["PostToolBatch"][0]["hooks"][0]["command"] == "camas mcp fix"


def test_write_hooks_removes_camas_only_groups(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_text(
		json.dumps(
			{
				"hooks": {
					"PostToolBatch": [{"hooks": [{"type": "command", "command": "camas mcp fix"}]}]
				}
			}
		)
	)
	assert write_hooks([]) == 0
	settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
	ptb = settings["hooks"]["PostToolBatch"]
	assert len(ptb) == 1
	assert ptb[0]["hooks"][0]["command"] == "camas mcp fix"


def test_write_claude_writes_all_four_files(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	assert write_claude([]) == 0
	# .mcp.json
	assert (tmp_path / ".mcp.json").exists()
	mcp = json.loads((tmp_path / ".mcp.json").read_text())
	assert mcp["mcpServers"]["camas"]["command"] == "camas"
	# .claude/settings.json
	assert (tmp_path / ".claude" / "settings.json").exists()
	settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
	assert "PostToolBatch" in settings["hooks"]
	# agent
	agent = (tmp_path / ".claude" / "agents" / "camas-fixer.md").read_text()
	assert "name: camas-fixer" in agent
	assert "mcp__camas__camas_gate" in agent
	assert "mcp__camas__camas_fix" in agent
	# skill
	skill = (tmp_path / ".claude" / "skills" / "gate" / "SKILL.md").read_text()
	assert "name: gate" in skill
	# consolidated output
	out = capsys.readouterr().out
	assert "Claude Code is configured" in out


def test_write_claude_stops_on_mcp_json_failure(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which())
	assert write_claude([]) == 2
	assert not (tmp_path / ".claude" / "settings.json").exists()
	assert not (tmp_path / ".claude" / "agents" / "camas-fixer.md").exists()


def test_write_claude_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	# First run
	assert write_claude([]) == 0
	mtime1 = (tmp_path / ".claude" / "agents" / "camas-fixer.md").stat().st_mtime
	# Second run should overwrite
	assert write_claude([]) == 0
	mtime2 = (tmp_path / ".claude" / "agents" / "camas-fixer.md").stat().st_mtime
	assert mtime2 > mtime1


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
	out = capsys.readouterr().out
	assert "camas mcp" in out
	assert "--claude" in out
	assert "--plain" in out


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


def test_mcp_cli_init_claude_routes_to_write_claude(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	from camas.mcp.cli import main

	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	with pytest.raises(SystemExit) as exc:
		main(["init", "--claude"])
	assert exc.value.code == 0
	assert (tmp_path / ".mcp.json").exists()
	assert (tmp_path / ".claude" / "settings.json").exists()
	assert (tmp_path / ".claude" / "agents" / "camas-fixer.md").exists()
	assert (tmp_path / ".claude" / "skills" / "gate" / "SKILL.md").exists()


def test_hooks_flag_warns_and_writes_only_mcp_json(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""``--hooks`` is removed; passing it warns on stderr and writes only ``.mcp.json``."""
	from camas.mcp.cli import main

	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	with pytest.raises(SystemExit) as exc:
		main(["init", "--hooks"])
	assert exc.value.code == 0
	assert (tmp_path / ".mcp.json").exists()
	assert not (tmp_path / ".claude" / "settings.json").exists()
	err = capsys.readouterr().err
	assert "--hooks was removed" in err
	assert "camas mcp init --claude" in err


def test_rich_and_plain_filtered_from_unexpected(
	capsys: pytest.CaptureFixture[str],
) -> None:
	"""``--rich`` and ``--plain`` are filtered out of the unexpected-arg check."""
	from camas.mcp.cli import main

	# --rich + bogus arg: only the bogus arg triggers the error
	with pytest.raises(SystemExit) as exc:
		main(["--rich", "bogus"])
	assert exc.value.code == 2
	err = capsys.readouterr().err
	assert "unexpected argument(s): bogus" in err

	# --plain + bogus arg
	with pytest.raises(SystemExit) as exc:
		main(["--plain", "bogus"])
	assert exc.value.code == 2
	err = capsys.readouterr().err
	assert "unexpected argument(s): bogus" in err


def test_write_agent_skill_templates_creates_dirs_and_files(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	write_agent_skill_templates()
	agent = (tmp_path / ".claude" / "agents" / "camas-fixer.md").read_text()
	assert "name: camas-fixer" in agent
	assert "mcp__camas__camas_gate" in agent
	skill = (tmp_path / ".claude" / "skills" / "gate" / "SKILL.md").read_text()
	assert "name: gate" in skill


def test_write_mcp_json_pinned_when_resolve_pin_returns_value(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""When ``resolve_pin()`` returns a requirement, the uvx launcher splices it into the spec."""
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("uvx"))
	monkeypatch.setattr("camas.mcp.scaffold.resolve_pin", lambda: "camas[mcp]>=0.1.18")
	assert write_mcp_json([]) == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert entry["command"] == "uvx"
	assert entry["args"] == ["camas[mcp]>=0.1.18", "mcp"]


def test_write_hooks_pinned_when_resolve_pin_returns_value(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""When ``resolve_pin()`` returns a requirement, the hook command uses the pinned launcher."""
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("uvx"))
	monkeypatch.setattr("camas.mcp.scaffold.resolve_pin", lambda: "camas[mcp]>=0.1.18")
	assert write_hooks([]) == 0
	out = capsys.readouterr().out
	assert "uvx 'camas[mcp]>=0.1.18' mcp fix" in out


def test_write_mcp_json_unpinned_when_no_tasks_py(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("uvx"))
	assert write_mcp_json([]) == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert entry["args"] == ["camas[mcp]", "mcp"]


def test_resolve_pin_no_tasks_py(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	"""No tasks.py in cwd or any ancestor → resolve_pin returns None."""
	monkeypatch.chdir(tmp_path)
	assert resolve_pin() is None


def test_resolve_pin_no_camas_in_tasks_py(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	"""tasks.py exists but has no camas dependency → resolve_pin returns None."""
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(
		'# /// script\n# dependencies = ["pytest"]\n# ///\nfrom camas import Task\n'
	)
	assert resolve_pin() is None


def test_resolve_pin_no_pep723_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	"""tasks.py exists but has no PEP 723 block → resolve_pin returns None."""
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text("from camas import Task\nlint = Task('echo hi')\n")
	assert resolve_pin() is None


def test_resolve_pin_returns_pinned_with_mcp_extra(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""tasks.py with a camas dependency → resolve_pin returns it with [mcp] extra."""
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(
		'# /// script\n# dependencies = ["camas>=0.1.8"]\n# ///\nfrom camas import Task\n'
	)
	assert resolve_pin() == "camas[mcp]>=0.1.8"


def test_write_claude_stops_on_hooks_failure(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""write_mcp_json succeeds but write_hooks fails on malformed settings.json → write_claude
	returns 2 and does NOT write agent/skill templates."""
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_text("{not valid json")
	assert write_claude([]) == 2
	assert (tmp_path / ".mcp.json").exists()
	assert not (tmp_path / ".claude" / "agents" / "camas-fixer.md").exists()
	assert not (tmp_path / ".claude" / "skills" / "gate" / "SKILL.md").exists()
