# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from camas.mcp.scaffold import (
	AGENT_TEMPLATES,
	installed_version_spec,
	launch_command,
	launch_command_str,
	parse_json_object,
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

_DEV_VERSION = "0.1.0.dev0+gabc1234"
_RELEASE_VERSION = "0.1.18"


def _which(*found: str) -> Callable[[str], str | None]:
	return lambda name: f"/usr/bin/{name}" if name in found else None


def _pin_installed_version(monkeypatch: pytest.MonkeyPatch, installed: str) -> None:
	"""Fix the running camas version ``launch_command``'s uvx fallback reads, so the unpinned/
	pinned split doesn't depend on whether the test happens to run from a tagged release."""

	def _version(_dist: str) -> str:
		return installed

	monkeypatch.setattr("camas.mcp.scaffold.version", _version)


def test_launch_command_uv_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "uv.lock").write_text("")
	monkeypatch.setattr("shutil.which", _which("uv", "camas"))
	assert launch_command() == ("uv", ["run", "camas", "mcp"])


def test_launch_command_uvx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("uvx"))
	_pin_installed_version(monkeypatch, _DEV_VERSION)
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
	_pin_installed_version(monkeypatch, _DEV_VERSION)
	assert launch_command_str() == "uvx 'camas[mcp]' mcp"


def test_launch_command_str_uvx_with_pin(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which("uvx"))
	assert launch_command_str(pin="camas[mcp]>=0.1.8") == "uvx 'camas[mcp]>=0.1.8' mcp"


def test_launch_command_str_none(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which())
	assert launch_command_str() is None


def test_installed_version_spec_pins_clean_release() -> None:
	assert installed_version_spec("0.1.18") == "camas[mcp]==0.1.18"


def test_installed_version_spec_dev_build_is_unpinned() -> None:
	assert installed_version_spec("0.1.22.dev3+g09f0fca") == "camas[mcp]"


def test_installed_version_spec_local_build_is_unpinned() -> None:
	assert installed_version_spec("0.1.18+g456ba4719") == "camas[mcp]"


def test_launch_command_uvx_no_pin_pins_to_installed_release(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""No PEP 723 pin, but the running camas is a clean release: pin the uvx fallback to it."""
	monkeypatch.setattr("shutil.which", _which("uvx"))
	_pin_installed_version(monkeypatch, _RELEASE_VERSION)
	assert launch_command() == ("uvx", [f"camas[mcp]=={_RELEASE_VERSION}", "mcp"])


def test_launch_command_uvx_no_pin_dev_build_stays_unpinned(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""No PEP 723 pin, and the running camas is a dev/local build not on PyPI: stay unpinned."""
	monkeypatch.setattr("shutil.which", _which("uvx"))
	_pin_installed_version(monkeypatch, _DEV_VERSION)
	assert launch_command() == ("uvx", ["camas[mcp]", "mcp"])


def test_launch_command_uvx_pep723_pin_wins_over_installed_version(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A PEP 723-derived pin is the project's SSOT — used as-is, even unpinned (bare ``camas``),
	regardless of the running camas version."""
	monkeypatch.setattr("shutil.which", _which("uvx"))
	_pin_installed_version(monkeypatch, _RELEASE_VERSION)
	assert launch_command(pin="camas[mcp]") == ("uvx", ["camas[mcp]", "mcp"])


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


def test_launch_command_launcher_uv_forced_uses_lock_project(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "uv.lock").write_text("")
	monkeypatch.setattr("shutil.which", _which("uv", "uvx", "camas"))
	assert launch_command(launcher="uv") == ("uv", ["run", "camas", "mcp"])


def test_launch_command_launcher_uv_forced_errors_without_uv(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr("shutil.which", _which("uvx", "camas"))
	assert launch_command(launcher="uv") is None


def test_launch_command_launcher_uv_forced_errors_without_lock_or_pep723(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("uv", "uvx", "camas"))
	assert launch_command(launcher="uv") is None


def test_launch_command_launcher_uvx_forced_even_with_uv_lock(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""``--launcher uvx`` forces uvx even though a uv.lock project would normally win."""
	monkeypatch.chdir(tmp_path)
	(tmp_path / "uv.lock").write_text("")
	monkeypatch.setattr("shutil.which", _which("uv", "uvx", "camas"))
	_pin_installed_version(monkeypatch, _DEV_VERSION)
	assert launch_command(launcher="uvx") == ("uvx", ["camas[mcp]", "mcp"])


def test_launch_command_launcher_uvx_forced_uses_pin(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which("uvx"))
	assert launch_command(pin="camas[mcp]>=0.1.8", launcher="uvx") == (
		"uvx",
		["camas[mcp]>=0.1.8", "mcp"],
	)


def test_launch_command_launcher_uvx_forced_errors_without_uvx(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr("shutil.which", _which("uv", "camas"))
	assert launch_command(launcher="uvx") is None


def test_launch_command_launcher_camas_forced(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which("uv", "uvx", "camas"))
	assert launch_command(launcher="camas") == ("camas", ["mcp"])


def test_launch_command_launcher_camas_forced_errors_without_path(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr("shutil.which", _which("uv", "uvx"))
	assert launch_command(launcher="camas") is None


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
	_pin_installed_version(monkeypatch, _DEV_VERSION)
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


def test_parse_json_object_invalid_utf8_returns_none(tmp_path: Path) -> None:
	"""An invalid-UTF-8 file (UnicodeDecodeError, a ValueError not an OSError) falls back
	cleanly instead of crashing."""
	corrupt = tmp_path / ".mcp.json"
	corrupt.write_bytes(b"\xff\xfe")
	assert parse_json_object(corrupt) is None


def test_write_mcp_json_invalid_utf8_errors(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / ".mcp.json").write_bytes(b"\xff\xfe")
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
	assert post_tool_batch["command"] == "camas mcp fix || exit 0"
	assert "FileChanged" not in settings["hooks"]
	assert "Wrote the camas autofix and Stop hooks" in capsys.readouterr().out


def test_write_hooks_writes_stop_fix_and_async_nudge_hooks(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""The two Stop hooks from #168's design: a plain (synchronous) settle-time fix, and an
	``async``/``asyncRewake`` check that nudges the main agent to launch the fixer ladder when
	the workspace is not green — coexisting with the unrelated PostToolBatch autofix hook."""
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	assert write_hooks([]) == 0
	settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
	stop_hooks = settings["hooks"]["Stop"][0]["hooks"]
	fix_hook, nudge_hook = stop_hooks
	assert fix_hook == {"type": "command", "command": "camas mcp fix || exit 0"}
	assert nudge_hook["type"] == "command"
	assert nudge_hook["command"] == "camas mcp gate --under 5s --nudge"
	assert nudge_hook["async"] is True
	assert nudge_hook["asyncRewake"] is True
	assert settings["hooks"]["PostToolBatch"][0]["hooks"][0]["command"] == "camas mcp fix || exit 0"
	out = capsys.readouterr().out
	assert "Stop (fix):         camas mcp fix || exit 0" in out
	assert "Stop (async nudge): camas mcp gate --under 5s --nudge" in out


def test_write_hooks_fix_hook_is_fail_safe_but_nudge_is_not(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""#221: the best-effort autofix hook trails ``|| exit 0`` so a launcher/env failure degrades to
	a no-op instead of blocking the turn; the nudge hook keeps its exit code — it is the signal."""
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	assert write_hooks([]) == 0
	hooks = json.loads((tmp_path / ".claude" / "settings.json").read_text())["hooks"]
	post = hooks["PostToolBatch"][0]["hooks"][0]["command"]
	stop_fix, stop_nudge = (h["command"] for h in hooks["Stop"][0]["hooks"])
	assert post.endswith("|| exit 0")
	assert stop_fix.endswith("|| exit 0")
	assert "|| exit 0" not in stop_nudge


def test_write_hooks_stop_hooks_are_idempotent(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	assert write_hooks([]) == 0
	assert write_hooks([]) == 0
	settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
	assert len(settings["hooks"]["Stop"]) == 1
	assert len(settings["hooks"]["Stop"][0]["hooks"]) == 2
	assert len(settings["hooks"]["PostToolBatch"]) == 1


def test_write_hooks_sweeps_stale_stop_hook_preserving_user_stop_hooks(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""A stale camas ``Stop`` hook (e.g. an older ``--nudge`` flag shape) is replaced on re-init,
	while a user's own unrelated ``Stop`` hook in the same group is preserved."""
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_text(
		json.dumps(
			{
				"hooks": {
					"Stop": [
						{
							"hooks": [
								{"type": "command", "command": "camas mcp gate --nudge"},
								{"type": "command", "command": "echo done"},
							]
						}
					]
				}
			}
		)
	)
	assert write_hooks([]) == 0
	hooks = json.loads((tmp_path / ".claude" / "settings.json").read_text())["hooks"]
	stop_commands = [h["command"] for g in hooks["Stop"] for h in g["hooks"]]
	assert "echo done" in stop_commands
	assert "camas mcp gate --nudge" not in stop_commands
	assert "camas mcp fix || exit 0" in stop_commands
	assert "camas mcp gate --under 5s --nudge" in stop_commands


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


def test_write_hooks_errors_on_invalid_utf8_settings(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_bytes(b"\xff\xfe")
	assert write_hooks([]) == 2
	assert "not valid JSON" in capsys.readouterr().err


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
	assert any(h["command"] == "camas mcp fix || exit 0" for g in ptb for h in g["hooks"])


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
		h["command"] == "camas mcp fix || exit 0"
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
	assert list(settings["hooks"].keys()) == ["PreToolUse", "PostToolBatch", "Stop"]
	assert "matcher" not in settings["hooks"]["PreToolUse"][0]
	assert "matcher" not in settings["hooks"]["PostToolBatch"][0]
	assert "matcher" not in settings["hooks"]["Stop"][0]
	assert settings["hooks"]["PostToolBatch"][0]["hooks"][0]["command"] == "camas mcp fix || exit 0"


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
	assert ptb[0]["hooks"][0]["command"] == "camas mcp fix || exit 0"


def _sweep_survivors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str) -> set[str]:
	"""The hook commands left under a neutral event after ``write_hooks`` re-sweeps a settings.json
	seeding ``command`` beside a non-camas sentinel. The neutral event is one write_hooks never
	writes to, so its own PostToolBatch/Stop hooks can't mask a swept ``command``: the sentinel
	always survives; ``command`` survives only when it is not recognized as a camas hook.
	"""
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	(tmp_path / ".claude").mkdir(parents=True)
	(tmp_path / ".claude" / "settings.json").write_text(
		json.dumps(
			{
				"hooks": {
					"PreToolUse": [
						{
							"hooks": [
								{"type": "command", "command": command},
								{"type": "command", "command": "echo sentinel"},
							]
						}
					]
				}
			}
		)
	)
	assert write_hooks([]) == 0
	settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
	return {
		h["command"] for group in settings["hooks"].get("PreToolUse", []) for h in group["hooks"]
	}


@pytest.mark.parametrize(
	"command",
	[
		"camas mcp fix",
		"uv run tasks.py mcp gate --under 5s --nudge",
		"uvx 'camas[mcp]' mcp fix",
	],
)
def test_camas_hook_matcher_sweeps_every_launcher_form(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
	survivors = _sweep_survivors(tmp_path, monkeypatch, command)
	assert command not in survivors
	assert "echo sentinel" in survivors


@pytest.mark.parametrize("command", ["my-camas-tool mcp gateway", "camas mcp fixture --list"])
def test_camas_hook_matcher_does_not_false_match(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
	survivors = _sweep_survivors(tmp_path, monkeypatch, command)
	assert command in survivors
	assert "echo sentinel" in survivors


def test_write_hooks_reinit_does_not_accumulate_pep723_hooks(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""A PEP 723 launcher (uv run tasks.py mcp, no bare "camas") must be swept on re-init, so
	repeated `camas mcp init --claude` never stacks duplicate Stop/PostToolBatch hooks."""
	monkeypatch.chdir(tmp_path)
	(tmp_path / "tasks.py").write_text(
		'# /// script\n# dependencies = ["camas[mcp]>=0.1.8"]\n# ///\nfrom camas import Task\n'
	)
	monkeypatch.setattr("shutil.which", _which("uv"))
	assert write_hooks([]) == 0
	assert write_hooks([]) == 0
	settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
	assert len(settings["hooks"]["PostToolBatch"]) == 1
	assert len(settings["hooks"]["Stop"]) == 1
	assert "uv run tasks.py mcp fix" in settings["hooks"]["PostToolBatch"][0]["hooks"][0]["command"]


_TIERED_AGENT_FILES = (
	"camas-lint-fixer-haiku.md",
	"camas-lint-fixer-sonnet.md",
	"camas-test-fixer.md",
)


def test_write_claude_writes_all_generated_files(
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
	assert "Stop" in settings["hooks"]
	# tiered agents
	for filename in _TIERED_AGENT_FILES:
		agent = (tmp_path / ".claude" / "agents" / filename).read_text()
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
	for filename in _TIERED_AGENT_FILES:
		assert not (tmp_path / ".claude" / "agents" / filename).exists()


def test_write_claude_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	# First run
	assert write_claude([]) == 0
	mtime1 = (tmp_path / ".claude" / "agents" / "camas-lint-fixer-haiku.md").stat().st_mtime
	# Second run should overwrite
	assert write_claude([]) == 0
	mtime2 = (tmp_path / ".claude" / "agents" / "camas-lint-fixer-haiku.md").stat().st_mtime
	assert mtime2 > mtime1


_PEP723_TASKS = (
	"# /// script\n"
	'# requires-python = ">=3.10"\n'
	'# dependencies = ["camas[mcp]"]\n'
	"# ///\n"
	"from camas import run_cli\n"
	"if __name__ == '__main__':\n"
	"\trun_cli(globals())\n"
)


def test_run_cli_script_entry_init_claude_matches_project_init_agent_set(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""The PEP 723 script entry (``uv run tasks.py mcp init --claude``) and the project init
	(``camas mcp init --claude``) write the same tiered agent set — the parity the e2e
	``test_pep723_init_claude_via_script_entry_writes_uv_launcher`` asserts, guarded here so
	plain CI catches a divergence without the harness.
	"""
	from camas.main.dispatch import run_cli

	monkeypatch.setattr("shutil.which", _which("uv"))
	script_dir = tmp_path / "script"
	project_dir = tmp_path / "project"
	for d in (script_dir, project_dir):
		d.mkdir()
		(d / "tasks.py").write_text(_PEP723_TASKS)
	monkeypatch.chdir(script_dir)
	monkeypatch.setattr("sys.argv", ["tasks.py", "mcp", "init", "--claude"])
	with pytest.raises(SystemExit) as exc:
		run_cli({})
	assert exc.value.code == 0
	monkeypatch.chdir(project_dir)
	assert write_claude([]) == 0
	script_agents = sorted(p.name for p in (script_dir / ".claude" / "agents").iterdir())
	project_agents = sorted(p.name for p in (project_dir / ".claude" / "agents").iterdir())
	assert script_agents == project_agents == sorted(dest for _, dest in AGENT_TEMPLATES)
	assert (script_dir / ".claude" / "skills" / "gate" / "SKILL.md").exists()


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
	assert "--launcher" in out
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
	for filename in _TIERED_AGENT_FILES:
		assert (tmp_path / ".claude" / "agents" / filename).exists()
	assert (tmp_path / ".claude" / "skills" / "gate" / "SKILL.md").exists()


def test_parse_launcher_absent_is_none() -> None:
	from camas.mcp.cli import parse_launcher

	assert parse_launcher([]) is None
	assert parse_launcher(["--claude"]) is None


def test_parse_launcher_valid_value() -> None:
	from camas.mcp.cli import parse_launcher

	assert parse_launcher(["--launcher", "uvx"]) == "uvx"
	assert parse_launcher(["--claude", "--launcher", "camas"]) == "camas"


def test_parse_launcher_missing_value_raises() -> None:
	from camas.mcp.cli import parse_launcher

	with pytest.raises(ValueError, match="--launcher"):
		parse_launcher(["--launcher"])


def test_parse_launcher_invalid_value_raises() -> None:
	from camas.mcp.cli import parse_launcher

	with pytest.raises(ValueError, match="bogus"):
		parse_launcher(["--launcher", "bogus"])


def test_mcp_cli_init_invalid_launcher_exits_2(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	from camas.mcp.cli import main

	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("camas"))
	with pytest.raises(SystemExit) as exc:
		main(["init", "--launcher", "bogus"])
	assert exc.value.code == 2
	assert "--launcher" in capsys.readouterr().err
	assert not (tmp_path / ".mcp.json").exists()


def test_mcp_cli_init_launcher_camas_writes_bare_camas(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""``--launcher camas`` forces the bare-``camas``-on-PATH launcher (for nix/flake-provided
	camas) even when uv and a uv.lock project are also available."""
	from camas.mcp.cli import main

	monkeypatch.chdir(tmp_path)
	(tmp_path / "uv.lock").write_text("")
	monkeypatch.setattr("shutil.which", _which("uv", "uvx", "camas"))
	with pytest.raises(SystemExit) as exc:
		main(["init", "--launcher", "camas"])
	assert exc.value.code == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert (entry["command"], entry["args"]) == ("camas", ["mcp"])


def test_mcp_cli_init_launcher_uvx_forces_uvx_even_with_uv_lock(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	from camas.mcp.cli import main

	monkeypatch.chdir(tmp_path)
	(tmp_path / "uv.lock").write_text("")
	monkeypatch.setattr("shutil.which", _which("uv", "uvx", "camas"))
	_pin_installed_version(monkeypatch, _DEV_VERSION)
	with pytest.raises(SystemExit) as exc:
		main(["init", "--launcher", "uvx"])
	assert exc.value.code == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert entry["command"] == "uvx"


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
	for filename in _TIERED_AGENT_FILES:
		agent = (tmp_path / ".claude" / "agents" / filename).read_text()
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
	"""No PEP 723 pin and a dev/local build (not on PyPI to pin against) → unpinned fallback."""
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("uvx"))
	_pin_installed_version(monkeypatch, _DEV_VERSION)
	assert write_mcp_json([]) == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert entry["args"] == ["camas[mcp]", "mcp"]


def test_write_mcp_json_pinned_to_installed_version_when_release(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""No PEP 723 pin, but the running camas is a clean release → pin the uvx fallback to it."""
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("uvx"))
	_pin_installed_version(monkeypatch, _RELEASE_VERSION)
	assert write_mcp_json([]) == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert entry["args"] == [f"camas[mcp]=={_RELEASE_VERSION}", "mcp"]


def test_write_mcp_json_launcher_camas_writes_bare_camas(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "uv.lock").write_text("")
	monkeypatch.setattr("shutil.which", _which("uv", "uvx", "camas"))
	assert write_mcp_json([], launcher="camas") == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert (entry["command"], entry["args"]) == ("camas", ["mcp"])


def test_write_mcp_json_launcher_uvx_forces_uvx_even_with_uv_lock(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "uv.lock").write_text("")
	monkeypatch.setattr("shutil.which", _which("uv", "uvx", "camas"))
	_pin_installed_version(monkeypatch, _DEV_VERSION)
	assert write_mcp_json([], launcher="uvx") == 0
	entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert entry["command"] == "uvx"


def test_write_mcp_json_launcher_uv_errors_without_lock_or_pep723(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("uv", "uvx", "camas"))
	assert write_mcp_json([], launcher="uv") == 2
	err = capsys.readouterr().err
	assert "--launcher uv" in err
	assert "PEP 723" in err
	assert not (tmp_path / ".mcp.json").exists()


def test_write_mcp_json_launcher_camas_errors_without_path(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which())
	assert write_mcp_json([], launcher="camas") == 2
	err = capsys.readouterr().err
	assert "--launcher camas" in err
	assert "PATH" in err


def test_write_mcp_json_launcher_uvx_errors_without_uvx(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which("uv", "camas"))
	assert write_mcp_json([], launcher="uvx") == 2
	err = capsys.readouterr().err
	assert "--launcher uvx" in err
	assert "uvx on PATH" in err
	assert not (tmp_path / ".mcp.json").exists()


def test_write_hooks_launcher_matches_chosen_command(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "uv.lock").write_text("")
	monkeypatch.setattr("shutil.which", _which("uv", "uvx", "camas"))
	assert write_hooks([], launcher="camas") == 0
	out = capsys.readouterr().out
	assert "camas mcp fix" in out
	settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
	command = settings["hooks"]["PostToolBatch"][0]["hooks"][0]["command"]
	assert command == "camas mcp fix || exit 0"


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
	for filename in _TIERED_AGENT_FILES:
		assert not (tmp_path / ".claude" / "agents" / filename).exists()
	assert not (tmp_path / ".claude" / "skills" / "gate" / "SKILL.md").exists()
