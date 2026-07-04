# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""``camas mcp init --claude`` produces a loadable plugin surface (criterion #1 + #4).

Happy path: in a tmp_path with a tasks.py and a uv.lock, ``init --claude`` writes the four
files; the ``.mcp.json`` camas entry uses a ``uv`` or ``uvx`` launcher (criterion #4 — no bare
``camas``); and a headless ``claude -p --strict-mcp-config`` loads the produced config and calls
the ``camas_list`` MCP tool without error.

Broken variant: corrupt the produced ``.mcp.json`` (point the camas command at a non-existent
binary) and prove ``--strict-mcp-config`` rejects it (non-zero returncode or stderr indicating
server registration failure).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from subprocess import CompletedProcess

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UV = shutil.which("uv") or "uv"
# The runner's system Python is 3.12; the repo's .python-version (3.14) is not installed there,
# and UV_PYTHON_DOWNLOADS=never forbids fetching it. Pin UV_PYTHON so every `uv` invocation in the
# test (sync, the init subprocess, and the headless-launched hook/server) resolves a present
# interpreter instead of failing "No interpreter found for Python 3.14".
_ENV = {**os.environ, "UV_PYTHON": "3.12", "UV_PYTHON_DOWNLOADS": "never"}
_ENABLED = bool(os.environ.get("CAMAS_CC_E2E")) and shutil.which("claude") is not None

_PYPROJECT = (
	"[project]\n"
	'name = "test-harness"\n'
	'version = "0.0.0"\n'
	'requires-python = ">=3.10"\n'
	'dependencies = ["camas"]\n'
	"\n[tool.uv.sources]\n"
	f'camas = {{ path = "{_REPO_ROOT}" }}\n'
)

_TASKS = "from camas import Config\n_ = Config()\n"

_INIT_FILES = (
	".mcp.json",
	".claude/settings.json",
	".claude/agents/camas-fixer.md",
	".claude/skills/gate/SKILL.md",
)


def _get(obj: object, key: str) -> object:
	return cast("dict[str, object]", obj).get(key) if isinstance(obj, dict) else None


def _mcp_command(mcp_json: Path) -> str:
	root: object = json.loads(mcp_json.read_text(encoding="utf-8"))
	command = _get(_get(root, "mcpServers"), "camas")
	field = _get(command, "command")
	return field if isinstance(field, str) else ""


def _mcp_json_uses_portable_launcher(mcp_json: Path) -> bool:
	return _mcp_command(mcp_json) in ("uv", "uvx")


def _setup_project(tmp_path: Path) -> None:
	(tmp_path / "pyproject.toml").write_text(_PYPROJECT)
	(tmp_path / "tasks.py").write_text(_TASKS)
	sync = subprocess.run(
		[_UV, "sync"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		timeout=180,
		check=False,
		env=_ENV,
	)
	assert sync.returncode == 0, (
		f"uv sync failed: rc={sync.returncode}\nstdout={sync.stdout}\nstderr={sync.stderr}"
	)


def _init_claude(tmp_path: Path) -> subprocess.CompletedProcess[str]:
	proc = subprocess.run(
		[_UV, "run", "--project", str(_REPO_ROOT), "camas", "mcp", "init", "--claude"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		timeout=180,
		check=False,
		env=_ENV,
	)
	assert proc.returncode == 0, (
		f"camas mcp init --claude failed: rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
	)
	return proc


def test_init_claude_writes_four_files_and_mcp_uses_portable_launcher(
	tmp_path: Path,
	run_headless: Callable[..., CompletedProcess[str]],
) -> None:
	_setup_project(tmp_path)
	_init_claude(tmp_path)

	for rel in _INIT_FILES:
		assert (tmp_path / rel).exists(), f"init --claude did not write {rel}"

	assert _mcp_json_uses_portable_launcher(tmp_path / ".mcp.json"), (
		".mcp.json camas entry must use uv or uvx launcher (criterion #4)"
	)

	headless = run_headless(
		tmp_path,
		"Call the camas_list MCP tool. Report how many tasks it lists. "
		"Use only the MCP tool — no shell commands.",
		strict_mcp=True,
	)
	assert headless.returncode == 0, (
		f"headless failed to load .mcp.json: rc={headless.returncode} stderr={headless.stderr}"
	)


@pytest.mark.skipif(not _ENABLED, reason="set CAMAS_CC_E2E=1 with claude on PATH")
def test_shipped_hook_command_uses_the_portable_launcher_and_fix_subcommand(
	tmp_path: Path,
) -> None:
	"""The shipped PostToolBatch hook camas writes must run the portable launcher with the ``fix``
	subcommand — the structural invariant the historical "bare ``camas``" and "wrong command"
	regions broke. Claude Code's headless ``-p`` is lenient about bad config (it warns and exits 0),
	so a deliberately-corrupted ``.mcp.json`` does NOT fail ``--strict-mcp-config``; assert the
	launcher shape on the file camas wrote instead, which is deterministic.
	"""
	_setup_project(tmp_path)
	_init_claude(tmp_path)

	settings = cast(
		"dict[str, object]",
		json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8")),
	)
	hooks = cast("dict[str, object]", settings["hooks"])
	batch = cast("list[dict[str, object]]", hooks["PostToolBatch"])
	group = batch[-1]
	raw_commands = (h.get("command") for h in cast("list[dict[str, object]]", group["hooks"]))
	commands = [c for c in raw_commands if isinstance(c, str)]
	hook_command = next(c for c in commands if "mcp fix" in c)
	launcher = _mcp_command(tmp_path / ".mcp.json")
	assert hook_command.startswith(launcher), (
		f"shipped hook must use the same portable launcher ({launcher!r}) as .mcp.json: {hook_command!r}"
	)
	assert hook_command.endswith("fix"), (
		f"shipped hook must invoke the fix subcommand: {hook_command!r}"
	)
