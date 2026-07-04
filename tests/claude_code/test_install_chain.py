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

if TYPE_CHECKING:
	from collections.abc import Callable
	from subprocess import CompletedProcess

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UV = shutil.which("uv") or "uv"

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
	subprocess.run(
		[_UV, "sync"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		timeout=120,
		check=False,
		env={**os.environ, "UV_PYTHON_DOWNLOADS": "never"},
	)
	(tmp_path / "tasks.py").write_text(_TASKS)


def test_init_claude_writes_four_files_and_mcp_uses_portable_launcher(
	tmp_path: Path,
	run_headless: Callable[..., CompletedProcess[str]],
) -> None:
	_setup_project(tmp_path)

	proc = subprocess.run(
		[_UV, "run", "--project", str(_REPO_ROOT), "camas", "mcp", "init", "--claude"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		timeout=120,
		check=False,
	)
	assert proc.returncode == 0, proc.stderr

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


def test_corrupt_mcp_json_fails_strict_mcp_config(
	tmp_path: Path,
	run_headless: Callable[..., CompletedProcess[str]],
) -> None:
	_setup_project(tmp_path)

	subprocess.run(
		[_UV, "run", "--project", str(_REPO_ROOT), "camas", "mcp", "init", "--claude"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		timeout=120,
		check=False,
	)

	mcp = cast(
		"dict[str, object]", json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
	)
	servers = cast("dict[str, object]", mcp.setdefault("mcpServers", {}))
	servers["camas"] = {"type": "stdio", "command": "no-such-binary-camas"}
	(tmp_path / ".mcp.json").write_text(json.dumps(mcp, indent=2) + "\n", encoding="utf-8")

	headless = run_headless(
		tmp_path,
		"Call the camas_list MCP tool. Report how many tasks it lists.",
		strict_mcp=True,
	)
	failed = headless.returncode != 0 or "failed" in (headless.stderr or "").lower()
	assert failed, (
		f"corrupt .mcp.json should fail strict-mcp-config: rc={headless.returncode}"
		f" stderr={headless.stderr}"
	)
