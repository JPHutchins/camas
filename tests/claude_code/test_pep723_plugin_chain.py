# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""A PEP 723 ``tasks.py`` (no ``uv.lock``, no installed camas) drives the whole plugin chain
through the script entry ``uv run --script tasks.py mcp …`` — the non-Python-repo compat the
recent ``run_cli`` ``mcp`` routing (``dea221e`` / #160) added.

Happy path: ``uv run --script tasks.py mcp init --claude`` writes the generated files; the
``.mcp.json`` camas entry uses the ``uv`` launcher (a PEP 723 ``tasks.py`` with a camas
dependency → ``launch_command`` emits ``uv run tasks.py mcp`` — criterion #4, no bare ``camas``).

Regression guard: ``uv run --script tasks.py mcp --help`` routes to the MCP CLI (output mentions
``camas mcp init``) and does NOT raise ``no task named 'mcp'`` — the pre-``dea221e`` failure mode
where ``run_cli`` bound ``mcp`` as a task. The headless server-load step (proving the
``uv run tasks.py mcp`` launcher Claude Code starts actually answers ``camas_list``) is opt-in via
``CAMAS_CC_PEP723_HEADLESS`` because it downloads camas from PyPI on each launch — slow and
network-bound, unlike the deterministic file/launcher assertions that always run.
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
_ENV = {**os.environ, "UV_PYTHON": "3.12", "UV_PYTHON_DOWNLOADS": "never"}
_HEADLESS = bool(os.environ.get("CAMAS_CC_PEP723_HEADLESS"))
_ENABLED = bool(os.environ.get("CAMAS_CC_E2E")) and shutil.which("claude") is not None

# A non-Python repo's tasks.py is a PEP 723 inline-script block pinning camas[mcp]; the ``[mcp]``
# extra is required because the ``mcp`` subcommand imports pydantic (a camas[mcp] dep), and
# ``run_cli(globals())`` is the script entry ``dea221e`` routed to ``mcp``.
_TASKS = (
	"# /// script\n"
	'# requires-python = ">=3.10"\n'
	'# dependencies = ["camas[mcp]"]\n'
	"# ///\n"
	"from camas import run_cli\n"
	"if __name__ == '__main__':\n"
	"\trun_cli(globals())\n"
)

_INIT_FILES = (
	".mcp.json",
	".claude/settings.json",
	".claude/agents/camas-lint-fixer-haiku.md",
	".claude/agents/camas-lint-fixer-sonnet.md",
	".claude/agents/camas-test-fixer.md",
	".claude/skills/gate/SKILL.md",
)


def _get(obj: object, key: str) -> object:
	return cast("dict[str, object]", obj).get(key) if isinstance(obj, dict) else None


def _mcp_command(mcp_json: Path) -> str:
	root: object = json.loads(mcp_json.read_text(encoding="utf-8"))
	command = _get(_get(root, "mcpServers"), "camas")
	field = _get(command, "command")
	return field if isinstance(field, str) else ""


def _init_via_script_entry(tmp_path: Path) -> subprocess.CompletedProcess[str]:
	proc = subprocess.run(
		[_UV, "run", "--script", "tasks.py", "mcp", "init", "--claude"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		timeout=180,
		check=False,
		env=_ENV,
	)
	assert proc.returncode == 0, (
		f"uv run --script tasks.py mcp init --claude failed: "
		f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
	)
	return proc


def test_pep723_init_claude_via_script_entry_writes_uv_launcher(
	tmp_path: Path,
	run_headless: Callable[..., CompletedProcess[str]],
) -> None:
	(tmp_path / "tasks.py").write_text(_TASKS)
	_init_via_script_entry(tmp_path)

	for rel in _INIT_FILES:
		assert (tmp_path / rel).exists(), f"init --claude did not write {rel}"

	assert _mcp_command(tmp_path / ".mcp.json") == "uv", (
		".mcp.json camas entry must use the uv launcher for a PEP 723 repo "
		"(a PEP 723 tasks.py with a camas dependency → uv run tasks.py mcp) — criterion #4"
	)


def test_pep723_init_claude_headless_server_loads(
	tmp_path: Path,
	run_headless: Callable[..., CompletedProcess[str]],
) -> None:
	if not _HEADLESS:
		pytest.skip(
			"set CAMAS_CC_PEP723_HEADLESS=1 to run the uvx server-load step (downloads camas)"
		)

	(tmp_path / "tasks.py").write_text(_TASKS)
	_init_via_script_entry(tmp_path)

	headless = run_headless(
		tmp_path,
		"Call the camas_list MCP tool. Report how many tasks it lists. "
		"Use only the MCP tool — no shell commands.",
		strict_mcp=True,
	)
	assert headless.returncode == 0, (
		f"headless failed to load the uvx-launched .mcp.json: "
		f"rc={headless.returncode} stderr={headless.stderr}"
	)


@pytest.mark.skipif(not _ENABLED, reason="set CAMAS_CC_E2E=1 with claude on PATH")
def test_pep723_script_entry_routes_mcp_not_task_binding(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text(_TASKS)
	proc = subprocess.run(
		[_UV, "run", "--script", "tasks.py", "mcp", "--help"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		timeout=120,
		check=False,
		env=_ENV,
	)
	assert proc.returncode == 0, proc.stderr
	combined = proc.stderr + proc.stdout
	assert "camas mcp init" in combined, "mcp --help did not route to the MCP CLI"
	assert "no task named" not in combined, (
		f"run_cli bound 'mcp' as a task instead of routing to camas.mcp.cli: {combined!r}"
	)
