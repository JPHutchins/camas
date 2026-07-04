# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Drive the ``camas-fixer`` subagent end to end against a failing scope, prove it reaches green
via the MCP gate tool (criterion #3).

The ``camas-fixer`` agent definition ships with ``tools: Read, Edit, mcp__camas__camas_gate,
mcp__camas__camas_fix`` and ``model: haiku`` — it has no Bash, so it physically cannot shell
out to the CLI; using the MCP tool is structurally enforced.

Happy path: set up a tasks.py whose check node fails when a file contains ``FORBIDDEN_TOKEN``
and whose fix node mechanically replaces it with ``ALLOWED_TOKEN``. After ``init --claude``,
run headless (``--permission-mode bypassPermissions --strict-mcp-config``), instruct the main
agent to write the forbidden token and spawn camas-fixer on the scope. Assert the marker is
fixed on disk.

Broken variant: overwrite the shipped ``camas-fixer.md`` with a copy whose ``tools:`` line
is ``Read, Edit`` (no MCP gate/fix tools). Re-run; assert the scope is NOT green — the fixer
without its gate tool cannot drive the scope.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable
	from subprocess import CompletedProcess

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UV = shutil.which("uv") or "uv"
_ENV = {**os.environ, "UV_PYTHON": "3.12", "UV_PYTHON_DOWNLOADS": "never"}

_PY = shlex.quote(sys.executable)

_PYPROJECT = (
	"[project]\n"
	'name = "test-harness"\n'
	'version = "0.0.0"\n'
	'requires-python = ">=3.10"\n'
	'dependencies = ["camas"]\n'
	"\n[tool.uv.sources]\n"
	f'camas = {{ path = "{_REPO_ROOT}" }}\n'
)

_CHECK_SCRIPT = (
	"import sys, pathlib\n"
	"bad = any('FORBIDDEN_TOKEN' in pathlib.Path(p).read_text() for p in sys.argv[1:])\n"
	"sys.exit(1 if bad else 0)\n"
)

_FIX_SCRIPT = (
	"import pathlib, sys\n"
	"for p in sys.argv[1:]:\n"
	"    fp = pathlib.Path(p)\n"
	'    fp.write_text(fp.read_text().replace("FORBIDDEN_TOKEN", "ALLOWED_TOKEN"))\n'
)

_TASKS = (
	"from camas import Claude, Config, Task\n"
	f'check = Task("{_PY} check_fail.py {{paths}}", name="check", paths=".")\n'
	f'fix = Task("{_PY} fixer.py {{paths}}", name="fix", mutates=True, paths=".")\n'
	"_ = Config(agent=Claude(fix=fix, check=check))\n"
)

_FIXER_MD_TOOLS_LINE = "tools: Read, Edit, mcp__camas__camas_gate, mcp__camas__camas_fix"
_FIXER_MD_BROKEN_TOOLS = "tools: Read, Edit"

_DELEGATE_PROMPT = (
	"Create a file named sentinel.txt containing exactly the word FORBIDDEN_TOKEN. "
	"Use the Write tool and nothing else. "
	"Then spawn the camas-fixer subagent with its scope set to paths=['sentinel.txt'] "
	"and wait for it to finish. "
	"Do NOT edit sentinel.txt yourself — let the subagent do its work."
)


def _setup_project(tmp_path: Path) -> None:
	(tmp_path / "pyproject.toml").write_text(_PYPROJECT)
	(tmp_path / "tasks.py").write_text(_TASKS)
	(tmp_path / "check_fail.py").write_text(_CHECK_SCRIPT)
	(tmp_path / "fixer.py").write_text(_FIX_SCRIPT)
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


def _init_claude(tmp_path: Path) -> None:
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


def _marker_is_fixed(sentinel: Path) -> bool:
	if not sentinel.exists():
		return False
	content = sentinel.read_text()
	return "FORBIDDEN_TOKEN" not in content and "ALLOWED_TOKEN" in content


def test_fixer_subagent_drives_scope_to_green_via_mcp_gate(
	tmp_path: Path,
	run_headless: Callable[..., CompletedProcess[str]],
) -> None:
	_setup_project(tmp_path)
	_init_claude(tmp_path)

	headless = run_headless(
		tmp_path,
		_DELEGATE_PROMPT,
		permission_mode="bypassPermissions",
		strict_mcp=True,
	)
	assert headless.returncode == 0, headless.stderr

	sentinel = tmp_path / "sentinel.txt"
	assert _marker_is_fixed(sentinel), (
		f"camas-fixer did not reach green: {sentinel.read_text() if sentinel.exists() else 'no file'}"
	)


def test_fixer_without_mcp_tools_cannot_reach_green(
	tmp_path: Path,
	run_headless: Callable[..., CompletedProcess[str]],
) -> None:
	_setup_project(tmp_path)
	_init_claude(tmp_path)

	fixer_md = tmp_path / ".claude" / "agents" / "camas-fixer.md"
	original = fixer_md.read_text(encoding="utf-8")
	assert _FIXER_MD_TOOLS_LINE in original, (
		f"shipped camas-fixer.md must contain the tools line: {_FIXER_MD_TOOLS_LINE!r}"
	)
	sabotaged = original.replace(_FIXER_MD_TOOLS_LINE, _FIXER_MD_BROKEN_TOOLS)
	assert _FIXER_MD_TOOLS_LINE not in sabotaged
	fixer_md.write_text(sabotaged, encoding="utf-8")

	run_headless(
		tmp_path,
		_DELEGATE_PROMPT,
		permission_mode="bypassPermissions",
		strict_mcp=True,
	)

	sentinel = tmp_path / "sentinel.txt"
	assert not _marker_is_fixed(sentinel), (
		"fixer without MCP gate tool should not reach green; "
		f"sentinel: {sentinel.read_text() if sentinel.exists() else 'no file'}"
	)
