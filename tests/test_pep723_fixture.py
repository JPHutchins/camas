# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import camas.mcp.cli
from camas import Task
from camas.main.dispatch import run_cli
from camas.main.tasks import load_tasks_from_py

if TYPE_CHECKING:
	import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "pep723" / "tasks.py"


def test_pep723_header_is_inert_to_loader() -> None:
	"""The PEP 723 header (and the ``__main__`` block) leave the loader's view of
	the module unchanged — it reads identically to a header-less ``tasks.py``."""
	loaded = load_tasks_from_py(FIXTURE)
	assert loaded.scope_effects == {}
	assert loaded.tasks["hello"] == Task(
		("python", "-c", "print('hello from pep723')"), name="hello"
	)


def test_pep723_runs_standalone() -> None:
	"""``python tasks.py hello`` dispatches through the ``run_cli(globals())`` entry
	point using the camas in the current environment. (PEP 723 dependency
	resolution is uv's concern, not camas's, so it is not exercised here.)"""
	result = subprocess.run(
		[
			sys.executable,
			str(FIXTURE),
			"hello",
			"--effects",
			"(Summary(show_passing=True),)",
		],
		capture_output=True,
		text=True,
		check=False,
	)
	assert result.returncode == 0, result.stderr
	assert "hello from pep723" in result.stdout


def test_pep723_mcp_subcommand_dispatches_to_mcp_cli() -> None:
	"""``python tasks.py mcp --help`` must route to the MCP CLI, not raise 'no task
	named mcp' — the regression test for issue #163."""
	result = subprocess.run(
		[sys.executable, str(FIXTURE), "mcp", "--help"],
		capture_output=True,
		text=True,
		check=False,
	)
	assert result.returncode == 0, result.stderr
	assert "camas mcp init" in result.stdout
	combined = result.stderr + result.stdout
	assert "no task named" not in combined


def test_pep723_mcp_init_runs_from_script_path(tmp_path: Path) -> None:
	"""``python tasks.py mcp init`` through the ``run_cli`` entry writes .mcp.json
	when a ``camas`` launcher is on PATH."""
	dest = tmp_path / "tasks.py"
	shutil.copy2(str(FIXTURE), str(dest))
	fake_camas = tmp_path / "camas"
	fake_camas.write_text("#!/bin/sh\nexit 0\n")
	fake_camas.chmod(0o755)
	env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
	result = subprocess.run(
		[sys.executable, str(dest), "mcp", "init"],
		cwd=tmp_path,
		env=env,
		capture_output=True,
		text=True,
		check=False,
	)
	assert result.returncode == 0, result.stderr
	mcp_json = tmp_path / ".mcp.json"
	assert mcp_json.exists()
	parsed = json.loads(mcp_json.read_text())
	assert "camas" in parsed["mcpServers"]


def test_run_cli_routes_mcp_subcommand_to_mcp_cli(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Regression test for issue #163: ``run_cli`` routes ``mcp`` to ``camas.mcp.cli``."""
	monkeypatch.setattr(sys, "argv", ["tasks.py", "mcp", "--help"])
	recorded: list[list[str]] = []

	def _fake_main(argv: list[str]) -> None:
		recorded.append(argv)

	monkeypatch.setattr(camas.mcp.cli, "main", _fake_main)
	run_cli({})
	assert recorded == [["--help"]]
