# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""End-to-end ``tasks.dhall`` loading against the real ``dhall`` binding.

Skipped where ``camas[dhall]`` is not installed (the binding ships wheels only through cp311; on
newer interpreters it needs a Rust build) — the pure loader logic is covered by
``tests/main/test_dhall.py`` without the binding.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

pytest.importorskip("dhall")

FIXTURE: Final = Path(__file__).parent / "fixtures" / "dhall-monorepo"
SHOW_OUTPUT: Final = "--effects=(Summary(show_passing=True),)"


def _camas(*args: str, github: bool = False) -> subprocess.CompletedProcess[str]:
	env = {
		k: v
		for k, v in os.environ.items()
		if k not in ("CLAUDECODE", "CAMAS_AGENT", "GITHUB_ACTIONS")
	}
	env["NO_COLOR"] = "1"
	if github:
		env["GITHUB_ACTIONS"] = "true"
	return subprocess.run(
		[sys.executable, "-m", "camas", *args],
		cwd=FIXTURE,
		capture_output=True,
		text=True,
		encoding="utf-8",
		env=env,
		stdin=subprocess.DEVNULL,
		check=False,
	)


def test_list_shows_tasks_and_composed_namespace() -> None:
	r = _camas("--list")
	assert r.returncode == 0, r.stderr
	for name in ("format", "lint", "typecheck", "check", "gate", "all", "matrix", "libs"):
		assert name in r.stdout, r.stdout
	for name in ("libs.lint", "libs.test", "libs.check"):
		assert name in r.stdout, r.stdout


def test_tree_expands_matrix_and_nested_groups() -> None:
	r = _camas("--tree", "matrix")
	assert r.returncode == 0, r.stderr
	for py in ("3.13", "3.14", "3.15"):
		assert f"PY={py}" in r.stdout, r.stdout
	assert "typecheck" in r.stdout, r.stdout


def test_dry_run_rebases_child_cwd() -> None:
	r = _camas("--dry-run", "libs.check")
	assert r.returncode == 0, r.stderr
	assert f"(cwd: {Path('libs')})" in r.stdout, r.stdout


def test_bare_runs_agent_default_when_configured() -> None:
	r = _camas("--dry-run")
	assert r.returncode == 0, r.stderr
	assert "ruff format --check {paths}" not in r.stdout or "ruff check --fix" in r.stdout, r.stdout


def test_check_matches_python_definition() -> None:
	r = _camas("--dry-run", "check")
	assert r.returncode == 0, r.stderr
	assert "mypy ." in r.stdout, r.stdout
	assert "pyright src tests" in r.stdout, r.stdout
