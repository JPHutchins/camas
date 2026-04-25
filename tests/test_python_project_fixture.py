# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

FIXTURE: Final = Path(__file__).parent / "fixtures" / "python-project"


def _camas(*args: str) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		[sys.executable, "-m", "camas", *args],
		cwd=FIXTURE,
		capture_output=True,
		text=True,
		env={**os.environ, "NO_COLOR": "1"},
	)


def test_list_shows_pyproject_tasks() -> None:
	r = _camas("--list")
	assert r.returncode == 0
	for name in ("all", "check", "format", "lint", "mypy", "test", "typecheck", "build"):
		assert name in r.stdout, r.stdout


def test_check_dry_run_expands_refs() -> None:
	r = _camas("--dry-run", "check")
	assert r.returncode == 0
	for cmd in ("ruff format --check .", "ruff check .", "mypy src tests", "pytest"):
		assert cmd in r.stdout, r.stdout


def test_build_matrix_expands_sdist_and_wheel() -> None:
	r = _camas("--dry-run", "build")
	assert r.returncode == 0
	assert "[FLAG=--sdist]" in r.stdout
	assert "[FLAG=--wheel]" in r.stdout


def test_missing_task_fails() -> None:
	r = _camas("nonexistent_task_xyz")
	assert r.returncode == 2
	assert "nonexistent_task_xyz" in r.stderr
