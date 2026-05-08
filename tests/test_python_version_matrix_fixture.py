# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

FIXTURE: Final = Path(__file__).parent / "fixtures" / "python-version-matrix"


def _camas(*args: str) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		[sys.executable, "-m", "camas", *args],
		cwd=FIXTURE,
		capture_output=True,
		text=True,
		encoding="utf-8",
		env={**os.environ, "NO_COLOR": "1"},
	)


def _python_versions() -> tuple[str, ...]:
	return tuple(
		stripped
		for line in (FIXTURE / ".python-version").read_text().splitlines()
		if (stripped := line.strip()) and not stripped.startswith("#")
	)


def test_list_shows_check_with_matrix_annotation() -> None:
	r = _camas("--list")
	assert r.returncode == 0, r.stderr
	assert "check" in r.stdout, r.stdout
	assert "[matrix: PY]" in r.stdout, r.stdout


def test_check_dry_run_expands_one_clone_per_python_version() -> None:
	versions = _python_versions()
	assert versions, "fixture .python-version is empty"
	r = _camas("--dry-run", "check")
	assert r.returncode == 0, r.stderr
	for version in versions:
		assert f"[PY={version}]" in r.stdout, r.stdout
	header_count = sum("check [PY=" in line for line in r.stdout.splitlines())
	assert header_count == len(versions), r.stdout
