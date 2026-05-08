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
	assert "[matrix: PY×6 (3.10..3.15)]" in r.stdout, r.stdout


def test_per_axis_flag_overrides_matrix() -> None:
	r = _camas("--dry-run", "check", "--PY", "3.13")
	assert r.returncode == 0, r.stderr
	assert "[PY=3.13]" in r.stdout, r.stdout
	for v in ("3.10", "3.11", "3.12", "3.14", "3.15"):
		assert f"[PY={v}]" not in r.stdout, r.stdout


def test_per_axis_flag_accepts_comma_separated_values() -> None:
	r = _camas("--dry-run", "check", "--PY", "3.13,3.14")
	assert r.returncode == 0, r.stderr
	assert "[PY=3.13]" in r.stdout
	assert "[PY=3.14]" in r.stdout
	assert "[PY=3.12]" not in r.stdout


def test_generic_matrix_flag_overrides_axis() -> None:
	r = _camas("--dry-run", "check", "--matrix", "PY=3.13")
	assert r.returncode == 0, r.stderr
	assert "[PY=3.13]" in r.stdout
	assert "[PY=3.14]" not in r.stdout


def test_unknown_axis_errors() -> None:
	r = _camas("check", "--matrix", "XX=1")
	assert r.returncode == 2
	assert "unknown matrix axis 'XX'" in r.stderr, r.stderr


def test_per_task_help_lists_axes() -> None:
	r = _camas("check", "--help")
	assert r.returncode == 0, r.stderr
	assert "--PY VAL[,VAL...]" in r.stdout, r.stdout
	assert "Matrix axes" in r.stdout, r.stdout


def test_check_dry_run_expands_one_clone_per_python_version() -> None:
	versions = _python_versions()
	assert versions, "fixture .python-version is empty"
	r = _camas("--dry-run", "check")
	assert r.returncode == 0, r.stderr
	for version in versions:
		assert f"[PY={version}]" in r.stdout, r.stdout
	header_count = sum("check [PY=" in line for line in r.stdout.splitlines())
	assert header_count == len(versions), r.stdout
