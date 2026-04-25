# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

FIXTURE: Final = Path(__file__).parent / "fixtures" / "tauri-app"


def _camas(*args: str) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		[sys.executable, "-m", "camas", *args],
		cwd=FIXTURE,
		capture_output=True,
		text=True,
		env={**os.environ, "NO_COLOR": "1"},
	)


def test_list_shows_top_level_tasks() -> None:
	r = _camas("--list")
	assert r.returncode == 0
	for name in ("all", "fix", "build"):
		assert name in r.stdout, r.stdout


def test_all_dry_run_includes_all_tools() -> None:
	r = _camas("--dry-run", "all")
	assert r.returncode == 0
	for cmd in (
		"prettier --write .",
		"tsc --noEmit",
		"eslint src/",
		"vitest run",
		"cargo fmt --all",
		"cargo clippy",
		"cargo test",
		"uv run ruff check --fix .",
		"uv run ruff format .",
		"uv run mypy .",
		"uv run pytest",
	):
		assert cmd in r.stdout, r.stdout


def test_python_sdk_tasks_have_sdk_cwd() -> None:
	r = _camas("--dry-run", "all")
	assert r.returncode == 0
	assert "(cwd: python-sdk)" in r.stdout


def test_cargo_tasks_have_src_tauri_cwd() -> None:
	r = _camas("--dry-run", "all")
	assert r.returncode == 0
	assert "(cwd: src-tauri)" in r.stdout


def test_tauri_build_matrix_expands_debug_and_release() -> None:
	r = _camas("--dry-run", "build")
	assert r.returncode == 0
	assert "[FLAG=-- --debug]" in r.stdout
	assert "[FLAG=]" in r.stdout


def test_missing_task_fails() -> None:
	r = _camas("nonexistent_task_xyz")
	assert r.returncode == 2
	assert "nonexistent_task_xyz" in r.stderr
