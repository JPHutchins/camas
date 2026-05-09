# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

FIXTURE: Final = Path(__file__).parent / "fixtures" / "effect-plugin"


def _camas(*args: str) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		[sys.executable, "-m", "camas", *args],
		cwd=FIXTURE,
		capture_output=True,
		text=True,
		encoding="utf-8",
		env={**os.environ, "NO_COLOR": "1"},
	)


def test_effects_listing_includes_user_and_builtin() -> None:
	r = _camas("--effects")
	assert r.returncode == 0, r.stderr
	assert "Tail" in r.stdout, r.stdout
	assert "Summary" in r.stdout, r.stdout
	assert "Termtree" in r.stdout, r.stdout


def test_user_effect_dry_run_accepts_at_parser() -> None:
	r = _camas("--dry-run", "check", "--effects=(Tail(),)")
	assert r.returncode == 0, r.stderr


def test_user_effect_runs_end_to_end() -> None:
	r = _camas("check", "--effects=(Tail(),)")
	assert r.returncode == 0, r.stderr
	for expected in ("fast: fast 0", "fast: fast 1", "slow: slow 0", "slow: slow 2", "done: done"):
		assert expected in r.stdout, r.stdout


def test_user_effect_mixed_with_builtin() -> None:
	r = _camas("check", "--effects=(Tail(), Summary())")
	assert r.returncode == 0, r.stderr
	# Tail's per-task output appears alongside Summary's final tree.
	assert "fast: fast 0" in r.stdout, r.stdout
	assert "done: done" in r.stdout, r.stdout
	assert "PASS" in r.stdout, r.stdout


def test_fixture_typechecks_under_strict_mypy() -> None:
	mypy: Final = shutil.which("mypy")
	if mypy is None:  # pragma: no cover
		pytest.skip("mypy not on PATH")
	r = subprocess.run(
		[mypy, "--strict", "tasks.py"],
		cwd=FIXTURE,
		capture_output=True,
		text=True,
	)
	assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
