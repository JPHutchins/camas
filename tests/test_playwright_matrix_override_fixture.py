# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

FIXTURE: Final = Path(__file__).parent / "fixtures" / "playwright-matrix-override"


def _camas(*args: str) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		[sys.executable, "-m", "camas", *args],
		cwd=FIXTURE,
		capture_output=True,
		text=True,
		encoding="utf-8",
		env={**os.environ, "NO_COLOR": "1"},
		check=False,
	)


def _count_combos(stdout: str) -> int:
	return sum("[BROWSER=" in line for line in stdout.splitlines())


def test_list_shows_two_axis_matrix_annotation() -> None:
	r = _camas("--list")
	assert r.returncode == 0, r.stderr
	assert "[matrix: BROWSER×3 (chromium..webkit) VIEWPORT×2 (desktop..mobile)]" in r.stdout, (
		r.stdout
	)


def test_e2e_default_expands_to_six_combos() -> None:
	r = _camas("--dry-run", "e2e")
	assert r.returncode == 0, r.stderr
	assert _count_combos(r.stdout) == 6, r.stdout


def test_pin_browser_collapses_to_two() -> None:
	r = _camas("--dry-run", "e2e", "--BROWSER", "chromium")
	assert r.returncode == 0, r.stderr
	assert _count_combos(r.stdout) == 2, r.stdout
	assert "[BROWSER=chromium, VIEWPORT=desktop]" in r.stdout
	assert "[BROWSER=chromium, VIEWPORT=mobile]" in r.stdout
	assert "BROWSER=firefox" not in r.stdout
	assert "BROWSER=webkit" not in r.stdout


def test_pin_viewport_collapses_to_three() -> None:
	r = _camas("--dry-run", "e2e", "--VIEWPORT", "mobile")
	assert r.returncode == 0, r.stderr
	assert _count_combos(r.stdout) == 3, r.stdout
	assert "VIEWPORT=desktop" not in r.stdout


def test_pin_both_axes_collapses_to_one() -> None:
	r = _camas("--dry-run", "e2e", "--BROWSER", "chromium", "--VIEWPORT", "mobile")
	assert r.returncode == 0, r.stderr
	assert _count_combos(r.stdout) == 1, r.stdout
	assert "[BROWSER=chromium, VIEWPORT=mobile]" in r.stdout


def test_comma_browser_expands_to_four() -> None:
	r = _camas("--dry-run", "e2e", "--BROWSER", "chromium,firefox")
	assert r.returncode == 0, r.stderr
	assert _count_combos(r.stdout) == 4, r.stdout
	assert "BROWSER=webkit" not in r.stdout


def test_per_task_help_lists_both_axes() -> None:
	r = _camas("e2e", "--help")
	assert r.returncode == 0, r.stderr
	assert "--BROWSER VAL[,VAL...]" in r.stdout, r.stdout
	assert "--VIEWPORT VAL[,VAL...]" in r.stdout, r.stdout
	assert "chromium, firefox, webkit" in r.stdout
	assert "desktop, mobile" in r.stdout


def test_unknown_axis_errors() -> None:
	r = _camas("e2e", "--matrix", "OS=linux")
	assert r.returncode == 2
	assert "unknown matrix axis 'OS'" in r.stderr, r.stderr
