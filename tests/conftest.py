# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Fixtures shared across the suite."""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
	from pathlib import Path


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
	"""A committed one-file repository the drift-gate pins run inside; the identity and
	signing configs are pinned so ambient git config cannot fail the commit. Skips when git
	is absent — the nix build's hermetic test phase has none on PATH."""
	if shutil.which("git") is None:  # pragma: no cover — only hermetic builds lack git
		pytest.skip("the pins run the drift check against a real git repository")
	subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
	(tmp_path / "tracked.txt").write_text("original\n", encoding="utf-8")
	subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
	subprocess.run(
		[
			"git",
			"-c",
			"user.name=t",
			"-c",
			"user.email=t@t",
			"-c",
			"commit.gpgSign=false",
			"commit",
			"-qm",
			"init",
		],
		cwd=tmp_path,
		check=True,
	)
	return tmp_path


@pytest.fixture
def unforced_color(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Clear the color environment, so a test reads camas's own decision rather than whatever the
	developer's shell or the CI runner exported into it.
	"""
	for name in ("NO_COLOR", "FORCE_COLOR", "CLICOLOR_FORCE"):
		monkeypatch.delenv(name, raising=False)
