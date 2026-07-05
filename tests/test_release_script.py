# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``.github/scripts/release.py`` helper: assert a clean synced main, bump VERSION, commit, tag."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="requires git")

SCRIPT = Path(__file__).resolve().parent.parent / ".github" / "scripts" / "release.py"


def _git(repo: Path, *args: str) -> str:
	return subprocess.run(
		["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
	).stdout.strip()


def _release(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		[sys.executable, str(SCRIPT), *args],
		cwd=repo,
		capture_output=True,
		text=True,
		check=False,
	)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
	origin = tmp_path / "origin.git"
	subprocess.run(
		["git", "init", "--bare", "--quiet", str(origin)], check=True, capture_output=True
	)
	repo = tmp_path / "repo"
	subprocess.run(
		["git", "init", "--quiet", "-b", "main", str(repo)], check=True, capture_output=True
	)
	_git(repo, "config", "user.email", "test@example.com")
	_git(repo, "config", "user.name", "Test")
	_git(repo, "config", "core.autocrlf", "false")
	(repo / "VERSION").write_text("0.1.0\n", encoding="utf-8", newline="\n")
	_git(repo, "add", "VERSION")
	_git(repo, "commit", "--quiet", "-m", "init")
	_git(repo, "remote", "add", "origin", str(origin))
	_git(repo, "push", "--quiet", "-u", "origin", "main")
	return repo


def test_release_bumps_commits_and_tags(repo: Path) -> None:
	result = _release(repo, "0.2.0")
	assert result.returncode == 0, result.stderr
	assert (repo / "VERSION").read_text(encoding="utf-8") == "0.2.0\n"
	assert _git(repo, "log", "-1", "--format=%s") == "release: 0.2.0"
	assert "0.2.0" in _git(repo, "tag", "--list")
	assert _git(repo, "status", "--porcelain") == ""
	assert "git push origin main 0.2.0" in result.stdout


def test_release_rejects_non_main_branch(repo: Path) -> None:
	_git(repo, "checkout", "--quiet", "-b", "feature")
	result = _release(repo, "0.2.0")
	assert result.returncode != 0
	assert "not main" in result.stderr


def test_release_rejects_dirty_tree(repo: Path) -> None:
	(repo / "stray.txt").write_text("dirty", encoding="utf-8")
	result = _release(repo, "0.2.0")
	assert result.returncode != 0
	assert "not clean" in result.stderr


def test_release_rejects_out_of_sync_main(repo: Path) -> None:
	(repo / "VERSION").write_text("0.1.1\n", encoding="utf-8", newline="\n")
	_git(repo, "commit", "--quiet", "-am", "ahead of origin")
	result = _release(repo, "0.2.0")
	assert result.returncode != 0
	assert "not in sync" in result.stderr


def test_release_rejects_current_version(repo: Path) -> None:
	result = _release(repo, "0.1.0")
	assert result.returncode != 0
	assert "already 0.1.0" in result.stderr


def test_release_rejects_existing_tag(repo: Path) -> None:
	_git(repo, "tag", "0.2.0")
	result = _release(repo, "0.2.0")
	assert result.returncode != 0
	assert "already exists" in result.stderr


@pytest.mark.parametrize("version", ["v0.2.0", "0.2.0rc1", "0.2.0.dev1", ""])
def test_release_rejects_non_release_version(repo: Path, version: str) -> None:
	result = _release(repo, version)
	assert result.returncode != 0
	assert "not a release version" in result.stderr


def test_release_requires_exactly_one_arg(repo: Path) -> None:
	result = _release(repo)
	assert result.returncode != 0
	assert "usage" in result.stderr
