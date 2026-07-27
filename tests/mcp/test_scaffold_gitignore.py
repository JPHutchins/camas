# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""``camas mcp init`` warns when git excludes the files it just wrote, so a ``--claude`` loop that
can never reach a teammate does not pass for a shared one (#243).

The tests that need a real ``git check-ignore`` skip without git on PATH — the nix check sandbox
runs pytest without it.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from camas.mcp.scaffold import (
	CLAUDE_TARGETS,
	Excluded,
	check_ignore,
	excluded_roots,
	gitignore_warning,
	unignore,
	warn_uncommittable,
	write_claude,
	write_mcp_json,
)

if TYPE_CHECKING:
	from collections.abc import Callable
	from pathlib import Path

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="requires git")


def _repo(root: Path, gitignore: str) -> None:
	subprocess.run(["git", "init", "--quiet", str(root)], check=True, capture_output=True)
	(root / ".gitignore").write_text(gitignore, encoding="utf-8")


def _git(root: Path, *args: str) -> str:
	return subprocess.run(
		["git", "-C", str(root), *args], check=True, capture_output=True, text=True
	).stdout


def _which_git(git: str | None, *found: str) -> Callable[[str], str | None]:
	"""``shutil.which`` resolving git to ``git`` — the real binary, an unrunnable path, or nothing —
	and faking the launcher probe for ``found``.
	"""
	return lambda name: git if name == "git" else (f"/usr/bin/{name}" if name in found else None)


@requires_git
def test_check_ignore_reports_the_excluding_rule(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	_repo(tmp_path, "build/\n.claude/\n")
	(tmp_path / ".claude").mkdir()
	monkeypatch.chdir(tmp_path)
	assert check_ignore((".mcp.json", ".claude", ".claude/settings.json")) == (
		Excluded(path=".claude", source=".gitignore", line="2", pattern=".claude/"),
		Excluded(path=".claude/settings.json", source=".gitignore", line="2", pattern=".claude/"),
	)


@requires_git
def test_check_ignore_skips_a_tracked_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	"""A force-added file is committable, and git says so — the warning must not cry over it."""
	_repo(tmp_path, ".mcp.json\n")
	(tmp_path / ".mcp.json").write_text("{}\n", encoding="utf-8")
	_git(tmp_path, "add", "-f", ".mcp.json")
	monkeypatch.chdir(tmp_path)
	assert check_ignore((".mcp.json",)) == ()


@requires_git
def test_check_ignore_skips_a_negated_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_repo(tmp_path, ".claude/*\n!.claude/settings.json\n")
	monkeypatch.chdir(tmp_path)
	assert check_ignore((".claude/settings.json", ".claude/agents")) == (
		Excluded(path=".claude/agents", source=".gitignore", line="1", pattern=".claude/*"),
	)


@requires_git
def test_check_ignore_outside_a_repository_is_empty(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	assert check_ignore((".mcp.json",)) == ()


def test_check_ignore_without_git_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _which_git(None))
	assert check_ignore((".mcp.json",)) == ()


def test_check_ignore_survives_an_unrunnable_git(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr("shutil.which", _which_git(str(tmp_path / "no-such-git")))
	assert check_ignore((".mcp.json",)) == ()


@requires_git
def test_excluded_roots_reports_the_outermost_directory_once(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	_repo(tmp_path, ".claude/\n")
	(tmp_path / ".claude").mkdir()
	monkeypatch.chdir(tmp_path)
	assert excluded_roots(CLAUDE_TARGETS) == (
		Excluded(path=".claude", source=".gitignore", line="1", pattern=".claude/"),
	)


@requires_git
def test_excluded_roots_skips_a_committable_path(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	_repo(tmp_path, ".claude/\n")
	(tmp_path / ".claude").mkdir()
	monkeypatch.chdir(tmp_path)
	assert excluded_roots((".mcp.json", ".claude/settings.json")) == (
		Excluded(path=".claude", source=".gitignore", line="1", pattern=".claude/"),
	)


@requires_git
def test_gitignore_warning_names_the_rule_and_the_lines_that_undo_it(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	_repo(tmp_path, ".claude/\n")
	(tmp_path / ".claude").mkdir()
	monkeypatch.chdir(tmp_path)
	assert gitignore_warning(CLAUDE_TARGETS, consequence="teammates lose the loop") == (
		"warning: git will not commit what camas just wrote — teammates lose the loop.\n"
		"  .claude is excluded by .gitignore:1 (`.claude/`) — git cannot re-include a path whose "
		"parent directory is excluded, so that rule has to stop matching .claude itself, e.g.:\n"
		"      .claude/*\n"
		"      !.claude/settings.json\n"
		"      !.claude/agents/\n"
		"      !.claude/skills/"
	)


@requires_git
def test_gitignore_warning_reports_every_excluded_root(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	_repo(tmp_path, ".mcp.json\n.claude/\n")
	(tmp_path / ".claude").mkdir()
	monkeypatch.chdir(tmp_path)
	warning = gitignore_warning((".mcp.json", *CLAUDE_TARGETS), consequence="nothing is shared")
	assert warning is not None
	assert warning.splitlines() == [
		"warning: git will not commit what camas just wrote — nothing is shared.",
		"  .mcp.json is excluded by .gitignore:1 (`.mcp.json`) — drop that rule, or negate it "
		"below:",
		"      !.mcp.json",
		"  .claude is excluded by .gitignore:2 (`.claude/`) — git cannot re-include a path whose "
		"parent directory is excluded, so that rule has to stop matching .claude itself, e.g.:",
		"      .claude/*",
		"      !.claude/settings.json",
		"      !.claude/agents/",
		"      !.claude/skills/",
	]


def test_gitignore_warning_is_none_when_git_cannot_say(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which_git(None))
	assert gitignore_warning(CLAUDE_TARGETS, consequence="teammates lose the loop") is None


@requires_git
def test_the_suggested_lines_are_what_makes_the_files_addable(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""The advice is load-bearing: applying the printed lines verbatim is what gets every file
	``--claude`` wrote staged, and it leaves ``.camas`` out.
	"""
	_repo(tmp_path, ".claude/\n")
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which_git(shutil.which("git"), "camas"))
	assert write_claude([]) == 0
	(root,) = excluded_roots(CLAUDE_TARGETS)
	(tmp_path / ".gitignore").write_text(
		"\n".join(unignore(root, CLAUDE_TARGETS).lines) + "\n", encoding="utf-8"
	)
	_git(tmp_path, "add", "-A")
	staged = _git(tmp_path, "diff", "--cached", "--name-only").split()
	assert set(CLAUDE_TARGETS) <= set(staged)
	assert not [path for path in staged if path.startswith(".camas/")]


@requires_git
def test_a_rule_from_outside_a_gitignore_is_named_as_clone_local(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	_repo(tmp_path, "")
	(tmp_path / ".git" / "info" / "exclude").write_text(".claude/\n", encoding="utf-8")
	(tmp_path / ".claude").mkdir()
	monkeypatch.chdir(tmp_path)
	(root,) = excluded_roots(CLAUDE_TARGETS)
	assert root == Excluded(
		path=".claude", source=".git/info/exclude", line="1", pattern=".claude/"
	)
	fix = unignore(root, CLAUDE_TARGETS)
	assert fix.advice.startswith("that file is local to this clone")
	assert fix.lines[0] == "!.claude/"


@requires_git
def test_the_suggested_lines_work_for_a_rule_outside_the_repo(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""A ``core.excludesFile`` rule cannot be edited into shape — only a ``.gitignore`` outranks it,
	and it has to re-include the directory before narrowing back down, or git never descends and
	nothing underneath is reachable. The narrowing still has to leave unrelated ``.claude`` state out.
	"""
	_repo(tmp_path, "")
	excludes = tmp_path / "global-ignore"
	excludes.write_text(".claude/\n", encoding="utf-8")
	_git(tmp_path, "config", "core.excludesFile", excludes.as_posix())
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which_git(shutil.which("git"), "camas"))
	assert write_claude([]) == 0
	(tmp_path / ".claude" / "settings.local.json").write_text("{}\n", encoding="utf-8")
	(root,) = excluded_roots(CLAUDE_TARGETS)
	(tmp_path / ".gitignore").write_text(
		"\n".join(unignore(root, CLAUDE_TARGETS).lines) + "\n", encoding="utf-8"
	)
	_git(tmp_path, "add", "-A")
	staged = _git(tmp_path, "diff", "--cached", "--name-only").split()
	assert set(CLAUDE_TARGETS) <= set(staged)
	assert ".claude/settings.local.json" not in staged


@requires_git
def test_warn_uncommittable_reports_to_stderr(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	_repo(tmp_path, ".mcp.json\n")
	monkeypatch.chdir(tmp_path)
	warn_uncommittable((".mcp.json",), consequence="nobody else gets the server entry")
	captured = capsys.readouterr()
	assert captured.out == ""
	assert captured.err == (
		"warning: git will not commit what camas just wrote — nobody else gets the server "
		"entry.\n"
		"  .mcp.json is excluded by .gitignore:1 (`.mcp.json`) — drop that rule, or negate it "
		"below:\n"
		"      !.mcp.json\n"
	)


def test_warn_uncommittable_is_silent_when_committable(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which_git(None))
	warn_uncommittable((".mcp.json",), consequence="nobody else gets the server entry")
	captured = capsys.readouterr()
	assert (captured.out, captured.err) == ("", "")


@requires_git
def test_write_mcp_json_warns_when_the_entry_is_gitignored(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	_repo(tmp_path, "*.json\n")
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which_git(shutil.which("git"), "camas"))
	assert write_mcp_json([]) == 0
	captured = capsys.readouterr()
	assert "Wrote the 'camas' MCP server" in captured.out
	assert "will not get the camas MCP server entry" in captured.err
	assert "      !.mcp.json" in captured.err


@requires_git
def test_write_claude_warns_that_the_loop_stays_per_developer(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	_repo(tmp_path, ".claude/\n")
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which_git(shutil.which("git"), "camas"))
	assert write_claude([]) == 0
	captured = capsys.readouterr()
	assert "Claude Code is configured" in captured.out
	assert "the autofix/gate loop stays per-developer" in captured.err
	assert "      !.claude/agents/" in captured.err


@requires_git
def test_write_claude_is_silent_when_the_generated_files_are_committable(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	_repo(tmp_path, "build/\n")
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which_git(shutil.which("git"), "camas"))
	assert write_claude([]) == 0
	assert capsys.readouterr().err == ""


@requires_git
def test_claude_targets_are_exactly_what_write_claude_wrote(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""``CLAUDE_TARGETS`` drives the check, so it has to stay the set of generated ``.claude``
	files — a new template that skips it would go unwarned.
	"""
	_repo(tmp_path, "build/\n")
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr("shutil.which", _which_git(shutil.which("git"), "camas"))
	assert write_claude([]) == 0
	_git(tmp_path, "add", "-A")
	written = _git(tmp_path, "diff", "--cached", "--name-only").split()
	assert sorted(p for p in written if p.startswith(".claude/")) == sorted(CLAUDE_TARGETS)
