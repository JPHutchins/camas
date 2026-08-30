# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The default drift check module: exit codes and diagnostics, driven with a monkeypatched
git."""

from __future__ import annotations

import runpy
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from camas._git_porcelain import main

if TYPE_CHECKING:
	from collections.abc import Callable


def _git_result(returncode: int, stdout: str = "", stderr: str = "") -> SimpleNamespace:
	return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _fake_git(result: SimpleNamespace) -> Callable[..., SimpleNamespace]:
	"""A ``subprocess.run`` stand-in returning ``result`` for any invocation."""

	def run(*args: object, **kwargs: object) -> SimpleNamespace:
		return result

	return run


def test_main_returns_zero_for_a_clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("camas._git_porcelain.subprocess.run", _fake_git(_git_result(0)))
	assert main() == 0


def test_main_fails_and_prints_the_status_when_dirty(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_fake_git(_git_result(0, stdout=" M tracked.txt\n")),
	)
	assert main() == 1
	assert capsys.readouterr().out == " M tracked.txt\n"


def test_main_fails_and_forwards_stderr_when_git_errors(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_fake_git(_git_result(128, stderr="fatal: not a git repository\n")),
	)
	assert main() == 1
	assert capsys.readouterr().err == "fatal: not a git repository\n"


def test_main_fails_with_a_hint_when_git_is_absent(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	def raises(*args: object, **kwargs: object) -> SimpleNamespace:
		raise FileNotFoundError("git")

	monkeypatch.setattr("camas._git_porcelain.subprocess.run", raises)
	assert main() == 1
	assert "git is required on PATH" in capsys.readouterr().err


def test_main_scrubs_ambient_git_environment(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""GIT_DIR and friends override directory discovery; the check runs with them scrubbed."""
	seen_env: dict[str, str] = {}

	def capture(*args: object, env: dict[str, str], **kwargs: object) -> SimpleNamespace:
		seen_env.update(env)
		return _git_result(0)

	monkeypatch.setattr("camas._git_porcelain.subprocess.run", capture)
	monkeypatch.setenv("GIT_DIR", "/elsewhere")
	monkeypatch.setenv("GIT_WORK_TREE", "/elsewhere")
	monkeypatch.setenv("KEEP_ME", "yes")
	assert main() == 0
	assert "GIT_DIR" not in seen_env
	assert "GIT_WORK_TREE" not in seen_env
	assert seen_env["KEEP_ME"] == "yes"


def test_main_scrubs_git_environment_case_insensitively(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Windows environment names are case-insensitive, so a lowercase git_dir still
	overrides discovery; the scrub matches on the upper-cased key."""
	seen_env: dict[str, str] = {}

	def capture(*args: object, env: dict[str, str], **kwargs: object) -> SimpleNamespace:
		seen_env.update(env)
		return _git_result(0)

	monkeypatch.setattr("camas._git_porcelain.subprocess.run", capture)
	monkeypatch.setenv("git_dir", "/elsewhere")
	monkeypatch.setenv("keep_me", "yes")
	assert main() == 0
	lower_keys = {key.lower(): value for key, value in seen_env.items()}
	assert "git_dir" not in lower_keys
	assert lower_keys["keep_me"] == "yes"


def test_module_main_exits_with_mains_code(monkeypatch: pytest.MonkeyPatch) -> None:
	"""The ``__main__`` block wraps :func:`main`'s exit code in ``SystemExit``. The
	``subprocess.run`` patch survives ``runpy``'s module re-execution, unlike a ``main``
	patch, which the re-execution would rebind."""
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_fake_git(_git_result(0, stdout=" M tracked.txt\n")),
	)
	with pytest.raises(SystemExit) as excinfo:
		runpy.run_module("camas._git_porcelain", run_name="__main__")
	assert excinfo.value.code == 1
