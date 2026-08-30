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
