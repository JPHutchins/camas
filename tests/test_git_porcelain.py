# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The default drift check module: exit codes and diagnostics, driven with a monkeypatched
git."""

from __future__ import annotations

import io
import runpy
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from camas._git_porcelain import main
from camas.core.platform import env_case_insensitive

if TYPE_CHECKING:
	from collections.abc import Callable


def _git_result(returncode: int, stdout: str = "", stderr: str = "") -> SimpleNamespace:
	return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


_NO_DIFF = SimpleNamespace(returncode=0, stdout="", stderr="")
"""A ``git diff`` with nothing to show: the default answer of :func:`_fake_git`."""


def _fake_git(
	result: SimpleNamespace,
	worktree: SimpleNamespace | OSError = _NO_DIFF,
	cached: SimpleNamespace | OSError = _NO_DIFF,
) -> Callable[..., SimpleNamespace]:
	"""A ``subprocess.run`` stand-in answering ``git status`` with ``result``, and ``git diff``
	with ``cached`` for the index or ``worktree`` for the working tree, raising either when it
	is an ``OSError``."""

	def run(argv: list[str], **kwargs: object) -> SimpleNamespace:
		answer = result if argv[1] == "status" else cached if "--cached" in argv else worktree
		if isinstance(answer, OSError):
			raise answer
		return answer

	return run


def _recording_run(
	seen_env: dict[str, str] | None = None,
	seen_kwargs: dict[str, object] | None = None,
	result: SimpleNamespace | None = None,
) -> Callable[..., SimpleNamespace]:
	"""A ``subprocess.run`` stand-in recording the ``env`` and/or kwargs it is passed."""

	def run(*args: object, env: dict[str, str] | None = None, **kwargs: object) -> SimpleNamespace:
		if seen_env is not None:
			seen_env.update(env or {})
		if seen_kwargs is not None:
			seen_kwargs.update(kwargs)
		return _git_result(0) if result is None else result

	return run


def _failing_run(exc: OSError) -> Callable[..., SimpleNamespace]:
	"""A ``subprocess.run`` stand-in raising ``exc``."""

	def run(*args: object, **kwargs: object) -> SimpleNamespace:
		raise exc

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


def test_main_prints_the_patch_then_the_status_when_dirty(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""The patch shows what changed in the tracked files; the status lines after it name every
	drifted file, untracked ones included, and survive a consumer that keeps only the tail
	(#338)."""
	patch = "diff --git a/tracked.txt b/tracked.txt\n+drift\n"
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_fake_git(
			_git_result(0, stdout=" M tracked.txt\n?? new.txt\n"), _git_result(0, stdout=patch)
		),
	)
	assert main() == 1
	assert capsys.readouterr().out == patch + " M tracked.txt\n?? new.txt\n"


def test_main_prints_the_staged_patch_before_the_unstaged_one(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""A generator that stages its output leaves drift a working-tree diff cannot show."""
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_fake_git(
			_git_result(0, stdout="M  staged.txt\nMM both.txt\n"),
			worktree=_git_result(0, stdout="+unstaged\n"),
			cached=_git_result(0, stdout="+staged\n"),
		),
	)
	assert main() == 1
	assert capsys.readouterr().out == "+staged\n+unstaged\nM  staged.txt\nMM both.txt\n"


def test_main_runs_no_diff_for_untracked_drift(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_fake_git(
			_git_result(0, stdout="?? new.txt\n"),
			worktree=FileNotFoundError("git"),
			cached=FileNotFoundError("git"),
		),
	)
	assert main() == 1
	captured = capsys.readouterr()
	assert captured.out == "?? new.txt\n"
	assert captured.err == ""


def test_the_patches_share_the_scrubbed_env_and_ignore_git_config(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""The patch calls read the same repository as the status call, and git config's color,
	external diff tool, and textconv filters stay out of a captured leaf."""
	calls: list[tuple[list[str], dict[str, str]]] = []

	def run(argv: list[str], *, env: dict[str, str], **kwargs: object) -> SimpleNamespace:
		calls.append((argv, env))
		return _git_result(0, stdout="MM tracked.txt\n") if argv[1] == "status" else _NO_DIFF

	monkeypatch.setattr("camas._git_porcelain.subprocess.run", run)
	monkeypatch.setenv("GIT_DIR", "/elsewhere")
	assert main() == 1
	assert [argv for argv, _ in calls] == [
		["git", "status", "--porcelain", "--untracked-files=normal"],
		["git", "diff", "--cached", "--no-color", "--no-ext-diff", "--no-textconv"],
		["git", "diff", "--no-color", "--no-ext-diff", "--no-textconv"],
	]
	assert all("GIT_DIR" not in env for _, env in calls)


@pytest.mark.parametrize(
	("diff", "err"),
	[
		(FileNotFoundError("git"), "git diff could not run (git)\n"),
		(_git_result(128, stderr="fatal: bad index\n"), "fatal: bad index\n"),
		(_git_result(1), "git diff exited with code 1\n"),
	],
	ids=("diff-cannot-start", "diff-errors", "diff-exits-silently"),
)
def test_a_failed_diff_keeps_the_status_verdict(
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
	diff: SimpleNamespace | OSError,
	err: str,
) -> None:
	"""The patch is a diagnostic: its failure is reported, never a pass and never a lost status."""
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_fake_git(_git_result(0, stdout=" M tracked.txt\n"), diff),
	)
	assert main() == 1
	captured = capsys.readouterr()
	assert captured.out == " M tracked.txt\n"
	assert captured.err == err


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
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run", _failing_run(FileNotFoundError("git"))
	)
	assert main() == 1
	assert "git is required on PATH" in capsys.readouterr().err


def test_main_scrubs_ambient_git_environment(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""GIT_DIR and friends override directory discovery; the check runs with them scrubbed."""
	seen_env: dict[str, str] = {}
	monkeypatch.setattr("camas._git_porcelain.subprocess.run", _recording_run(seen_env))
	monkeypatch.setenv("GIT_DIR", "/elsewhere")
	monkeypatch.setenv("GIT_WORK_TREE", "/elsewhere")
	monkeypatch.setenv("KEEP_ME", "yes")
	assert main() == 0
	assert "GIT_DIR" not in seen_env
	assert "GIT_WORK_TREE" not in seen_env
	assert seen_env["KEEP_ME"] == "yes"


@pytest.mark.skipif(
	env_case_insensitive(),
	reason=(
		"on a case-insensitive env platform the lowercase key is scrubbed (Windows 3.10/3.11 "
		"preserve the case; 3.12+ upper-cases it), so the exact-case assertion cannot run there"
	),
)
def test_main_scrubs_git_environment_exact_case_on_posix(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""POSIX git's getenv is case-sensitive, so the scrub is exact: a lowercase git_dir
	survives, since git never reads it."""
	seen_env: dict[str, str] = {}
	monkeypatch.setattr("camas._git_porcelain.subprocess.run", _recording_run(seen_env))
	monkeypatch.setenv("git_dir", "/elsewhere")
	monkeypatch.setenv("keep_me", "yes")
	assert main() == 0
	assert seen_env["git_dir"] == "/elsewhere"
	assert seen_env["keep_me"] == "yes"


def test_main_scrubs_git_environment_case_insensitively_on_windows(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""The case-insensitive branch is dead on POSIX CI: patch the platform predicate seam and
	pin it directly, so a revert to the exact match fails here."""
	seen_env: dict[str, str] = {}
	monkeypatch.setattr("camas._git_porcelain.subprocess.run", _recording_run(seen_env))
	monkeypatch.setattr("camas._git_porcelain.env_case_insensitive", lambda: True)
	monkeypatch.delenv("keep_me", raising=False)
	monkeypatch.delenv("KEEP_ME", raising=False)
	monkeypatch.setenv("git_dir", "/elsewhere")
	monkeypatch.setenv("keep_me", "yes")
	assert main() == 0
	lower_keys = {key.lower(): value for key, value in seen_env.items()}
	assert "git_dir" not in lower_keys
	assert lower_keys["keep_me"] == "yes"


def test_main_runs_git_with_lenient_utf8_decode(monkeypatch: pytest.MonkeyPatch) -> None:
	"""The exit-code logic never reads the decoded bytes, so the lenient UTF-8 decode must not
	be trimmed back out."""
	seen_kwargs: dict[str, object] = {}
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_recording_run(seen_kwargs=seen_kwargs, result=_git_result(1, stderr="fatal: bad byte\n")),
	)
	assert main() == 1
	assert seen_kwargs["errors"] == "replace"
	assert seen_kwargs["encoding"] == "utf-8"


@pytest.mark.skipif(
	env_case_insensitive(),
	reason="a native git child surfaces Windows status codes, not POSIX signals; that form has its own pin",
)
def test_main_reports_a_signal_death_when_git_dies_without_output(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setattr("camas._git_porcelain.subprocess.run", _fake_git(_git_result(-9)))
	assert main() == 1
	assert capsys.readouterr().err == "git status killed by signal 9\n"


def test_main_reports_a_native_child_crash_code_as_an_exit(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setattr("camas._git_porcelain.env_case_insensitive", lambda: True)
	monkeypatch.setattr("camas._git_porcelain.subprocess.run", _fake_git(_git_result(-1073741510)))
	assert main() == 1
	assert capsys.readouterr().err == "git status exited with code -1073741510\n"


def test_main_falls_back_when_stderr_is_whitespace_only(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run", _fake_git(_git_result(1, stderr=" \n"))
	)
	assert main() == 1
	assert capsys.readouterr().err == "git status exited with code 1\n"


def test_main_newline_terminates_a_trailing_failure(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run", _fake_git(_git_result(1, stderr="fatal: bad"))
	)
	assert main() == 1
	assert capsys.readouterr().err == "fatal: bad\n"


def test_main_forwards_warnings_from_a_clean_run(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_fake_git(_git_result(0, stderr="warning: something\n")),
	)
	assert main() == 0
	assert capsys.readouterr().err == "warning: something\n"


def test_main_newline_terminates_a_trailing_warning(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_fake_git(_git_result(0, stderr="warning: something")),
	)
	assert main() == 0
	assert capsys.readouterr().err == "warning: something\n"


def test_main_survives_an_undecodable_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
	"""The status bytes reach stdout as UTF-8 — the renderer's codec — so a replacement
	character survives a strict-ASCII stdout instead of raising."""
	sink = io.TextIOWrapper(io.BytesIO(), encoding="ascii", errors="strict")
	monkeypatch.setattr("sys.stdout", sink)
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_fake_git(_git_result(0, stdout=" M �.txt\n")),
	)
	assert main() == 1
	assert sink.buffer.getvalue() == " M �.txt\n".encode()


def test_main_writes_a_dirty_tree_without_a_buffer(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""An embedding may redirect stdout to a text-only stream; the write falls back to it."""
	sink = io.StringIO()
	monkeypatch.setattr("sys.stdout", sink)
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_fake_git(_git_result(0, stdout=" M tracked.txt\n")),
	)
	assert main() == 1
	assert sink.getvalue() == " M tracked.txt\n"


class _StrictAsciiText(io.StringIO):
	"""A buffer-less text stream whose codec cannot represent non-ASCII."""

	encoding = "ascii"

	def write(self, s: str) -> int:
		s.encode("ascii")
		return super().write(s)


def test_main_survives_a_dirty_tree_on_a_strict_text_stream(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""The buffer-less fallback sanitizes through the stream's own codec."""
	sink = _StrictAsciiText()
	monkeypatch.setattr("sys.stdout", sink)
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_fake_git(_git_result(0, stdout=" M �.txt\n")),
	)
	assert main() == 1
	assert sink.getvalue() == " M ?.txt\n"


def test_main_writes_failures_as_utf8_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
	"""git's stderr reaches the renderer as UTF-8 bytes, whatever stderr's own codec."""
	sink = io.TextIOWrapper(io.BytesIO(), encoding="ascii", errors="strict")
	monkeypatch.setattr("sys.stderr", sink)
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_fake_git(_git_result(1, stderr="fatal: bäd\n")),
	)
	assert main() == 1
	assert sink.buffer.getvalue() == "fatal: bäd\n".encode()


def test_main_writes_failures_to_a_bufferless_stderr(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	sink = io.StringIO()
	monkeypatch.setattr("sys.stderr", sink)
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_fake_git(_git_result(1, stderr="fatal: bad")),
	)
	assert main() == 1
	assert sink.getvalue() == "fatal: bad\n"


def test_main_fails_with_a_hint_when_git_cannot_execute(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""A git shim that fails to exec (WinError 193) is still a git that cannot run — and the
	hint carries the exception itself."""
	monkeypatch.setattr(
		"camas._git_porcelain.subprocess.run",
		_failing_run(OSError("not a valid Win32 application")),
	)
	assert main() == 1
	err = capsys.readouterr().err
	assert "git is required on PATH" in err
	assert "not a valid Win32 application" in err


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
