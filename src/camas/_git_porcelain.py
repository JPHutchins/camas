# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The default :func:`camas.Clean` drift check — ``python -m camas._git_porcelain``."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Final

from .core.platform import env_case_insensitive


def _git_env_var(key: str) -> bool:
	"""Whether ``key`` names a GIT_* variable under this platform's env case rules."""
	return key.upper().startswith("GIT_") if env_case_insensitive() else key.startswith("GIT_")


def _write_line(text: str) -> None:
	"""``text`` to stderr as UTF-8 bytes — the renderer's codec — newline-terminated, via
	``stderr.buffer`` or the text stream when it has none.
	"""
	line = text if text.endswith("\n") else text + "\n"
	buffer = getattr(sys.stderr, "buffer", None)
	if buffer is None:
		sys.stderr.write(line)
	else:
		buffer.write(line.encode("utf-8", "replace"))


def _write_stdout(text: str) -> None:
	"""``text`` to stdout as UTF-8 bytes — the renderer's codec — via ``stdout.buffer``, or
	sanitized through the text stream's own codec when it has no buffer.
	"""
	buffer = getattr(sys.stdout, "buffer", None)
	if buffer is None:
		codec: Final = getattr(sys.stdout, "encoding", None) or "utf-8"
		sys.stdout.write(text.encode(codec, "replace").decode(codec))
	else:
		buffer.write(text.encode("utf-8", "replace"))


_PLAIN_PATCH: Final = ("--no-color", "--no-ext-diff", "--no-textconv")


def _git(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		["git", *args],
		capture_output=True,
		text=True,
		encoding="utf-8",
		errors="replace",
		check=False,
		env=env,
	)


def _report(proc: subprocess.CompletedProcess[str], what: str) -> None:
	"""``proc``'s stderr, or the exit that left none."""
	if proc.stderr.strip():
		_write_line(proc.stderr)
	elif proc.returncode != 0:
		code: Final = proc.returncode
		_write_line(
			f"{what} killed by signal {-code}"
			if code < 0 and not env_case_insensitive()
			else f"{what} exited with code {code}"
		)


def _patch_flags(status: str) -> tuple[tuple[str, ...], ...]:
	r"""The ``git diff`` flags whose patches show the drift ``status`` lists: the index's, then
	the working tree's. Untracked files have none.

	>>> _patch_flags(" M a.txt\n?? b.txt\n")
	((),)
	>>> _patch_flags("M  a.txt\nMM b.txt\n")
	(('--cached',), ())
	>>> _patch_flags("?? new.txt\n")
	()
	"""
	tracked: Final = tuple(line for line in status.splitlines() if not line.startswith("??"))
	return tuple(
		flags
		for flags, column in ((("--cached",), 0), ((), 1))
		if any(line[column] != " " for line in tracked)
	)


def _write_patch(env: dict[str, str], *flags: str) -> None:
	try:
		diff: Final = _git(env, "diff", *flags, *_PLAIN_PATCH)
	except OSError as exc:
		_write_line(f"git diff could not run ({exc})")
		return
	_write_stdout(diff.stdout)
	_report(diff, "git diff")


def main() -> int:
	env: Final = {key: value for key, value in os.environ.items() if not _git_env_var(key)}
	try:
		run: Final = _git(env, "status", "--porcelain", "--untracked-files=normal")
	except OSError as exc:
		_write_line(f"git is required on PATH ({exc})")
		return 1
	_report(run, "git status")
	if run.returncode != 0:
		return 1
	if run.stdout.strip():
		for flags in _patch_flags(run.stdout):
			_write_patch(env, *flags)
		_write_stdout(run.stdout)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
