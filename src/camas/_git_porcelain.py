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


def main() -> int:
	env: Final = {key: value for key, value in os.environ.items() if not _git_env_var(key)}
	try:
		run: Final = subprocess.run(
			["git", "status", "--porcelain", "--untracked-files=normal"],
			capture_output=True,
			text=True,
			encoding="utf-8",
			errors="replace",
			check=False,
			env=env,
		)
	except OSError as exc:
		_write_line(f"git is required on PATH ({exc})")
		return 1
	if run.returncode != 0:
		if run.stderr.strip():
			_write_line(run.stderr)
		else:
			code: Final = run.returncode
			_write_line(
				f"git status killed by signal {-code}"
				if code < 0 and not env_case_insensitive()
				else f"git status exited with code {code}"
			)
		return 1
	if run.stderr.strip():
		_write_line(run.stderr)
	if run.stdout.strip():
		_write_stdout(run.stdout)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
