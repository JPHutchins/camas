# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The default :func:`camas.Clean` drift check — ``python -m camas._git_porcelain``."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Final

from .core.execution import env_case_insensitive

_PLATFORM: Final = sys.platform


def _git_env_var(key: str) -> bool:
	"""Whether ``key`` names a GIT_* variable under this platform's env case rules."""
	return key.upper().startswith("GIT_") if env_case_insensitive() else key.startswith("GIT_")


def _write_line(text: str) -> None:
	"""``text`` to stderr, newline-terminated."""
	sys.stderr.write(text if text.endswith("\n") else text + "\n")


def _write_stdout(text: str) -> None:
	"""``text`` to stdout as UTF-8 bytes, the codec the renderer decodes leaf output with, via
	``stdout.buffer`` or the text stream when it has none.
	"""
	buffer = getattr(sys.stdout, "buffer", None)
	if buffer is None:
		sys.stdout.write(text)
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
				if code < 0 and _PLATFORM != "win32"
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
