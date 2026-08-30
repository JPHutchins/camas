# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The default :func:`camas.Clean` drift check — ``python -m camas._git_porcelain``."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Final


def _git_env_var(key: str) -> bool:
	"""Whether ``key`` names a GIT_* variable: case-insensitive where the environment is
	(Windows and MSYS/Cygwin), exact elsewhere.
	"""
	return (
		key.upper().startswith("GIT_")
		if sys.platform in ("win32", "msys", "cygwin")
		else key.startswith("GIT_")
	)


def main() -> int:
	env: Final = {key: value for key, value in os.environ.items() if not _git_env_var(key)}
	try:
		run: Final = subprocess.run(
			["git", "status", "--porcelain", "--untracked-files=normal"],
			capture_output=True,
			text=True,
			errors="replace",
			check=False,
			env=env,
		)
	except OSError as exc:
		sys.stderr.write(f"git is required on PATH ({exc})\n")
		return 1
	if run.returncode != 0:
		sys.stderr.write(run.stderr or f"git status exited with code {run.returncode}\n")
		return 1
	if run.stderr:
		sys.stderr.write(run.stderr)
	if run.stdout.strip():
		sys.stdout.write(run.stdout)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
