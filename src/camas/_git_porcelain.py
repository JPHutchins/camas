# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The default :func:`camas.Clean` drift check — ``python -m camas._git_porcelain``."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Final


def main() -> int:
	env: Final = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
	try:
		run: Final = subprocess.run(
			["git", "status", "--porcelain", "--untracked-files=normal"],
			capture_output=True,
			text=True,
			check=False,
			env=env,
		)
	except FileNotFoundError:
		sys.stderr.write("git is required on PATH\n")
		return 1
	if run.returncode != 0:
		sys.stderr.write(run.stderr)
		return 1
	if run.stdout.strip():
		sys.stdout.write(run.stdout)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
