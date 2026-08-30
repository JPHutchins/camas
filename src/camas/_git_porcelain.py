# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The default :func:`camas.Clean` drift check — ``python -m camas._git_porcelain``."""

from __future__ import annotations

import subprocess
import sys
from typing import Final


def main() -> int:
	try:
		run: Final = subprocess.run(
			["git", "status", "--porcelain", "--untracked-files=normal"],
			capture_output=True,
			text=True,
			check=False,
		)
	except OSError as exc:
		sys.stderr.write(f"git is required on PATH: {exc}\n")
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
