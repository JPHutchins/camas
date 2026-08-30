# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The default :func:`camas.Clean` drift check, invoked as ``python -m camas._git_porcelain``:
exit nonzero when the working tree's porcelain status is non-empty — tracked edits and
untracked files alike, immune to the repository's ``status.showUntrackedFiles`` config — or
when git itself cannot report. The leaf prints the status as its drift diagnostic; a git
failure forwards its stderr instead of silently greening the gate.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Final


def main() -> int:
	"""Run the check; the exit code is the gate's signal."""
	run: Final = subprocess.run(
		["git", "status", "--porcelain", "--untracked-files=normal"],
		capture_output=True,
		text=True,
		check=False,
	)
	if run.returncode != 0:
		sys.stderr.write(run.stderr)
		return 1
	if run.stdout.strip():
		sys.stdout.write(run.stdout)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
