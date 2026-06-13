# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""``--init``: scaffold a commented starter ``tasks.py``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

STARTER: Final = (Path(__file__).parent / "starter.py").read_text(encoding="utf-8")
"""The packaged :mod:`camas.main.starter` template, scaffolded verbatim.
mypyc ships the original ``.py`` next to the compiled artifact, so the read
works identically from a compiled wheel."""


def write_starter_tasks_py(directory: Path) -> int:
	"""Write :data:`STARTER` to ``directory/tasks.py`` — the exit code for
	``camas --init``. Exclusive create: an existing file is never touched, and
	any ``OSError`` (exists, permissions, ...) becomes a clean error exit.
	"""
	target: Final = directory / "tasks.py"
	try:
		with target.open("x", encoding="utf-8") as f:
			f.write(STARTER)
	except OSError as e:
		print(f"error: {e}", file=sys.stderr)
		return 2
	print(f"Wrote {target}\n\nTry:\n  camas --list\n  camas greet --help\n  camas")
	return 0
