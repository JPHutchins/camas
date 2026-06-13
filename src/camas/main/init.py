# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""``--init``: scaffold a commented starter ``tasks.py``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final


def starter_text() -> str:
	"""The packaged :mod:`camas.main.starter` template, scaffolded verbatim.

	Read at call time, not import time: mypyc-compiled modules may not define
	``__file__`` while the module body executes (nixpkgs' mypyc doesn't), and
	mypyc ships the original ``.py`` next to the compiled artifact, so the
	call-time read works identically from a compiled wheel.
	"""
	return (Path(__file__).parent / "starter.py").read_text(encoding="utf-8")


def write_starter_tasks_py(directory: Path) -> int:
	"""Write :func:`starter_text` to ``directory/tasks.py`` — the exit code for
	``camas --init``. Exclusive create: an existing file is never touched, and
	any ``OSError`` (exists, permissions, ...) becomes a clean error exit.
	"""
	target: Final = directory / "tasks.py"
	try:
		with target.open("x", encoding="utf-8") as f:
			f.write(starter_text())
	except OSError as e:
		print(f"error: {e}", file=sys.stderr)
		return 2
	print(f"Wrote {target}\n\nTry:\n  camas --list\n  camas greet --help\n  camas")
	return 0
