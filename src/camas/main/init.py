# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""``--init``: scaffold a commented starter ``tasks.py``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from ..core.timings import ensure_camas_dir
from ..v0.config import DEFAULT_CAMAS_DIR


def starter_text() -> str:
	"""The packaged :mod:`camas.main.starter` template, scaffolded verbatim.

	Read at call time, not import time: mypyc-compiled modules may not define
	``__file__`` while the module body executes (nixpkgs' mypyc doesn't), and
	mypyc ships the original ``.py`` next to the compiled artifact, so the
	call-time read works identically from a compiled wheel.
	"""
	return (Path(__file__).parent / "starter.py").read_text(encoding="utf-8")


def create_starter_tasks_py(directory: Path) -> Path:
	"""Exclusive-create the starter ``tasks.py`` in ``directory`` and the camas directory beside
	it; return the ``tasks.py`` path. Never overwrites — an existing ``tasks.py`` raises
	``FileExistsError``; any other IO failure raises its ``OSError``.
	"""
	target: Final = directory / "tasks.py"
	with target.open("x", encoding="utf-8") as f:
		f.write(starter_text())
	ensure_camas_dir(directory / DEFAULT_CAMAS_DIR)
	return target


def write_starter_tasks_py(directory: Path) -> int:
	"""Write :func:`starter_text` to ``directory/tasks.py`` and create the camas
	directory beside it — the exit code for ``camas --init``. Exclusive create: an
	existing ``tasks.py`` is never touched, and any ``OSError`` (exists, permissions,
	...) becomes a clean error exit.
	"""
	camas_dir: Final = directory / DEFAULT_CAMAS_DIR
	try:
		target = create_starter_tasks_py(directory)
	except OSError as e:
		print(f"error: {e}", file=sys.stderr)
		return 2
	print(
		f"Wrote {target}\n"
		f"Created {camas_dir} for run logs and timing estimates; delete it to opt out.\n\n"
		"Try:\n  camas --list\n  camas greet --help\n  camas"
	)
	return 0
