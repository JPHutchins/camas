# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from ..core.timings import ensure_camas_dir
from ..v0.config import DEFAULT_CAMAS_DIR


def starter_text(*, verbose: bool = False) -> str:
	name = "starter_verbose.py" if verbose else "starter.py"
	return (Path(__file__).parent / name).read_text(encoding="utf-8")


def create_starter_tasks_py(directory: Path, *, verbose: bool = False) -> Path:
	target: Final = directory / "tasks.py"
	with target.open("x", encoding="utf-8") as f:
		f.write(starter_text(verbose=verbose))
	ensure_camas_dir(directory / DEFAULT_CAMAS_DIR)
	return target


def write_starter_tasks_py(directory: Path, *, verbose: bool = False) -> int:
	camas_dir: Final = directory / DEFAULT_CAMAS_DIR
	try:
		target = create_starter_tasks_py(directory, verbose=verbose)
	except OSError as e:
		print(f"error: {e}", file=sys.stderr)
		return 2
	print(
		f"Wrote {target}\n"
		f"Created {camas_dir} for run logs and timing estimates; delete it to opt out.\n\n"
		"Try:\n  camas --list\n  camas greet --help\n  camas"
	)
	return 0
