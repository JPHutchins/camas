# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

from .source import Unresolved

PINNED: Final = {
	"COLUMNS": "100",
	"NO_COLOR": "1",
	"PYTHONUTF8": "1",
	"CLAUDECODE": "",
	"CAMAS_AGENT": "",
	"CAMAS_NO_MCP_HINT": "1",
	"GITHUB_ACTIONS": "",
}
ANSI: Final = re.compile(r"\x1b\[[0-9;]*m")
SOURCE_LINE: Final = re.compile(r"\s*Source \(docs live here\):.*")


def cli_section(directory: str, args: tuple[str, ...]) -> tuple[str, ...] | Unresolved:
	if directory == "-":
		with tempfile.TemporaryDirectory() as scratch:
			return session(Path(scratch), "...", args)
	path = Path(directory)
	if not path.is_dir():
		return Unresolved(f"no directory {path.as_posix()!r}")
	return session(path, f".../{path.resolve().name}", args)


def session(cwd: Path, shown: str, args: tuple[str, ...]) -> tuple[str, ...] | Unresolved:
	result = subprocess.run(
		[sys.executable, "-m", "camas", *args],
		cwd=cwd,
		capture_output=True,
		text=True,
		encoding="utf-8",
		check=False,
		env={**os.environ, **PINNED},
	)
	if result.returncode != 0:
		return Unresolved(
			f"`camas {shlex.join(args)}` exited {result.returncode}: {result.stderr.strip()[-300:]}"
		)
	resolved = str(cwd.resolve())
	text = (
		ANSI.sub("", result.stdout).replace(resolved + os.sep, f"{shown}/").replace(resolved, shown)
	)
	return (
		"```console",
		f"$ camas {shlex.join(args)}",
		*(
			line.rstrip().expandtabs(4)
			for line in text.rstrip("\n").split("\n")
			if not SOURCE_LINE.fullmatch(line)
		),
		"```",
	)
