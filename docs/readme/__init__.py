# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Render README.md from docs/README.md.in; `--check` reports a stale README.md."""

from __future__ import annotations

import difflib
import sys
from pathlib import Path
from typing import Final, NamedTuple

from .template import render

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

TEMPLATE: Final = Path("docs/README.md.in")
README: Final = Path("README.md")
COMMAND: Final = "python -m docs.readme"
BANNER: Final = f"<!-- Generated from {TEMPLATE.as_posix()} by `{COMMAND}`; edit that file. -->"


class Write(NamedTuple):
	pass


class Check(NamedTuple):
	pass


def main(argv: list[str]) -> int:
	match argv:
		case []:
			return run(Write())
		case ["--check"]:
			return run(Check())
		case _:
			print(f"usage: {COMMAND} [--check]", file=sys.stderr)
			return 2


def run(mode: Write | Check) -> int:
	rendered = render(TEMPLATE.read_text(encoding="utf-8"))
	if not isinstance(rendered, str):
		print(
			*(f"{TEMPLATE.as_posix()}:{e.line}: {e.message}" for e in rendered),
			sep="\n",
			file=sys.stderr,
		)
		return 2
	expected = f"{BANNER}\n{rendered}"
	committed = README.read_text(encoding="utf-8") if README.is_file() else ""
	match mode:
		case Write():
			if expected != committed:
				README.write_text(expected, encoding="utf-8", newline="\n")
				print(f"{README}: regenerated from {TEMPLATE.as_posix()}")
			return 0
		case Check():
			if expected == committed:
				return 0
			print(
				*difflib.unified_diff(
					committed.splitlines(),
					expected.splitlines(),
					f"{README} (committed)",
					f"{README} (rendered from {TEMPLATE.as_posix()})",
					lineterm="",
				),
				sep="\n",
			)
			print(f"{README}: stale; run `{COMMAND}`", file=sys.stderr)
			return 1
		case _:
			assert_never(mode)
