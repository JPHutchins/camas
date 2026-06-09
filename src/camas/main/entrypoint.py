# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``camas`` console-script entry point."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast

from .dispatch import dispatch, resolve_tasks_source

if TYPE_CHECKING:
	import io


def main() -> None:
	"""Console script entry: resolves tasks source and dispatches.

	Reconfigures stdout/stderr to UTF-8 so Windows consoles (cp1252 by default) can
	render the box-drawing characters used in the tree output.
	"""
	for stream in (sys.stdout, sys.stderr):
		cast("io.TextIOWrapper", stream).reconfigure(encoding="utf-8", errors="replace")
	dispatch(*resolve_tasks_source(sys.argv[1:]))


if __name__ == "__main__":
	main()
