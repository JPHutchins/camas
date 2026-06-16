# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``camas`` console-script entry point."""

from __future__ import annotations

import sys

from .dispatch import dispatch, reconfigure_stdio_utf8, resolve_tasks_source


def main() -> None:
	"""Console script entry: resolves tasks source and dispatches."""
	reconfigure_stdio_utf8()
	argv = sys.argv[1:]
	if argv and argv[0] == "mcp":
		from camas.mcp.serve import serve_stdio

		serve_stdio(argv[1:])
		return
	dispatch(*resolve_tasks_source(argv))


if __name__ == "__main__":
	main()
