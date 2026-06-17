# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Dispatch ``camas mcp`` subcommands: ``init`` scaffolds ``.mcp.json``; otherwise serve."""

from __future__ import annotations

import sys
from typing import Final

HELP: Final = """\
camas mcp — serve this project's tasks to AI agents over the Model Context Protocol.

Usage:
  camas mcp [--rich]        run the MCP stdio server (an MCP client launches this)
  camas mcp init [--rich]   write this project's .mcp.json entry for the camas server

Options:
  --rich        emit the 2025-11-25 tool fields (title, annotations, outputSchema) and
                structuredContent; off by default because some clients drop tools that
                include them (Claude Code #25081)
  -h, --help    show this help
"""


def main(argv: list[str]) -> None:
	"""Route ``camas mcp [-h|init|--rich|...]``; ``init`` exits with its own code."""
	if "-h" in argv or "--help" in argv:
		print(HELP)
		return
	if argv and argv[0] == "init":
		from .scaffold import write_mcp_json

		sys.exit(write_mcp_json(argv[1:]))
	from .serve import serve_stdio

	serve_stdio(argv)
