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
  camas mcp fix [--paths P]… run the registered agent fix node (Config.agent.fix) over the
                              changed paths — the FileChanged autofix; no-op if unregistered
  camas mcp gate [task]     run the gate once, headless — print the verdict as JSON, exit
    [--paths P]… [--under N]  0 (continue) / 2 (block); scope to --paths or a PostToolBatch
                              event on stdin (the command-hook entry; parallel + benchmark)

Options:
  --rich        emit the 2025-11-25 tool fields (title, annotations, outputSchema) and
                structuredContent; off by default because some clients drop tools that
                include them (Claude Code #25081)
  -h, --help    show this help
"""


def main(argv: list[str]) -> None:
	"""Route ``camas mcp [-h|init|--rich]``; ``init`` and an unexpected argument exit with a code.

	Raises:
		ModuleNotFoundError: if a dependency other than the optional ``mcp`` extra is missing.
	"""
	if "-h" in argv or "--help" in argv:
		print(HELP)
		return
	if argv and argv[0] == "init":
		from .scaffold import write_mcp_json

		sys.exit(write_mcp_json(argv[1:]))
	if argv and argv[0] == "fix":
		from ..main.dispatch import fix_cli

		sys.exit(fix_cli(argv[1:]))
	if argv and argv[0] == "gate":
		try:
			from .serve import gate_cli
		except ModuleNotFoundError as e:
			if e.name != "mcp":
				raise
			print("camas mcp gate: requires feature camas[mcp]", file=sys.stderr)
			sys.exit(2)
		sys.exit(gate_cli(argv[1:]))
	unexpected = [arg for arg in argv if arg != "--rich"]
	if unexpected:
		hint = " (did you mean 'camas mcp init'?)" if "--init" in unexpected else ""
		print(
			f"camas mcp: unexpected argument(s): {' '.join(unexpected)}{hint}\n"
			"Usage: camas mcp [--rich]  or  camas mcp init [--rich]  (camas mcp --help for more)",
			file=sys.stderr,
		)
		sys.exit(2)
	try:
		from .serve import serve_stdio
	except ModuleNotFoundError as e:
		if e.name != "mcp":
			raise
		print("camas mcp: requires feature camas[mcp]", file=sys.stderr)
		sys.exit(2)

	serve_stdio(argv)
