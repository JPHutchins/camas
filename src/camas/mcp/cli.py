# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Dispatch ``camas mcp`` subcommands: ``init`` scaffolds ``.mcp.json``; otherwise serve."""

from __future__ import annotations

import sys
from typing import Final

HELP: Final = """\
camas mcp — serve this project's tasks to AI agents over the Model Context Protocol.

Usage:
  camas mcp [--plain]        run the MCP stdio server (rich output by default)
  camas mcp init [--claude]  write this project's .mcp.json entry for the camas server;
                              with --claude also configure Claude Code: .mcp.json +
                              PostToolBatch autofix hook + camas-fixer subagent + gate skill
  camas mcp fix [--paths P]… run the registered agent fix node (Config.agent.fix) over the
                              changed paths (--paths, else a piped PostToolBatch event) — the
                              autofix hook; no-op if unregistered
  camas mcp gate [task]     run the gate once, headless — print the verdict as JSON, exit
    [--paths P]… [--under N]  0 (continue) / 2 (block); scope to --paths or a piped
                              PostToolBatch event (the camas-fixer subagent + benchmark)

Options:
  --rich        accepted for back-compat; rich output is the default
  --plain       disable rich output (opt out of the 2025-11-25 tool fields)
  -h, --help    show this help
"""


def main(argv: list[str]) -> None:
	"""Route ``camas mcp [-h|init|--rich|--plain]``; ``init`` and an unexpected argument exit with a code.

	Raises:
		ModuleNotFoundError: if a dependency other than the optional ``mcp`` extra is missing.
	"""
	if "-h" in argv or "--help" in argv:
		print(HELP)
		return
	if argv and argv[0] == "init":
		from .scaffold import write_claude, write_mcp_json

		if "--claude" in argv:
			sys.exit(write_claude(argv[1:]))
		if "--hooks" in argv:
			print(
				"warning: --hooks was removed; use `camas mcp init --claude` to write the hook, "
				"agent, and skill. Only .mcp.json will be written for this invocation.",
				file=sys.stderr,
			)
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
	unexpected = [arg for arg in argv if arg not in ("--rich", "--plain")]
	if unexpected:
		hint = " (did you mean 'camas mcp init'?)" if "--init" in unexpected else ""
		print(
			f"camas mcp: unexpected argument(s): {' '.join(unexpected)}{hint}\n"
			"Usage: camas mcp [--rich|--plain]  or  camas mcp init [--claude]  "
			"(camas mcp --help for more)",
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
