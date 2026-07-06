# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Dispatch ``camas mcp`` subcommands: ``init`` scaffolds ``.mcp.json``; otherwise serve."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Final, NoReturn

if TYPE_CHECKING:
	from .scaffold import Launcher

HELP: Final = """\
camas mcp — serve this project's tasks to AI agents over the Model Context Protocol.

Usage:
  camas mcp [--plain]        run the MCP stdio server (rich output by default)
  camas mcp init [--claude]  write this project's .mcp.json entry for the camas server;
    [--launcher uv|uvx|camas]  with --claude also configure Claude Code: .mcp.json +
                              PostToolBatch/Stop autofix hooks + async Stop nudge hook +
                              the tiered camas-fixer agents + gate skill; --launcher forces
                              the launch strategy instead of auto-detecting
  camas mcp fix [--paths P]… run the registered agent fix node (Config.agent.fix) over the
                              changed paths (--paths, else a piped PostToolBatch/Stop event) —
                              the autofix hook; no-op if unregistered
  camas mcp gate [task]     run the gate once, headless — print the verdict as JSON, exit
    [--paths P]… [--under D]  0 (continue) / 2 (block); scope to --paths or a piped
    [--jobs N] [--nudge]      PostToolBatch/Stop event (the camas-fixer agents + benchmark);
                              --under takes a duration (5, 1.5s, 500ms, 2m, 1h); --nudge emits
                              the async Stop-hook nudge text instead of the JSON verdict

Options:
  --rich        accepted for back-compat; rich output is the default
  --plain       disable rich output (opt out of the 2025-11-25 tool fields)
  -h, --help    show this help
"""


def parse_launcher(argv: list[str]) -> Launcher | None:
	"""The ``--launcher`` value from ``camas mcp init`` arguments, or ``None`` when absent.

	Raises:
		ValueError: the flag has no value, or names an unknown launcher.
	"""
	import argparse

	from .scaffold import LAUNCHERS

	parser = argparse.ArgumentParser(prog="camas mcp init", add_help=False, exit_on_error=False)
	parser.add_argument("--launcher", choices=LAUNCHERS, default=None)
	try:
		namespace, _ = parser.parse_known_args(argv)
	except argparse.ArgumentError as e:
		raise ValueError(f"camas mcp init: {e}") from None
	launcher: Launcher | None = namespace.launcher
	return launcher


def import_failure_diagnostic(
	exception: ImportError, executable: str, pythonpath: str | None
) -> str:
	"""The actionable stderr diagnostic for a broken ``camas[mcp]`` dependency import — names the
	failing module, the interpreter that failed to import it, and, when ``PYTHONPATH`` is set,
	that a leaked ``PYTHONPATH`` (e.g. from a nix ``mkShell``) can shadow the server's own
	dependencies.

	>>> "pydantic_core" in import_failure_diagnostic(
	...     ModuleNotFoundError("No module named 'pydantic_core'", name="pydantic_core"),
	...     "/usr/bin/python3",
	...     None,
	... )
	True

	>>> "unset PYTHONPATH" in import_failure_diagnostic(
	...     ImportError("boom"), "/usr/bin/python3", "/leaked/path"
	... )
	True
	"""
	name = exception.name or str(exception)
	hint = (
		f"\n  PYTHONPATH is set ({pythonpath!r}) — a leaked PYTHONPATH (e.g. from a nix mkShell) "
		"can shadow camas[mcp]'s own dependencies. Try: unset PYTHONPATH"
		if pythonpath is not None
		else ""
	)
	return (
		f"camas mcp: failed to import {name!r} under {executable} — a broken or shadowed "
		f"install, not a missing camas[mcp] extra.{hint}"
	)


def report_import_failure(e: ImportError, *, feature_hint: str) -> NoReturn:
	"""Print the right diagnostic for a ``.serve`` import failure and exit 2: the terse
	``feature_hint`` for the plain missing-``mcp``-extra case, else the full diagnostic.
	"""
	if isinstance(e, ModuleNotFoundError) and e.name == "mcp":
		print(feature_hint, file=sys.stderr)
	else:
		print(
			import_failure_diagnostic(e, sys.executable, os.environ.get("PYTHONPATH")),
			file=sys.stderr,
		)
	sys.exit(2)


def main(argv: list[str]) -> None:
	"""Route ``camas mcp [-h|init|--rich|--plain]``; ``init`` and an unexpected argument exit with a code."""
	if "-h" in argv or "--help" in argv:
		print(HELP)
		return
	if argv and argv[0] == "init":
		from .scaffold import write_claude, write_mcp_json

		try:
			launcher = parse_launcher(argv[1:])
		except ValueError as e:
			print(str(e), file=sys.stderr)
			sys.exit(2)
		if "--claude" in argv:
			sys.exit(write_claude(argv[1:], launcher=launcher))
		if "--hooks" in argv:
			print(
				"warning: --hooks was removed; use `camas mcp init --claude` to write the hook, "
				"agent, and skill. Only .mcp.json will be written for this invocation.",
				file=sys.stderr,
			)
		sys.exit(write_mcp_json(argv[1:], launcher=launcher))
	if argv and argv[0] == "fix":
		from ..main.dispatch import fix_cli

		sys.exit(fix_cli(argv[1:]))
	if argv and argv[0] == "gate":
		try:
			from .serve import gate_cli
		except ImportError as e:
			report_import_failure(e, feature_hint="camas mcp gate: requires feature camas[mcp]")
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
	except ImportError as e:
		report_import_failure(e, feature_hint="camas mcp: requires feature camas[mcp]")

	serve_stdio(argv)
