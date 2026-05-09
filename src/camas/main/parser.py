# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import argparse
import importlib.metadata
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from ..core.effect import Effect
from ..core.render import color_on
from ..core.task import TaskNode
from .format import (
	format_available_effects,
	format_reference,
	format_task_summary_listing,
	format_try_hint,
)


class CamasArgumentParser(argparse.ArgumentParser):
	"""``ArgumentParser`` whose ``--help`` appends the discovered tasks listing,
	the discovered Effects listing, and the Try hint after argparse's standard
	output. ``tasks``/``source``/``scope_effects`` are populated by
	``build_parser`` so the override has access without nesting.
	"""

	tasks: Mapping[str, TaskNode] | None
	source: Path | None
	scope_effects: Mapping[str, type[Effect[Any]]]

	def format_help(self) -> str:
		color = color_on()
		sections = [super().format_help().rstrip()]
		if self.tasks:
			sections.append(format_task_summary_listing(self.tasks, self.source, color=color))
		effects_listing = format_available_effects(color=color, scope_effects=self.scope_effects)
		if effects_listing:
			sections.append(effects_listing)
		sections.append(format_try_hint(color))
		sections.append(format_reference(color))
		return "\n\n".join(sections) + "\n"


def build_parser(
	tasks: Mapping[str, TaskNode] | None = None,
	source: Path | None = None,
	scope_effects: Mapping[str, type[Effect[Any]]] = {},
) -> argparse.ArgumentParser:
	"""Build the CLI argument parser.

	When ``tasks`` is provided, known task names appear in the positional
	metavar so the usage line reads like a list of subcommands.

	>>> from camas.core.task import Task
	>>> parser = build_parser()
	>>> args = parser.parse_args(['Task("echo hi")'])
	>>> args.expression
	'Task("echo hi")'
	>>> parser.parse_args(["--list"]).list
	True
	>>> "task | expression" in build_parser({"all": Task("x")}).format_usage()
	True
	"""
	parser: Final = CamasArgumentParser(
		prog="camas",
		description="Generic parallel/sequential task runner with TUI output.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.tasks = tasks
	parser.source = source
	parser.scope_effects = scope_effects
	parser.add_argument(
		"expression",
		nargs="?",
		metavar=expression_metavar(tasks),
		help="name of a defined task, or a camas expression "
		"(trailing '-- ARGS' append to a Task's command)",
	)
	parser.add_argument(
		"--version",
		action="version",
		version=f"camas {importlib.metadata.version('camas')}",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="print the task tree without executing",
	)
	parser.add_argument(
		"--list",
		action="store_true",
		help="list all defined tasks and exit (also the default with no args)",
	)
	parser.add_argument(
		"--tree",
		action="store_true",
		help="print every defined task's expanded tree and exit",
	)
	parser.add_argument(
		"--effects",
		nargs="?",
		default="(Termtree(),)",
		const="",
		help="tuple of Effect instances; pass with no value to list available Effects",
	)
	parser.add_argument(
		"--matrix",
		action="append",
		default=[],
		metavar="KEY=VAL[,VAL...]",
		help="override a matrix axis (repeatable; e.g. --matrix PY=3.13)",
	)
	return parser


RESERVED_FLAGS: Final = frozenset(
	{"help", "version", "dry-run", "list", "tree", "effects", "matrix"}
)


def expression_metavar(tasks: Mapping[str, TaskNode] | None) -> str:
	"""Build the positional metavar.

	>>> from camas.core.task import Task
	>>> expression_metavar(None)
	'expression'
	>>> expression_metavar({"all": Task("x")})
	'task | expression'
	"""
	return "task | expression" if tasks else "expression"
