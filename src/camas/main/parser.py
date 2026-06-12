# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""argparse construction: help is generated from the loaded tasks and effects."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from typing import TYPE_CHECKING, Any, Final

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..core.render import color_on
from .check import describe_check_help
from .format import (
	format_available_effects,
	format_load_error_hint,
	format_reference,
	format_task_summary_listing,
	format_try_hint,
)
from .state import EMPTY_STATE, LoadErr, LoadOk, TasksState

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..v0.effect import Effect
	from ..v0.task import TaskNode


def default_effects_expr() -> str:
	"""Default ``--effects`` for the environment: ``Status('github')`` under GitHub
	Actions (``GITHUB_ACTIONS=true``, collapsed workflow groups), else live ``Termtree``.

	>>> import os
	>>> _saved = os.environ.pop("GITHUB_ACTIONS", None)
	>>> default_effects_expr()
	'(Termtree(),)'
	>>> os.environ["GITHUB_ACTIONS"] = "true"
	>>> default_effects_expr()
	'(Status(StatusOptions(output_mode="github")),)'
	>>> os.environ.pop("GITHUB_ACTIONS", None)
	'true'
	>>> _ = os.environ.update({"GITHUB_ACTIONS": _saved}) if _saved is not None else None
	"""
	if os.environ.get("GITHUB_ACTIONS") == "true":
		return '(Status(StatusOptions(output_mode="github")),)'
	return "(Termtree(),)"


class CamasArgumentParser(argparse.ArgumentParser):
	"""``ArgumentParser`` whose ``--help`` appends the discovered tasks listing
	(or load-error hint), the Effects listing, and the Try hint after argparse's
	standard output. ``state`` is populated by ``build_parser`` so the override
	has access without nesting.
	"""

	state: TasksState

	def format_help(self) -> str:
		color = color_on()
		sections = [super().format_help().rstrip()]
		match self.state:
			case LoadOk(tasks=tasks, source=source) if tasks:
				sections.append(format_task_summary_listing(tasks, source, color=color))
			case LoadErr(source=source, exception=exc):
				sections.append(format_load_error_hint(source, exc))
			case LoadOk():
				pass
			case _:
				assert_never(self.state)
		effects: Mapping[str, type[Effect[Any]]] = (
			self.state.scope_effects if isinstance(self.state, LoadOk) else {}
		)
		effects_listing = format_available_effects(color=color, scope_effects=effects)
		if effects_listing:
			sections.append(effects_listing)
		sections.append(format_try_hint(color))
		sections.append(format_reference(color))
		return "\n\n".join(sections) + "\n"


def build_parser(state: TasksState = EMPTY_STATE) -> argparse.ArgumentParser:
	"""Build the CLI argument parser.

	When ``state`` is :class:`LoadOk` with tasks, known task names appear in the
	positional metavar so the usage line reads like a list of subcommands.

	>>> from camas import Task
	>>> from camas.main.state import LoadOk
	>>> parser = build_parser()
	>>> parser.parse_args(['Task("echo hi")']).expression
	'Task("echo hi")'
	>>> parser.parse_args(["--list"]).list
	True
	>>> "task | expression" in build_parser(LoadOk({"all": Task("x")}, None, {})).format_usage()
	True
	"""
	tasks_for_metavar: Mapping[str, TaskNode] = state.tasks if isinstance(state, LoadOk) else {}
	parser: Final = CamasArgumentParser(
		prog="camas",
		description="Generic parallel/sequential task runner with TUI output.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.state = state
	parser.add_argument(
		"expression",
		nargs="?",
		metavar=expression_metavar(tasks_for_metavar),
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
		"--check",
		action="store_true",
		help=describe_check_help(),
	)
	parser.add_argument(
		"--effects",
		nargs="?",
		default=default_effects_expr(),
		const="",
		help="tuple of Effect instances; pass with no value to list available Effects "
		"(default: Termtree, or Status('github') when GITHUB_ACTIONS=true)",
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
	{"help", "version", "dry-run", "list", "tree", "check", "effects", "matrix"}
)


def expression_metavar(tasks: Mapping[str, TaskNode] | None) -> str:
	"""Positional metavar: ``task | expression`` when tasks exist, else ``expression``."""
	return "task | expression" if tasks else "expression"
