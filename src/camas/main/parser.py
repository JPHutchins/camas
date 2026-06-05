# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from collections.abc import Mapping
from typing import Any, Final, NamedTuple

from argtree import arg, from_namespace
from argtree import build_parser as argtree_build_parser

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..core.effect import Effect
from ..core.render import color_on
from ..core.task import TaskNode
from .check import describe_check_help
from .format import (
	format_available_effects,
	format_load_error_hint,
	format_reference,
	format_task_summary_listing,
	format_try_hint,
)
from .state import EMPTY_STATE, LoadErr, LoadOk, TasksState


def default_effects_expr() -> str:
	"""Pick the default ``--effects`` expression based on the runtime environment.

	GitHub Actions sets ``GITHUB_ACTIONS=true`` reliably; in that case prefer
	``Status`` with the ``github`` mode (collapsed workflow groups) over the
	live ``Termtree`` (which renders garbage in non-interactive log capture).

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


class Cli(NamedTuple):
	"""The ``camas`` command line as an :mod:`argtree` spec — one field per
	``argparse`` argument, recovered as a typed value by :func:`reconstruct`.

	``effects`` is ``str | None`` rather than carrying the environment-sensitive
	default directly: a field default is frozen at class-definition time, but the
	``--effects`` default depends on ``GITHUB_ACTIONS`` *at parse time*, so the
	resolution is deferred to :func:`camas.main.dispatch` (``None`` ⇒ absent ⇒
	:func:`default_effects_expr`, ``""`` ⇒ given without a value ⇒ list Effects).
	"""

	expression: str | None = arg(
		positional=True,
		nargs="?",
		help="name of a defined task, or a camas expression "
		"(trailing '-- ARGS' append to a Task's command)",
	)
	version: bool = arg(
		"--version",
		action="version",
		version=f"camas {importlib.metadata.version('camas')}",
	)
	dry_run: bool = arg("--dry-run", help="print the task tree without executing")
	list_: bool = arg(
		"--list",
		help="list all defined tasks and exit (also the default with no args)",
	)
	tree: bool = arg("--tree", help="print every defined task's expanded tree and exit")
	check: bool = arg("--check", help=describe_check_help())
	effects: str | None = arg(
		"--effects",
		nargs="?",
		const="",
		help="tuple of Effect instances; pass with no value to list available Effects "
		"(default: Termtree, or Status('github') when GITHUB_ACTIONS=true)",
	)
	matrix: list[str] = arg(
		"--matrix",
		action="append",
		default=[],
		metavar="KEY=VAL[,VAL...]",
		help="override a matrix axis (repeatable; e.g. --matrix PY=3.13)",
	)


DESCRIPTION: Final = "Generic parallel/sequential task runner with TUI output."


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

	The argument surface is declared once on :class:`Cli`; :func:`argtree.build_parser`
	constructs it straight into a :class:`CamasArgumentParser` (via ``parser_class=``)
	so the custom ``--help`` (tasks/Effects/Try listings) layers on top. When ``state``
	is :class:`LoadOk` with tasks, known task names appear in the positional metavar so
	the usage line reads like a list of subcommands.

	>>> from camas.core.task import Task
	>>> from camas.main.state import LoadOk
	>>> reconstruct(build_parser().parse_args(['Task("echo hi")'])).expression
	'Task("echo hi")'
	>>> reconstruct(build_parser().parse_args(["--list"])).list_
	True
	>>> "task | expression" in build_parser(LoadOk({"all": Task("x")}, None, {})).format_usage()
	True
	"""
	tasks_for_metavar: Mapping[str, TaskNode] = state.tasks if isinstance(state, LoadOk) else {}
	parser: Final = argtree_build_parser(
		Cli,
		parser_class=CamasArgumentParser,
		prog="camas",
		description=DESCRIPTION,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.state = state
	metavar: Final = expression_metavar(tasks_for_metavar)
	for action in parser._actions:
		if not action.option_strings:
			action.metavar = metavar
	return parser


def reconstruct(namespace: argparse.Namespace) -> Cli:
	"""Rebuild the typed :class:`Cli` from a parsed argparse namespace.

	The dynamically-added per-matrix-axis flags (e.g. ``--PY``) are not part of
	:class:`Cli`; :func:`argtree.from_namespace` ignores them, and the dispatcher
	reads them off the raw namespace by their (clean) dest.
	"""
	return from_namespace(Cli, namespace)


RESERVED_FLAGS: Final = frozenset(
	{"help", "version", "dry-run", "list", "tree", "check", "effects", "matrix"}
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
