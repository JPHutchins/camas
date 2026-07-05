# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""argparse construction: help is generated from the loaded tasks and effects."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import sys
from typing import TYPE_CHECKING, Any, Final

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..core.color import BOLD_CYAN
from ..core.render import color_on
from ..v0.config import Config
from .check import describe_check_help
from .color import maybe_color
from .effects import (
	default_effect_names,
	format_effects_expr,
	resolve_default_effects,
	running_under_agent,
)
from .format import (
	format_available_effects,
	format_load_error_hint,
	format_reference,
	format_task_summary_listing,
	format_try_hint,
	task_summary,
)
from .state import EMPTY_STATE, LoadErr, LoadOk, TasksState

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..v0.effect import Effect
	from ..v0.task import TaskNode


def effects_help(config: Config, *, github: bool, agent: bool = False) -> str:
	"""``--effects`` help: names the environment's resolved default, and the other
	environment's default when it differs (``Termtree`` locally vs. ``Status``
	under GitHub Actions, unless the :class:`Config` overrides them).

	>>> effects_help(Config(), github=False)
	"tuple of Effect instances; pass with no value to list available Effects (default: (Termtree(),); (Status(output_mode='github'),) under GITHUB_ACTIONS=true)"
	>>> effects_help(Config(), github=True)
	"tuple of Effect instances; pass with no value to list available Effects (default: (Status(output_mode='github'),); (Termtree(),) off GitHub Actions)"
	"""
	active = format_effects_expr(resolve_default_effects(config, github=github, agent=agent))
	other = format_effects_expr(resolve_default_effects(config, github=not github, agent=agent))
	where = "under GITHUB_ACTIONS=true" if not github else "off GitHub Actions"
	default = active if active == other else f"{active}; {other} {where}"
	return (
		"tuple of Effect instances; pass with no value to list available Effects "
		f"(default: {default})"
	)


def positional_help(config: Config, *, github: bool, color: bool = False) -> str:
	"""Positional help: names the task a bare ``camas`` runs, when one is configured.
	A named task is rendered in bold cyan (matching ``--list``); an anonymous one
	falls back to its one-line summary.

	>>> from camas import Task
	>>> positional_help(Config(default_task=Task("echo hi", name="greet")), github=False)
	"name of a defined task, a camas expression, or 'mcp' (trailing '-- ARGS' append to a Task's command); with no argument, runs greet"
	>>> positional_help(Config(), github=False)
	"name of a defined task, a camas expression, or 'mcp' (trailing '-- ARGS' append to a Task's command)"
	"""
	base = "name of a defined task, a camas expression, or 'mcp' (trailing '-- ARGS' append to a Task's command)"
	bare = config.bare_task(github=github)
	if bare is None:
		return base
	label = (
		maybe_color(bare.name, BOLD_CYAN, color)
		if bare.name is not None
		else task_summary(bare, frozenset())
	)
	return f"{base}; with no argument, runs {label}"


def positive_jobs(raw: str) -> int:
	"""Validate ``--jobs`` as an argparse ``type``: a positive integer.

	Raises:
		ArgumentTypeError: when ``raw`` is not an integer ``>= 1``.
	"""
	try:
		n = int(raw)
	except ValueError:
		raise argparse.ArgumentTypeError(f"--jobs expects an integer, got {raw!r}") from None
	if n < 1:
		raise argparse.ArgumentTypeError(f"--jobs must be >= 1, got {n}")
	return n


_DURATION: Final = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)?\s*")


def parse_duration(raw: str) -> float:
	"""Validate ``--under`` as an argparse ``type``: a positive duration in seconds.

	Accepts a bare number (seconds) or a ``ms``/``s``/``m``/``h`` suffix.

	Raises:
		ArgumentTypeError: when ``raw`` is not a positive duration.

	>>> parse_duration("1.5s")
	1.5
	>>> parse_duration("500ms")
	0.5
	>>> parse_duration("2m")
	120.0
	>>> parse_duration("3")
	3.0
	"""
	m = _DURATION.fullmatch(raw)
	if m is None:
		raise argparse.ArgumentTypeError(
			f"--under expects a duration like '1s', '500ms', '2m', got {raw!r}"
		)
	value = float(m.group(1))
	if value <= 0:
		raise argparse.ArgumentTypeError(f"--under must be positive, got {raw!r}")
	match m.group(2):
		case "ms":
			return value / 1000.0
		case "m":
			return value * 60.0
		case "h":
			return value * 3600.0
		case _:
			return value


def resolve_jobs(cli_jobs: int | None) -> int | None:
	"""Effective ``--jobs`` cap: the CLI flag wins, else ``CAMAS_JOBS``, else
	``None`` (unbounded).

	Raises:
		ValueError: when ``CAMAS_JOBS`` is set but not a positive integer.
	"""
	if cli_jobs is not None:
		return cli_jobs
	raw = os.environ.get("CAMAS_JOBS")
	if not raw:
		return None
	try:
		n = int(raw)
	except ValueError:
		raise ValueError(f"CAMAS_JOBS expects a positive integer, got {raw!r}") from None
	if n < 1:
		raise ValueError(f"CAMAS_JOBS must be >= 1, got {n}")
	return n


class CamasArgumentParser(argparse.ArgumentParser):
	"""``ArgumentParser`` whose ``--help`` appends the discovered tasks listing
	(or load-error hint), the Effects listing, and the Try hint after argparse's
	standard output. ``state`` is populated by ``build_parser`` so the override
	has access without nesting.
	"""

	state: TasksState

	def format_help(self) -> str:
		color = color_on()
		config = (
			self.state.config
			if isinstance(self.state, LoadOk) and self.state.config is not None
			else Config()
		)
		github = os.environ.get("GITHUB_ACTIONS") == "true"
		agent = running_under_agent()
		bare = config.bare_task(github=github)
		sections = [super().format_help().rstrip()]
		match self.state:
			case LoadOk(tasks=tasks, source=source) if tasks:
				sections.append(
					format_task_summary_listing(
						tasks,
						source,
						color=color,
						default_task_name=bare.name if bare is not None else None,
						camas_dir=config.camas_path(source.parent) if source is not None else None,
					)
				)
			case LoadErr(source=source, exception=exc):
				sections.append(format_load_error_hint(source, exc))
			case LoadOk():
				pass
			case _:
				assert_never(self.state)
		effects: Mapping[str, type[Effect[Any]]] = (
			self.state.scope_effects if isinstance(self.state, LoadOk) else {}
		)
		effects_listing = format_available_effects(
			color=color,
			scope_effects=effects,
			default_effect_names=default_effect_names(config, github=github, agent=agent),
		)
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
	>>> build_parser().parse_args(["fix", "--paths", "a.py", "--paths", "b.py"]).paths
	['a.py', 'b.py']
	"""
	tasks_for_metavar: Mapping[str, TaskNode] = state.tasks if isinstance(state, LoadOk) else {}
	config: Final = (
		state.config if isinstance(state, LoadOk) and state.config is not None else Config()
	)
	github: Final = os.environ.get("GITHUB_ACTIONS") == "true"
	agent: Final = running_under_agent()
	parser: Final = CamasArgumentParser(
		prog="camas",
		description="A task runner with parallel execution, matrix expansion, MCP, and pluggable output effects.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.state = state
	parser.add_argument(
		"expression",
		nargs="?",
		metavar=expression_metavar(tasks_for_metavar),
		help=positional_help(config, github=github, color=color_on()),
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
		help="list all defined tasks and exit",
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
		"--init",
		action="store_true",
		help="write a commented starter tasks.py into the current directory and exit",
	)
	parser.add_argument(
		"--effects",
		nargs="?",
		default=None,
		const="",
		help=effects_help(config, github=github, agent=agent),
	)
	parser.add_argument(
		"--matrix",
		action="append",
		default=[],
		metavar="KEY=VAL[,VAL...]",
		help="override a matrix axis (repeatable; e.g. --matrix PY=3.13)",
	)
	parser.add_argument(
		"--jobs",
		type=positive_jobs,
		default=None,
		metavar="N",
		help="cap concurrently running leaf subprocesses at N (also: CAMAS_JOBS). "
		"A throttle, not a speedup — the tree never runs wider than its own "
		"fan-out, so --jobs can only slow a run down; reach for it only when a "
		"wide matrix oversubscribes the machine (CPU/RAM/disk)",
	)
	parser.add_argument(
		"--under",
		type=parse_duration,
		default=None,
		metavar="DURATION",
		help="run only the task's leaves whose timed estimate fits DURATION (e.g. 1s, "
		"500ms, 2m), mutating leaves (formatters) first then the read-only rest in "
		"parallel; untimed leaves run (and are thereby measured), only leaves measured "
		"over budget are skipped, so a cold cache runs the whole tree. Operates on the "
		"named task, or the Config default when none is given",
	)
	parser.add_argument(
		"--paths",
		action="append",
		default=None,
		metavar="PATH[,PATH...]",
		help="scope the run to these changed paths (repeatable): inject them into each "
		"leaf's {paths} and drop leaves that match none (a leaf's when= predicate also "
		"prunes on no match), e.g. camas check --paths src/a.py. "
		"A PostToolBatch hook drives the agent fix node this way, feeding camas mcp fix the "
		"changed files on stdin",
	)
	return parser


RESERVED_FLAGS: Final = frozenset(
	{
		"help",
		"version",
		"dry-run",
		"list",
		"tree",
		"check",
		"init",
		"effects",
		"matrix",
		"jobs",
		"under",
		"paths",
	}
)


def expression_metavar(tasks: Mapping[str, TaskNode] | None) -> str:
	"""Positional metavar: prepends ``task`` when tasks exist; ``mcp`` is always reserved."""
	return "task | expression | mcp" if tasks else "expression | mcp"
