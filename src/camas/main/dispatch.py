# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""CLI dispatch: resolve the tasks source, parse argv, then run or print."""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..core.execution import run
from ..core.matrix import matrix_axes, override_matrix
from ..core.render import print_tree
from ..v0.config import Config
from .argv import apply_passthrough, parse_axis_values, parse_matrix_kv, split_passthrough
from .effects import default_effect_names, resolve_effects
from .expression import parse_expression
from .format import (
	format_load_error_hint,
	print_available_effects,
	print_task_help,
	print_task_summary_listing,
	print_task_trees,
)
from .init import write_starter_tasks_py
from .parser import RESERVED_FLAGS, build_parser, resolve_jobs
from .state import EMPTY_STATE, LoadErr, LoadOk, TasksState
from .tasks import (
	load_tasks,
	load_tasks_from_py,
	name_scope_bindings,
	name_scope_config,
	name_scope_effects,
)

if TYPE_CHECKING:
	import io
	from collections.abc import Mapping

	from ..v0.task import TaskNode


_NAME_LIKE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
"""A bare task name, hyphens allowed — distinct from a camas expression, which
carries parens, quotes, or braces. A ``-`` is taken as part of the name (the
convention alias), never as subtraction."""


def dispatch_arg(arg: str, tasks: Mapping[str, TaskNode]) -> TaskNode:
	"""Interpret a CLI arg: task name ⇒ lookup, else inline expression.

	The name is matched verbatim first (so a hyphenated ``[tool.camas.tasks]``
	key resolves directly), then with hyphens folded to underscores — so the
	``tasks.py`` binding ``test_all`` is reachable as the conventional
	``camas test-all``, which would otherwise parse as a subtraction.
	"""
	for candidate in (arg, arg.replace("-", "_")):
		if candidate in tasks:
			return tasks[candidate]
	if _NAME_LIKE.match(arg):
		known = ", ".join(sorted(tasks)) or "none"
		print(f"error: no task named {arg!r} (known: {known})", file=sys.stderr)
		sys.exit(2)
	return parse_expression(arg, tasks=tasks)


def print_interrupt_banner(count: int) -> None:
	"""Print the white CLI exit line for a Ctrl-C'd run."""
	from ..core.color import RESET, WHITE
	from ..core.render import color_on

	line = f"Ctrl-C ({count}) received - exiting"
	print(f"{WHITE}{line}{RESET}" if color_on() else line)


def reconfigure_stdio_utf8() -> None:
	"""UTF-8 with ``errors="replace"`` on stdout/stderr so Windows consoles and
	pipes (ANSI code page by default before Python 3.15) can carry the
	box-drawing tree output and ``×`` matrix annotations.
	"""
	for stream in (sys.stdout, sys.stderr):
		cast("io.TextIOWrapper", stream).reconfigure(encoding="utf-8", errors="replace")


def run_cli(scope: Mapping[str, object]) -> None:
	"""Collect Task/Sequential/Parallel and Effect bindings from ``scope`` (skipping
	``_``-prefixed names) and dispatch CLI args, citing ``scope['__file__']`` as the
	:class:`LoadOk` ``source`` for per-task help and ``--check``.

	The standalone entry point for a PEP 723 ``tasks.py`` run via ``python tasks.py``
	or ``uv run --script tasks.py`` — hand it the module globals::

	    if __name__ == "__main__":
	        from camas import run_cli

	        run_cli(globals())
	"""
	reconfigure_stdio_utf8()
	source_obj = scope.get("__file__")
	source = Path(source_obj) if isinstance(source_obj, (str, Path)) else None
	dispatch(
		LoadOk(
			tasks=name_scope_bindings(scope),
			source=source,
			scope_effects=name_scope_effects(scope),
			config=name_scope_config(scope),
		)
	)


def exit_for_load_err(err: LoadErr) -> None:
	"""Print a minimal trace + opportunistic typecheck for ``err`` and exit ``1``."""
	from .check import report_eval_error

	sys.exit(report_eval_error(err.source, err.exception))


def dispatch(state: TasksState, argv: list[str] | None = None) -> None:
	"""Parse ``argv`` (defaulting to sys.argv) against ``state`` and run the dispatched task.

	Dispatch is structured around the state sum type:

	* :class:`LoadOk` — task-running path. ``camas`` with no expression runs the
	  :class:`Config`'s task for the environment, else prints help and exits ``2``.
	* :class:`LoadErr` — meta actions (``--list`` / ``--tree`` / ``--effects`` /
	  ``--init``) still work; everything else delegates to :func:`exit_for_load_err`.
	"""
	split: Final = split_passthrough(sys.argv[1:] if argv is None else argv)
	tasks: Mapping[str, TaskNode] = state.tasks if isinstance(state, LoadOk) else {}

	if (
		len(split.head) >= 2
		and split.head[0] in tasks
		and any(a in ("-h", "--help") for a in split.head[1:])
	):
		print_task_help(split.head[0], tasks[split.head[0]])
		sys.exit(0)

	parser: Final = build_parser(state)

	match state:
		case LoadErr() as err:
			# No per-task matrix axes to augment; parse strictly so typos in
			# flags surface instead of being silently consumed.
			args = parser.parse_args(split.head)
			if args.init:
				sys.exit(write_starter_tasks_py(Path.cwd()))
			if args.list or args.tree:
				print(format_load_error_hint(err.source, err.exception))
				sys.exit(0)
			if args.effects == "":
				print_available_effects(
					{},
					default_effect_names(
						Config(), github=os.environ.get("GITHUB_ACTIONS") == "true"
					),
				)
				sys.exit(0)
			exit_for_load_err(err)

		case LoadOk(tasks=tasks, source=source, scope_effects=scope_effects, config=config):
			args, _leftover = parser.parse_known_args(split.head)
			in_github: Final = os.environ.get("GITHUB_ACTIONS") == "true"
			effective_config: Final = config if config is not None else Config()
			default_node: Final = effective_config.bare_task(github=in_github)
			axis_node: Final = (
				tasks[args.expression]
				if isinstance(args.expression, str) and args.expression in tasks
				else default_node
			)
			augmented_axes: dict[str, tuple[str, ...]] = {}
			if axis_node is not None:
				for name, values in matrix_axes(axis_node).items():
					if name.lower() in RESERVED_FLAGS:
						continue
					parser.add_argument(
						f"--{name}",
						dest=name,
						action="store",
						default=None,
						metavar="VAL[,VAL...]",
						help=f"override matrix axis {name!r} (current: {', '.join(values)})",
					)
					augmented_axes[name] = values
			args = parser.parse_args(split.head)

			if args.init:
				sys.exit(write_starter_tasks_py(Path.cwd()))

			if args.list:
				print_task_summary_listing(
					tasks,
					source,
					default_task_name=default_node.name if default_node is not None else None,
				)
				sys.exit(0)

			if args.tree:
				print_task_trees(tasks, source)
				sys.exit(0)

			if args.check:
				from .check import run_typecheck_only

				sys.exit(run_typecheck_only(source))

			if args.effects == "":
				print_available_effects(
					scope_effects, default_effect_names(effective_config, github=in_github)
				)
				sys.exit(0)

			resolved: TaskNode
			if args.expression is None:
				if default_node is None:
					parser.print_help()
					sys.stdout.flush()
					print(
						f"{parser.prog}: error: task or expression is required "
						"(define a Config with default_task to set a default)",
						file=sys.stderr,
					)
					sys.exit(2)
				resolved = default_node
			else:
				resolved = dispatch_arg(args.expression, tasks)

			try:
				effects: Final = resolve_effects(
					args.effects, effective_config, github=in_github, scope_effects=scope_effects
				)
			except ValueError as e:
				print(f"error: --effects: {e}", file=sys.stderr)
				sys.exit(2)

			overrides: dict[str, tuple[str, ...]] = {}
			for raw in args.matrix:
				try:
					k, v = parse_matrix_kv(raw)
				except ValueError as e:
					print(f"error: {e}", file=sys.stderr)
					sys.exit(2)
				overrides[k] = v
			for axis_name in augmented_axes:
				val = getattr(args, axis_name, None)
				if val is not None:
					values = parse_axis_values(val)
					if not values:
						print(f"error: --{axis_name}: at least one value required", file=sys.stderr)
						sys.exit(2)
					overrides[axis_name] = values

			if overrides:
				try:
					resolved = override_matrix(resolved, overrides)
				except ValueError as e:
					print(f"error: {e}", file=sys.stderr)
					sys.exit(2)
			try:
				task: Final = (
					apply_passthrough(resolved, split.passthrough)
					if split.passthrough
					else resolved
				)
			except ValueError as e:
				print(f"error: {e}", file=sys.stderr)
				sys.exit(2)
			if args.dry_run:
				print_tree(task, show_cmd=True)
				sys.exit(0)
			try:
				jobs: Final = resolve_jobs(args.jobs)
			except ValueError as e:
				print(f"error: {e}", file=sys.stderr)
				sys.exit(2)
			result: Final = asyncio.run(run(task, effects=effects, jobs=jobs))
			if result.interrupt_count:
				print_interrupt_banner(result.interrupt_count)
			sys.exit(result.returncode)

		case _:
			assert_never(state)


def _load_py(path: Path) -> TasksState:
	"""Evaluate ``path`` and return a :class:`LoadOk` / :class:`LoadErr`."""
	try:
		return load_tasks_from_py(path)
	except Exception as e:
		return LoadErr(source=path, exception=e)


def resolve_tasks_source(argv: list[str]) -> tuple[TasksState, list[str]]:
	"""Locate the tasks source and return ``(state, remaining_argv)``.

	If ``argv[0]`` ends in ``.py`` it is consumed as an explicit file path.
	Otherwise walks upward from cwd: ``tasks.py`` wins over ``pyproject.toml``
	at the same level; a ``pyproject.toml`` without ``[tool.camas.tasks]`` keeps
	the walk going. A ``tasks.py`` whose evaluation raises returns
	:class:`LoadErr` so meta operations (``--help``, ``--list``) still work.
	"""
	if argv and argv[0].endswith(".py"):
		path = Path(argv[0])
		if not path.is_file():
			print(f"error: {path}: no such file", file=sys.stderr)
			sys.exit(2)
		return _load_py(path), argv[1:]

	start: Final = Path.cwd()
	for candidate in (start, *start.parents):
		tasks_py = candidate / "tasks.py"
		if tasks_py.is_file():
			return _load_py(tasks_py), argv
		pyproject = candidate / "pyproject.toml"
		if pyproject.is_file():
			try:
				loaded = load_tasks(pyproject)
			except ValueError as e:
				print(f"error: {pyproject}: {e}", file=sys.stderr)
				sys.exit(2)
			if loaded.tasks:
				return loaded, argv

	return EMPTY_STATE, argv
