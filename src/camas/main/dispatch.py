# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""CLI dispatch: resolve the tasks source, parse argv, then run or print."""

from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..core.execution import run
from ..core.matrix import matrix_axes, override_matrix
from ..core.render import color_on, print_tree
from .argv import apply_passthrough, parse_axis_values, parse_matrix_kv, split_passthrough
from .effects import parse_effects
from .expression import parse_expression
from .format import (
	format_load_error_hint,
	format_reference,
	format_try_hint,
	print_available_effects,
	print_task_help,
	print_task_summary_listing,
	print_task_trees,
)
from .parser import RESERVED_FLAGS, build_parser
from .state import EMPTY_STATE, LoadErr, LoadOk, TasksState
from .tasks import load_tasks, load_tasks_from_py, name_scope_bindings, name_scope_effects

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..v0.task import TaskNode


def dispatch_arg(arg: str, tasks: Mapping[str, TaskNode]) -> TaskNode:
	"""Interpret a CLI arg: task name (possibly hyphenated) ⇒ lookup, else inline expression."""
	if arg in tasks:
		return tasks[arg]
	try:
		parsed = ast.parse(arg, mode="eval")
	except SyntaxError:
		return parse_expression(arg, tasks=tasks)
	if isinstance(parsed.body, ast.Name):
		name = parsed.body.id
		known = ", ".join(sorted(tasks)) or "none"
		print(f"error: no task named {name!r} (known: {known})", file=sys.stderr)
		sys.exit(2)
	return parse_expression(arg, tasks=tasks)


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
	source_obj = scope.get("__file__")
	source = Path(source_obj) if isinstance(source_obj, (str, Path)) else None
	dispatch(
		LoadOk(
			tasks=name_scope_bindings(scope),
			source=source,
			scope_effects=name_scope_effects(scope),
		)
	)


def exit_for_load_err(err: LoadErr) -> None:
	"""Print a minimal trace + opportunistic typecheck for ``err`` and exit ``1``."""
	from .check import report_eval_error

	sys.exit(report_eval_error(err.source, err.exception))


def dispatch(state: TasksState, argv: list[str] | None = None) -> None:
	"""Parse ``argv`` (defaulting to sys.argv) against ``state`` and run the dispatched task.

	Dispatch is structured around the state sum type:

	* :class:`LoadOk` — task-running path. ``camas`` with no expression prints
	  the listing and exits ``2`` (a la ``just`` / ``task``).
	* :class:`LoadErr` — meta actions (``--list`` / ``--tree`` / ``--effects``)
	  still render; everything else delegates to :func:`exit_for_load_err`.
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
			if args.list or args.tree:
				print(format_load_error_hint(err.source, err.exception))
				sys.exit(0)
			if args.effects == "":
				print_available_effects({})
				sys.exit(0)
			exit_for_load_err(err)

		case LoadOk(tasks=tasks, source=source, scope_effects=scope_effects):
			args, _leftover = parser.parse_known_args(split.head)
			augmented_axes: dict[str, tuple[str, ...]] = {}
			if isinstance(args.expression, str) and args.expression in tasks:
				for name, values in matrix_axes(tasks[args.expression]).items():
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

			if args.list:
				print_task_summary_listing(tasks, source)
				sys.exit(0)

			if args.tree:
				print_task_trees(tasks, source)
				sys.exit(0)

			if args.check:
				from .check import run_typecheck_only

				sys.exit(run_typecheck_only(source))

			if args.effects == "":
				print_available_effects(scope_effects)
				sys.exit(0)

			if args.expression is None:
				if tasks:
					print(parser.format_usage().rstrip())
					print()
					print_task_summary_listing(tasks, source)
					print()
					print(format_try_hint(color_on()))
					print()
					print(format_reference(color_on()))
					print()
					sys.stdout.flush()
					print(f"{parser.prog}: error: task or expression is required", file=sys.stderr)
					sys.exit(2)
				parser.error("task or expression is required (no tasks defined)")

			try:
				effects: Final = parse_effects(args.effects, scope_effects)
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

			resolved = dispatch_arg(args.expression, tasks)
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
			sys.exit(asyncio.run(run(task, effects=effects)).returncode)

		case _:
			assert_never(state)


def _load_py(path: Path) -> TasksState:
	"""Evaluate ``path`` and return a :class:`LoadOk` / :class:`LoadErr`."""
	try:
		tasks, scope_effects = load_tasks_from_py(path)
	except Exception as e:
		return LoadErr(source=path, exception=e)
	return LoadOk(tasks=tasks, source=path, scope_effects=scope_effects)


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
				tasks, _ = load_tasks(pyproject)
			except ValueError as e:
				print(f"error: {pyproject}: {e}", file=sys.stderr)
				sys.exit(2)
			if tasks:
				return LoadOk(tasks=tasks, source=pyproject, scope_effects={}), argv

	return EMPTY_STATE, argv
