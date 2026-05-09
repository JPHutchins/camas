# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast
import asyncio
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from ..core.execution import run
from ..core.matrix import matrix_axes, override_matrix
from ..core.render import color_on
from ..core.task import TaskNode
from .argv import apply_passthrough, parse_axis_values, parse_matrix_kv, split_passthrough
from .effects import parse_effects
from .expression import parse_expression
from .format import (
	format_reference,
	format_try_hint,
	print_available_effects,
	print_task_help,
	print_task_summary_listing,
	print_task_trees,
	print_tree,
)
from .parser import RESERVED_FLAGS, build_parser
from .tasks import load_tasks, load_tasks_from_py, name_scope_bindings


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
	"""Collect Task/Sequential/Parallel values from ``scope`` and dispatch CLI args.

	Names starting with ``_`` and non-task values are skipped. Public bindings
	are named by their binding identifier, and nested references inherit those
	names (consistent with ``[tool.camas.tasks]`` in pyproject.toml).
	"""
	dispatch(name_scope_bindings(scope))


def dispatch(
	tasks: Mapping[str, TaskNode],
	argv: list[str] | None = None,
	source: Path | None = None,
) -> None:
	"""Parse ``argv`` (defaulting to sys.argv) against ``tasks`` and run the dispatched task.

	When invoked with no expression and tasks are defined, prints a compact listing
	of available tasks and exits 0 — the ``camas`` (no-args) ergonomic default,
	mirroring ``just`` and ``task``.
	"""
	split: Final = split_passthrough(sys.argv[1:] if argv is None else argv)

	if (
		len(split.head) >= 2
		and split.head[0] in tasks
		and any(a in ("-h", "--help") for a in split.head[1:])
	):
		print_task_help(split.head[0], tasks[split.head[0]])
		sys.exit(0)

	parser: Final = build_parser(tasks, source)
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

	if args.effects == "":
		print_available_effects()
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
		effects: Final = parse_effects(args.effects)
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
			apply_passthrough(resolved, split.passthrough) if split.passthrough else resolved
		)
	except ValueError as e:
		print(f"error: {e}", file=sys.stderr)
		sys.exit(2)
	if args.dry_run:
		print_tree(task, show_cmd=True)
		sys.exit(0)
	sys.exit(asyncio.run(run(task, effects=effects)).returncode)


def looks_like_py_file(arg: str) -> bool:
	"""A CLI arg refers to a tasks file when it ends in ``.py`` — expressions always end in ``)``.

	>>> looks_like_py_file("tasks.py")
	True
	>>> looks_like_py_file("./sub/my_tasks.py")
	True
	>>> looks_like_py_file('Task("pytest x.py")')
	False
	>>> looks_like_py_file("lint")
	False
	"""
	return arg.endswith(".py")


def resolve_tasks_source(
	argv: list[str],
) -> tuple[dict[str, TaskNode], list[str], Path | None]:
	"""Locate the tasks source and return (tasks, remaining_argv, source_path).

	If ``argv[0]`` ends in ``.py`` it is consumed as an explicit file path.
	Otherwise walks upward from cwd and returns tasks from the nearest directory
	that defines any; ``tasks.py`` wins over ``pyproject.toml`` at the same level.
	A ``pyproject.toml`` without ``[tool.camas.tasks]`` is not a match — the walk
	continues upward. ``source_path`` is the file the tasks were loaded from, or
	``None`` when no tasks file was found.
	"""
	if argv and looks_like_py_file(argv[0]):
		path = Path(argv[0])
		if not path.is_file():
			print(f"error: {path}: no such file", file=sys.stderr)
			sys.exit(2)
		try:
			return load_tasks_from_py(path), argv[1:], path
		except Exception as e:
			print(f"error: {path}: {e}", file=sys.stderr)
			sys.exit(2)

	start: Final = Path.cwd()
	for candidate in (start, *start.parents):
		tasks_py = candidate / "tasks.py"
		if tasks_py.is_file():
			try:
				return load_tasks_from_py(tasks_py), argv, tasks_py
			except Exception as e:
				print(f"error: {tasks_py}: {e}", file=sys.stderr)
				sys.exit(2)
		pyproject = candidate / "pyproject.toml"
		if pyproject.is_file():
			try:
				tasks = load_tasks(pyproject)
			except ValueError as e:
				print(f"error: {pyproject}: {e}", file=sys.stderr)
				sys.exit(2)
			if tasks:
				return tasks, argv, pyproject

	return {}, argv, None
