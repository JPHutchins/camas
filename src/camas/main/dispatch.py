# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""CLI dispatch: resolve the tasks source, parse argv, then run or print."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..core import timings
from ..core.budget import plan_under
from ..core.execution import run
from ..core.hook_event import stdin_changed
from ..core.matrix import expand_matrix, matrix_axes, override_matrix
from ..core.render import print_tree, render_tree_lines
from ..core.scope import scope_to_changed, to_changed, with_default_paths
from ..core.task import task_label
from ..v0.config import Config
from .argv import apply_passthrough, parse_axis_values, parse_matrix_kv, split_passthrough
from .discover import load_py_state, state_from_scope
from .effects import default_effect_names, resolve_effects, running_under_agent
from .expression import parse_expression
from .format import (
	format_load_error_hint,
	format_scope_warnings,
	print_available_effects,
	print_task_help,
	print_task_summary_listing,
	print_task_trees,
)
from .init import write_starter_tasks_py
from .parser import RESERVED_FLAGS, build_parser, resolve_jobs
from .state import EMPTY_STATE, LoadErr, LoadOk, TasksState
from .tasks import load_tasks

if TYPE_CHECKING:
	import io
	from collections.abc import Mapping, Sequence

	from ..core.budget import BudgetPlan
	from ..core.completion import RunResult
	from ..v0.effect import Effect
	from ..v0.task import TaskNode


_NAME_LIKE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*$")
"""A bare task name, hyphens and dotted namespace segments allowed — distinct
from a camas expression, which carries parens, quotes, or braces. A ``-`` is
taken as part of the name (the convention alias), never as subtraction."""


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


def budget_summary_lines(plan: BudgetPlan) -> list[str]:
	"""The ``--under`` summary: how many leaves run (and which are unmeasured), and what was
	excluded as measured-over-budget.
	"""
	lines = [
		f"Time budget {plan.budget_s:.2f}s — running {len(plan.fits) + len(plan.untimed)} leaf(s) "
		f"({len(plan.untimed)} unmeasured), excluded {len(plan.over_budget)} over budget."
	]
	if plan.over_budget:
		lines.append(
			"  over budget: "
			+ ", ".join(f"{task_label(o.task)} ~{o.estimated_s:.2f}s" for o in plan.over_budget)
		)
	if plan.untimed:
		lines.append(
			"  unmeasured (running to record an estimate): "
			+ ", ".join(task_label(u.task) for u in plan.untimed)
		)
	if plan.node is None:
		lines.append("All leaves exceed the budget — nothing to run.")
	return lines


def run_under(
	source: TaskNode,
	budget_s: float,
	*,
	camas_dir: Path | None,
	effects: Sequence[Effect[Any]],
	jobs: int | None,
	dry_run: bool,
	passthrough: tuple[str, ...],
	base: Path | None = None,
) -> int:
	"""Plan and run the leaves of ``source`` that fit ``budget_s``; return the exit code."""
	if passthrough:
		print("error: -- passthrough args cannot be combined with --under", file=sys.stderr)
		return 2
	plan = plan_under(source, budget_s, timings.load(camas_dir) if camas_dir is not None else {})
	for line in budget_summary_lines(plan):
		print(line)
	if plan.node is None:
		return 0
	if dry_run:
		print_tree(with_default_paths(plan.node), show_cmd=True)
		return 0
	return finish_run(asyncio.run(run(plan.node, effects=effects, jobs=jobs, base=base)))


def finish_run(result: RunResult) -> int:
	"""Print the interrupt banner for a Ctrl-C'd run and return the run's exit code."""
	if result.interrupt_count:
		print_interrupt_banner(result.interrupt_count)
	return result.returncode


def fix_cli(argv: list[str]) -> int:
	"""``camas mcp fix [--paths P]…``: run the project's *registered* agent fix node
	(``Config.agent.fix`` — whatever the user named it), scoped to the changed paths — taken from
	``--paths`` or, failing that, the ``PostToolBatch`` event piped on stdin (the Claude Code
	autofix hook). In the ``camas mcp`` namespace so it never collides with a user's own
	``camas <task>``. A clean no-op (exit 0) when no fix node is registered — without registration
	there is simply nothing for the hook to run.
	"""
	parser = argparse.ArgumentParser(
		prog="camas mcp fix", description="Run the registered agent fix node."
	)
	parser.add_argument(
		"--paths",
		action="append",
		default=[],
		metavar="PATH[,PATH...]",
		help="changed paths to scope to (repeatable, comma-separated)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		default=False,
		help="print the resolved path-scoped leaf plan without executing",
	)
	args = parser.parse_args(argv)
	state, _ = resolve_tasks_source([])
	if not isinstance(state, LoadOk) or state.config is None:
		return 0
	node = state.config.gate_fix()
	if node is None:
		return 0
	base = state.source.parent if state.source is not None else Path.cwd()
	stdin = stdin_changed() if not args.paths else None
	if not args.paths and stdin is not None and not stdin:
		return 0
	changed = to_changed(args.paths or (stdin or ()), base)
	expanded = expand_matrix(node)
	scoped = scope_to_changed(expanded, changed) if changed else with_default_paths(expanded)
	if args.dry_run:
		if scoped is None:
			print("No leaves cover the changed paths — nothing would run.")
		else:
			plan = "\n".join(render_tree_lines(scoped, show_cmd=True, color=False))
			print(f"Dry run — resolved path-scoped plan, nothing executed:\n{plan}")
		return 0
	if scoped is None:
		return 0
	return finish_run(asyncio.run(run(scoped, effects=(), jobs=None, base=base)))


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
	"""Intercept the ``mcp`` subcommand (routing to :mod:`camas.mcp.cli`), then
	dispatch CLI args against ``scope`` loaded as a :class:`~.state.TasksState`
	(:func:`state_from_scope` — ``scope``'s own bindings, composed with its
	discovered descendants, citing ``scope['__file__']`` as the source for
	per-task help and ``--check``).

	The standalone entry point for a PEP 723 ``tasks.py`` run via ``python tasks.py``
	or ``uv run --script tasks.py`` — hand it the module globals::

	    if __name__ == "__main__":
	        from camas import run_cli

	        run_cli(globals())
	"""
	reconfigure_stdio_utf8()
	argv = sys.argv[1:]
	if argv and argv[0] == "mcp":
		importlib.import_module("camas.mcp.cli").main(argv[1:])
		return
	dispatch(state_from_scope(scope))


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
						Config(),
						github=os.environ.get("GITHUB_ACTIONS") == "true",
						agent=running_under_agent(),
					),
				)
				sys.exit(0)
			exit_for_load_err(err)

		case LoadOk(tasks=tasks, source=source, scope_effects=scope_effects, config=config):
			args, _leftover = parser.parse_known_args(split.head)
			in_github: Final = os.environ.get("GITHUB_ACTIONS") == "true"
			in_agent: Final = running_under_agent()
			effective_config: Final = config if config is not None else Config()
			camas_dir: Final = (
				effective_config.camas_path(source.parent) if source is not None else None
			)
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
					camas_dir=camas_dir,
				)
				sys.exit(0)

			if args.tree:
				print_task_trees(tasks, source, camas_dir=camas_dir)
				sys.exit(0)

			if args.check:
				from .check import run_typecheck_only

				warnings = format_scope_warnings(tasks)
				if warnings:
					print(warnings, file=sys.stderr)
				sys.exit(run_typecheck_only(source))

			if args.effects == "":
				print_available_effects(
					scope_effects,
					default_effect_names(effective_config, github=in_github, agent=in_agent),
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
					args.effects,
					effective_config,
					github=in_github,
					agent=in_agent,
					scope_effects=scope_effects,
					base=source.parent if source is not None else None,
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

			if args.paths is not None:
				base = source.parent if source is not None else Path.cwd()
				changed = to_changed(args.paths, base)
				scoped = scope_to_changed(expand_matrix(resolved), changed) if changed else None
				if scoped is None:
					print(
						f"No task leaf covers {', '.join(changed) or '(no paths given)'}"
						" — nothing to run."
					)
					sys.exit(0)
				resolved = scoped

			if args.under is not None:
				try:
					budget_jobs: Final = resolve_jobs(args.jobs)
				except ValueError as e:
					print(f"error: {e}", file=sys.stderr)
					sys.exit(2)
				sys.exit(
					run_under(
						resolved,
						args.under,
						camas_dir=camas_dir,
						effects=effects,
						jobs=budget_jobs,
						dry_run=args.dry_run,
						passthrough=split.passthrough,
						base=source.parent if source is not None else None,
					)
				)

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
				print_tree(with_default_paths(task), show_cmd=True)
				sys.exit(0)
			try:
				jobs: Final = resolve_jobs(args.jobs)
			except ValueError as e:
				print(f"error: {e}", file=sys.stderr)
				sys.exit(2)
			sys.exit(
				finish_run(
					asyncio.run(
						run(
							task,
							effects=effects,
							jobs=jobs,
							base=source.parent if source is not None else None,
						)
					)
				)
			)

		case _:
			assert_never(state)


def _load_py(path: Path) -> TasksState:
	"""Evaluate ``path`` and return a :class:`LoadOk` / :class:`LoadErr`."""
	return load_py_state(path)


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
