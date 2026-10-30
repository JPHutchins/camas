# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Final, NoReturn

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..core.attached import Exited, Signaled, die_of, run_attached, scoped_body
from ..core.color import BOLD_CYAN, CAMAS_VIOLET
from ..core.gate import strip_agent_only_pipes
from ..core.render import color_on, print_tree
from ..core.scope import with_default_paths
from ..core.traversal import flatten_leaves
from ..v0.task import Task
from .argv import apply_passthrough
from .color import maybe_color

if TYPE_CHECKING:
	import argparse
	from collections.abc import Iterable, Mapping
	from pathlib import Path

	from ..core.attached import Outcome
	from ..v0.command import Command
	from ..v0.task import TaskNode


PLACEHOLDER: Final = re.compile(r"\{(\w+)\}")


def command_name(token: str, commands: Mapping[str, Command]) -> str | None:
	"""
	>>> from camas.v0.command import Command
	>>> command_name("flash-board", {"flash_board": Command("x")}), command_name("lint", {})
	('flash_board', None)
	"""
	return next((name for name in (token, token.replace("-", "_")) if name in commands), None)


def run_command_cli(
	command: Command,
	args: argparse.Namespace,
	passthrough: tuple[str, ...],
	*,
	axes: Iterable[str],
	base: Path | None,
	agent: bool,
) -> NoReturn:
	refused: Final = refused_flag(args, axes)
	if refused is not None:
		print(f"error: {refused} does not apply to a Command", file=sys.stderr)
		sys.exit(2)
	try:
		node: Final = command_node(command, passthrough, agent=agent)
	except ValueError as e:
		print(f"error: {e}", file=sys.stderr)
		sys.exit(2)
	if args.dry_run:
		print_tree(node, show_cmd=True)
		sys.exit(0)
	exit_with(run_attached(node, base))


def refused_flag(args: argparse.Namespace, axes: Iterable[str]) -> str | None:
	"""
	>>> from camas.main.parser import build_parser
	>>> refused_flag(build_parser().parse_args(["flash", "--dry-run"]), ())
	>>> refused_flag(build_parser().parse_args(["flash", "--jobs", "2"]), ())
	'--jobs'
	"""
	flags: Final = (
		("--effects", args.effects is not None),
		("--jobs", args.jobs is not None),
		("--under", args.under is not None),
		("--paths", args.paths is not None),
		("--github-matrix", args.github_matrix is not None),
		("--matrix", bool(args.matrix)),
		*((f"--{axis}", getattr(args, axis, None) is not None) for axis in axes),
	)
	return next((flag for flag, given in flags if given), None)


def command_node(command: Command, passthrough: tuple[str, ...], *, agent: bool) -> TaskNode:
	"""
	>>> from camas.v0.command import Command
	>>> from camas.v0.task import Pipe
	>>> command_node(Command("pytest"), ("-x",), agent=False)
	Task(cmd='pytest -x', name=None, env={}, cwd=None)
	>>> command_node(Command(Pipe("lint", "sarif", agent_only=True)), (), agent=False) == Pipe("lint")
	True
	"""
	body: Final = scoped_body(command)
	scoped: Final = with_default_paths(body if agent else strip_agent_only_pipes(body))
	return apply_passthrough(scoped, passthrough) if passthrough else scoped


def exit_with(outcome: Outcome) -> NoReturn:
	match outcome:
		case Exited(code=code):
			sys.exit(code)
		case Signaled(signum=signum):
			die_of(signum)
		case _:
			assert_never(outcome)


def print_command_help(name: str, command: Command) -> None:
	passthrough: Final = " [-- ARG ...]" if isinstance(command.body, Task) else ""
	print(f"usage: camas {name} [-h] [--dry-run]{passthrough}")
	if command.help is not None:
		print()
		print(command.help)
	print()
	print_tree(scoped_body(command), show_cmd=True)


def print_command_trees(commands: Mapping[str, Command]) -> None:
	if commands:
		print(maybe_color("Commands:", CAMAS_VIOLET, color_on()))
	for name, command in sorted(commands.items()):
		print()
		print(maybe_color(name, BOLD_CYAN, color_on()))
		print_tree(scoped_body(command), show_cmd=True)


def format_command_listing(commands: Mapping[str, Command], color: bool) -> str:
	"""
	>>> from camas.v0.command import Command
	>>> print(format_command_listing({"flash": Command("probe-rs run", help="Flash the board")}, color=False))
	Commands:
	  flash  Flash the board
	>>> format_command_listing({}, color=False)
	''
	"""
	if not commands:
		return ""
	width: Final = max(map(len, commands))
	return "\n".join(
		(
			maybe_color("Commands:", CAMAS_VIOLET, color),
			*(
				f"  {maybe_color(name.ljust(width), BOLD_CYAN, color)}  {summary(command)}"
				for name, command in sorted(commands.items())
			),
		)
	)


def summary(command: Command) -> str:
	"""
	>>> from camas.v0.command import Command
	>>> summary(Command(Task("build") + Task(("probe-rs", "run"))))
	'build; probe-rs run'
	"""
	if command.help is not None:
		return command.help
	return "; ".join(leaf_cmd(info.task) for info in flatten_leaves(command.body))


def leaf_cmd(task: Task) -> str:
	return task.cmd if isinstance(task.cmd, str) else " ".join(task.cmd)


def placeholder_warnings(commands: Mapping[str, Command]) -> tuple[str, ...]:
	"""
	>>> from camas.v0.command import Command
	>>> placeholder_warnings({"flash": Command("probe-rs run --chip {CHIP} {paths}")})
	("command 'flash' runs {CHIP} literally: a Command has no matrix axes",)
	"""
	return tuple(
		f"command {name!r} runs {{{token}}} literally: a Command has no matrix axes"
		for name, command in sorted(commands.items())
		for token in sorted(
			{
				token
				for info in flatten_leaves(command.body)
				for token in PLACEHOLDER.findall(leaf_cmd(info.task))
			}
			- {"paths"}
		)
	)
