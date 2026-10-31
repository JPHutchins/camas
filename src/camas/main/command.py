# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast
import re
import sys
from contextlib import suppress
from typing import TYPE_CHECKING, Final, NoReturn

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..core.attached import Exited, Signaled, die_of, run_attached, scoped_body
from ..core.color import BOLD_CYAN, CAMAS_VIOLET
from ..core.gate import strip_agent_only_pipes
from ..core.render import color_on, print_tree
from ..core.scope import PATHS_TOKEN, with_default_paths
from ..core.traversal import flatten_leaves
from ..v0.task import Task
from .argv import apply_passthrough
from .color import maybe_color
from .format import colorize_summary, task_summary

if TYPE_CHECKING:
	import argparse
	from collections.abc import Collection, Mapping
	from pathlib import Path

	from ..core.attached import Outcome
	from ..v0.command import Command
	from ..v0.task import TaskNode


PLACEHOLDER: Final = re.compile(r"\{\w+\}")

SHELL_UNEXPANDED_PLACEHOLDER: Final = re.compile(r"(?<!\$)\{\w+\}")

TOLERATED_ARGUMENTS: Final = frozenset({"expression", "dry_run", "verbose"})


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
	parser: argparse.ArgumentParser,
	axes: Collection[str],
	base: Path | None,
	agent: bool,
) -> NoReturn:
	refused: Final = refused_flag(args, parser, axes)
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


def refused_flag(
	args: argparse.Namespace, parser: argparse.ArgumentParser, axes: Collection[str]
) -> str | None:
	"""
	>>> from camas.main.parser import build_parser
	>>> parser = build_parser()
	>>> refused_flag(parser.parse_args(["flash", "--dry-run", "--verbose"]), parser, ())
	>>> refused_flag(parser.parse_args(["flash", "--github-matrix=auto"]), parser, ())
	'--github-matrix'
	"""
	return next(
		(
			f"--{dest}" if dest in axes else f"--{dest.replace('_', '-')}"
			for dest, value in vars(args).items()
			if dest not in TOLERATED_ARGUMENTS and value != parser.get_default(dest)
		),
		None,
	)


def composed_command(expression: str, commands: Collection[str]) -> str | None:
	"""
	>>> composed_command("lint | fw.flash", {"fw.flash"}), composed_command('Task("flash")', {"flash"})
	('fw.flash', None)
	"""
	with suppress(SyntaxError):
		return next(
			(
				ref
				for node in ast.walk(ast.parse(expression, mode="eval"))
				if isinstance(node, ast.Name | ast.Attribute)
				and (ref := ast.unparse(node)) in commands
			),
			None,
		)
	return None


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
				f"  {maybe_color(name.ljust(width), BOLD_CYAN, color)}  {summary(command, color)}"
				for name, command in sorted(commands.items())
			),
		)
	)


def summary(command: Command, color: bool) -> str:
	"""
	>>> from camas.v0.command import Command
	>>> from camas.v0.task import Pipe
	>>> summary(Command(Task("build") + Pipe("gen", Task(("probe-rs", "run")))), color=False)
	'build, gen | probe-rs run'
	"""
	if command.help is not None:
		return command.help
	return colorize_summary(task_summary(command.body, frozenset()), color)


def placeholder_warnings(commands: Mapping[str, Command]) -> tuple[str, ...]:
	"""
	>>> from camas.v0.command import Command
	>>> placeholder_warnings({"flash": Command("probe-rs run --chip {CHIP} {paths}")})
	("command 'flash' runs {CHIP} literally: a Command has no matrix axes",)
	>>> placeholder_warnings({"flash": Command("probe-rs run", env={"PROBE": "{PROBE}"})})
	("command 'flash' runs {PROBE} literally: a Command has no matrix axes",)
	>>> placeholder_warnings({"flash": Command("sh -c 'cd ${HOME}'", cwd="boards/{CHIP}")})
	("command 'flash' runs {CHIP} literally: a Command has no matrix axes",)
	>>> placeholder_warnings({"flash": Command("probe-rs run", env={"PROBE": "${PROBE}"})})
	("command 'flash' runs {PROBE} literally: a Command has no matrix axes",)
	"""
	return tuple(
		f"command {name!r} runs {token} literally: a Command has no matrix axes"
		for name, command in sorted(commands.items())
		for token in sorted(literal_placeholders(command))
	)


def literal_placeholders(command: Command) -> set[str]:
	body: Final = scoped_body(command)
	unexpanded: Final = (
		*(value for info in flatten_leaves(body) for value in info.task.env.values()),
		*(str(info.task.cwd) for info in flatten_leaves(body) if info.task.cwd is not None),
	)
	return {
		*SHELL_UNEXPANDED_PLACEHOLDER.findall(task_summary(body, frozenset())),
		*(token for text in unexpanded for token in PLACEHOLDER.findall(text)),
	} - {PATHS_TOKEN}
