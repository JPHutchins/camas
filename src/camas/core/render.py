# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""ANSI primitives and tree-layout rendering shared by the display Effects."""

from __future__ import annotations

import os
import re
import sys
from typing import TYPE_CHECKING, Final, NamedTuple, TypeAlias

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..v0.task import Parallel, Sequential, Task, TaskNode
from .leaf_state import ChainLink, LeafInfo
from .matrix import expand_matrix
from .task import task_label

if TYPE_CHECKING:
	from collections.abc import Iterator, Mapping
	from pathlib import Path

BOLD: Final = "\033[1m"
CYAN: Final = "\033[36m"
GREY: Final = "\033[90m"
GREEN: Final = "\033[32m"
RED: Final = "\033[31m"
BLUE: Final = "\033[34m"
YELLOW: Final = "\033[33m"
VIOLET: Final = "\033[95m"
RESET: Final = "\033[0m"

ANSI_ESCAPE: Final = re.compile(
	r"\x1b(?:"
	r"\[[0-?]*[ -/]*[@-~]"  # CSI sequences (colors, cursor movement, etc.)
	r"|\][^\x07]*\x07"  # OSC terminated by BEL (hyperlinks, window title)
	r"|\][^\x1b]*\x1b\\"  # OSC terminated by ST
	r"|[()*+][\x20-\x7e]"  # ISO-2022 character-set designation (e.g. ESC(B = G0 ASCII)
	r"|[@-Z\\-_]"  # two-character Fe sequences (after OSC: ] is in range)
	r")"
	r"|[\x00-\x08\x0b-\x1f\x7f]"  # ASCII control chars except \t (\x09) and \n (\x0a)
)


def strip_ansi(text: str) -> str:
	r"""Remove ANSI escape sequences and ASCII control characters from a string.

	Tab (``\t``) and newline (``\n``) are preserved — they're load-bearing for
	formatted log output. Carriage return (``\r``), BEL, BS, and other
	control chars are stripped.

	>>> strip_ansi("\x1b[32mgreen\x1b[0m text")
	'green text'
	>>> strip_ansi("\x1b]8;;https://example.com\x07link\x1b]8;;\x07 text")
	'link text'
	>>> strip_ansi("no escapes")
	'no escapes'
	>>> strip_ansi("line one\nline two\tcol")
	'line one\nline two\tcol'
	>>> strip_ansi("\r\x1b[2Kprogress")
	'progress'
	>>> strip_ansi("\x1b[1mbold\x1b(B\x1b[m done")
	'bold done'
	"""
	return ANSI_ESCAPE.sub("", text)


def color_on() -> bool:
	"""True when stdout is a TTY and ``NO_COLOR`` is unset."""
	return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def colored(text: str, code: str, on: bool) -> str:
	"""Wrap ``text`` in ANSI ``code``...``RESET`` when ``on`` and text is non-empty."""
	return f"{code}{text}{RESET}" if on and text else text


class GroupHeader(NamedTuple):
	"""Display row for a Sequential or Parallel group header.

	>>> GroupHeader("ci", 0, ())
	GroupHeader(label='ci', depth=0, is_last_chain=())
	"""

	label: str
	depth: int
	is_last_chain: tuple[ChainLink, ...]


DisplayRow: TypeAlias = LeafInfo | GroupHeader


SEQ_SUFFIX: Final = " →"
PAR_SUFFIX: Final = " ∥"


def group_display_name(tasks: tuple[TaskNode, ...], separator: str) -> str:
	"""Derive a display label for a group by joining children's names.

	>>> group_display_name((Task("a"), Task("b")), " | ")
	'a | b'
	>>> group_display_name((Task("build"), Task("test")), " → ")
	'build → test'
	"""
	parts: list[str] = []
	for t in tasks:
		match t:
			case Task():
				parts.append(task_label(t))
			case Sequential(name=name) | Parallel(name=name):
				parts.append(
					name
					if name is not None
					else f"({group_display_name(t.tasks, ' | ' if isinstance(t, Parallel) else ' → ')})"
				)
			case _:
				assert_never(t)
	return separator.join(parts)


def render_tree_prefix(depth: int, is_last_chain: tuple[ChainLink, ...]) -> str:
	"""Reconstitute the ASCII tree prefix from structural position data.

	Children of a ``Sequential`` get ``├─`` / ``└─`` branches with ``│`` continuations —
	the sequence has an ordering and a terminator. Children of a ``Parallel`` get a
	plain ``┃`` column with no ``├``/``└`` distinction, since parallel siblings have
	no order.

	>>> render_tree_prefix(0, ())
	''
	>>> render_tree_prefix(1, (ChainLink(True, False),))
	'└─ '
	>>> render_tree_prefix(1, (ChainLink(False, False),))
	'├─ '
	>>> render_tree_prefix(1, (ChainLink(False, True),))
	'┃ '
	>>> render_tree_prefix(1, (ChainLink(True, True),))
	'┃ '
	>>> render_tree_prefix(2, (ChainLink(False, False), ChainLink(True, True)))
	'│ ┃ '
	>>> render_tree_prefix(2, (ChainLink(True, False), ChainLink(False, False)))
	'  ├─ '
	"""
	if depth == 0:
		return ""
	parts: list[str] = []
	for link in is_last_chain[:-1]:
		if link.parent_is_parallel:
			parts.append("┃ ")
		else:
			parts.append("  " if link.is_last else "│ ")
	last: Final = is_last_chain[-1]
	if last.parent_is_parallel:
		parts.append("┃ ")
	else:
		parts.append("└─ " if last.is_last else "├─ ")
	return "".join(parts)


def iter_rows(
	node: TaskNode,
	depth: int = 0,
	is_last_chain: tuple[ChainLink, ...] = (),
) -> Iterator[DisplayRow]:
	"""Walk a task tree depth-first, yielding one DisplayRow per node (groups + leaves)."""
	match node:
		case Task():
			yield LeafInfo(node, depth, is_last_chain)
		case Sequential(tasks=children, name=name):
			seq_label = (
				f"{name}{SEQ_SUFFIX}" if name is not None else group_display_name(children, " → ")
			)
			yield GroupHeader(seq_label, depth, is_last_chain)
			seq_last = len(children) - 1
			for i, child in enumerate(children):
				link = ChainLink(is_last=i == seq_last, parent_is_parallel=False)
				yield from iter_rows(child, depth + 1, (*is_last_chain, link))
		case Parallel(tasks=children, name=name):
			par_label = (
				f"{name}{PAR_SUFFIX}" if name is not None else group_display_name(children, " | ")
			)
			yield GroupHeader(par_label, depth, is_last_chain)
			par_last = len(children) - 1
			for i, child in enumerate(children):
				link = ChainLink(is_last=i == par_last, parent_is_parallel=True)
				yield from iter_rows(child, depth + 1, (*is_last_chain, link))
		case _:
			assert_never(node)


def flatten_rows(task: TaskNode) -> tuple[DisplayRow, ...]:
	"""Flatten a task tree into display rows (GroupHeaders + LeafInfos) in DFS order.

	>>> rows = flatten_rows(Parallel(Task("a"), Task("b")))
	>>> len(rows)
	3
	>>> isinstance(rows[0], GroupHeader)
	True
	"""
	return tuple(iter_rows(task))


def env_diff(env: Mapping[str, str], ancestor_env: Mapping[str, str]) -> dict[str, str]:
	return {k: v for k, v in env.items() if ancestor_env.get(k) != v}


def walk_with_context(
	node: TaskNode,
	depth: int = 0,
	is_last_chain: tuple[ChainLink, ...] = (),
	ancestor_env: Mapping[str, str] = {},
	ancestor_cwd: Path | None = None,
) -> Iterator[tuple[DisplayRow, dict[str, str], Path | None]]:
	"""Walk the expanded tree yielding ``(row, env_introduced_here, cwd_introduced_here)``.

	Env entries and cwd are each reported only at the node that introduces or
	changes them, so they render exactly once in the tree.
	"""
	match node:
		case Task(env=env, cwd=cwd):
			yield (
				LeafInfo(node, depth, is_last_chain),
				env_diff(env, ancestor_env),
				cwd if cwd != ancestor_cwd else None,
			)
		case Sequential(tasks=children, name=name, env=env, cwd=cwd):
			here_env = env_diff(env, ancestor_env)
			here_cwd = cwd if cwd is not None and cwd != ancestor_cwd else None
			label = (
				f"{name}{SEQ_SUFFIX}" if name is not None else group_display_name(children, " → ")
			)
			yield GroupHeader(label, depth, is_last_chain), here_env, here_cwd
			last_i = len(children) - 1
			new_env = {**ancestor_env, **env}
			new_cwd = cwd if cwd is not None else ancestor_cwd
			for i, child in enumerate(children):
				link = ChainLink(is_last=i == last_i, parent_is_parallel=False)
				yield from walk_with_context(
					child, depth + 1, (*is_last_chain, link), new_env, new_cwd
				)
		case Parallel(tasks=children, name=name, env=env, cwd=cwd):
			here_env = env_diff(env, ancestor_env)
			here_cwd = cwd if cwd is not None and cwd != ancestor_cwd else None
			label = (
				f"{name}{PAR_SUFFIX}" if name is not None else group_display_name(children, " | ")
			)
			yield GroupHeader(label, depth, is_last_chain), here_env, here_cwd
			last_i = len(children) - 1
			new_env = {**ancestor_env, **env}
			new_cwd = cwd if cwd is not None else ancestor_cwd
			for i, child in enumerate(children):
				link = ChainLink(is_last=i == last_i, parent_is_parallel=True)
				yield from walk_with_context(
					child, depth + 1, (*is_last_chain, link), new_env, new_cwd
				)
		case _:
			assert_never(node)


def leaf_label(task: Task, show_cmd: bool, color: bool) -> str:
	label = task_label(task)
	base = colored(label, BOLD, color)
	if show_cmd and task.name is not None:
		cmd = task.cmd if isinstance(task.cmd, str) else " ".join(task.cmd)
		if cmd != task.name:
			base = f"{base}: {colored(cmd, CYAN, color)}"
	return base


def render_tree_lines(task: TaskNode, show_cmd: bool = False, color: bool = False) -> list[str]:
	"""Build the tree-display lines (group headers + leaves) for an expanded task tree.

	Pure: returns a list of strings without printing. ``print_tree`` (in
	``main.format``) is the printing wrapper.

	>>> render_tree_lines(Task("echo hi"))
	['echo hi']
	>>> render_tree_lines(Task("echo hi", name="greet"), show_cmd=True)
	['greet: echo hi']
	"""
	lines: list[str] = []
	for row, env_new, cwd_new in walk_with_context(expand_matrix(task)):
		prefix = render_tree_prefix(row.depth, row.is_last_chain)
		meta: list[str] = []
		if show_cmd and cwd_new is not None:
			meta.append(colored(f"(cwd: {cwd_new})", GREY, color))
		if show_cmd and env_new:
			meta.append(colored(" ".join(f"{k}={v}" for k, v in env_new.items()), GREY, color))
		meta_str = f"  {' '.join(meta)}" if meta else ""
		match row:
			case GroupHeader(label=label):
				lines.append(f"{colored(prefix, GREY, color)}{label}{meta_str}")
			case LeafInfo(task=leaf_task):
				lines.append(
					f"{colored(prefix, GREY, color)}"
					f"{leaf_label(leaf_task, show_cmd, color)}{meta_str}"
				)
			case _:
				assert_never(row)
	return lines
