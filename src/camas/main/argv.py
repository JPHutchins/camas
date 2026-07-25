# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""argv pre-processing: ``--`` passthrough splitting and matrix axis overrides."""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import PurePath
from typing import TYPE_CHECKING, Final, NamedTuple

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..core.matrix import resolve_cmd
from ..v0.task import Parallel, Sequential, Task, TaskNode

if TYPE_CHECKING:
	from collections.abc import Sequence


class SplitArgv(NamedTuple):
	"""argv partitioned at the first ``--`` separator.

	>>> SplitArgv(("mytask",), ("-v",))
	SplitArgv(head=('mytask',), passthrough=('-v',))
	"""

	head: tuple[str, ...]
	"""Args before ``--`` (or the whole argv if no ``--`` is present)."""
	passthrough: tuple[str, ...]
	"""Args after ``--``, to be appended to a leaf Task's command."""


def split_passthrough(argv: Sequence[str]) -> SplitArgv:
	"""Split ``argv`` at the first ``--``; args after it are pass-through to the Task.

	>>> split_passthrough(["mytask"])
	SplitArgv(head=('mytask',), passthrough=())
	>>> split_passthrough(["mytask", "--", "-v", "--tb=short"])
	SplitArgv(head=('mytask',), passthrough=('-v', '--tb=short'))
	"""
	argv_copy: Final = tuple(argv)
	try:
		idx: Final = argv_copy.index("--")
	except ValueError:
		return SplitArgv(argv_copy, ())
	return SplitArgv(argv_copy[:idx], argv_copy[idx + 1 :])


def apply_passthrough(task: TaskNode, args: tuple[str, ...]) -> Task:
	"""Append ``args`` to a leaf ``Task``'s command, preserving ``cmd`` shape: tuples
	stay tuples (appended), strings stay strings (args shell-joined) so dry-run/tree
	output keeps the user's quoting. Errors on Sequential/Parallel.

	Raises:
		ValueError: if ``task`` is a ``Sequential`` or ``Parallel``.

	>>> apply_passthrough(Task("pytest"), ("-v",))
	Task(cmd='pytest -v', name=None, env={}, cwd=None)
	>>> apply_passthrough(Task(("pytest",), name="t"), ("-v", "-k", "x"))
	Task(cmd=('pytest', '-v', '-k', 'x'), name='t', env={}, cwd=None)
	>>> apply_passthrough(Task("pytest"), ("-k", "a b"))
	Task(cmd="pytest -k 'a b'", name=None, env={}, cwd=None)
	"""
	match task:
		case Task(
			cmd=cmd,
			name=name,
			env=env,
			cwd=cwd,
			help=help,
			mutates=mutates,
			paths=paths,
			when=when,
			agent_format=agent_format,
		):
			return Task(
				cmd=f"{cmd} {shlex.join(args)}" if isinstance(cmd, str) else cmd + args,
				name=name,
				env=env,
				cwd=cwd,
				help=help,
				mutates=mutates,
				paths=paths,
				when=when,
				agent_format=agent_format,
			)
		case Sequential() | Parallel():
			raise ValueError(
				f"pass-through args (--) only apply to Task, got {type(task).__name__}"
			)
		case _:
			assert_never(task)


def parse_matrix_kv(raw: str) -> tuple[str, tuple[str, ...]]:
	"""Parse ``KEY=VAL[,VAL...]`` into ``(key, values)``.

	Raises:
		ValueError: on a missing or empty key, or an empty value list.

	>>> parse_matrix_kv("PY=3.13")
	('PY', ('3.13',))
	>>> parse_matrix_kv("PY=3.13,3.14")
	('PY', ('3.13', '3.14'))
	"""
	if "=" not in raw:
		raise ValueError(f"--matrix expects KEY=VAL[,VAL...], got {raw!r}")
	key, _, rest = raw.partition("=")
	if not key:
		raise ValueError(f"--matrix expects KEY=VAL[,VAL...], got {raw!r}")
	values = parse_axis_values(rest)
	if not values:
		raise ValueError(f"--matrix {key!r}: at least one value required")
	return key, values


def parse_axis_values(raw: str) -> tuple[str, ...]:
	"""Comma-separated values into a tuple, trimming whitespace and dropping empties.

	>>> parse_axis_values("3.13, 3.14")
	('3.13', '3.14')
	>>> parse_axis_values("")
	()
	"""
	return tuple(s for v in raw.split(",") if (s := v.strip()))


NAME_LIKE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*$")
"""A bare task name, hyphens and dotted namespace segments allowed — distinct from a camas
expression, which carries parens, quotes, or braces. A ``-`` is taken as part of the name (the
convention alias), never as subtraction."""

_LAUNCHERS: Final = frozenset({"run", "exec", "tool", "uvx", "-m", "-c", "--"})
"""Tokens a camas invocation may sit behind: ``uv run camas``, ``uvx camas``, ``nix develop -c
camas``, ``python -m camas``. Anything else with ``camas`` as a mere argument (``echo camas``) is
not an invocation."""


def _invoked_at(tokens: tuple[str, ...], index: int) -> bool:
	"""Whether the token at ``index`` is camas being *invoked* rather than merely named: it is the
	command itself, or sits behind a launcher — skipping that launcher's own flags, so
	``uv run --frozen camas lint`` still counts.

	>>> _invoked_at(("camas", "lint"), 0), _invoked_at(("uv", "run", "--frozen", "camas"), 3)
	(True, True)
	>>> _invoked_at(("echo", "camas"), 1)
	False
	"""
	for token in reversed(tokens[:index]):
		if token in _LAUNCHERS:
			return True
		if not token.startswith("-"):
			return False
	return index == 0


def camas_task_arg(cmd: str | tuple[str, ...]) -> str | None:
	"""The task name a leaf command re-enters camas with — ``camas lint``, ``uv run camas lint``,
	``python -m camas lint`` — or ``None`` when the command does not invoke camas with a bare task
	name. A flag, an inline expression, and the ``mcp`` subcommand are not task lookups.

	A heuristic, deliberately narrow: it exists so ``--check`` can warn about a fan-out that
	dispatches a name nothing resolves ("passes locally, 404s in CI"), and a false negative
	(``uv run tasks.py lint``) only means no warning.

	>>> camas_task_arg("camas lint")
	'lint'
	>>> camas_task_arg("uv run camas libs.build")
	'libs.build'
	>>> camas_task_arg("nix develop -c camas test-all")
	'test-all'
	>>> camas_task_arg("uvx camas==0.1.28 lint") is None   # a pinned uvx spec is not the camas binary
	True
	>>> camas_task_arg("uvx camas lint"), camas_task_arg("uv run --frozen camas lint")
	('lint', 'lint')
	>>> camas_task_arg("camas --list") is None
	True
	>>> camas_task_arg("camas mcp fix") is None
	True
	>>> camas_task_arg("echo camas lint") is None
	True
	>>> camas_task_arg(("camas", "check"))
	'check'
	"""
	tokens: Final = resolve_cmd(cmd)
	for i, token in enumerate(tokens):
		if PurePath(token).stem != "camas" or not _invoked_at(tokens, i):
			continue
		arg = tokens[i + 1] if i + 1 < len(tokens) else ""
		return arg if arg != "mcp" and NAME_LIKE.match(arg) else None
	return None
