# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

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
from ..v0.task import Parallel, Pipe, Sequential, Task, TaskNode

if TYPE_CHECKING:
	from collections.abc import Sequence


class SplitArgv(NamedTuple):
	"""
	>>> SplitArgv(("mytask",), ("-v",))
	SplitArgv(head=('mytask',), passthrough=('-v',))
	"""

	head: tuple[str, ...]
	passthrough: tuple[str, ...]


def split_passthrough(argv: Sequence[str]) -> SplitArgv:
	"""
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
	"""
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
		case Sequential() | Parallel() | Pipe():
			raise ValueError(
				f"pass-through args (--) only apply to Task, got {type(task).__name__}"
			)
		case _:
			assert_never(task)


def parse_matrix_kv(raw: str) -> tuple[str, tuple[str, ...]]:
	"""
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
	"""
	>>> parse_axis_values("3.13, 3.14")
	('3.13', '3.14')
	>>> parse_axis_values("")
	()
	"""
	return tuple(s for v in raw.split(",") if (s := v.strip()))


NAME_LIKE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*$")

_LAUNCHERS: Final = frozenset({"run", "exec", "tool", "uvx", "-m", "-c", "--"})


def _invoked_at(tokens: tuple[str, ...], index: int) -> bool:
	"""
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
	"""
	>>> camas_task_arg("camas lint")
	'lint'
	>>> camas_task_arg("uv run camas libs.build")
	'libs.build'
	>>> camas_task_arg("nix develop -c camas test-all")
	'test-all'
	>>> camas_task_arg("uvx camas==0.1.28 lint") is None
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
	>>> camas_task_arg('camas "lint') is None
	True
	"""
	try:
		tokens: Final = resolve_cmd(cmd)
	except ValueError:
		return None
	for i, token in enumerate(tokens):
		if PurePath(token).stem != "camas" or not _invoked_at(tokens, i):
			continue
		arg = tokens[i + 1] if i + 1 < len(tokens) else ""
		return arg if arg != "mcp" and NAME_LIKE.match(arg) else None
	return None
