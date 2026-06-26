# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""argv pre-processing: ``--`` passthrough splitting and matrix axis overrides."""

from __future__ import annotations

import shlex
import sys
from typing import TYPE_CHECKING, Final, NamedTuple

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

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
			output_kind=output_kind,
		):
			return Task(
				cmd=f"{cmd} {shlex.join(args)}" if isinstance(cmd, str) else cmd + args,
				name=name,
				env=env,
				cwd=cwd,
				help=help,
				mutates=mutates,
				paths=paths,
				output_kind=output_kind,
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
