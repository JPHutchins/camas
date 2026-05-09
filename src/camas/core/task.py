# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple, TypeAlias


class VarBinding(NamedTuple):
	"""A single matrix variable bound to a concrete value.

	>>> VarBinding("PY", "3.14")
	VarBinding(name='PY', value='3.14')
	"""

	name: str
	value: str


MatrixBinding: TypeAlias = tuple[VarBinding, ...]


_EMPTY_ENV: Mapping[str, str] = MappingProxyType({})
"""Read-only sentinel used as the default for ``Task.env``: shared across
instances (NamedTuple stores defaults on the class), but immutable so a
caller can't accidentally mutate other Tasks via ``task.env``."""


class Task(NamedTuple):
	"""A leaf task that executes a shell command.

	``env`` is a ``Mapping`` (read-only contract). The default is a shared
	``MappingProxyType({})``; user-provided dicts are stored as-is.

	``help`` is an optional one-line description shown in ``--list`` output and
	``camas <task> --help`` instead of the bare command.

	>>> Task("echo hi")
	Task(cmd='echo hi', name=None, env={}, cwd=None)
	>>> Task(("ruff", "check", "."), name="lint")
	Task(cmd=('ruff', 'check', '.'), name='lint', env={}, cwd=None)
	>>> Task("cargo test", cwd=Path("src-tauri")).cwd == Path("src-tauri")
	True
	>>> Task("ruff check .", help="Lint all sources").help
	'Lint all sources'
	>>> hash(Task("a")) == hash(Task("a"))
	True
	>>> {Task("a", env={"K": "v"}), Task("a", env={"K": "v"})} == {Task("a", env={"K": "v"})}
	True
	"""

	cmd: str | tuple[str, ...]
	name: str | None = None
	env: Mapping[str, str] = _EMPTY_ENV
	cwd: Path | None = None
	help: str | None = None

	def __hash__(self) -> int:
		return hash((self.cmd, self.name, tuple(sorted(self.env.items())), self.cwd, self.help))

	def __repr__(self) -> str:
		base = (
			f"Task(cmd={self.cmd!r}, name={self.name!r}, env={dict(self.env)!r}, cwd={self.cwd!r}"
		)
		return f"{base}, help={self.help!r})" if self.help is not None else f"{base})"


@dataclass(frozen=True, slots=True, init=False, repr=False)
class Group:
	"""Shared base for ``Sequential`` and ``Parallel``: variadic ``*tasks`` (with
	``str`` → ``Task`` coercion), identical kwargs, hashable. Use
	``isinstance(x, Group)`` to test for "either kind of grouping node";
	pattern-match on the concrete subclass to discriminate.

	>>> isinstance(Sequential("a"), Group) and isinstance(Parallel("a"), Group)
	True
	>>> hash(Sequential("a")) == hash(Sequential("a"))
	True
	"""

	tasks: tuple[TaskNode, ...]
	name: str | None
	matrix: dict[str, tuple[str, ...]] | None
	env: dict[str, str]
	cwd: Path | None
	help: str | None

	def __init__(
		self,
		*tasks: TaskNode | str,
		name: str | None = None,
		matrix: dict[str, tuple[str, ...]] | None = None,
		env: dict[str, str] | None = None,
		cwd: Path | None = None,
		help: str | None = None,
	) -> None:
		put = object.__setattr__
		put(self, "tasks", tuple(Task(cmd=t) if isinstance(t, str) else t for t in tasks))
		put(self, "name", name)
		put(self, "matrix", matrix)
		put(self, "env", env if env is not None else {})
		put(self, "cwd", cwd)
		put(self, "help", help)

	def __hash__(self) -> int:
		matrix_key = None if self.matrix is None else tuple(sorted(self.matrix.items()))
		return hash(
			(
				self.tasks,
				self.name,
				matrix_key,
				tuple(sorted(self.env.items())),
				self.cwd,
				self.help,
			)
		)

	def __repr__(self) -> str:
		base = (
			f"{type(self).__name__}(tasks={self.tasks!r}, name={self.name!r}, "
			f"matrix={self.matrix!r}, env={self.env!r}, cwd={self.cwd!r}"
		)
		return f"{base}, help={self.help!r})" if self.help is not None else f"{base})"


class Sequential(Group):  # pyrefly: ignore[bad-class-definition]
	"""A group of tasks that run one after another, short-circuiting on failure.

	>>> Sequential("build", "test", name="ci").tasks
	(Task(cmd='build', name=None, env={}, cwd=None), Task(cmd='test', name=None, env={}, cwd=None))
	"""

	__slots__ = ()


class Parallel(Group):  # pyrefly: ignore[bad-class-definition]
	"""A group of tasks that run concurrently.

	>>> Parallel("lint", "typecheck").tasks
	(Task(cmd='lint', name=None, env={}, cwd=None), Task(cmd='typecheck', name=None, env={}, cwd=None))
	"""

	__slots__ = ()


TaskNode: TypeAlias = Task | Sequential | Parallel


def task_label(task: Task) -> str:
	"""Return a task's identifying label: the explicit `name` or the full command string.

	This is a data accessor with no concept of display width — callers that render
	into a column-constrained terminal are responsible for truncation.

	>>> task_label(Task("echo hi", name="greet"))
	'greet'
	>>> task_label(Task("echo hi"))
	'echo hi'
	>>> task_label(Task(("python", "-c", "pass")))
	'python -c pass'
	"""
	if task.name is not None:
		return task.name
	return task.cmd if isinstance(task.cmd, str) else " ".join(task.cmd)
