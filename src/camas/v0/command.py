# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""A side-effectful action or interactive program."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from .task import Group, Parallel, Pipe, ProjectRef, Sequential, Task

if TYPE_CHECKING:
	from collections.abc import Mapping


CommandBody: TypeAlias = Task | Pipe | Sequential

NEVER_COMPOSES: Final = "a Command never composes into a task tree"

_EMPTY_ENV: Mapping[str, str] = MappingProxyType({})


def body_problem(node: object) -> str | None:
	"""
	>>> body_problem(Sequential("build", Pipe("gen", "upper")))
	>>> body_problem(Sequential("a", Parallel("b", "c")))
	'a Command body cannot hold a Parallel: its children would share one terminal'
	>>> body_problem(Sequential("t {X}", matrix={"X": ("1",)}))
	'a Command body cannot fan out: matrix= and variants= expand to a Parallel'
	>>> from camas.v0.task import Project
	>>> body_problem(Sequential(Project("libs")))
	'a Command body cannot hold a Project'
	"""
	match node:
		case Task():
			return None
		case Parallel():
			return "a Command body cannot hold a Parallel: its children would share one terminal"
		case Group(matrix=None, variants=None, tasks=children):
			return next((p for p in map(body_problem, children) if p is not None), None)
		case Group():
			return "a Command body cannot fan out: matrix= and variants= expand to a Parallel"
		case ProjectRef():
			return "a Command body cannot hold a Project"
		case _:
			return (
				f"a Command body holds Task, Pipe, and Sequential nodes, not {type(node).__name__}"
			)


@dataclass(frozen=True, slots=True, init=False, repr=False)
class Command:
	"""A side-effectful action or interactive program, run attached to the terminal.

	>>> Command("cargo run --release")
	Command(body=Task(cmd='cargo run --release', name=None, env={}, cwd=None), name=None, env={}, cwd=None)
	>>> Command(("cargo", "run"), cwd="fw").cwd == Path("fw")
	True
	>>> Command(Task("build") + Task("flash")).body == Sequential("build", "flash")
	True
	>>> Command(Parallel("a", "b"))  # type: ignore[arg-type]
	Traceback (most recent call last):
	    ...
	ValueError: a Command body cannot hold a Parallel: its children would share one terminal
	>>> hash(Command("a", env={"K": "v"})) == hash(Command("a", env={"K": "v"}))
	True

	Raises:
		ValueError: when the body holds a ``Parallel``, a fan-out, or a ``Project``.
	"""

	body: CommandBody
	name: str | None
	env: Mapping[str, str]
	cwd: Path | None
	help: str | None

	def __init__(
		self,
		body: str | tuple[str, ...] | CommandBody,
		*,
		name: str | None = None,
		env: Mapping[str, str] = _EMPTY_ENV,
		cwd: str | Path | None = None,
		help: str | None = None,
	) -> None:
		node: Final = Task(body) if isinstance(body, (str, tuple)) else body
		problem: Final = body_problem(node)
		if problem is not None:
			raise ValueError(problem)
		put = object.__setattr__
		put(self, "body", node)
		put(self, "name", name)
		put(self, "env", env)
		put(self, "cwd", Path(cwd) if isinstance(cwd, str) else cwd)
		put(self, "help", help)

	def __hash__(self) -> int:
		return hash((self.body, self.name, tuple(sorted(self.env.items())), self.cwd, self.help))

	def __repr__(self) -> str:
		parts = (
			f"body={self.body!r}",
			f"name={self.name!r}",
			f"env={dict(self.env)!r}",
			f"cwd={self.cwd!r}",
			*([f"help={self.help!r}"] if self.help is not None else []),
		)
		return f"Command({', '.join(parts)})"
