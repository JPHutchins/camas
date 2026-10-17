# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The task tree."""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import (
	TYPE_CHECKING,
	Final,
	Literal,
	NamedTuple,
	TypeAlias,
	TypeVar,
	cast,
	get_args,
	overload,
)

from .ref import Ref

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping
	from typing import Any

	from typing_extensions import Self


PathScope: TypeAlias = "Callable[[tuple[str, ...]], tuple[str, ...]]"
"""The changed paths to the arguments injected at ``{paths}``; ``()`` in is a full run."""


WhenPredicate: TypeAlias = "Callable[[tuple[str, ...]], bool]"
"""Whether a leaf runs for the changed paths; never called on a full run."""


Nodes: TypeAlias = "TaskNode | str | Sequence[TaskNode]"
"""One node or a sequence of nodes; a ``str`` is a ``Task``."""


def _prefix(value: str | Path) -> str:
	"""
	>>> _prefix(Path("code-gen")), _prefix("src")
	('code-gen', 'src')
	"""
	return value.as_posix() if isinstance(value, Path) else value


def _coerce_when(
	when: str | Path | tuple[str | Path, ...] | WhenPredicate | None,
) -> str | tuple[str, ...] | WhenPredicate | None:
	"""
	>>> _coerce_when(Path("code-gen"))
	'code-gen'
	>>> _coerce_when((Path("src"), "include"))
	('src', 'include')
	>>> _coerce_when("src"), _coerce_when(None)
	('src', None)
	"""
	match when:
		case str() | Path():
			return _prefix(when)
		case tuple():
			return tuple(
				_prefix(w)  # ty: ignore[invalid-argument-type]
				for w in when
			)
		case _:
			return when


def by_suffix(suffixes: tuple[str, ...], default: tuple[str, ...] = (".",)) -> PathScope:
	"""A ``PathScope`` keeping the changed files that end in one of ``suffixes``.

	>>> f = by_suffix((".c", ".h"), default=("src", "include"))
	>>> f(())
	('src', 'include')
	>>> f(("a.c", "b.py", "c.h"))
	('a.c', 'c.h')
	>>> f(("b.py",))
	()
	"""
	return lambda changed: (
		default if not changed else tuple(c for c in changed if c.endswith(suffixes))
	)


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
	"""
	>>> rx = _glob_to_regex("homeassistant/**/*.py")
	>>> bool(rx.fullmatch("homeassistant/core.py")), bool(rx.fullmatch("homeassistant/a/b.py"))
	(True, True)
	>>> bool(rx.fullmatch("tests/core.py")), bool(rx.fullmatch("homeassistant/core.pyi"))
	(False, False)
	>>> bool(_glob_to_regex("src/*.py").fullmatch("src/a/b.py"))
	False
	>>> bool(_glob_to_regex("logs/**").fullmatch("logs/a/b.txt"))
	True
	>>> tuple(bool(_glob_to_regex("v?.txt").fullmatch(c)) for c in ("v1.txt", "v10.txt"))
	(True, False)
	>>> _glob_to_regex("")
	Traceback (most recent call last):
	ValueError: by_glob pattern must not be empty

	Raises:
		ValueError: when ``pattern`` is empty.
	"""
	if not pattern:
		raise ValueError("by_glob pattern must not be empty")
	out: list[str] = []
	i, n = 0, len(pattern)
	while i < n:
		if pattern.startswith("**/", i):
			out.append("(?:.*/)?")
			i += 3
		elif pattern.startswith("**", i):
			out.append(".*")
			i += 2
		elif pattern[i] == "*":
			out.append("[^/]*")
			i += 1
		elif pattern[i] == "?":
			out.append("[^/]")
			i += 1
		else:
			out.append(re.escape(pattern[i]))
			i += 1
	return re.compile("".join(out))


def by_glob(patterns: tuple[str, ...], default: tuple[str, ...] = (".",)) -> PathScope:
	"""A ``PathScope`` keeping the changed files that match one of the shell-glob ``patterns``.

	>>> f = by_glob(("homeassistant/**/*.py", "homeassistant/**/*.pyi"), default=("homeassistant",))
	>>> f(())
	('homeassistant',)
	>>> f(("homeassistant/core.py", "homeassistant/x.pyi", "tests/t.py", "README.md"))
	('homeassistant/core.py', 'homeassistant/x.pyi')
	>>> f(("homeassistant/components/hue/light.py",))
	('homeassistant/components/hue/light.py',)
	>>> f(("README.md",))
	()
	"""
	compiled = tuple(_glob_to_regex(p) for p in patterns)
	return lambda changed: (
		default
		if not changed
		else tuple(c for c in changed if any(rx.fullmatch(c) for rx in compiled))
	)


def _check_variants(
	variants: tuple[dict[str, str], ...] | None, matrix: dict[str, tuple[str, ...]] | None
) -> None:
	"""
	Raises:
		ValueError: when a variant binds no keys, variants bind different keys, or a key is
			declared in both ``matrix`` and ``variants``.

	>>> _check_variants(({"target": "a"}, {"target": "b"}), {"PROFILE": ("debug",)})
	>>> _check_variants(({"target": "a"}, {}), None)
	Traceback (most recent call last):
	    ...
	ValueError: variants: variant 1 binds no keys
	>>> _check_variants(({"target": "a"}, {"target": "b", "cpu": "m4"}), None)
	Traceback (most recent call last):
	    ...
	ValueError: variants: variant 1 binds 'cpu', 'target' but variant 0 binds 'target'; every variant must bind the same keys
	>>> _check_variants(({"PY": "3.13"},), {"PY": ("3.13",)})
	Traceback (most recent call last):
	    ...
	ValueError: variants: 'PY' is declared in both matrix= and variants=
	"""
	if not variants:
		return
	first: Final = tuple(sorted(variants[0]))
	for i, variant in enumerate(variants):
		keys = tuple(sorted(variant))
		if not keys:
			raise ValueError(f"variants: variant {i} binds no keys")
		if keys != first:
			raise ValueError(
				f"variants: variant {i} binds {', '.join(repr(k) for k in keys)} but variant 0 "
				f"binds {', '.join(repr(k) for k in first)}; every variant must bind the same keys"
			)
	for key in first:
		if matrix is not None and key in matrix:
			raise ValueError(f"variants: {key!r} is declared in both matrix= and variants=")


OutputKind: TypeAlias = Literal["sarif", "rdjson", "lsp", "junit", "tap", "raw"]
"""The format a leaf's diagnostics are in."""


class AgentFormat(NamedTuple):
	"""A leaf's agent-only structured-output variant.

	>>> AgentFormat("--output-format sarif", "sarif")
	AgentFormat(args='--output-format sarif', kind='sarif', limit=8000)
	>>> AgentFormat("--junitxml {report}", "junit").args
	'--junitxml {report}'
	"""

	args: str
	"""Appended to the command on agent runs; a literal ``{report}`` names the file the gate reads."""
	kind: OutputKind
	limit: int = 8_000
	"""Characters of structured payload the gate inlines before pointing at the full output."""


_EMPTY_ENV: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True, slots=True, init=False, repr=False)
class Task:
	"""A leaf that runs one command.

	>>> Task("echo hi")
	Task(cmd='echo hi', name=None, env={}, cwd=None)
	>>> Task(("ruff", "check", "."), name="lint")
	Task(cmd=('ruff', 'check', '.'), name='lint', env={}, cwd=None)
	>>> Task("cargo test", cwd=Path("src-tauri")).cwd == Path("src-tauri")
	True
	>>> Task("cargo test", cwd="src-tauri").cwd == Path("src-tauri")
	True
	>>> Task("ruff check .", help="Lint all sources").help
	'Lint all sources'
	>>> Task("ruff format .", mutates=True)
	Task(cmd='ruff format .', name=None, env={}, cwd=None, mutates=True)
	>>> Task("ruff format {paths}", mutates=True, paths=".")
	Task(cmd='ruff format {paths}', name=None, env={}, cwd=None, mutates=True, paths='.')
	>>> Task("cargo build", when=("src", "include"))
	Task(cmd='cargo build', name=None, env={}, cwd=None, when=('src', 'include'))
	>>> Task("cargo build", when=Path("src")).when
	'src'
	>>> Task("cargo build", when=(Path("src"), "include")).when
	('src', 'include')
	>>> Task("ruff check .", agent_format=AgentFormat("--output-format sarif", "sarif"))
	Task(cmd='ruff check .', name=None, env={}, cwd=None, agent_format=AgentFormat(args='--output-format sarif', kind='sarif', limit=8000))
	>>> Task("ruff check .", agent_format=("--output-format sarif", "sarif")).agent_format
	AgentFormat(args='--output-format sarif', kind='sarif', limit=8000)
	>>> hash(Task("a")) == hash(Task("a"))
	True
	>>> {Task("a", env={"K": "v"}), Task("a", env={"K": "v"})} == {Task("a", env={"K": "v"})}
	True
	"""

	cmd: str | tuple[str, ...]
	"""Argv tokens, or a string split by POSIX shell rules on every OS."""
	name: str | None
	env: Mapping[str, str]
	cwd: Path | None
	help: str | None
	"""Shown by ``--list`` and ``--help`` in place of the command."""
	mutates: bool
	"""The leaf writes the workspace; ``--under`` orders it ahead of read-only leaves."""
	paths: str | PathScope | None
	"""The scope a ``{paths}`` command narrows to on a scoped run."""
	when: str | tuple[str, ...] | WhenPredicate | None
	"""Paths whose change keeps the leaf in a scoped run; unset, a leaf with a ``cwd`` gates on it."""
	agent_format: AgentFormat | None

	def __init__(
		self,
		cmd: str | tuple[str, ...],
		name: str | None = None,
		env: Mapping[str, str] = _EMPTY_ENV,
		cwd: str | Path | None = None,
		help: str | None = None,
		mutates: bool = False,
		paths: str | PathScope | None = None,
		when: str | Path | tuple[str | Path, ...] | WhenPredicate | None = None,
		agent_format: AgentFormat | tuple[str, OutputKind] | None = None,
	) -> None:
		put = object.__setattr__
		put(self, "cmd", cmd)
		put(self, "name", name)
		put(self, "env", env)
		put(self, "cwd", Path(cwd) if isinstance(cwd, str) else cwd)
		put(self, "help", help)
		put(self, "mutates", mutates)
		put(self, "paths", paths)
		put(self, "when", _coerce_when(when))
		put(
			self,
			"agent_format",
			agent_format
			if agent_format is None or isinstance(agent_format, AgentFormat)
			else AgentFormat(*agent_format),
		)

	def __hash__(self) -> int:
		return hash(
			(
				self.cmd,
				self.name,
				tuple(sorted(self.env.items())),
				self.cwd,
				self.help,
				self.mutates,
				self.paths,
				self.when,
				self.agent_format,
			)
		)

	def __repr__(self) -> str:
		parts = (
			f"cmd={self.cmd!r}",
			f"name={self.name!r}",
			f"env={dict(self.env)!r}",
			f"cwd={self.cwd!r}",
			*([f"help={self.help!r}"] if self.help is not None else []),
			*(["mutates=True"] if self.mutates else []),
			*([f"paths={self.paths!r}"] if self.paths is not None else []),
			*([f"when={self.when!r}"] if self.when is not None else []),
			*([f"agent_format={self.agent_format!r}"] if self.agent_format is not None else []),
		)
		return f"Task({', '.join(parts)})"

	def __or__(self, other: Nodes) -> Parallel:
		"""Compose in parallel.

		>>> (Task("format") | Task("lint")).tasks == (Task("format"), Task("lint"))
		True
		>>> Task("a") | Parallel("b", "c") == Parallel("a", Parallel("b", "c"))
		True
		"""
		return _composed(Parallel, self, other)

	def __add__(self, other: Nodes) -> Sequential:
		"""Compose in sequence; ``+`` binds tighter than ``|``.

		>>> (Task("build") + Task("test")).tasks == (Task("build"), Task("test"))
		True
		"""
		return _composed(Sequential, self, other)

	def __gt__(self, other: Nodes) -> Pipe:
		"""Pipe this leaf's stdout into ``other``; Python chains ``>``, so parenthesize longer pipes.

		>>> (Task("gen") > "upper").tasks == (Task("gen"), Task("upper"))
		True
		>>> ((Task("a") > "b") > "c").tasks == (Task("a"), Task("b"), Task("c"))
		True
		>>> bool(Task("a") > "b")
		True
		"""
		return _pipe_of(self, other)


@dataclass(frozen=True, slots=True, init=False, repr=False)
class Group:
	"""The shared base of ``Sequential``, ``Parallel``, and ``Pipe``.

	>>> isinstance(Sequential("a"), Group) and isinstance(Parallel("a"), Group)
	True
	>>> hash(Sequential("a")) == hash(Sequential("a"))
	True
	>>> Parallel(Task("ruff {paths}"), paths=".").paths
	'.'
	>>> Parallel(Task("cargo build -b {backend}"), variants=({"backend": "thumbv7"},)).variants
	({'backend': 'thumbv7'},)
	"""

	tasks: tuple[TaskNode, ...]
	name: str | None
	matrix: dict[str, tuple[str, ...]] | None
	"""A cross product: one copy of the group per combination of axis values."""
	variants: tuple[dict[str, str], ...] | None
	"""Coupled bundles of values, run as listed rather than crossed."""
	env: dict[str, str]
	cwd: Path | None
	help: str | None
	paths: str | PathScope | None
	"""The default ``{paths}`` scope for descendant leaves that set none."""
	when: str | tuple[str, ...] | WhenPredicate | None
	"""The default ``when`` for descendant leaves that set none."""

	def __init__(
		self,
		*tasks: TaskNode | str,
		name: str | None = None,
		matrix: dict[str, tuple[str, ...]] | None = None,
		variants: tuple[dict[str, str], ...] | None = None,
		env: dict[str, str] | None = None,
		cwd: str | Path | None = None,
		help: str | None = None,
		paths: str | PathScope | None = None,
		when: str | Path | tuple[str | Path, ...] | WhenPredicate | None = None,
	) -> None:
		_check_variants(variants, matrix)
		put = object.__setattr__
		put(self, "tasks", tuple(Task(cmd=t) if isinstance(t, str) else t for t in tasks))
		put(self, "name", name)
		put(self, "matrix", matrix or None)
		put(self, "variants", variants)
		put(self, "env", env if env is not None else {})
		put(self, "cwd", Path(cwd) if isinstance(cwd, str) else cwd)
		put(self, "help", help)
		put(self, "paths", paths)
		put(self, "when", _coerce_when(when))

	def __hash__(self) -> int:
		matrix_key = None if self.matrix is None else tuple(sorted(self.matrix.items()))
		variants_key = (
			None
			if self.variants is None
			else tuple(tuple(sorted(v.items())) for v in self.variants)
		)
		return hash(
			(
				self.tasks,
				self.name,
				matrix_key,
				variants_key,
				tuple(sorted(self.env.items())),
				self.cwd,
				self.help,
				self.paths,
				self.when,
			)
		)

	def __repr__(self) -> str:
		return f"{type(self).__name__}({', '.join(_group_repr_parts(self))})"

	def extend(self, nodes: Nodes) -> Self:
		"""This group with ``nodes`` appended inside its scope.

		>>> Parallel("a").extend(Parallel("b", "c")) == Parallel("a", Parallel("b", "c"))
		True
		>>> Parallel("a").extend(Parallel("b", "c").tasks) == Parallel("a", "b", "c")
		True
		>>> Parallel("a", cwd="w").extend("b") == Parallel("a", "b", cwd="w")
		True
		>>> Parallel("a").extend(("python", "-c", "print(1)"))
		Traceback (most recent call last):
		    ...
		TypeError: cannot compose 'python' from a sequence: a str there is ambiguous with a tuple command — pass Task((...)) for one command or Task(...) for each
		"""
		return rebuilt(self, *_nodes(self.tasks), *_children(nodes))

	def remove(self, nodes: Nodes) -> Self:
		"""This group without its direct children equal to one of ``nodes``.

		>>> Parallel("a", "b", "c").remove(Task("b")) == Parallel("a", "c")
		True
		>>> Sequential("a", "b", "a").remove((Task("a"),)) == Sequential("b")
		True
		>>> Parallel("a", Parallel("b")).remove("b")
		Traceback (most recent call last):
		    ...
		ValueError: cannot remove Task(cmd='b', name=None, env={}, cwd=None): not a direct child

		Raises:
			ValueError: when one of ``nodes`` is not a direct child.
		"""
		targets = _children(nodes)
		absent = next((target for target in targets if target not in self.tasks), None)
		if absent is not None:
			raise ValueError(f"cannot remove {absent!r}: not a direct child")
		return rebuilt(self, *(child for child in _nodes(self.tasks) if child not in targets))

	def __sub__(self, other: Nodes) -> Self:
		return self.remove(other)


class Sequential(Group):  # pyrefly: ignore[bad-class-definition]
	"""A group of tasks that run one after another, short-circuiting on failure.

	>>> Sequential("build", "test", name="ci").tasks
	(Task(cmd='build', name=None, env={}, cwd=None), Task(cmd='test', name=None, env={}, cwd=None))
	"""

	__slots__ = ()

	def __or__(self, other: Nodes) -> Parallel:
		"""Compose in parallel.

		>>> seq = Sequential("build", "test")
		>>> (seq | "lint").tasks == (seq, Task("lint"))
		True
		"""
		return _composed(Parallel, self, other)

	def __add__(self, other: Nodes) -> Sequential:
		"""Compose in sequence, extending a plain ``Sequential`` rather than nesting it.

		>>> (Sequential("build") + "test").tasks == (Task("build"), Task("test"))
		True
		>>> Sequential("a") + Sequential("b", "c") == Sequential("a", Sequential("b", "c"))
		True
		>>> Sequential("a") + Sequential("b", "c").tasks == Sequential("a", "b", "c")
		True
		>>> Sequential("a", cwd="w") + "b" == Sequential(Sequential("a", cwd="w"), "b")
		True
		"""
		return _composed(Sequential, self, other)

	def __gt__(self, other: Nodes) -> Pipe:
		"""Refuse: a group cannot be a pipe stage.

		>>> Sequential("a") > "b"
		Traceback (most recent call last):
		    ...
		ValueError: Pipe stages must be Tasks — a nested group would mean several commands sharing one stream
		"""
		return _pipe_of(self, other)


class Parallel(Group):  # pyrefly: ignore[bad-class-definition]
	"""A group of tasks that run concurrently.

	>>> Parallel("lint", "typecheck").tasks
	(Task(cmd='lint', name=None, env={}, cwd=None), Task(cmd='typecheck', name=None, env={}, cwd=None))
	"""

	__slots__ = ()

	def __or__(self, other: Nodes) -> Parallel:
		"""Compose in parallel, extending a plain ``Parallel`` rather than nesting it.

		>>> (Parallel("format") | "lint").tasks == (Task("format"), Task("lint"))
		True
		>>> Parallel("a") | Parallel("b", "c") == Parallel("a", Parallel("b", "c"))
		True
		>>> Parallel("a") | Parallel("b", "c").tasks == Parallel("a", "b", "c")
		True
		>>> Parallel("a", cwd="w") | "b" == Parallel(Parallel("a", cwd="w"), "b")
		True
		"""
		return _composed(Parallel, self, other)

	def __add__(self, other: Nodes) -> Sequential:
		"""Compose in sequence; ``+`` binds tighter than ``|``.

		>>> check = Parallel("format")
		>>> (check + "integration").tasks == (check, Task("integration"))
		True
		"""
		return _composed(Sequential, self, other)

	def __gt__(self, other: Nodes) -> Pipe:
		"""Refuse: a group cannot be a pipe stage.

		>>> Parallel("a") > "b"
		Traceback (most recent call last):
		    ...
		ValueError: Pipe stages must be Tasks — a nested group would mean several commands sharing one stream
		"""
		return _pipe_of(self, other)


STAGE_MESSAGE: Final = (
	"Pipe stages must be Tasks — a nested group would mean several commands sharing one stream"
)


@dataclass(frozen=True, slots=True, init=False, repr=False)
class Pipe(Group):
	"""Leaf stages piped stdout to stdin, each its own argv, with no shell.

	>>> Pipe("cargo clippy --message-format=json", "clippy-sarif").agent_only
	False
	>>> Pipe("a", "b", agent_only=True).tasks[0].cmd
	'a'
	>>> Pipe()
	Traceback (most recent call last):
	    ...
	ValueError: Pipe needs at least one stage
	>>> Pipe("a", "b")
	Pipe(tasks=(Task(cmd='a', name=None, env={}, cwd=None), Task(cmd='b', name=None, env={}, cwd=None)), name=None, matrix=None, env={}, cwd=None)
	>>> Pipe("a", Sequential("b"))
	Traceback (most recent call last):
	    ...
	ValueError: Pipe stages must be Tasks — a nested group would mean several commands sharing one stream
	"""

	agent_only: bool
	"""Pipe on agent runs only; a human run keeps the first stage alone."""

	def __init__(
		self,
		*tasks: TaskNode | str,
		agent_only: bool = False,
		name: str | None = None,
		matrix: dict[str, tuple[str, ...]] | None = None,
		variants: tuple[dict[str, str], ...] | None = None,
		env: dict[str, str] | None = None,
		cwd: str | Path | None = None,
		help: str | None = None,
		paths: str | PathScope | None = None,
		when: str | Path | tuple[str | Path, ...] | WhenPredicate | None = None,
	) -> None:
		"""
		Raises:
			ValueError: when there is no stage, a stage is not a ``Task``, or a stage repeats.
		"""
		# Explicit base call — zero-arg super() breaks under mypyc's compiled subclasses.
		Group.__init__(
			self,
			*tasks,
			name=name,
			matrix=matrix,
			variants=variants,
			env=env,
			cwd=cwd,
			help=help,
			paths=paths,
			when=when,
		)
		object.__setattr__(self, "agent_only", agent_only)
		if not self.tasks:
			raise ValueError("Pipe needs at least one stage")
		if any(not isinstance(t, (Task, Ref)) for t in self.tasks):
			raise ValueError(STAGE_MESSAGE)
		if len({id(t) for t in self.tasks}) != len(self.tasks):
			raise ValueError(
				"Pipe stages must be distinct Tasks — a repeated stage would collide in the "
				"engine's per-leaf slots"
			)

	def __hash__(self) -> int:
		return hash((Group.__hash__(self), self.agent_only))

	def __repr__(self) -> str:
		parts = (
			*_group_repr_parts(self),
			*(["agent_only=True"] if self.agent_only else []),
		)
		return f"Pipe({', '.join(parts)})"

	def __or__(self, other: Nodes) -> Parallel:
		"""Compose in parallel."""
		return _composed(Parallel, self, other)

	def __add__(self, other: Nodes) -> Sequential:
		"""Compose in sequence."""
		return _composed(Sequential, self, other)

	def __gt__(self, other: Nodes) -> Pipe:
		"""Append stages; a right-side ``Pipe`` with fields contributes only its ``.tasks``.

		>>> (Pipe("a") > "b").tasks == (Task("a"), Task("b"))
		True
		>>> (Pipe("a", env={"K": "v"}) > Pipe("b", "c")) == Pipe("a", "b", "c", env={"K": "v"})
		True
		>>> Pipe("a") > Pipe("b", agent_only=True)
		Traceback (most recent call last):
		    ...
		ValueError: cannot splice Pipe Pipe(tasks=(Task(cmd='b', name=None, env={}, cwd=None),), name=None, matrix=None, env={}, cwd=None, agent_only=True) into a pipe: its fields, agent_only, or subclass type would be dropped — pipe its .tasks instead: x > y.tasks
		"""
		return _pipe_of(self, other)


TaskNode: TypeAlias = Task | Sequential | Parallel | Pipe


UNRESOLVED_REF_MESSAGE: Final = "an unresolved Ref: task references must be resolved before a run"


def ref_problem(group: Group) -> str | None:
	if not any(isinstance(t, Ref) for t in group.tasks):
		return None
	return STAGE_MESSAGE if isinstance(group, Pipe) else UNRESOLVED_REF_MESSAGE


def reject_refs(group: Group) -> None:
	if (problem := ref_problem(group)) is not None:
		raise ValueError(problem)


def _group_repr_parts(group: Group) -> tuple[str, ...]:
	return (
		f"tasks={group.tasks!r}",
		f"name={group.name!r}",
		f"matrix={group.matrix!r}",
		*([f"variants={group.variants!r}"] if group.variants is not None else []),
		f"env={group.env!r}",
		f"cwd={group.cwd!r}",
		*([f"help={group.help!r}"] if group.help is not None else []),
		*([f"paths={group.paths!r}"] if group.paths is not None else []),
		*([f"when={group.when!r}"] if group.when is not None else []),
	)


GROUP_FIELDS: Final = ("name", "matrix", "variants", "env", "cwd", "help", "paths", "when")


G = TypeVar("G", bound=Group)


def rebuilt(group: G, *children: TaskNode, **changes: object) -> G:
	"""
	>>> rebuilt(Parallel(Task("a"), matrix={"x": ("1",)}), Task("b"), Task("c"))
	Parallel(tasks=(Task(cmd='b', name=None, env={}, cwd=None), Task(cmd='c', name=None, env={}, cwd=None)), name=None, matrix={'x': ('1',)}, env={}, cwd=None)
	>>> rebuilt(Sequential("a"), Task("b"), paths=".")
	Sequential(tasks=(Task(cmd='b', name=None, env={}, cwd=None),), name=None, matrix=None, env={}, cwd=None, paths='.')
	>>> rebuilt(Sequential("a", name="n"), Task("b"), name=None)
	Sequential(tasks=(Task(cmd='b', name=None, env={}, cwd=None),), name=None, matrix=None, env={}, cwd=None)

	Raises:
		TypeError: a ``changes`` name outside :data:`GROUP_FIELDS`.
	"""
	allowed = (*GROUP_FIELDS, *(("agent_only",) if isinstance(group, Pipe) else ()))
	unknown = [name for name in changes if name not in allowed]
	if unknown:
		raise TypeError(f"rebuilt() got an unexpected keyword argument {unknown[0]!r}")
	kwargs = {field: changes.get(field, getattr(group, field)) for field in GROUP_FIELDS}
	if isinstance(group, Pipe):
		kwargs["agent_only"] = changes.get("agent_only", group.agent_only)
	return type(group)(*children, **cast("dict[str, Any]", kwargs))


@dataclass(frozen=True, slots=True)
class ProjectRef:
	path: str

	def __or__(self, other: Nodes) -> Parallel:
		return _composed(Parallel, cast("TaskNode", self), other)

	def __add__(self, other: Nodes) -> Sequential:
		return _composed(Sequential, cast("TaskNode", self), other)

	def __gt__(self, other: Nodes) -> Pipe:
		return _pipe_of(cast("TaskNode", self), other)


def Project(path: str) -> TaskNode:  # noqa: N802  # constructor-style factory, like Task/Parallel
	"""Another ``tasks.py``, by path relative to this one, as a task node."""
	return cast("TaskNode", ProjectRef(path))


_NODE_KINDS: Final = (*get_args(TaskNode), ProjectRef)


def _node(child: object) -> TaskNode:
	if isinstance(child, str):
		return Task(cmd=child)
	if isinstance(child, _NODE_KINDS):
		return cast("TaskNode", child)
	raise TypeError(
		f"cannot compose {child!r} of type {type(child).__name__}: a child is a task node or str"
	)


def _nodes(children: tuple[TaskNode, ...]) -> tuple[TaskNode, ...]:
	return tuple(_node(child) for child in children)


def fieldless(group: Group) -> bool:
	return all(
		getattr(group, field) is None or (field == "env" and not getattr(group, field))
		for field in GROUP_FIELDS
	)


def _member(node: object) -> TaskNode:
	if isinstance(node, str):
		raise TypeError(
			f"cannot compose {node!r} from a sequence: a str there is ambiguous with a tuple "
			"command — pass Task((...)) for one command or Task(...) for each"
		)
	return _node(node)


def _children(nodes: Nodes) -> tuple[TaskNode, ...]:
	if isinstance(nodes, Sequence) and not isinstance(
		nodes, (str, bytes, bytearray, memoryview, range, Ref)
	):
		return tuple(_member(node) for node in nodes)
	return (_node(nodes),)


def _plain(group: Group, kind: type[Group]) -> bool:
	return (
		type(group) is kind
		and fieldless(group)
		and not (isinstance(group, Pipe) and group.agent_only)
	)


@overload
def _composed(kind: type[Parallel], left: TaskNode, right: Nodes) -> Parallel: ...
@overload
def _composed(kind: type[Sequential], left: TaskNode, right: Nodes) -> Sequential: ...
def _composed(
	kind: type[Parallel | Sequential], left: TaskNode, right: Nodes
) -> Parallel | Sequential:
	if isinstance(left, (Parallel, Sequential)) and _plain(left, kind):
		return left.extend(right)
	return kind(left).extend(right)


def _pipe_of(left: TaskNode, right: Nodes) -> Pipe:
	head: Final = left if isinstance(left, Pipe) else Pipe(left)
	if not isinstance(right, Pipe):
		return head.extend(right)
	if not _plain(right, Pipe):
		raise ValueError(
			f"cannot splice {type(right).__name__} {right!r} into a pipe: its fields, agent_only, "
			"or subclass type would be dropped — pipe its .tasks instead: x > y.tasks"
		)
	return head.extend(right.tasks)


def GIT_PORCELAIN() -> Task:  # noqa: N802  # constructor-style factory, like Task/Parallel
	"""A fresh copy of the default ``Clean`` drift check."""
	return Task((sys.executable, "-m", "camas._git_porcelain"), cwd=".")


_GIT_PORCELAIN: Final = GIT_PORCELAIN()


def _as_task(value: object, message: str) -> Task:
	node: Final = _node(value) if isinstance(value, str) else value
	if not isinstance(node, Task):
		raise ValueError(message)
	return node


def Clean(  # noqa: N802  # constructor-style factory, like Task/Parallel
	mutator: Task | str,
	*,
	check: Task | str = _GIT_PORCELAIN,
	before: bool = True,
	name: str | None = None,
) -> Sequential:
	"""A drift gate: fail when running ``mutator`` changes the working tree.

	>>> [t.name for t in Clean(Task("make gen", mutates=True)).tasks]
	['make gen-before', None, 'make gen-after']
	>>> Clean(Task("make gen", mutates=True), before=False).tasks[0].cmd
	'make gen'

	Raises:
		ValueError: on a ``mutator`` without ``mutates=True`` or a ``check`` with ``{paths}``.
	"""
	mutator = _as_task(
		mutator,
		"Clean's mutator must be a Task marked mutates=True — the drift gate exists to "
		"catch its writes",
	)
	check = _as_task(check, "Clean's check must be a Task whose exit 0 means the tree is clean")
	check_cmd: Final = check.cmd if isinstance(check.cmd, str) else " ".join(check.cmd)
	if "{paths}" in check_cmd:
		raise ValueError(
			"Clean's check must not carry {paths} — a scoped run would narrow the check "
			"while the mutator still writes"
		)
	if not mutator.mutates:
		raise ValueError(
			"Clean's mutator must be marked mutates=True — the drift gate exists to catch "
			"its writes"
		)

	def check_leaf(leaf_name: str) -> Task:
		return replace(check, name=leaf_name, when=".", paths=None)

	from camas.core.task import task_label

	mutator_label: Final = task_label(mutator)
	gate_name: Final = (
		name
		if name is not None
		else (f"{mutator_label}-{check.name}" if check.name is not None else mutator_label)
	)
	after: Final = check_leaf(f"{gate_name}-after")
	if not before:
		return Sequential(mutator, after, name=name)
	return Sequential(check_leaf(f"{gate_name}-before"), mutator, after, name=name)
