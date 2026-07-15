# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Task-tree AST: ``Task`` leaves composed by ``Sequential`` and ``Parallel`` groups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeAlias, cast

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping


PathScope: TypeAlias = "Callable[[tuple[str, ...]], tuple[str, ...]]"
"""Maps the changed paths to the args injected at ``{paths}``: called with ``()`` for a
full run (return the default target), with the changed set otherwise (``()`` → skip)."""


WhenPredicate: TypeAlias = "Callable[[tuple[str, ...]], bool]"
"""A run-if-changed predicate for :attr:`Task.when` / :attr:`Group.when`: receives the
changed set and is never called for a full run (``changed == ()``)."""


def _prefix(value: str | Path) -> str:
	"""A ``when`` prefix in its stored form: a ``Path`` as POSIX, a string unchanged.

	>>> _prefix(Path("code-gen")), _prefix("src")
	('code-gen', 'src')
	"""
	return value.as_posix() if isinstance(value, Path) else value


def _coerce_when(
	when: str | Path | tuple[str | Path, ...] | WhenPredicate | None,
) -> str | tuple[str, ...] | WhenPredicate | None:
	"""Coerce a ``when`` argument to its stored form: a ``Path`` (or a ``Path`` inside a tuple)
	becomes its POSIX prefix; a string, tuple of strings, callable, or ``None`` is unchanged.

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
	"""A ``PathScope`` that filters the changed files by suffix on a scoped run and returns
	``default`` on a full run, so a ``{paths}`` command never loses its arguments to an empty
	change set.

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


OutputKind: TypeAlias = Literal["sarif", "rdjson", "lsp", "junit", "tap", "raw"]
"""The standard a leaf's command emits its diagnostics in — the agent-facing format camas
tags and passes through verbatim, never parsing. ``raw`` (the default) is plain text."""


@dataclass(frozen=True)
class AgentFormat:
	"""A leaf's agent-only structured-output variant: ``args`` (a producing flag the user
	supplies — camas never infers it) appended to the command, and the ``kind`` of diagnostics
	it makes the tool emit. Applied only when an agent runs; a human run leaves the command as-is.

	``args`` containing the literal ``{report}`` switches the leaf to path mode: the gate
	substitutes it with an allocated file path and, after the leaf runs, reads that file for the
	payload instead of stdout — for a tool (``pytest --junitxml``, ``pytest-json-report``) that
	writes its diagnostics to a file rather than printing them.

	``limit`` bounds a structured (non-``raw``) payload — and any path-mode report file,
	``raw`` included — in characters. A payload over ``limit`` is neither dumped nor tailed —
	a truncated structured document is invalid — but replaced with a pointer to the full
	file/log instead; stdout ``raw`` is exempt, since the gate line-tails it. It must be a
	positive int (a bool or a value ``<= 0`` raises ``ValueError``).

	>>> AgentFormat("--output-format sarif", "sarif")
	AgentFormat(args='--output-format sarif', kind='sarif', limit=8000)
	>>> AgentFormat("--junitxml {report}", "junit").args
	'--junitxml {report}'
	>>> AgentFormat("--out", "sarif", 0)
	Traceback (most recent call last):
	ValueError: AgentFormat limit must be a positive int, got 0
	"""

	args: str
	kind: OutputKind
	limit: int = 8_000

	def __post_init__(self) -> None:
		if isinstance(self.limit, bool) or self.limit <= 0:
			raise ValueError(f"AgentFormat limit must be a positive int, got {self.limit!r}")


_EMPTY_ENV: Mapping[str, str] = MappingProxyType({})
"""Read-only sentinel used as the default for ``Task.env``: shared across
instances (NamedTuple stores defaults on the class), but immutable so a
caller can't accidentally mutate other Tasks via ``task.env``."""


@dataclass(frozen=True, slots=True, init=False, repr=False)
class Task:
	"""A leaf task that executes a shell command.

	``env`` is a ``Mapping`` (read-only contract). The default is a shared
	``MappingProxyType({})``; user-provided dicts are stored as-is.

	``cwd`` is stored as ``Path | None``; the constructor also accepts a bare
	``str`` and coerces it (``"src-tauri"`` ⇒ ``Path("src-tauri")``).

	``help`` is an optional one-line description shown in ``--list`` output and
	``camas <task> --help`` instead of the bare command.

	``mutates`` marks a leaf that writes the workspace (a formatter or auto-fixer).
	The ``--under`` budget scheduler runs such leaves sequentially, before the
	read-only group, so they never race a checker over the same files.

	``paths`` is the scope for a ``{paths}`` command (:mod:`camas.core.scope`): a
	directory-prefix string (``"."``) or a ``(changed) -> tuple[str, ...]`` callable that maps the
	changed files into the command. A ``Sequential``/``Parallel`` may set ``paths`` to supply the
	default to descendants that set none. A command without ``{paths}`` can't be narrowed, so its
	``paths`` is a no-op and the command always runs unless a ``when`` predicate (below) prunes it.

	``when`` is a run-if-changed predicate (:mod:`camas.core.scope`) for a leaf whose command
	can't take ``{paths}`` (``cargo build``, ``nix flake check``): a directory-prefix string or
	``Path`` (coerced to its POSIX prefix), a tuple of those (OR'd), or a ``(changed) -> bool``
	callable. On a scoped run a leaf whose ``when`` doesn't match the changed set is pruned; a
	full run never consults ``when``. A leaf that also sets ``paths``/``{paths}`` is gated by
	``when`` first, then narrowed as usual. When ``when`` is unset, a leaf with a ``cwd`` gates on
	its ``cwd`` directory (:func:`camas.core.matrix.expand_matrix`) — a monorepo file-tree default;
	set ``when="."`` to opt back into always-run.

	``agent_format`` is the agent-only structured-output variant (:class:`AgentFormat`): the gate
	appends its ``args`` and tags the diagnostics ``kind``; a human run leaves the command as-is.
	A bare ``(args, kind)`` tuple is coerced to an :class:`AgentFormat`.

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
	name: str | None
	env: Mapping[str, str]
	cwd: Path | None
	help: str | None
	mutates: bool
	paths: str | PathScope | None
	when: str | tuple[str, ...] | WhenPredicate | None
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


@dataclass(frozen=True, slots=True, init=False, repr=False)
class Group:
	"""Shared base for ``Sequential`` and ``Parallel``: variadic ``*tasks`` (with
	``str`` → ``Task`` coercion), identical kwargs, hashable. Use
	``isinstance(x, Group)`` to test for "either kind of grouping node";
	pattern-match on the concrete subclass to discriminate.

	``paths`` is the default path-scope for descendant ``{paths}`` leaves that set none
	(see :mod:`camas.core.scope`); ``env``/``cwd`` likewise propagate into leaves.

	``when`` is the default run-if-changed predicate (see :class:`Task`) for descendant
	leaves that set none — baked into leaves by :func:`camas.core.matrix.expand_matrix`,
	the same way ``paths``/``env``/``cwd`` propagate.

	>>> isinstance(Sequential("a"), Group) and isinstance(Parallel("a"), Group)
	True
	>>> hash(Sequential("a")) == hash(Sequential("a"))
	True
	>>> Parallel(Task("ruff {paths}"), paths=".").paths
	'.'
	"""

	tasks: tuple[TaskNode, ...]
	name: str | None
	matrix: dict[str, tuple[str, ...]] | None
	env: dict[str, str]
	cwd: Path | None
	help: str | None
	paths: str | PathScope | None
	when: str | tuple[str, ...] | WhenPredicate | None

	def __init__(
		self,
		*tasks: TaskNode | str,
		name: str | None = None,
		matrix: dict[str, tuple[str, ...]] | None = None,
		env: dict[str, str] | None = None,
		cwd: str | Path | None = None,
		help: str | None = None,
		paths: str | PathScope | None = None,
		when: str | Path | tuple[str | Path, ...] | WhenPredicate | None = None,
	) -> None:
		put = object.__setattr__
		put(self, "tasks", tuple(Task(cmd=t) if isinstance(t, str) else t for t in tasks))
		put(self, "name", name)
		put(self, "matrix", matrix)
		put(self, "env", env if env is not None else {})
		put(self, "cwd", Path(cwd) if isinstance(cwd, str) else cwd)
		put(self, "help", help)
		put(self, "paths", paths)
		put(self, "when", _coerce_when(when))

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
				self.paths,
				self.when,
			)
		)

	def __repr__(self) -> str:
		parts = (
			f"tasks={self.tasks!r}",
			f"name={self.name!r}",
			f"matrix={self.matrix!r}",
			f"env={self.env!r}",
			f"cwd={self.cwd!r}",
			*([f"help={self.help!r}"] if self.help is not None else []),
			*([f"paths={self.paths!r}"] if self.paths is not None else []),
			*([f"when={self.when!r}"] if self.when is not None else []),
		)
		return f"{type(self).__name__}({', '.join(parts)})"


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


class ProjectRef(NamedTuple):
	"""A :func:`Project` reference before the loader resolves it; never reaches the engine."""

	path: str


def Project(path: str) -> TaskNode:  # noqa: N802  # constructor-style factory, like Task/Parallel
	"""Another ``tasks.py`` as a task node — a private, immutable child project, referenced by
	``path`` relative to the importing file (a directory resolves its ``tasks.py``). Runs what a
	bare ``camas`` runs in that directory; bound at module scope, its tasks mount under the
	binding's name for dotted dispatch (``libs``, ``libs.search.lint``).
	"""
	return cast("TaskNode", ProjectRef(path))
