# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Task-tree AST: ``Task`` leaves composed by ``Sequential`` and ``Parallel`` groups."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, TypeAlias, TypeVar, cast, get_args

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping
	from typing import Any


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


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
	"""Compile a shell glob over POSIX paths to a full-match regex: ``**`` spans any directory
	depth (``**/`` also matching zero directories), ``*`` and ``?`` stay within a single path
	segment, and every other character is literal.

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
		ValueError: when ``pattern`` is empty — it would compile to a regex matching only the
			empty string, silently dropping every changed file.
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
	r"""A ``PathScope`` that keeps the changed files matching any of ``patterns`` — shell globs
	over the repo-relative POSIX path, where ``**`` spans any directory depth and ``*``/``?`` stay
	within a segment — on a scoped run, and returns ``default`` on a full run so a ``{paths}``
	command never loses its arguments to an empty change set. Ports a pre-commit ``files:`` filter
	— a directory prefix plus a suffix, e.g. ``^(homeassistant|pylint)/.+\.(py|pyi)$`` — to a
	single scope, where ``by_suffix`` (suffix-only) can't express the prefix.

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
	"""Validate a ``variants`` declaration against its node's ``matrix``.

	Raises:
		ValueError: when a variant binds no keys, when the variants disagree on which keys they
			bind — a ragged bundle would leave a ``{placeholder}`` unsubstituted in some cells —
			or when a key is declared in both ``matrix`` and ``variants``.

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
"""The standard a leaf's command emits its diagnostics in — the agent-facing format camas
tags and passes through verbatim, never parsing. ``raw`` (the default) is plain text."""


class AgentFormat(NamedTuple):
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
	file/log instead; stdout ``raw`` is exempt, since the gate line-tails it.

	>>> AgentFormat("--output-format sarif", "sarif")
	AgentFormat(args='--output-format sarif', kind='sarif', limit=8000)
	>>> AgentFormat("--junitxml {report}", "junit").args
	'--junitxml {report}'
	"""

	args: str
	kind: OutputKind
	limit: int = 8_000


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
	The ``--under`` budget scheduler serializes mutating subtrees ahead of a
	``Parallel``'s pure read-only siblings, so a mutator never runs concurrently
	with a checker over the same files.

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

	def __or__(self, other: TaskNode | str) -> Parallel:
		"""``|`` composes nodes in parallel: a :class:`Parallel` of this leaf and ``other``
		(a right-side ``Parallel`` contributes its children and carries its fields).

		>>> (Task("format") | Task("lint")).tasks == (Task("format"), Task("lint"))
		True
		"""
		return _parallel_of(self, other)

	def __add__(self, other: TaskNode | str) -> Sequential:
		"""``+`` composes nodes in sequence: a :class:`Sequential` of this leaf then ``other``
		(a right-side ``Sequential`` contributes its children and carries its fields). ``+``
		binds tighter than ``|`` — parenthesize a mixed chain to control its shape.

		>>> (Task("build") + Task("test")).tasks == (Task("build"), Task("test"))
		True
		"""
		return _sequential_of(self, other)


@dataclass(frozen=True, slots=True, init=False, repr=False)
class Group:
	"""Shared base for ``Sequential``, ``Parallel``, and ``Pipe``: variadic ``*tasks`` (with
	``str`` → ``Task`` coercion), identical kwargs, hashable. Use
	``isinstance(x, Group)`` to test for "some kind of grouping node";
	pattern-match on the concrete subclass to discriminate.

	``paths`` is the default path-scope for descendant ``{paths}`` leaves that set none
	(see :mod:`camas.core.scope`); ``env``/``cwd`` likewise propagate into leaves.

	``when`` is the default run-if-changed predicate (see :class:`Task`) for descendant
	leaves that set none — baked into leaves by :func:`camas.core.matrix.expand_matrix`,
	the same way ``paths``/``env``/``cwd`` propagate.

	``matrix`` is a cross product: each axis varies independently, and the node expands to
	one cell per combination. ``variants`` is the *coupled* form — a tuple of bundles, each
	binding the same keys to values that only make sense together (a backend with its
	platform and rustup target). The two compose: the run-set is the ``matrix`` cross
	product × the ``variants``. Overriding a ``matrix`` axis from the CLI *replaces* its
	values; overriding a ``variants`` key *filters* to the variants that bind it, since a
	replacement would fabricate a bundle the project never declared. Every variant must
	bind the same keys, and a key belongs to ``matrix`` or ``variants``, not both.
	``variants=()`` declares zero variants — a run rejects it rather than silently doing
	nothing, exactly as ``matrix={"axis": ()}`` does.

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
	variants: tuple[dict[str, str], ...] | None
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
		parts = (
			f"tasks={self.tasks!r}",
			f"name={self.name!r}",
			f"matrix={self.matrix!r}",
			*([f"variants={self.variants!r}"] if self.variants is not None else []),
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

	def __or__(self, other: TaskNode | str) -> Parallel:
		"""``|`` composes in parallel: a :class:`Parallel` of this whole group and ``other``
		(a right-side ``Parallel`` contributes its children and carries its fields).

		>>> seq = Sequential("build", "test")
		>>> (seq | "lint").tasks == (seq, Task("lint"))
		True
		"""
		return _parallel_of(self, other)

	def __add__(self, other: TaskNode | str) -> Sequential:
		"""``+`` appends ``other`` to this sequence (a right-side ``Sequential`` contributes
		its children). Fields and type carry from the operand that brings them: the left's,
		except when the left carries only constructor defaults and the right carries fields or
		a non-plain (subclass) type — then the right's. A subclass operand must accept the
		Group constructor kwargs for its type to carry. ``+`` binds tighter than ``|`` —
		parenthesize a mixed chain to control its shape.

		>>> (Sequential("build") + "test").tasks == (Task("build"), Task("test"))
		True
		"""
		return _sequential_of(self, other)


class Parallel(Group):  # pyrefly: ignore[bad-class-definition]
	"""A group of tasks that run concurrently.

	>>> Parallel("lint", "typecheck").tasks
	(Task(cmd='lint', name=None, env={}, cwd=None), Task(cmd='typecheck', name=None, env={}, cwd=None))
	"""

	__slots__ = ()

	def __or__(self, other: TaskNode | str) -> Parallel:
		"""``|`` appends ``other`` to this group (a right-side ``Parallel`` contributes its
		children). Fields and type carry from the operand that brings them: the left's, except
		when the left carries only constructor defaults and the right carries fields or a
		non-plain (subclass) type — then the right's. A subclass operand must accept the Group
		constructor kwargs for its type to carry.

		>>> (Parallel("format") | "lint").tasks == (Task("format"), Task("lint"))
		True
		"""
		return _parallel_of(self, other)

	def __add__(self, other: TaskNode | str) -> Sequential:
		"""``+`` builds a :class:`Sequential` that runs this whole group first, then ``other``
		(a right-side ``Sequential`` contributes its children and carries its fields). ``+``
		binds tighter than ``|`` — parenthesize a mixed chain to control its shape.

		>>> check = Parallel("format")
		>>> (check + "integration").tasks == (check, Task("integration"))
		True
		"""
		return _sequential_of(self, other)


@dataclass(frozen=True, slots=True, init=False, repr=False)
class Pipe(Group):
	"""A group of leaf stages piped fd-to-fd, each stage its own argv vector — no shell.

	Every stage runs concurrently and to completion (a dying stage feeds EOF downstream);
	the pipeline fails ``pipefail``-style: any stage's non-zero exit fails the run, so a
	producer failing under a succeeding last stage is still a failure. Stages must be
	leaves — a nested group would mean several commands sharing one stream.

	``agent_only=True`` runs the full pipeline on an agent run and collapses to the first
	stage alone on a human run — the plain human-readable output for a human, the piped
	structured output for the agent, selected by the same runner identity that appends
	``agent_format`` args (:func:`camas.core.gate.strip_agent_only_pipes`).

	>>> Pipe("cargo clippy --message-format=json", "clippy-sarif").agent_only
	False
	>>> Pipe("a", "b", agent_only=True).tasks[0].cmd
	'a'
	>>> Pipe("a", "b")
	Pipe(tasks=(Task(cmd='a', name=None, env={}, cwd=None), Task(cmd='b', name=None, env={}, cwd=None)), name=None, matrix=None, env={}, cwd=None)
	>>> Pipe("a", Sequential("b"))
	Traceback (most recent call last):
	    ...
	ValueError: Pipe stages must be Tasks — a nested group would mean several commands sharing one stream
	"""

	agent_only: bool

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
		if any(not isinstance(t, Task) for t in self.tasks):
			raise ValueError(
				"Pipe stages must be Tasks — a nested group would mean several commands "
				"sharing one stream"
			)

	def __repr__(self) -> str:
		parts = (
			f"tasks={self.tasks!r}",
			f"name={self.name!r}",
			f"matrix={self.matrix!r}",
			*([f"variants={self.variants!r}"] if self.variants is not None else []),
			f"env={self.env!r}",
			f"cwd={self.cwd!r}",
			*([f"help={self.help!r}"] if self.help is not None else []),
			*([f"paths={self.paths!r}"] if self.paths is not None else []),
			*([f"when={self.when!r}"] if self.when is not None else []),
			*(["agent_only=True"] if self.agent_only else []),
		)
		return f"Pipe({', '.join(parts)})"

	def __or__(self, other: TaskNode | str) -> Parallel:
		"""``|`` composes in parallel: a :class:`Parallel` of this whole pipe and ``other``."""
		return _parallel_of(self, other)

	def __add__(self, other: TaskNode | str) -> Sequential:
		"""``+`` builds a :class:`Sequential` that runs this whole pipe first, then ``other``."""
		return _sequential_of(self, other)


TaskNode: TypeAlias = Task | Sequential | Parallel | Pipe


GROUP_FIELDS: Final = ("name", "matrix", "variants", "env", "cwd", "help", "paths", "when")
"""Every ``Group.__init__`` keyword other than the variadic tasks, in signature order — the drift
test fails when a new field lands on Group but not here."""


G = TypeVar("G", bound=Group)


def rebuilt(group: G, *children: TaskNode, **changes: object) -> G:
	"""``group`` rebuilt around ``children``: fields named in ``changes`` take the new value, every
	other field is carried verbatim — so a Group field added later is carried by construction at
	every rebuild site, instead of being listed (and missable) at each. A ``changes`` name outside
	:data:`GROUP_FIELDS` is a ``TypeError``, as the spelled-out constructors raised it.

	>>> rebuilt(Parallel(Task("a"), matrix={"x": ("1",)}), Task("b"), Task("c"))
	Parallel(tasks=(Task(cmd='b', name=None, env={}, cwd=None), Task(cmd='c', name=None, env={}, cwd=None)), name=None, matrix={'x': ('1',)}, env={}, cwd=None)
	>>> rebuilt(Sequential("a"), Task("b"), paths=".")
	Sequential(tasks=(Task(cmd='b', name=None, env={}, cwd=None),), name=None, matrix=None, env={}, cwd=None, paths='.')
	>>> rebuilt(Sequential("a", name="n"), Task("b"), name=None)
	Sequential(tasks=(Task(cmd='b', name=None, env={}, cwd=None),), name=None, matrix=None, env={}, cwd=None)

	Raises:
		TypeError: a ``changes`` name outside :data:`GROUP_FIELDS`.
	"""
	unknown = [name for name in changes if name not in GROUP_FIELDS]
	if unknown:
		raise TypeError(f"rebuilt() got an unexpected keyword argument {unknown[0]!r}")
	return type(group)(
		*children,
		**cast(
			"dict[str, Any]",
			{field: changes.get(field, getattr(group, field)) for field in GROUP_FIELDS},
		),
	)


@dataclass(frozen=True, slots=True)
class ProjectRef:
	"""A :func:`Project` reference before the loader resolves it; never reaches the engine."""

	path: str

	def __or__(self, other: TaskNode | str) -> Parallel:
		"""Composes like a task node — see :meth:`Task.__or__`."""
		return _parallel_of(cast("TaskNode", self), other)

	def __add__(self, other: TaskNode | str) -> Sequential:
		"""Composes like a task node — see :meth:`Task.__add__`."""
		return _sequential_of(cast("TaskNode", self), other)


def Project(path: str) -> TaskNode:  # noqa: N802  # constructor-style factory, like Task/Parallel
	"""Another ``tasks.py`` as a task node — a private, immutable child project, referenced by
	``path`` relative to the importing file (a directory resolves its ``tasks.py``). Runs what a
	bare ``camas`` runs in that directory; bound at module scope, its tasks mount under the
	binding's name for dotted dispatch (``libs``, ``libs.search.lint``).
	"""
	return cast("TaskNode", ProjectRef(path))


_NODE_KINDS: Final = (*get_args(TaskNode), ProjectRef)


def _node(child: object) -> TaskNode:
	"""A composition operand as a task node: a ``str`` as its :class:`Task`, a task node (a
	:class:`Project` reference included — it resolves in the loader) as itself, anything else
	a :class:`TypeError` naming the operand. ``object``-typed so the guard is the check, not
	the annotation.

	Raises:
		TypeError: when ``child`` is neither a node nor a string.
	"""
	if isinstance(child, str):
		return Task(cmd=child)
	if isinstance(child, _NODE_KINDS):
		return cast("TaskNode", child)
	raise TypeError(
		f"cannot compose {child!r} of type {type(child).__name__}: a child is a task node or str"
	)


def _nodes(children: tuple[TaskNode, ...]) -> tuple[TaskNode, ...]:
	"""Every child back through the operand guard — a group operand's children came through the
	lenient constructor, so composition re-checks them.
	"""
	return tuple(_node(child) for child in children)


def fieldless(group: Group) -> bool:
	"""Whether every :data:`GROUP_FIELDS` value is the constructor default: ``None``, or an
	empty mapping for ``env``.
	"""
	return all(
		getattr(group, field) is None or (field == "env" and not getattr(group, field))
		for field in GROUP_FIELDS
	)


def _parallel_of(left: TaskNode | str, right: TaskNode | str) -> Parallel:
	"""The ``|`` composition — see :meth:`Sequential.__or__` and :meth:`Parallel.__or__`."""
	left_node, right_node = _node(left), _node(right)
	if isinstance(left_node, Parallel):
		if isinstance(right_node, Parallel):
			if fieldless(left_node) and (
				not fieldless(right_node) or type(right_node) is not Parallel
			):
				return rebuilt(right_node, *_nodes(left_node.tasks), *_nodes(right_node.tasks))
			return rebuilt(left_node, *_nodes(left_node.tasks), *_nodes(right_node.tasks))
		return rebuilt(left_node, *_nodes(left_node.tasks), right_node)
	if isinstance(right_node, Parallel):
		return rebuilt(right_node, left_node, *_nodes(right_node.tasks))
	return Parallel(left_node, right_node)


def _sequential_of(left: TaskNode | str, right: TaskNode | str) -> Sequential:
	"""The ``+`` composition — see :meth:`Sequential.__add__` and :meth:`Parallel.__add__`."""
	left_node, right_node = _node(left), _node(right)
	if isinstance(left_node, Sequential):
		if isinstance(right_node, Sequential):
			if fieldless(left_node) and (
				not fieldless(right_node) or type(right_node) is not Sequential
			):
				return rebuilt(right_node, *_nodes(left_node.tasks), *_nodes(right_node.tasks))
			return rebuilt(left_node, *_nodes(left_node.tasks), *_nodes(right_node.tasks))
		return rebuilt(left_node, *_nodes(left_node.tasks), right_node)
	if isinstance(right_node, Sequential):
		return rebuilt(right_node, left_node, *_nodes(right_node.tasks))
	return Sequential(left_node, right_node)


def GIT_PORCELAIN() -> Task:  # noqa: N802  # constructor-style factory, like Task/Parallel
	"""A fresh default drift check for :func:`Clean` — a factory, not an instance, so a
	``tasks.py`` importing it registers no runnable task. Runs the maintained
	:mod:`camas._git_porcelain` module under camas's own interpreter; ``git`` must be on
	PATH.
	"""
	return Task((sys.executable, "-m", "camas._git_porcelain"), cwd=".")


_GIT_PORCELAIN: Final = GIT_PORCELAIN()
"""Kept private so importing camas into a ``tasks.py`` registers no runnable task."""


def _as_task(value: object, message: str) -> Task:
	"""Coerce a bare command string to a Task, or reject any other non-Task with ``message``.

	Raises:
		ValueError: ``value`` is neither a ``str`` nor a ``Task``.
	"""
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
	"""A codegen drift gate: run the mutating generator, then fail if the working tree is not
	exactly as it was.

	Expands to ``Sequential``: the ``check`` as ``<label>-before``, then ``mutator``, then
	``check`` as ``<label>-after``; ``label`` is ``name``, else the mutator's label joined
	with the check's own name. A dirty tree fails the before-check and the blocker skips the
	generator; the after-check's failure output is the drift diagnostic. The check leaves
	always run — ``when="."`` and ``paths=None`` override the check's own scoping — and the
	check reads git's view, so paths git ignores are outside its contract. The default check
	scrubs ambient GIT_* environment (case-insensitively on Windows and MSYS/Cygwin); a
	user-supplied ``check`` runs with the ambient environment. Under ``--under``
	the gate keeps its ordering (#306); a check leaf measured over budget is excluded
	outright, and a mutator measured over budget drops like any leaf, leaving the checks to
	run around an un-run generator, so drift goes undetected.

	Raises:
		ValueError: ``mutator`` or ``check`` is neither a ``str`` nor a ``Task``; ``mutator``
			lacks ``mutates=True``; ``check`` carries a ``{paths}`` token.

	>>> Clean(Task("make update-openapi", mutates=True), before=False).tasks[0].cmd
	'make update-openapi'
	>>> Clean(Task("make update-openapi", mutates=True)).tasks[0].name
	'make update-openapi-before'
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
