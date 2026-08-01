# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins
"""``camas --check`` action: eval the tasks file and run a type checker.

The same eval-error reporting (minimal user-frame trace + opportunistic
typecheck) is reused by the normal-task path on eval failure — type checking
is paid for only when something's already broken, except when ``--check``
explicitly requests it.
"""

from __future__ import annotations

import linecache
import os
import shutil
import site
import subprocess
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, TypeAlias

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..v0.task import TaskNode


CheckerName: TypeAlias = Literal["ty", "mypy"]


def unsatisfiable_declaration_warnings(tasks: Mapping[str, TaskNode]) -> tuple[str, ...]:
	"""One warning per task declaring ``variants=()`` — zero coupled bundles, so that branch expands
	to no leaves and silently does nothing. Deduplicated, preserving order.

	An empty *matrix axis* is deliberately not warned about: ``matrix={"version": ()}`` is the
	documented required-input form, filled per run with ``--AXIS VALUE``, so flagging it would
	report an intentional pattern as a mistake. ``variants=()`` has no such filling — no CLI
	override can supply a whole bundle — so it is always an authoring error.

	>>> from camas import Parallel, Task
	>>> unsatisfiable_declaration_warnings({"qemu": Parallel(Task("t"), variants=(), name="qemu")})
	("task 'qemu' declares variants=() on 'qemu' — that branch expands to no leaves, so it silently does nothing; give it at least one variant",)
	>>> unsatisfiable_declaration_warnings({"release": Parallel(Task("t"), matrix={"version": ()})})
	()
	"""
	from ..core.matrix import empty_variant_labels

	return tuple(
		dict.fromkeys(
			f"task {name!r} declares variants=() on {label!r} — that branch expands to no leaves, "
			"so it silently does nothing; give it at least one variant"
			for name, node in tasks.items()
			for label in empty_variant_labels(node)
		)
	)


def pep723_run_cli_warning(source: Path | None) -> tuple[str, ...]:
	r"""Warn when a ``tasks.py`` declares a PEP 723 camas dependency but has no
	``run_cli(globals())`` guard — the standalone flow (``uv run tasks.py …``)
	imports and exits without dispatching, so every invocation (including
	``--check``) silently passes.

	>>> from pathlib import Path
	>>> import tempfile, os
	>>> d = tempfile.mkdtemp()

	A file with a PEP 723 camas dep and no guard:

	>>> p = Path(d) / "tasks.py"
	>>> _ = p.write_text('# /// script\n# dependencies = ["camas"]\n# ///\nfrom camas import Task\n')
	>>> len(pep723_run_cli_warning(p))
	1

	Same file with the guard added:

	>>> _ = p.write_text('# /// script\n# dependencies = ["camas"]\n# ///\nfrom camas import Task, run_cli\nif __name__ == "__main__":\n    run_cli(globals())\n')
	>>> pep723_run_cli_warning(p)
	()

	No PEP 723 block at all:

	>>> _ = p.write_text('from camas import Task\n')
	>>> pep723_run_cli_warning(p)
	()

	PEP 723 block without a camas dependency (``uv run`` would fail with
	ImportError, not silently pass):

	>>> _ = p.write_text('# /// script\n# dependencies = ["other"]\n# ///\nfrom camas import Task\n')
	>>> pep723_run_cli_warning(p)
	()

	Non-.py source or None:

	>>> pep723_run_cli_warning(None)
	()
	>>> pep723_run_cli_warning(Path("pyproject.toml"))
	()
	"""
	if source is None or source.suffix.lower() != ".py":
		return ()
	try:
		text = source.read_text(encoding="utf-8")
	except (OSError, ValueError):
		return ()
	from .pep723 import parse_camas_requirement

	if parse_camas_requirement(text) is None:
		return ()
	if "run_cli(" in text:
		return ()
	return (
		f"{source.name} has a PEP 723 script header with a camas dependency "
		"but no run_cli(globals()) guard — `uv run "
		f"{source.name} …` imports and exits "
		"without dispatching, so every invocation (including --check) silently "
		"passes; add:\n"
		'    if __name__ == "__main__":\n'
		"        run_cli(globals())",
	)


def unresolved_dispatch_warnings(tasks: Mapping[str, TaskNode]) -> tuple[str, ...]:
	"""One warning per leaf command that re-enters camas with a task name nothing resolves —
	the fan-out that "passes locally, 404s in CI", where a matrix axis or a hand-written command
	names a task that isn't dispatchable. Deduplicated, preserving order.

	>>> from camas import Parallel, Task
	>>> fan = Parallel(Task("camas {crate}"), matrix={"crate": ("core", "gone")})
	>>> unresolved_dispatch_warnings({"fan": fan, "core": Task("cargo test -p core")})
	("task 'fan' runs `camas gone`, but no task named 'gone' exists (known: core, fan) — that command fails wherever it runs, CI included",)
	>>> unresolved_dispatch_warnings({"core": Task("camas core")})
	()
	"""
	from ..core.matrix import expand_matrix
	from ..core.task import did_you_mean
	from ..core.traversal import flatten_leaves
	from .argv import camas_task_arg

	def resolves(arg: str) -> bool:
		return arg in tasks or arg.replace("-", "_") in tasks

	known: Final = ", ".join(sorted(tasks)) or "none"
	return tuple(
		dict.fromkeys(
			f"task {name!r} runs `camas {arg}`, but no task named {arg!r} exists"
			f"{did_you_mean(arg, tasks)} (known: {known}) — that command fails wherever it runs, "
			"CI included"
			for name, node in tasks.items()
			for info in flatten_leaves(expand_matrix(node))
			if (arg := camas_task_arg(info.task.cmd)) is not None and not resolves(arg)
		)
	)


class FoundChecker(NamedTuple):
	"""A located type checker: its identity and an executable path."""

	name: CheckerName
	path: Path


class CheckerOk(NamedTuple):
	"""Type-check outcome: checker exited 0."""

	name: CheckerName


class CheckerErr(NamedTuple):
	"""Type-check outcome: checker exited non-zero."""

	name: CheckerName
	output: str
	"""Combined stdout + stderr of the checker process."""


class CheckerNotFound(NamedTuple):
	"""Type-check outcome: no ty / mypy available internally or on PATH."""


TypeCheckResult: TypeAlias = CheckerOk | CheckerErr | CheckerNotFound


CHECKER_PRIORITY: Final[tuple[CheckerName, ...]] = ("ty", "mypy")


INSTALL_HINT: Final = (
	"no type checker found; install with the [check] extra "
	"(e.g. `pip install camas[check]`) to bundle ty, or put ty / mypy on PATH"
)


def describe_check_help() -> str:
	"""Dynamic ``--help`` text for the ``--check`` flag — reflects the *current*
	environment so the user sees which checker would run, or how to install one.

	>>> isinstance(describe_check_help(), str)
	True
	"""
	if (found := find_typechecker()) is None:
		return (
			"check the task definition (to include type checking, add ty or mypy "
			"to PATH or install camas[check])"
		)
	return f"check the task definition ({found.name})"


def find_typechecker() -> FoundChecker | None:
	"""Locate the highest-priority available type checker.

	For each ``name`` in ``CHECKER_PRIORITY``: prefer ``<sys.executable>/../<name>``
	(the user opted in via ``pip install camas[check]``), then ``shutil.which(name)``.

	>>> result = find_typechecker()
	>>> result is None or (result.name in ("ty", "mypy") and result.path.is_file())
	True
	"""
	suffix = ".exe" if sys.platform == "win32" else ""
	internal_dir = Path(sys.executable).parent
	for name in CHECKER_PRIORITY:
		internal = internal_dir / f"{name}{suffix}"
		if internal.is_file():
			return FoundChecker(name=name, path=internal)
		if (found := shutil.which(name)) is not None:
			return FoundChecker(name=name, path=Path(found))
	return None


class CheckerInvocation(NamedTuple):
	"""How to run a located checker so that ``tasks_py``'s ``import camas`` resolves."""

	argv: tuple[str, ...]
	env: dict[str, str]
	"""Variables to overlay on the inherited environment; empty when ``argv`` carries it all.

	A builtin rather than ``Mapping`` so the annotation resolves at runtime: under PEP 563 a
	``NamedTuple`` field annotated with a ``TYPE_CHECKING``-only name makes ``get_type_hints``
	raise ``NameError``, and importing ``Mapping`` unconditionally would be moved back under
	``TYPE_CHECKING`` by the ``TC`` lint rules on the next ``camas fix``."""


def camas_search_path() -> Path:
	"""The directory holding the running ``camas`` package.

	Resolved at call time, not import time: mypyc-compiled modules may not define ``__file__``
	while the module body executes (nixpkgs' mypyc doesn't) — which is the very install this
	path exists to serve.

	>>> (camas_search_path() / "camas" / "__init__.py").is_file()
	True
	"""
	return Path(__file__).parents[2]


def in_interpreter_site_packages(path: Path) -> bool:
	"""Whether ``path`` is one of *this* interpreter's site-packages directories.

	Asked only for mypy, which resolves imports from the site-packages of the interpreter it runs
	under: a mypy there already sees camas, and ``modulefinder`` refuses to start when ``MYPYPATH``
	holds one of those directories ("… is in the MYPYPATH. Please remove it."). For mypy the two
	conditions coincide, so skipping is right rather than an evasion of the guard. It is not asked
	for ty, which resolves against an environment it *discovers* — possibly the project's venv
	rather than camas's — where camas can be absent from ty's view while sitting in this
	interpreter's site-packages, exactly the case a wheel-installed camas is in.

	>>> in_interpreter_site_packages(Path(site.getsitepackages()[0]))
	True
	>>> in_interpreter_site_packages(Path("nowhere-in-particular"))
	False
	"""
	return str(path) in (*site.getsitepackages(), site.getusersitepackages())


def checker_invocation(
	found: FoundChecker, tasks_py: Path, camas_root: Path, inherited: Mapping[str, str]
) -> CheckerInvocation:
	"""Build the per-tool invocation: ``ty check <path>`` vs ``mypy <path>``, each told where
	``camas_root`` is so the checker resolves ``import camas`` from the camas that is running
	rather than from whatever environment it discovers for itself.

	``inherited`` is the environment the checker would run under, so each tool reads whatever it
	needs from it here rather than having the runner know which variable belongs to which tool.

	>>> checker_invocation(FoundChecker("ty", Path("ty")), Path("tasks.py"), Path("site"), {})
	CheckerInvocation(argv=('ty', 'check', '--extra-search-path', 'site', 'tasks.py'), env={})
	>>> checker_invocation(FoundChecker("mypy", Path("mypy")), Path("tasks.py"), Path("site"), {})
	CheckerInvocation(argv=('mypy', 'tasks.py'), env={'MYPYPATH': 'site'})

	mypy is told nothing when camas sits in a site-packages of its own, which it rejects outright
	and where it needs no telling (:func:`in_interpreter_site_packages`):

	>>> checker_invocation(
	...     FoundChecker("mypy", Path("mypy")), Path("tasks.py"), Path(site.getsitepackages()[0]), {}
	... )
	CheckerInvocation(argv=('mypy', 'tasks.py'), env={})

	A non-empty inherited ``MYPYPATH`` keeps its priority; ``camas_root`` is appended behind it:

	>>> checker_invocation(
	...     FoundChecker("mypy", Path("mypy")), Path("tasks.py"), Path("site"), {"MYPYPATH": "stubs"}
	... ).env["MYPYPATH"].split(os.pathsep)
	['stubs', 'site']
	"""
	match found.name:
		case "ty":
			return CheckerInvocation(
				(str(found.path), "check", "--extra-search-path", str(camas_root), str(tasks_py)),
				{},
			)
		case "mypy":
			return CheckerInvocation(
				(str(found.path), str(tasks_py)),
				{}
				if in_interpreter_site_packages(camas_root)
				else {
					"MYPYPATH": os.pathsep.join(
						p for p in (inherited.get("MYPYPATH"), str(camas_root)) if p
					)
				},
			)
		case _:
			assert_never(found.name)


def run_typecheck(tasks_py: Path) -> TypeCheckResult:
	"""Run the highest-priority available type checker against ``tasks_py``."""
	found = find_typechecker()
	if found is None:
		return CheckerNotFound()
	invocation = checker_invocation(found, tasks_py, camas_search_path(), os.environ)
	proc = subprocess.run(
		invocation.argv,
		capture_output=True,
		text=True,
		encoding="utf-8",
		errors="replace",
		check=False,
		env={**os.environ, **invocation.env} if invocation.env else None,
	)
	if proc.returncode == 0:
		return CheckerOk(name=found.name)
	return CheckerErr(name=found.name, output=proc.stdout + proc.stderr)


def deepest_user_frame(exc: Exception, tasks_py: Path) -> traceback.FrameSummary | None:
	"""Find the deepest frame in ``exc``'s traceback whose file is ``tasks_py``.

	Returns ``None`` if no frame matches (e.g. exception raised entirely inside
	a module imported by ``tasks_py``, with no in-tasks_py frame on the stack).
	"""
	target = tasks_py.resolve()
	stack = traceback.TracebackException.from_exception(exc).stack
	user = [fs for fs in stack if Path(fs.filename).resolve() == target]
	return user[-1] if user else None


def caret_line(colno: int, end_colno: int, raw_source: str) -> str | None:
	"""Build the ``    ^^^^`` line pointing at PEP 657 column info, or ``None`` when the
	offset falls inside ``raw_source``'s leading whitespace.

	>>> caret_line(11, 14, "x = foo(bar)")
	'               ^^^'
	>>> caret_line(2, 6, "        deep_indent()") is None
	True
	"""
	caret_col = colno - (len(raw_source) - len(raw_source.lstrip()))
	if caret_col < 0:
		return None
	return "    " + " " * caret_col + "^" * max(1, end_colno - colno)


def format_minimal_trace(exc: Exception, tasks_py: Path) -> str:
	"""Render a minimal user-frame trace for an eval failure in ``tasks_py``.

	Format::

	    error: <path>:<line>
	        <source line>
	        <caret marking the offending span on Python 3.11+>
	    <ExceptionType>: <message>

	If no frame inside ``tasks_py`` is on the traceback, falls back to the
	full :mod:`traceback` formatting so we never silently drop information.
	"""
	frame = deepest_user_frame(exc, tasks_py)
	if frame is None or frame.lineno is None:
		return "".join(traceback.format_exception(exc))
	raw = linecache.getline(frame.filename, frame.lineno).rstrip("\n")
	stripped = raw.lstrip()
	colno = getattr(frame, "colno", None)
	end_colno = getattr(frame, "end_colno", None)
	parts = [f"error: {frame.filename}:{frame.lineno}"]
	if stripped:
		parts.append(f"    {stripped}")
		if (
			isinstance(colno, int)
			and isinstance(end_colno, int)
			and (caret := caret_line(colno, end_colno, raw)) is not None
		):
			parts.append(caret)
	parts.append(f"{type(exc).__name__}: {exc}")
	return "\n".join(parts) + "\n"


def format_checker_output(result: TypeCheckResult, *, after_trace: bool) -> str:
	r"""Format ``result`` for stderr.

	``after_trace`` flips two behaviours used to merge typechecker output with
	a preceding eval trace: it prepends a blank-line separator before a
	:class:`CheckerErr` block, and silences :class:`CheckerNotFound` (the
	install hint is noise when an eval traceback is already on screen).

	>>> format_checker_output(CheckerOk(name="ty"), after_trace=False)
	''
	>>> format_checker_output(CheckerErr("ty", "msg"), after_trace=True)
	'\nty:\nmsg'
	"""
	match result:
		case CheckerErr(name=name, output=out):
			separator = "\n" if after_trace else ""
			return f"{separator}{name}:\n{out}"
		case CheckerNotFound():
			return "" if after_trace else f"{INSTALL_HINT}\n"
		case CheckerOk():
			return ""
		case _:
			assert_never(result)


def report_eval_error(tasks_py: Path, exc: Exception) -> int:
	"""Print a minimal trace for ``exc`` and run the typechecker; return exit code 1.

	Shared by the normal-task path (on tasks.py eval failure) and ``--check``.
	"""
	sys.stderr.write(format_minimal_trace(exc, tasks_py))
	sys.stderr.write(format_checker_output(run_typecheck(tasks_py), after_trace=True))
	return 1


def run_typecheck_only(source: Path | None) -> int:
	"""Run only the type-checker (eval is assumed to have already passed).

	A non-``.py`` source (pyproject.toml) or missing source short-circuits to 0.
	"""
	if source is None or source.suffix != ".py":
		return 0
	result = run_typecheck(source)
	sys.stderr.write(format_checker_output(result, after_trace=False))
	return 0 if isinstance(result, CheckerOk) else 1
