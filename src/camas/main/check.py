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
import subprocess
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, TypeAlias

from ..paths import camas_package_dir

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
	"""The directory holding the running ``camas`` package, for a checker's search path.

	An install reached through a symlink tree (a Nix profile, a symlinked editable install) is
	handed to the checker as the location it will read the package out of, rather than as whichever
	spelling of it the import came through.

	>>> (camas_search_path() / "camas" / "__init__.py").is_file()
	True
	"""
	return camas_package_dir().resolve().parent


UNRESOLVED_CAMAS: Final = {
	"ty": ("Cannot resolve imported module `camas`", "Cannot resolve imported module `camas."),
	"mypy": ('module named "camas"', 'module named "camas.'),
}
"""Each checker's phrasing for #277: exact for the package, dot-bounded for a submodule spelling."""


def cannot_resolve_camas(name: CheckerName, output: str) -> bool:
	"""Whether the checker reported the ``import camas`` in a tasks file unresolved — the #277
	failure, and the only answer that earns a second run.

	Whether a checker resolves camas on its own is not a question camas can answer about it. mypy
	searches the site-packages of the interpreter *it* runs under, which a mypy beside this
	interpreter shares but so does one in a venv built with ``--system-site-packages``; ty searches an
	environment it discovers for itself. So each is asked, and asked bare first, because being told
	costs something in every layout that did not need it: mypy refuses a ``MYPYPATH`` at or under its
	own site-packages, and ty searches an ``--extra-search-path`` *before* its discovered environment,
	which would let camas's site-packages shadow the project's for every module both hold.

	Matching their wording can only be wrong in the cheap direction. A tasks file whose own source
	contains the phrase (mypy echoes source lines) buys one extra run that only adds a search path and
	changes no verdict; a future rewording stops the second run from happening, which is where this
	was before the fix rather than worse than it.

	>>> cannot_resolve_camas("mypy", 't.py:1: error: … stub for module named "camas"')
	True
	>>> cannot_resolve_camas("ty", "error[unresolved-import]: Cannot resolve imported module `camas`")
	True
	>>> cannot_resolve_camas("mypy", 't.py:1: error: … stub for module named "camas.mcp"')
	True
	>>> cannot_resolve_camas("ty", "error[unresolved-import]: Cannot resolve imported module `camas.mcp`")
	True
	>>> cannot_resolve_camas("mypy", 't.py:1: error: … stub for module named "camas_extra"')
	False
	>>> cannot_resolve_camas("ty", "error[unresolved-import]: Cannot resolve imported module `camas_extra`")
	False
	>>> cannot_resolve_camas("ty", "error[invalid-assignment]: `int` is not assignable to `str`")
	False
	"""
	from ..core.render import strip_ansi

	stripped = strip_ansi(output)
	return any(phrase in stripped for phrase in UNRESOLVED_CAMAS[name])


def refuses_the_search_path(output: str) -> bool:
	"""Whether mypy declined to start because the directory camas named is one of its own
	site-packages, which ``modulefinder`` rejects outright ("… is in the MYPYPATH. Please
	remove it.").

	Only reachable on the second run, and only from a layout that holds camas *under* one of mypy's
	site-packages rather than in it — where mypy can neither resolve camas nor accept being told. The
	first run's diagnostic is the honest one there, so it is what gets reported — with the refusal
	explained (:func:`refusal_note`).

	>>> refuses_the_search_path("/x/site-packages is in the MYPYPATH. Please remove it.")
	True
	>>> refuses_the_search_path('t.py:1: error: Name "x" is undefined')
	False
	"""
	from ..core.render import strip_ansi

	return "is in the MYPYPATH" in strip_ansi(output)


def refusal_note(camas_root: Path) -> str:
	"""Appended to the kept first answer when the told run is refused."""
	return (
		f"camas is installed at {camas_root}, but mypy refuses to search a directory at or under "
		"its own site-packages; run mypy directly, or install camas somewhere mypy can be told "
		"about (e.g. pipx)"
	)


def mypypath(inherited: str, camas_root: Path) -> str:
	"""``camas_root`` behind whatever the environment already asked for, empty entries dropped so
	that the spellings of "nothing inherited" overlay alike.

	>>> mypypath("", Path("site"))
	'site'
	>>> mypypath(os.pathsep, Path("site"))
	'site'
	>>> mypypath(f"stubs{os.pathsep}", Path("site")).split(os.pathsep)
	['stubs', 'site']
	"""
	return os.pathsep.join((*(p for p in inherited.split(os.pathsep) if p), str(camas_root)))


def checker_invocation(found: FoundChecker, tasks_py: Path) -> CheckerInvocation:
	"""How a checker is asked the first time: bare, exactly as camas asked before #277 had a fix.

	Neither hint is offered up front, because being told costs something wherever it was not needed —
	see :func:`cannot_resolve_camas`. Only a checker that answers that it cannot find camas is told
	(:func:`told_where_camas_is`).

	>>> checker_invocation(FoundChecker("ty", Path("ty")), Path("tasks.py"))
	CheckerInvocation(argv=('ty', 'check', 'tasks.py'), env={})
	>>> checker_invocation(FoundChecker("mypy", Path("mypy")), Path("tasks.py"))
	CheckerInvocation(argv=('mypy', 'tasks.py'), env={})
	"""
	match found.name:
		case "ty":
			return CheckerInvocation((str(found.path), "check", str(tasks_py)), {})
		case "mypy":
			return CheckerInvocation((str(found.path), str(tasks_py)), {})
		case _:
			assert_never(found.name)


def told_where_camas_is(
	found: FoundChecker, tasks_py: Path, camas_root: Path, inherited: str
) -> CheckerInvocation:
	"""The same run with ``camas_root`` named the way each tool takes it — a flag for ty, a variable
	for mypy, which has no equivalent flag. Both are additive.

	``inherited`` keeps its priority in mypy's search path, so an entry there holding a camas of its
	own shadows the running one: explicit configuration outranks camas's hint.

	>>> told_where_camas_is(FoundChecker("ty", Path("ty")), Path("tasks.py"), Path("site"), "")
	CheckerInvocation(argv=('ty', 'check', '--extra-search-path', 'site', 'tasks.py'), env={})
	>>> told_where_camas_is(FoundChecker("mypy", Path("mypy")), Path("tasks.py"), Path("site"), "")
	CheckerInvocation(argv=('mypy', 'tasks.py'), env={'MYPYPATH': 'site'})
	>>> told_where_camas_is(
	...     FoundChecker("mypy", Path("mypy")), Path("t.py"), Path("site"), "stubs"
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
				{"MYPYPATH": mypypath(inherited, camas_root)},
			)
		case _:
			assert_never(found.name)


def run_checker(found: FoundChecker, invocation: CheckerInvocation) -> CheckerOk | CheckerErr:
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


def run_typecheck(tasks_py: Path) -> TypeCheckResult:
	"""Run the highest-priority available type checker against ``tasks_py``, naming camas's location
	only to a checker that answers that it cannot find camas.

	So the second run is paid by the installs #277 is about — a Nix or pipx camas, whose checker
	resolves against an environment camas does not live in. Where the checker already sees camas the
	answer is the first run's, and camas's own location is never even resolved.

	A mypy that will neither resolve camas nor accept being told about it — camas installed *under*
	one of mypy's site-packages — keeps its first answer, the diagnostic camas gave before any of
	this existed, with the refusal explained.
	"""
	found = find_typechecker()
	if found is None:
		return CheckerNotFound()
	result = run_checker(found, checker_invocation(found, tasks_py))
	if not isinstance(result, CheckerErr) or not cannot_resolve_camas(found.name, result.output):
		return result
	told = run_checker(
		found,
		told_where_camas_is(found, tasks_py, camas_search_path(), os.environ.get("MYPYPATH", "")),
	)
	if isinstance(told, CheckerErr) and refuses_the_search_path(told.output):
		return CheckerErr(
			name=result.name,
			output=result.output + "\n" + refusal_note(camas_search_path()),
		)
	return told


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
