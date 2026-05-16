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
import runpy
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Final, Literal, NamedTuple, TypeAlias

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never


CheckerName: TypeAlias = Literal["ty", "mypy"]


class EvalOk(NamedTuple):
	"""Eval outcome: the tasks file executed without raising ``Exception``."""


class EvalErr(NamedTuple):
	"""Eval outcome: the tasks file raised an ``Exception`` during execution."""

	exception: Exception
	"""The captured exception, still attached to its traceback."""


EvalResult: TypeAlias = EvalOk | EvalErr


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


def run_eval(tasks_py: Path) -> EvalResult:
	"""Execute ``tasks_py`` via :mod:`runpy`; capture any :class:`Exception`.

	>>> import tempfile
	>>> with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
	...     _ = f.write("x = 1\\n")
	...     ok = Path(f.name)
	>>> isinstance(run_eval(ok), EvalOk)
	True
	>>> with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
	...     _ = f.write("raise ValueError('boom')\\n")
	...     bad = Path(f.name)
	>>> r = run_eval(bad)
	>>> isinstance(r, EvalErr) and isinstance(r.exception, ValueError)
	True
	"""
	try:
		runpy.run_path(str(tasks_py))
	except Exception as e:
		return EvalErr(exception=e)
	return EvalOk()


def checker_argv(found: FoundChecker, tasks_py: Path) -> list[str]:
	"""Build the per-tool argv: ``ty check <path>`` vs ``mypy <path>``.

	>>> checker_argv(FoundChecker("ty", Path("ty")), Path("tasks.py"))
	['ty', 'check', 'tasks.py']
	>>> checker_argv(FoundChecker("mypy", Path("mypy")), Path("tasks.py"))
	['mypy', 'tasks.py']
	"""
	match found.name:
		case "ty":
			return [str(found.path), "check", str(tasks_py)]
		case "mypy":
			return [str(found.path), str(tasks_py)]
		case _:
			assert_never(found.name)


def run_typecheck(tasks_py: Path) -> TypeCheckResult:
	"""Run the highest-priority available type checker against ``tasks_py``."""
	found = find_typechecker()
	if found is None:
		return CheckerNotFound()
	proc = subprocess.run(
		checker_argv(found, tasks_py),
		capture_output=True,
		text=True,
		encoding="utf-8",
		errors="replace",
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
	"""Build the ``    ^^^^`` line that points at PEP 657 column info.

	Returns ``None`` when the col offset is inside ``raw_source``'s leading
	whitespace — defensive: shouldn't happen for real Python tracebacks (the
	column points at the offending token, not indentation).

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
	"""Format ``result`` for stderr.

	``after_trace`` flips two behaviours used to merge typechecker output with
	a preceding eval trace: it prepends a blank-line separator before a
	:class:`CheckerErr` block, and silences :class:`CheckerNotFound` (the
	install hint is noise when an eval traceback is already on screen).

	>>> format_checker_output(CheckerOk(name="ty"), after_trace=False)
	''
	>>> format_checker_output(CheckerErr("ty", "msg"), after_trace=True)
	'\\nty:\\nmsg'
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
	"""Print a minimal trace for ``exc`` and opportunistically run the typechecker.

	Used by both ``resolve_tasks_source`` (normal-task path, on tasks.py eval
	failure) and ``--check`` (which always runs the typechecker, including on
	eval failure). Returns exit code ``1``.
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
