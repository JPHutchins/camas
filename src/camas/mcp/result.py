# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Build MCP wire responses from camas's run results and load state."""

from __future__ import annotations

import shlex
import sys
from typing import TYPE_CHECKING, Literal, NamedTuple

from ..core.matrix import expand_matrix
from ..core.render import strip_ansi
from ..core.traversal import flatten_leaves
from ..main.check import (
	INSTALL_HINT,
	CheckerErr,
	CheckerNotFound,
	CheckerOk,
	format_checker_output,
	format_minimal_trace,
	run_typecheck,
)
from ..main.state import LoadErr, LoadOk
from ..v0.completion import Finished, Skipped, Stopped
from . import wire

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

if TYPE_CHECKING:
	from collections.abc import Sequence
	from pathlib import Path

	from ..core.completion import RunResult, TaskResult
	from ..main.state import TasksState
	from ..v0.completion import Completion
	from ..v0.task import Task, TaskNode

Verbosity = Literal["summary", "failures", "full"]


def to_run_response(
	node: TaskNode, result: RunResult, *, verbosity: Verbosity = "failures", tail: int = 50
) -> wire.RunResponse:
	"""Assemble the wire ``RunResponse`` from a finished run and its pre-expansion task tree."""
	leaves = tuple(info.task for info in flatten_leaves(expand_matrix(node)))
	reports = [
		_report(task, tr, verbosity=verbosity, tail=tail)
		for task, tr in zip(leaves, result.results, strict=True)
	]
	completions = tuple(tr.completion for tr in result.results)
	passed = sum(1 for c in completions if isinstance(c, Finished) and c.returncode == 0)
	skipped = sum(1 for c in completions if isinstance(c, Skipped))
	return wire.RunResponse(
		returncode=result.returncode,
		elapsed=result.elapsed,
		passed=passed,
		failed=len(completions) - passed - skipped,
		skipped=skipped,
		interrupt_count=result.interrupt_count,
		leaves=reports,
		truncated=any(report.truncated for report in reports),
	)


def to_plan_response(node: TaskNode) -> wire.RunResponse:
	"""A ``RunResponse`` for a dry run: every resolved leaf, none executed.

	Lets ``camas_run dry_run=true`` satisfy the advertised ``outputSchema`` — the
	structured counterpart of the rendered plan, with each leaf reported as skipped.
	"""
	leaves = [
		wire.LeafReport(
			name=info.task.name if info.task.name is not None else _command(info.task),
			command=_command(info.task),
			cwd=str(info.task.cwd) if info.task.cwd is not None else None,
			completion=wire.Skipped(returncode=0, blocked_by=None),
		)
		for info in flatten_leaves(expand_matrix(node))
	]
	return wire.RunResponse(
		returncode=0,
		elapsed=0.0,
		passed=0,
		failed=0,
		skipped=len(leaves),
		interrupt_count=0,
		leaves=leaves,
	)


def to_check_response(state: TasksState) -> wire.CheckResponse:
	"""Validate a freshly-resolved tasks source: did it evaluate, and does it type-check."""
	match state:
		case LoadErr(source=source, exception=exception):
			trace = format_minimal_trace(exception, source)
			checker = (
				format_checker_output(run_typecheck(source), after_trace=True)
				if source.suffix == ".py"
				else ""
			)
			return wire.CheckResponse(
				status="load_error", source=str(source), diagnostics=trace + checker
			)
		case LoadOk(tasks=tasks, source=source):
			if source is None:
				return wire.CheckResponse(status="no_tasks")
			if source.suffix != ".py":
				return wire.CheckResponse(status="ok", source=str(source), task_count=len(tasks))
			return _typecheck_response(source, len(tasks))
		case _:
			assert_never(state)


def _typecheck_response(source: Path, task_count: int) -> wire.CheckResponse:
	"""Run the type checker against a loaded ``.py`` tasks source and map its outcome."""
	result = run_typecheck(source)
	match result:
		case CheckerOk(name=name):
			return wire.CheckResponse(
				status="ok", source=str(source), task_count=task_count, checker=name
			)
		case CheckerErr(name=name, output=output):
			return wire.CheckResponse(
				status="type_error",
				source=str(source),
				task_count=task_count,
				checker=name,
				diagnostics=output,
			)
		case CheckerNotFound():
			return wire.CheckResponse(
				status="no_checker",
				source=str(source),
				task_count=task_count,
				diagnostics=INSTALL_HINT,
			)
		case _:
			assert_never(result)


class _Decoded(NamedTuple):
	lines: list[str]
	truncated: bool


def _command(task: Task) -> str:
	"""The task's command as one shell-readable string."""
	return task.cmd if isinstance(task.cmd, str) else shlex.join(task.cmd)


def _decode(output: Sequence[bytes], tail: int, *, include: bool) -> _Decoded:
	"""Decode merged stdout/stderr to ANSI-free lines, tail-truncated to ``tail``."""
	if not include:
		return _Decoded([], truncated=False)
	lines = [strip_ansi(b.decode("utf-8", errors="replace")).rstrip("\n") for b in output]
	if len(lines) > tail:
		return _Decoded(lines[-tail:], truncated=True)
	return _Decoded(lines, truncated=False)


def _report(task: Task, result: TaskResult, *, verbosity: Verbosity, tail: int) -> wire.LeafReport:
	comp: Completion = result.completion
	include = verbosity == "full" or (verbosity == "failures" and comp.returncode != 0)
	match comp:
		case Finished(returncode=rc, elapsed=el, output=out):
			decoded = _decode(out, tail, include=include)
			completion: wire.Completion = wire.Finished(
				returncode=rc, elapsed=el, output=decoded.lines
			)
		case Stopped(returncode=rc, elapsed=el, output=out):
			decoded = _decode(out, tail, include=include)
			completion = wire.Stopped(returncode=rc, elapsed=el, output=decoded.lines)
		case Skipped(returncode=rc, blocked_by=bb):
			decoded = _Decoded([], truncated=False)
			completion = wire.Skipped(returncode=rc, blocked_by=bb)
		case _:
			assert_never(comp)
	return wire.LeafReport(
		name=result.name,
		command=_command(task),
		cwd=str(task.cwd) if task.cwd is not None else None,
		completion=completion,
		truncated=decoded.truncated,
	)
