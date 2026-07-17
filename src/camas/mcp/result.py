# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Build MCP wire responses from camas's run results and load state."""

from __future__ import annotations

import shlex
import sys
from typing import TYPE_CHECKING, Literal, NamedTuple

from ..core.gate import decision_of
from ..core.matrix import expand_matrix
from ..core.render import strip_ansi
from ..core.scope import scope_warning_messages
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
from ..main.github_matrix import to_matrix_object
from ..main.state import LoadErr, LoadOk
from ..v0.completion import Errored, Finished, Skipped, Stopped
from . import wire

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

if TYPE_CHECKING:
	from collections.abc import Sequence
	from pathlib import Path

	from ..core.completion import RunResult, TaskResult
	from ..core.gate import GateOutcome
	from ..main.state import TasksState
	from ..v0.completion import Completion
	from ..v0.task import OutputKind, Task, TaskNode

Verbosity = Literal["summary", "failures", "full"]


def to_run_response(
	node: TaskNode, result: RunResult, *, verbosity: Verbosity = "failures", tail: int = 50
) -> wire.RunResponse:
	"""Assemble the wire ``RunResponse`` from a finished run and its pre-expansion task tree."""
	leaves = tuple(info.task for info in flatten_leaves(expand_matrix(node)))
	reports = tuple(
		report(task, tr, verbosity=verbosity, tail=tail)
		for task, tr in zip(leaves, result.results, strict=True)
	)
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
	leaves = tuple(
		wire.LeafReport(
			name=info.task.name if info.task.name is not None else command_of(info.task),
			command=command_of(info.task),
			cwd=str(info.task.cwd) if info.task.cwd is not None else None,
			completion=wire.Skipped(returncode=0, blocked_by=None),
		)
		for info in flatten_leaves(expand_matrix(node))
	)
	return wire.RunResponse(
		returncode=0,
		elapsed=0.0,
		passed=0,
		failed=0,
		skipped=len(leaves),
		interrupt_count=0,
		leaves=leaves,
	)


def to_github_matrix_response(name: str, node: TaskNode) -> wire.GithubMatrixResponse:
	"""The wire ``GithubMatrixResponse`` for ``node`` — its faithful GHA object-of-arrays, or the
	``ValueError`` :func:`camas.main.github_matrix.to_matrix_object` raises for a task with no
	matrix or a non-cross-product run-set (the handler surfaces it as a tool error).
	"""
	return wire.GithubMatrixResponse(task=name, matrix=to_matrix_object(node))


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
			warnings = scope_warning_messages(tasks.values())
			if source is None:
				return wire.CheckResponse(status="no_tasks", warnings=warnings)
			if source.suffix != ".py":
				return wire.CheckResponse(
					status="ok", source=str(source), task_count=len(tasks), warnings=warnings
				)
			return typecheck_response(source, len(tasks), warnings)
		case _:
			assert_never(state)


def typecheck_response(
	source: Path, task_count: int, warnings: tuple[str, ...] = ()
) -> wire.CheckResponse:
	"""Run the type checker against a loaded ``.py`` tasks source and map its outcome."""
	result = run_typecheck(source)
	match result:
		case CheckerOk(name=name):
			return wire.CheckResponse(
				status="ok",
				source=str(source),
				task_count=task_count,
				checker=name,
				warnings=warnings,
			)
		case CheckerErr(name=name, output=output):
			return wire.CheckResponse(
				status="type_error",
				source=str(source),
				task_count=task_count,
				checker=name,
				diagnostics=output,
				warnings=warnings,
			)
		case CheckerNotFound():
			return wire.CheckResponse(
				status="no_checker",
				source=str(source),
				task_count=task_count,
				diagnostics=INSTALL_HINT,
				warnings=warnings,
			)
		case _:
			assert_never(result)


class Decoded(NamedTuple):
	"""Decoded leaf output: ANSI-free lines and whether they are a tail excerpt."""

	lines: list[str]
	truncated: bool


def command_of(task: Task) -> str:
	"""The task's command as one shell-readable string."""
	return task.cmd if isinstance(task.cmd, str) else shlex.join(task.cmd)


def decode(output: Sequence[bytes], tail: int | None) -> Decoded:
	"""Decode merged stdout/stderr to ANSI-free lines, tail-truncated to ``tail`` (``None`` keeps
	all lines — for a structured payload that must pass verbatim).
	"""
	lines = [strip_ansi(b.decode("utf-8", errors="replace")).rstrip("\n") for b in output]
	if tail is not None and len(lines) > tail:
		return Decoded(lines[-tail:], truncated=True)
	return Decoded(lines, truncated=False)


def report(task: Task, result: TaskResult, *, verbosity: Verbosity, tail: int) -> wire.LeafReport:
	comp: Completion = result.completion
	include = verbosity == "full" or (verbosity == "failures" and comp.returncode != 0)
	match comp:
		case Finished(returncode=rc, elapsed=el, output=out):
			decoded = decode(out, tail) if include else Decoded([], truncated=False)
			completion: wire.Completion = wire.Finished(
				returncode=rc, elapsed=el, output=decoded.lines
			)
		case Stopped(returncode=rc, elapsed=el, output=out):
			decoded = decode(out, tail) if include else Decoded([], truncated=False)
			completion = wire.Stopped(returncode=rc, elapsed=el, output=decoded.lines)
		case Skipped(returncode=rc, blocked_by=bb):
			decoded = Decoded([], truncated=False)
			completion = wire.Skipped(returncode=rc, blocked_by=bb)
		case Errored(returncode=rc, message=message):
			decoded = Decoded([], truncated=False)
			completion = wire.Errored(returncode=rc, message=message)
		case _:
			assert_never(comp)
	return wire.LeafReport(
		name=result.name,
		command=command_of(task),
		cwd=str(task.cwd) if task.cwd is not None else None,
		completion=completion,
		truncated=decoded.truncated,
	)


def over_limit_pointer(kind: OutputKind, limit: int, report_path: Path | None) -> str:
	"""The payload substituted when a structured payload exceeds ``limit`` — a truncated
	structured document is invalid, so the agent is pointed at the full file instead of
	receiving a corrupt excerpt.
	"""
	if report_path is not None:
		return (
			f"{kind} payload exceeds the {limit}-character limit — a truncated structured "
			f"document would be invalid; read the full file directly: {report_path}"
		)
	return (
		f"{kind} payload exceeds the {limit}-character limit — a truncated structured document "
		"would be invalid, so it was omitted. Raise this task's agent_format limit=, or switch "
		'it to path mode (agent_format=("... {report}", kind)) so the full output lives in a '
		"file instead of this response; see camas_docs."
	)


def to_agent_envelope(
	task: Task, result: TaskResult, *, tail: int = 50, report_path: Path | None = None
) -> wire.AgentEnvelope:
	"""``result``'s AgentEnvelope: stdout tail-capped for ``raw``, verbatim for a structured
	``kind`` — read from ``report_path`` (path mode) only when the tool ran to its own exit
	(``Finished``) and the file holds a payload, else from stdout/message, so a tool killed
	(``Stopped``) or that never started (``Errored``) still delivers its captured diagnostics
	rather than a partial or empty report. A payload over its ``agent_format.limit`` — any
	structured ``kind``, or a ``raw`` report file (stdout ``raw`` is already tail-capped) — is
	replaced with :func:`over_limit_pointer` rather than dumped or tailed, since a truncated
	structured document is invalid; the pointer names the report file only when the payload came
	from it.
	"""
	comp = result.completion
	fmt = task.agent_format
	kind = fmt.kind if fmt is not None else "raw"
	report_payload = (
		report_path.read_text(encoding="utf-8", errors="replace")
		if report_path is not None and report_path.is_file()
		else ""
	)
	used_report = bool(report_payload) and isinstance(comp, Finished)
	if used_report:
		payload, truncated = report_payload, False
	else:
		match comp:
			case Finished(output=output) | Stopped(output=output):
				decoded = decode(output, tail if kind == "raw" else None)
			case Skipped():
				decoded = Decoded([], truncated=False)
			case Errored(message=message):
				decoded = Decoded([message], truncated=False)
			case _:
				assert_never(comp)
		payload, truncated = "\n".join(decoded.lines), decoded.truncated
	log = str(report_path) if report_path is not None else None
	if fmt is not None and len(payload) > fmt.limit and (kind != "raw" or used_report):
		return wire.AgentEnvelope(
			name=result.name,
			exit_code=comp.returncode,
			output_kind=kind,
			payload=over_limit_pointer(kind, fmt.limit, report_path if used_report else None),
			truncated=True,
			log=log,
		)
	return wire.AgentEnvelope(
		name=result.name,
		exit_code=comp.returncode,
		output_kind=kind,
		payload=payload,
		truncated=truncated,
		log=log,
	)


def agent_envelopes(
	node: TaskNode, result: RunResult, report_paths: tuple[Path | None, ...] = ()
) -> tuple[wire.AgentEnvelope, ...]:
	"""The failing leaves of a finished run as AgentJSON envelopes, in DFS order. A ``Skipped``
	leaf never ran, so it is not a failure to surface — only its blocker is. ``report_paths``
	(DFS-aligned with ``node``'s leaves, from :func:`camas.core.gate.with_agent_format`) supplies
	each path-mode leaf's report file; omit it for a node with no path-mode leaf.
	"""
	leaves = tuple(info.task for info in flatten_leaves(expand_matrix(node)))
	paths = report_paths or (None,) * len(leaves)
	return tuple(
		to_agent_envelope(task, tr, report_path=rp)
		for task, tr, rp in zip(leaves, result.results, paths, strict=True)
		if not isinstance(tr.completion, Skipped) and tr.completion.returncode != 0
	)


def has_failing_leaf_without_agent_format(node: TaskNode, result: RunResult) -> bool:
	"""True when a non-skipped failing leaf's task has no ``agent_format`` — its output is
	unstructured raw text rather than a tagged, machine-readable diagnostic.
	"""
	leaves = tuple(info.task for info in flatten_leaves(expand_matrix(node)))
	return any(
		task.agent_format is None
		for task, tr in zip(leaves, result.results, strict=True)
		if not isinstance(tr.completion, Skipped) and tr.completion.returncode != 0
	)


def to_gate_response(
	outcome: GateOutcome, budget: wire.BudgetReport | None, rerun: wire.GateRerun
) -> wire.GateResponse:
	diagnostics = (
		agent_envelopes(outcome.node, outcome.result, outcome.report_paths)
		if outcome.residual_class == "needs_reasoning"
		and outcome.node is not None
		and outcome.result is not None
		else None
	)
	return wire.GateResponse(
		decision=decision_of(outcome.residual_class),
		residual_class=outcome.residual_class,
		diagnostics=diagnostics,
		budget=budget,
		rerun=rerun,
	)
