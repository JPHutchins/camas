# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``camas mcp`` stdio server: exposes a project's camas tasks over the Model Context Protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import sys
import textwrap
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple, cast

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.server import NotificationOptions
from pydantic import AnyUrl, BaseModel, ValidationError

from ..core import timings
from ..core.budget import plan_under
from ..core.execution import run
from ..core.gate import run_gate
from ..core.matrix import override_matrix
from ..core.render import render_tree_lines, strip_ansi
from ..core.scope import to_changed
from ..core.task import task_label
from ..main.argv import apply_passthrough
from ..main.format import format_load_error_hint
from ..main.state import EMPTY_STATE, LoadErr, LoadOk, TasksState
from ..main.tasks import load_tasks, load_tasks_from_py
from ..v0.completion import Finished, Skipped, Stopped
from ..v0.config import Config
from . import wire
from .catalog import to_list_response
from .docs import to_docs_response
from .result import to_check_response, to_gate_response, to_plan_response, to_run_response

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

	from ..core.budget import BudgetPlan
	from ..core.completion import RunResult, TaskResult
	from ..v0.task import TaskNode


class ToolName(Enum):
	"""The tools this server exposes; each member's value is its on-the-wire name."""

	LIST = "camas_list"
	RUN = "camas_run"
	CHECK = "camas_check"
	DOCS = "camas_docs"
	GATE = "camas_gate"


NO_ARGS_SCHEMA: Final[dict[str, Any]] = {
	"type": "object",
	"properties": {},
	"additionalProperties": False,
}
LIST_ANNOTATIONS: Final = types.ToolAnnotations(readOnlyHint=True, openWorldHint=False)
RUN_ANNOTATIONS: Final = types.ToolAnnotations(
	readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True
)
CHECK_ANNOTATIONS: Final = types.ToolAnnotations(readOnlyHint=True, openWorldHint=False)
DOCS_ANNOTATIONS: Final = types.ToolAnnotations(readOnlyHint=True, openWorldHint=False)
GATE_ANNOTATIONS: Final = types.ToolAnnotations(readOnlyHint=True, openWorldHint=False)


class Compat(NamedTuple):
	"""Client-compatibility switches. ``emit_structured`` gates the 2025-11-25 ``title``,
	``annotations``, and ``outputSchema`` tool fields plus ``structuredContent`` — all
	default-OFF because emitting them makes some Claude Code versions silently drop every tool
	(issue #25081, closed unfixed); the load-bearing ``TextContent`` summary carries everything.
	When ``--rich`` opts in, ``outputSchema`` is the raw ``model_json_schema()`` with ``$ref``s
	intact — current Claude Code resolves them; ref-blind clients are not a target.
	"""

	emit_structured: bool = False


@dataclass(slots=True)
class Session:
	"""Server state: the resolved project, its root, the compat switches, a per-run counter
	naming each run's log directory, and the source mtime that gates lazy re-resolution.
	"""

	project: TasksState
	base: Path
	compat: Compat
	runs: int = 0
	source_mtime_ns: int | None = field(default=None, init=False)

	def __post_init__(self) -> None:
		self.source_mtime_ns = source_mtime_ns(self.project)

	@property
	def camas_dir(self) -> Path:
		"""The resolved camas directory for run logs and the timing cache."""
		config = self.project.config if isinstance(self.project, LoadOk) else None
		return (config if config is not None else Config()).camas_path(self.base)

	def refresh(self) -> None:
		"""Re-resolve the project if its source file changed on disk since it was pinned."""
		if source_mtime_ns(self.project) != self.source_mtime_ns:
			self.project = resolve_project_quiet(self.base)
			self.source_mtime_ns = source_mtime_ns(self.project)

	def reserve_run(self) -> int:
		"""Claim the next run's sequence number (atomic between ``await`` points)."""
		self.runs += 1
		return self.runs


def project_source(state: TasksState) -> Path | None:
	"""The tasks file a state was resolved from, or ``None`` when none was found."""
	match state:
		case LoadOk(source=source) | LoadErr(source=source):
			return source
		case _:
			assert_never(state)


def source_mtime_ns(state: TasksState) -> int | None:
	"""The mtime of a state's source file, or ``None`` if it has none or is gone."""
	source = project_source(state)
	if source is None:
		return None
	try:
		return source.stat().st_mtime_ns
	except OSError:
		return None


def serve_stdio(argv: list[str]) -> None:  # pragma: no cover
	"""Entry point for ``camas mcp``: resolve the project, build the server, serve over stdio."""
	os.environ["NO_COLOR"] = "1"
	base = project_base()
	session = Session(resolve_project_quiet(base), base, Compat(emit_structured="--rich" in argv))
	asyncio.run(run_over_stdio(build_server(session)))


async def run_over_stdio(server: Server[object]) -> None:  # pragma: no cover
	"""Wire the low-level server to the stdio transport and run until the client closes it."""
	from mcp.server.stdio import stdio_server

	options = server.create_initialization_options(NotificationOptions(tools_changed=True))
	async with stdio_server() as (read, write):
		await server.run(read, write, options)


def build_server(session: Session) -> Server[object]:
	"""A low-level MCP ``Server`` with the camas tool handlers registered."""
	server: Server[object] = Server(
		"camas",
		version=version("camas"),
		instructions=textwrap.dedent("""\
			camas is the single source of truth for this project's tasks: the same
			definitions drive local development and CI, so running a task through camas
			reproduces CI exactly. Use the camas tools to discover, run, and validate tasks
			instead of invoking the underlying commands by hand — camas_list to see the
			available tasks and what each one runs, camas_run to run one, camas_check to
			validate the tasks definition after you edit it, and camas_docs for how to
			author tasks. Call camas_list first; it is the source of truth for task names.
			If the project has no tasks file yet, scaffold a commented starter with
			`camas --init` before authoring one by hand.
		""").strip(),
	)

	async def list_handler() -> list[types.Tool]:
		return list(tools(task_names(session.project), session.compat))

	async def call_handler(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
		before = task_names(session.project)
		session.refresh()
		result = await call(session, name, arguments)
		if task_names(session.project) != before:
			await server.request_context.session.send_tool_list_changed()
		return result

	server.list_tools()(list_handler)  # type: ignore[no-untyped-call]
	server.call_tool(validate_input=False)(call_handler)
	return server


async def call(session: Session, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
	"""Dispatch a ``tools/call`` to the matching handler, or a tool error for an unknown name."""
	match parse_tool(name):
		case ToolName.LIST:
			return list_call(session, arguments)
		case ToolName.RUN:
			return await run_call(session, arguments)
		case ToolName.CHECK:
			return check_call(session)
		case ToolName.DOCS:
			return docs_call(session)
		case ToolName.GATE:
			return await gate_call(session, arguments)
		case _:
			known = ", ".join(repr(tool.value) for tool in ToolName)
			return error_result(f"unknown tool {name!r}; expected one of: {known}")


def parse_tool(name: str) -> ToolName | None:
	"""The tool matching ``name``, or ``None`` if the client named an unknown tool."""
	return next((tool for tool in ToolName if tool.value == name), None)


def task_names(project: TasksState) -> tuple[str, ...]:
	"""Discovered task names, sorted; empty when the project failed to load."""
	return tuple(sorted(project.tasks)) if isinstance(project, LoadOk) else ()


class Tools(NamedTuple):
	"""The fixed set of tools the server exposes — a known shape, not an open list."""

	discovery: types.Tool
	execution: types.Tool
	validation: types.Tool
	docs: types.Tool
	gate: types.Tool


def tool_def(
	compat: Compat,
	*,
	name: str,
	description: str,
	input_schema: dict[str, Any],
	output_model: type[BaseModel],
	title: str,
	annotations: types.ToolAnnotations,
) -> types.Tool:
	"""One tool definition; the 2025-11-25 rich fields are emitted only under ``--rich``."""
	rich: Final = compat.emit_structured
	return types.Tool(
		name=name,
		description=description,
		inputSchema=input_schema,
		outputSchema=output_model.model_json_schema() if rich else None,
		title=title if rich else None,
		annotations=annotations if rich else None,
	)


def tools(task_names: tuple[str, ...], compat: Compat) -> Tools:
	"""The four tool definitions, with ``task_names`` spliced into ``camas_run``'s ``task`` enum."""
	return Tools(
		discovery=tool_def(
			compat,
			name=ToolName.LIST.value,
			description=textwrap.dedent("""\
				List THIS project's camas-defined tasks: each task's name, help text, its
				command expression, and any matrix axes it expands across (versions, targets,
				toolchains, platforms — whatever the project declares). Matrix tasks show the
				unexpanded template by default (their axis values are listed separately, which
				keeps discovery compact); pass expand_matrix=true to inline every resolved leaf
				recursively. Two tasks are flagged: the default (what the developer runs while
				working) and the github default — the exact task this project's CI runs (camas
				is the single definition shared by local and CI), so running it locally before
				you commit or push reproduces CI exactly. Tasks whose leaves have been timed also
				carry an estimated duration and their slowest leaf, so you can pick a quick task
				for the inner loop and the thorough one before committing. Use this first to
				discover valid task names for camas_run; it is the source of truth. Read-only;
				runs nothing.
			""").strip(),
			input_schema=wire.ListRequest.model_json_schema(),
			output_model=wire.ListResponse,
			title="List camas tasks",
			annotations=LIST_ANNOTATIONS,
		),
		execution=tool_def(
			compat,
			name=ToolName.RUN.value,
			description=textwrap.dedent("""\
				Run THIS project's defined tasks exactly as composed by the project — honoring
				its concurrency, matrix expansion across whatever axes it declares, and
				per-task cwd/env. Use this INSTEAD of invoking the underlying commands by hand:
				it runs the project-sanctioned command set, in the right order, in parallel
				where declared, and returns a structured per-task pass/fail report. A failed
				leaf's output is included inline (the last N lines) — read its log path only
				when you see the truncation marker, or pass verbosity='full' to inline every
				leaf's full output. For a tight inner loop, pass args=[…] to append flags to a
				single-leaf task (camas's -- passthrough, e.g. args=['tests/test_x.py::test_y',
				'-x'] to run and fail-fast on one test); composite tasks reject args, so target
				a leaf. For a time-boxed inner loop, pass under=<seconds> to run only the leaves
				whose recorded estimate fits — mutating leaves (formatters) first, then the
				read-only rest in parallel; omit task to budget the project default. Compact
				failures-first summary by default; dry_run=true previews the fully-resolved plan
				without executing. A non-zero result means a task failed — a normal result, not a
				tool error.
			""").strip(),
			input_schema=wire.run_input_schema(task_names),
			output_model=wire.RunResponse,
			title="Run camas tasks",
			annotations=RUN_ANNOTATIONS,
		),
		validation=tool_def(
			compat,
			name=ToolName.CHECK.value,
			description=textwrap.dedent("""\
				Validate THIS project's tasks definition: re-read tasks.py from disk, evaluate
				it (catching import and runtime errors), then run a static type checker (ty or
				mypy) over it. Call this right after you write or edit tasks.py to confirm it
				loads and type-checks before running anything — the tight authoring loop.
				Returns a structured verdict (ok / load error / type error / no checker
				available / no tasks file) carrying the exact diagnostics to fix. A broken file
				is a normal result, not a tool error. On success the task catalog is refreshed,
				so a newly-added task becomes runnable via camas_run without restarting the
				server.
			""").strip(),
			input_schema=NO_ARGS_SCHEMA,
			output_model=wire.CheckResponse,
			title="Check camas tasks",
			annotations=CHECK_ANNOTATIONS,
		),
		docs=tool_def(
			compat,
			name=ToolName.DOCS.value,
			description=textwrap.dedent("""\
				How to author or edit THIS project's tasks.py. Returns camas's own installed
				source path and its authoring tutorial — served live from the package, so it
				never drifts: the Task / Sequential / Parallel / Config API, matrix expansion,
				per-leaf cwd and env, and custom output Effects, with worked examples. Read this
				before writing tasks.py, then validate your work with camas_check. The cited
				source path is the API source of truth — read it for exact signatures and the
				examples directory.
			""").strip(),
			input_schema=NO_ARGS_SCHEMA,
			output_model=wire.DocsResponse,
			title="How to author camas tasks",
			annotations=DOCS_ANNOTATIONS,
		),
		gate=tool_def(
			compat,
			name=ToolName.GATE.value,
			description=textwrap.dedent("""\
				The SA-delegation gate, for a PostToolBatch hook: scope THIS project's checks to the
				files just changed, run them, and return a binary verdict. It does not mutate — the
				deterministic fixers run separately on FileChanged (camas fix). residual_class is
				'green' (decision 'continue') when the checks pass, or 'needs_reasoning' (decision
				'block') when a check still fails — then diagnostics carries the failing leaves. Pass
				paths=[…] (the changed files) to scope; omit to gate the whole check node. under=<seconds>
				time-boxes the checks: leaves measured to exceed it are skipped, untimed leaves run.
			""").strip(),
			input_schema=wire.gate_input_schema(task_names),
			output_model=wire.GateResponse,
			title="SA-delegation gate",
			annotations=GATE_ANNOTATIONS,
		),
	)


def list_call(session: Session, arguments: dict[str, Any]) -> types.CallToolResult:
	"""Handle ``camas_list``: the project's task catalog, or a load error."""
	try:
		req = wire.ListRequest.model_validate(arguments)
	except ValidationError as e:
		return error_result(f"invalid camas_list arguments:\n{e}")
	match session.project:
		case LoadErr(source=source, exception=exception):
			return error_result(format_load_error_hint(source, exception))
		case LoadOk(tasks=tasks, config=config):
			resp = to_list_response(
				tasks, config, timings.load(session.camas_dir), expand=req.expand_matrix
			)
			return success(list_text(resp, expand_matrix=req.expand_matrix), resp, session.compat)
		case _:
			assert_never(session.project)


def check_call(session: Session) -> types.CallToolResult:
	"""Handle ``camas_check``: re-resolve and re-pin the project from disk, then validate it."""
	session.project = resolve_project_quiet(session.base)
	resp = to_check_response(session.project)
	return success(check_text(resp), resp, session.compat)


def docs_call(session: Session) -> types.CallToolResult:
	"""Handle ``camas_docs``: the camas authoring tutorial."""
	resp = to_docs_response()
	return success(docs_text(resp), resp, session.compat)


async def run_call(session: Session, arguments: dict[str, Any]) -> types.CallToolResult:
	"""Handle ``camas_run``: validate, resolve the node, then dry-run or execute in-process."""
	match session.project:
		case LoadErr(source=source, exception=exception):
			return error_result(format_load_error_hint(source, exception))
		case LoadOk(tasks=tasks, config=config):
			return await run_for(session, tasks, config, arguments)
		case _:
			assert_never(session.project)


async def run_for(
	session: Session,
	tasks: Mapping[str, TaskNode],
	config: Config | None,
	arguments: dict[str, Any],
) -> types.CallToolResult:
	"""Validate, resolve the node, then dry-run or execute in the server's event loop."""
	try:
		req = wire.RunRequest.model_validate(arguments)
	except ValidationError as e:
		return error_result(f"invalid camas_run arguments:\n{e}")
	if req.under is not None:
		return await run_budget(session, tasks, config, req, req.under)
	try:
		name, node = resolve_run_node(tasks, req)
	except ValueError as e:
		return error_result(str(e))
	if req.dry_run:
		return success(dry_run_text(node), to_plan_response(node), session.compat)
	result = await run(node, jobs=req.jobs, interactive=False)
	logs = write_logs(create_run_log_dir(session.camas_dir, name, session.reserve_run()), result)
	timings.record_run(session.camas_dir, result)
	resp = attach_logs(to_run_response(node, result, verbosity=req.verbosity), logs)
	return success(
		run_text(name, resp, logs), resp, session.compat, links=failing_log_links(resp, logs)
	)


def resolve_run_node(tasks: Mapping[str, TaskNode], req: wire.RunRequest) -> tuple[str, TaskNode]:
	"""The ``(name, node)`` to run: looked up by name, then matrix-overridden and
	passthrough-appended.

	Raises:
		ValueError: when ``req.task`` is omitted (only ``under`` may omit it), names no
			task, an override targets an unknown matrix axis, or passthrough args are
			applied to a non-leaf task.
	"""
	if req.task is None:
		raise ValueError("camas_run requires 'task' (or pass 'under' to budget the default task)")
	if req.task not in tasks:
		known = ", ".join(sorted(tasks)) or "none"
		raise ValueError(f"no task named {req.task!r} (known: {known})")
	node = tasks[req.task]
	if req.matrix_overrides:
		node = override_matrix(node, {k: tuple(v) for k, v in req.matrix_overrides.items()})
	if req.args:
		node = apply_passthrough(node, tuple(req.args))
	return req.task, node


async def run_budget(
	session: Session,
	tasks: Mapping[str, TaskNode],
	config: Config | None,
	req: wire.RunRequest,
	budget_s: float,
) -> types.CallToolResult:
	"""Handle ``camas_run`` with ``under``: budget the source task's leaves, then dry-run or execute."""
	if req.args:
		return error_result("camas_run: 'args' (passthrough) cannot be combined with 'under'")
	try:
		source = budget_source(tasks, config, req.task)
		if req.matrix_overrides:
			source = override_matrix(source, {k: tuple(v) for k, v in req.matrix_overrides.items()})
	except ValueError as e:
		return error_result(str(e))
	plan = plan_under(source, budget_s, timings.load(session.camas_dir))
	report = to_budget_report(plan)
	label = req.task or "default task"
	if plan.node is None:
		empty = wire.RunResponse(
			returncode=0, elapsed=0.0, passed=0, failed=0, skipped=0, interrupt_count=0, leaves=()
		)
		text = f"{budget_headline(report)}\n\nNothing ran — no leaf fit the budget."
		return success(text, attach_budget(empty, report), session.compat)
	if req.dry_run:
		resp = attach_budget(to_plan_response(plan.node), report)
		return success(
			f"{budget_headline(report)}\n\n{dry_run_text(plan.node)}", resp, session.compat
		)
	result = await run(plan.node, jobs=req.jobs, interactive=False)
	logs = write_logs(create_run_log_dir(session.camas_dir, label, session.reserve_run()), result)
	timings.record_run(session.camas_dir, result)
	resp = attach_budget(
		attach_logs(to_run_response(plan.node, result, verbosity=req.verbosity), logs), report
	)
	return success(
		f"{budget_headline(report)}\n\n{run_text(label, resp, logs)}",
		resp,
		session.compat,
		links=failing_log_links(resp, logs),
	)


def budget_source(
	tasks: Mapping[str, TaskNode], config: Config | None, task: str | None
) -> TaskNode:
	"""The task a budget run filters: the named task, else the project's default task.

	Raises:
		ValueError: when ``task`` names no task, or no task is named and no
			:class:`Config` default exists to budget.
	"""
	if task is not None:
		if task not in tasks:
			known = ", ".join(sorted(tasks)) or "none"
			raise ValueError(f"no task named {task!r} (known: {known})")
		return tasks[task]
	default = config.default_task if config is not None else None
	if default is None:
		raise ValueError("no task given and no Config default_task to budget; name a task to run")
	return default


def gate_source(tasks: Mapping[str, TaskNode], config: Config | None, task: str | None) -> TaskNode:
	"""The node the gate checks: the named task, else the agent's ``check`` node (or the default).

	Raises:
		ValueError: when ``task`` names no task, or no task is named and no check node
			(``Config.agent.check`` or ``default_task``) exists.
	"""
	if task is not None:
		if task not in tasks:
			known = ", ".join(sorted(tasks)) or "none"
			raise ValueError(f"no task named {task!r} (known: {known})")
		return tasks[task]
	check = config.gate_check(github=False) if config is not None else None
	if check is None:
		raise ValueError(
			"no task given and no check node (Config.agent.check or default_task) to gate"
		)
	return check


def to_budget_report(plan: BudgetPlan) -> wire.BudgetReport:
	"""The wire ``BudgetReport`` for a plan: the leaves that run (fitting + unmeasured) and the
	over-budget leaves that don't.
	"""
	return wire.BudgetReport(
		budget_s=plan.budget_s,
		selected=(
			*(task_label(f.task) for f in plan.fits),
			*(task_label(u.task) for u in plan.untimed),
		),
		unmeasured=tuple(task_label(u.task) for u in plan.untimed),
		excluded=tuple(
			wire.ExcludedLeaf(
				name=task_label(o.task), reason="over_budget", estimated_s=o.estimated_s
			)
			for o in plan.over_budget
		),
	)


def attach_budget(resp: wire.RunResponse, report: wire.BudgetReport) -> wire.RunResponse:
	"""Thread the budget partition into the run response's ``budget`` field."""
	return resp.model_copy(update={"budget": report})


def budget_headline(report: wire.BudgetReport) -> str:
	"""The load-bearing budget summary: leaves running (and which are unmeasured), and which
	were excluded as measured-over-budget.
	"""
	lines = [
		f"Time budget {report.budget_s:.2f}s — running {len(report.selected)} leaf(s) "
		f"({len(report.unmeasured)} unmeasured), excluded {len(report.excluded)} over budget."
	]
	if report.excluded:
		lines.append("  over budget: " + ", ".join(excluded_note(e) for e in report.excluded))
	if report.unmeasured:
		lines.append(
			"  unmeasured (running to record an estimate): " + ", ".join(report.unmeasured)
		)
	return "\n".join(lines)


def excluded_note(leaf: wire.ExcludedLeaf) -> str:
	return f"{leaf.name} ~{leaf.estimated_s:.2f}s" if leaf.estimated_s is not None else leaf.name


def create_run_log_dir(camas_dir: Path, task: str, seq: int) -> Path:
	"""Create and return ``camas_dir/runs/<task>/<seq>/`` (creating ``camas_dir`` if needed)."""
	timings.ensure_camas_dir(camas_dir)
	run_dir = camas_dir / "runs" / slug(task) / str(seq)
	run_dir.mkdir(parents=True, exist_ok=True)
	return run_dir


def write_logs(run_dir: Path, result: RunResult) -> tuple[Path | None, ...]:
	"""Write each leaf's full output to ``run_dir``, one entry per leaf in DFS order.

	An entry is ``None`` when that leaf produced no log — it was skipped, or ran silently.
	Positionally aligned with ``RunResponse.leaves`` so the two zip ``strict``.
	"""
	return tuple(write_leaf_log(run_dir, i, leaf) for i, leaf in enumerate(result.results))


def write_leaf_log(run_dir: Path, index: int, leaf: TaskResult) -> Path | None:
	match leaf.completion:
		case Finished(output=output) | Stopped(output=output):
			if not output:
				return None
			path = run_dir / f"{index:03d}_{slug(leaf.name)}.log"
			path.write_text(decode_log(output), encoding="utf-8")
			return path
		case Skipped():
			return None
		case _:
			assert_never(leaf.completion)


def failing_log_links(
	resp: wire.RunResponse, logs: tuple[Path | None, ...]
) -> tuple[types.ResourceLink, ...]:
	"""``resource_link``s to the on-disk logs of the leaves that did not pass."""
	return tuple(
		types.ResourceLink(
			type="resource_link",
			name=leaf.name,
			uri=AnyUrl(log.as_uri()),
			mimeType="text/plain",
			description=f"Full output of task {leaf.name!r}",
		)
		for leaf, log in zip(resp.leaves, logs, strict=True)
		if log is not None and leaf.completion.returncode != 0
	)


def list_text(resp: wire.ListResponse, *, expand_matrix: bool) -> str:
	"""The load-bearing agent-facing summary for ``camas_list``."""
	lines = [f"{len(resp.tasks)} task(s) defined. Call camas_run with one of these names."]
	if resp.default is not None:
		lines.append(f"default (developer's task): {resp.default}")
	if resp.github_default is not None:
		lines.append(
			f"github default (exactly what CI runs; run before committing/pushing): "
			f"{resp.github_default}. A bare `camas` runs it under GitHub Actions, so a CI "
			f"step is just `camas` — naming the task there is redundant."
		)
	lines.append("")
	width = max((len(t.name) for t in resp.tasks), default=0)
	for t in resp.tasks:
		marks = "".join(
			mark
			for mark, on in ((" [default]", t.is_default), (" [github]", t.is_github_default))
			if on
		)
		help_note = f"  {t.help}" if t.help else ""
		matrix = matrix_note(t.matrix_axes) if t.matrix_axes else ""
		timing = timing_note(t) if t.estimated_s is not None else ""
		lines.append(f"  {t.name.ljust(width)}{marks}{help_note}{matrix}{timing}")
		lines.append(f"      {t.command_preview}")
	if not expand_matrix and any(t.matrix_axes for t in resp.tasks):
		lines.append("")
		lines.append(
			"Matrix previews collapsed to templates; call camas_list with expand_matrix=true "
			"for the fully-expanded per-leaf tree."
		)
	return "\n".join(lines)


def matrix_note(axes: dict[str, list[str]]) -> str:
	rendered = ", ".join(f"{name}={'/'.join(values)}" for name, values in axes.items())
	return f"  (matrix: {rendered})"


def timing_note(task: wire.TaskInfo) -> str:
	"""The estimated-duration annotation: ``~Ns``, the slowest leaf, and the sample count."""
	slowest = (
		f", slowest {task.slowest_leaf} {task.slowest_s:.2f}s"
		if task.slowest_leaf != task.name
		else ""
	)
	return f"  [~{task.estimated_s:.2f}s{slowest}, n={task.samples}]"


def run_text(task: str, resp: wire.RunResponse, logs: tuple[Path | None, ...]) -> str:
	"""The load-bearing agent-facing summary for ``camas_run`` — failures carry their log path."""
	verdict = "PASSED" if resp.returncode == 0 else "FAILED"
	lines = [
		f"camas_run {task!r}: {verdict} (returncode={resp.returncode}) in {resp.elapsed:.2f}s — "
		f"{resp.passed} passed, {resp.failed} failed, {resp.skipped} skipped",
		"",
	]
	for leaf, log in zip(resp.leaves, logs, strict=True):
		lines.extend(leaf_lines(leaf, log))
	return "\n".join(lines)


def dry_run_text(node: TaskNode) -> str:
	"""The fully-resolved (post-matrix) plan for ``dry_run=true`` — nothing is executed."""
	plan = "\n".join(render_tree_lines(node, show_cmd=True, color=False))
	return f"Dry run — fully-resolved plan, nothing executed:\n{plan}"


def check_text(resp: wire.CheckResponse) -> str:
	"""The load-bearing agent-facing summary for ``camas_check`` — verdict, then diagnostics."""
	match resp.status:
		case "ok":
			how = (
				f"type-checked clean with {resp.checker}"
				if resp.checker is not None
				else "no type checker ran"
			)
			return f"camas_check: OK — {resp.source} loads ({resp.task_count} task(s); {how})."
		case "type_error":
			return (
				f"camas_check: TYPE ERRORS — {resp.source} loads ({resp.task_count} task(s)) but "
				f"{resp.checker} reported:\n\n{resp.diagnostics}"
			)
		case "load_error":
			return (
				f"camas_check: LOAD ERROR — {resp.source} failed to evaluate:\n\n{resp.diagnostics}"
			)
		case "no_checker":
			return (
				f"camas_check: {resp.source} loads ({resp.task_count} task(s)), but no type "
				f"checker is available.\n{resp.diagnostics}"
			)
		case "no_tasks":
			return (
				"camas_check: no tasks.py or [tool.camas.tasks] found in this project. "
				"Run `camas --init` to scaffold a commented starter, or call camas_docs to author one."
			)
		case _:
			assert_never(resp.status)


def docs_text(resp: wire.DocsResponse) -> str:
	"""The load-bearing agent-facing summary for ``camas_docs`` — source pointer, then tutorial."""
	return (
		"camas authoring guide. The API source of truth is the installed camas package at:\n"
		f"  {resp.source}\n"
		"Read its v0/ submodules for exact Task/Sequential/Parallel/Config signatures and the "
		"examples/ directory for full project layouts; validate your tasks.py with camas_check."
		f"\n\n{resp.tutorial}"
	)


def success(
	text: str,
	model: wire.ListResponse
	| wire.RunResponse
	| wire.CheckResponse
	| wire.DocsResponse
	| wire.GateResponse,
	compat: Compat,
	*,
	links: tuple[types.ResourceLink, ...] = (),
) -> types.CallToolResult:
	"""A non-error result: the load-bearing ``TextContent`` plus gated ``structuredContent``."""
	return types.CallToolResult(
		content=[types.TextContent(type="text", text=text), *links],
		structuredContent=model.model_dump(mode="json") if compat.emit_structured else None,
		isError=False,
	)


def attach_logs(resp: wire.RunResponse, logs: tuple[Path | None, ...]) -> wire.RunResponse:
	"""Thread each written log path into its ``LeafReport.log`` for ``structuredContent``."""
	return resp.model_copy(
		update={
			"leaves": tuple(
				leaf.model_copy(update={"log": str(log)}) if log is not None else leaf
				for leaf, log in zip(resp.leaves, logs, strict=True)
			)
		}
	)


def error_result(text: str) -> types.CallToolResult:
	"""A tool-execution error (``isError=true``) with self-correcting text for the agent."""
	return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=True)


def resolve_project(base: Path) -> TasksState:
	"""Walk up from ``base`` for a ``tasks.py`` or ``[tool.camas.tasks]`` pyproject.

	Never exits or prints (unlike the CLI's resolver): a broken tasks file becomes a
	:class:`LoadErr` the handlers turn into a tool error.
	"""
	for candidate in (base, *base.parents):
		tasks_py = candidate / "tasks.py"
		if tasks_py.is_file():
			try:
				return load_tasks_from_py(tasks_py)
			except Exception as e:
				return LoadErr(source=tasks_py, exception=e)
		pyproject = candidate / "pyproject.toml"
		if pyproject.is_file():
			try:
				loaded = load_tasks(pyproject)
			except ValueError as e:
				return LoadErr(source=pyproject, exception=e)
			if loaded.tasks:
				return loaded
	return EMPTY_STATE


def resolve_project_quiet(base: Path) -> TasksState:
	"""Resolve the project with stdout redirected to stderr.

	``tasks.py`` runs as Python via ``runpy``; a module-level ``print`` there would
	otherwise land on the stdout that the JSON-RPC transport owns and corrupt the stream.
	"""
	with redirect_stdout(sys.stderr):
		return resolve_project(base)


def project_base() -> Path:
	"""The absolute project root: ``CLAUDE_PROJECT_DIR`` when Claude Code sets it, else cwd."""
	return (Path(env) if (env := os.environ.get("CLAUDE_PROJECT_DIR")) else Path.cwd()).resolve()


def leaf_lines(leaf: wire.LeafReport, log: Path | None) -> list[str]:
	"""The summary lines for one leaf: a status header, any output excerpt, and its log path."""
	match leaf.completion:
		case wire.Finished(returncode=0, elapsed=el, output=output):
			return [f"ok     {leaf.name} ({el:.2f}s)", *indent_lines(output)]
		case wire.Finished(returncode=rc, elapsed=el, output=output):
			head = f"FAIL   {leaf.name} (exit {rc}, {el:.2f}s)"
			return [head, *indent_lines(output), *tail_lines(leaf.truncated, log)]
		case wire.Stopped(returncode=rc, elapsed=el, output=output):
			head = f"STOP   {leaf.name} (interrupted, exit {rc}, {el:.2f}s)"
			return [head, *indent_lines(output), *tail_lines(leaf.truncated, log)]
		case wire.Skipped(blocked_by=blocked_by):
			blame = f" — blocked by {blocked_by!r}" if blocked_by is not None else ""
			return [f"SKIP   {leaf.name}{blame}"]
		case _:
			assert_never(leaf.completion)


def indent_lines(output: list[str]) -> list[str]:
	return [f"    {line}" for line in output]


def tail_lines(truncated: bool, log: Path | None) -> list[str]:
	"""The trailing lines under a failed leaf: a truncation marker then the full-log path."""
	lines = ["    … earlier output truncated; see full log"] if truncated else []
	if log is not None:
		lines.append(f"    full log: {log}")
	return lines


def slug(name: str) -> str:
	return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:80]


def decode_log(output: Sequence[bytes]) -> str:
	return strip_ansi("".join(chunk.decode("utf-8", errors="replace") for chunk in output))


async def gate_call(session: Session, arguments: dict[str, Any]) -> types.CallToolResult:
	"""Handle ``camas_gate``: scope to the changed paths, autofix, run the checks, classify."""
	match session.project:
		case LoadErr(source=source, exception=exception):
			return error_result(format_load_error_hint(source, exception))
		case LoadOk(tasks=tasks, config=config):
			return await gate_for(session, tasks, config, arguments)
		case _:
			assert_never(session.project)


async def gate_for(
	session: Session,
	tasks: Mapping[str, TaskNode],
	config: Config | None,
	arguments: dict[str, Any],
) -> types.CallToolResult:
	"""Validate, resolve the gated task, run the gate, and build the verdict."""
	try:
		req = wire.GateRequest.model_validate(arguments)
	except ValidationError as e:
		return error_result(f"invalid camas_gate arguments:\n{e}")
	try:
		node = gate_source(tasks, config, req.task)
	except ValueError as e:
		return error_result(str(e))
	changed = to_changed(req.paths, session.base)
	outcome = await run_gate(
		node,
		changed,
		under=req.under,
		jobs=req.jobs,
		timings=timings.load(session.camas_dir),
	)
	budget = to_budget_report(outcome.budget) if outcome.budget is not None else None
	rerun = wire.GateRerun(task=req.task, paths=changed, under=req.under)
	resp = to_gate_response(outcome, budget, rerun)
	return success(gate_text(resp), resp, session.compat)


def gate_text(resp: wire.GateResponse) -> str:
	"""The load-bearing agent-facing summary for ``camas_gate`` — verdict, budget, residual."""
	headline = (
		"CONTINUE — checks green; no residual needs reasoning"
		if resp.decision == "continue"
		else "BLOCK — a residual needs reasoning"
	)
	lines: list[str] = [f"camas_gate: {headline} (residual_class={resp.residual_class})"]
	if resp.budget is not None:
		lines.extend(["", budget_headline(resp.budget)])
	if resp.diagnostics is not None:
		lines.extend(["", "Residual (failing checks):"])
		for env in resp.diagnostics:
			lines.append(f"  {env.name} (exit {env.exit_code}, {env.output_kind})")
			lines.extend(f"    {line}" for line in env.payload.splitlines())
			if env.truncated:
				lines.append("    … earlier output truncated")
	if resp.decision == "block":
		lines.extend(["", f"Re-gate this scope: {rerun_command(resp.rerun)}"])
	return "\n".join(lines)


def rerun_command(rerun: wire.GateRerun) -> str:
	"""The ``camas mcp gate`` invocation that reproduces a verdict — the handle a higher tier
	or the main agent re-issues against the same scope.
	"""
	task = (rerun.task,) if rerun.task is not None else ()
	paths = tuple(arg for p in rerun.paths for arg in ("--paths", p))
	under = ("--under", str(rerun.under)) if rerun.under is not None else ()
	return shlex.join(("camas", "mcp", "gate", *task, *paths, *under))


class GateArgs(NamedTuple):
	"""Parsed ``camas mcp gate`` arguments."""

	task: str | None
	paths: tuple[str, ...]
	under: float | None
	jobs: int | None


def parse_gate_args(argv: list[str]) -> GateArgs:
	"""Parse ``camas mcp gate [task] [--paths P]… [--under N] [--jobs N]``."""
	parser = argparse.ArgumentParser(
		prog="camas mcp gate", description="Run the gate once, headless."
	)
	parser.add_argument(
		"task", nargs="?", default=None, help="task to gate; omit for the check node"
	)
	parser.add_argument(
		"--paths",
		action="append",
		default=[],
		metavar="PATH",
		help="changed path to scope to (repeatable)",
	)
	parser.add_argument(
		"--under", type=float, default=None, metavar="SECONDS", help="wall-clock budget"
	)
	parser.add_argument("--jobs", type=int, default=None, metavar="N", help="max concurrent leaves")
	ns = parser.parse_args(argv)
	return GateArgs(task=ns.task, paths=tuple(ns.paths), under=ns.under, jobs=ns.jobs)


def _event_get(obj: object, key: str) -> object:
	"""``obj[key]`` for a parsed-JSON object, else ``None`` — narrows JSON's ``Any`` to ``object``."""
	return cast("dict[str, object]", obj).get(key) if isinstance(obj, dict) else None


def changed_from_stdin() -> tuple[str, ...]:
	"""The edited files in a ``PostToolBatch`` event piped on stdin (the command-hook delivery),
	de-duplicated in order; ``()`` when stdin is a tty, empty, or not such an event.
	"""
	if sys.stdin.isatty():
		return ()
	raw = sys.stdin.read().strip()
	if not raw:
		return ()
	try:
		event: object = json.loads(raw)
	except json.JSONDecodeError:
		return ()
	calls = _event_get(event, "tool_calls")
	if not isinstance(calls, list):
		return ()
	edited = (
		_event_get(_event_get(call, "tool_input"), key)
		for call in cast("list[object]", calls)
		for key in ("file_path", "path", "notebook_path")
	)
	return tuple(dict.fromkeys(f for f in edited if isinstance(f, str)))


def gate_cli(argv: list[str]) -> int:
	"""Run the gate once, headless: scope this project's checks to the changed paths (``--paths``,
	else the files in a ``PostToolBatch`` event on stdin), print the ``GateResponse`` as JSON to
	stdout, and exit ``0`` (continue) / ``2`` (block) — on a block the agent-facing summary goes
	to stderr. The process-isolated, machine-readable gate entry: the ``PostToolBatch`` command
	hook, parallel fixers, and the benchmark all drive it.
	"""
	args = parse_gate_args(argv)
	base = project_base()
	state = resolve_project_quiet(base)
	match state:
		case LoadErr(source=source, exception=exception):
			print(f"camas mcp gate: cannot load {source}: {exception}", file=sys.stderr)
			return 2
		case LoadOk(tasks=tasks, config=config):
			return run_gate_cli(args, base, tasks, config)
		case _:
			assert_never(state)


def run_gate_cli(
	args: GateArgs, base: Path, tasks: Mapping[str, TaskNode], config: Config | None
) -> int:
	"""Resolve the check node, run the gate over the changed paths, emit the verdict."""
	try:
		node = gate_source(tasks, config, args.task)
	except ValueError as e:
		print(f"camas mcp gate: {e}", file=sys.stderr)
		return 2
	changed = to_changed(args.paths or changed_from_stdin(), base)
	camas_dir = (config if config is not None else Config()).camas_path(base)
	outcome = asyncio.run(
		run_gate(node, changed, under=args.under, jobs=args.jobs, timings=timings.load(camas_dir))
	)
	budget = to_budget_report(outcome.budget) if outcome.budget is not None else None
	rerun = wire.GateRerun(task=args.task, paths=changed, under=args.under)
	resp = to_gate_response(outcome, budget, rerun)
	print(resp.model_dump_json(indent=2))
	if resp.decision == "block":
		print(gate_text(resp), file=sys.stderr)
		return 2
	return 0
