# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import shlex
import shutil
import sys
import tempfile
import textwrap
import time
from contextlib import redirect_stdout, suppress
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple, TypeAlias

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.server import NotificationOptions
from pydantic import AnyUrl, BaseModel, ValidationError

from camas.paths import camas_package_dir

from ..core import timings
from ..core.budget import (
	BudgetCensus,
	BudgetRun,
	NothingToRun,
	census_of,
	outcome_lines,
	plan_under,
	resolve_budget,
	summary_lines,
)
from ..core.execution import run
from ..core.gate import GateOutcome, run_gate, settle
from ..core.hook_event import (
	NO_EVENT,
	STALE_TEMP_MAX_AGE_S,
	HookEvent,
	await_settled,
	event_from_stdin,
)
from ..core.jobs import JOBS_UNIT
from ..core.matrix import (
	empty_variant_labels,
	expand_matrix,
	override_matrix,
	unfilled_required_axes,
)
from ..core.render import render_tree_lines, strip_ansi
from ..core.scope import coverage_message, requested_but_unusable, scope_to_changed, to_changed
from ..core.task import did_you_mean, task_label
from ..main.argv import apply_passthrough
from ..main.compose import load_py_tasks_state
from ..main.format import (
	format_empty_variants_error,
	format_load_error_hint,
	format_version_skew_hint,
)
from ..main.github_matrix import format_matrix_json
from ..main.init import create_starter_tasks_py, starter_text
from ..main.parser import parse_duration, positive_jobs
from ..main.pep723 import camas_requirement_from, version_specifier
from ..main.state import EMPTY_STATE, LoadErr, LoadOk, TasksState
from ..main.tasks import load_tasks
from ..v0.completion import Errored, Finished, Skipped, Stopped
from ..v0.config import Config
from ..v0.task import Task
from . import wire
from .catalog import to_list_response
from .docs import to_docs_response
from .result import (
	has_failing_leaf_without_agent_format,
	to_check_response,
	to_gate_response,
	to_github_matrix_response,
	to_plan_response,
	to_run_response,
)

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

	from ..core.budget import BudgetSummary, OverBudget
	from ..core.completion import RunResult, TaskResult
	from ..v0.task import TaskNode


class ToolName(Enum):
	LIST = "camas_list"
	RUN = "camas_run"
	CHECK = "camas_check"
	DOCS = "camas_docs"
	GATE = "camas_gate"
	FIX = "camas_fix"
	INIT = "camas_init"
	GITHUB_MATRIX = "camas_github_matrix"


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
GATE_ANNOTATIONS: Final = types.ToolAnnotations(
	readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True
)
FIX_ANNOTATIONS: Final = types.ToolAnnotations(
	readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True
)
INIT_ANNOTATIONS: Final = types.ToolAnnotations(
	readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
GITHUB_MATRIX_ANNOTATIONS: Final = types.ToolAnnotations(readOnlyHint=True, openWorldHint=False)
RUN_LOG_KEEP: Final = 10
NO_RUN_DEFAULT_MSG: Final = (
	"camas_run has no task to run: name a task, or set a project default in tasks.py — "
	"Config(default_task=…), Config(github_task=…), or "
	"Config(agent=Claude(fix=…, default=…))."
)


class Compat(NamedTuple):
	emit_structured: bool = True


@dataclass(slots=True)
class Session:
	project: TasksState
	base: Path
	compat: Compat
	runs: int = 0
	version_warning: str | None = field(default=None, init=False)

	def __post_init__(self) -> None:
		self.version_warning = check_version_pin(self.project)

	@property
	def camas_dir(self) -> Path:
		config = self.project.config if isinstance(self.project, LoadOk) else None
		return (config if config is not None else Config()).camas_path(self.base)

	def refresh(self) -> None:
		self.project = resolve_project_quiet(self.base)
		self.version_warning = check_version_pin(self.project)

	def reserve_run(self) -> int:
		self.runs += 1
		return self.runs


def project_source(state: TasksState) -> Path | None:
	match state:
		case LoadOk(source=source) | LoadErr(source=source):
			return source
		case _:
			assert_never(state)


def base_from(source: Path | None, fallback: Path) -> Path:
	return source.parent if source is not None else fallback


def base_for(session: Session) -> Path:
	return base_from(project_source(session.project), session.base)


class VersionSkew(NamedTuple):
	running: str
	spec: str


def check_version_pin(state: TasksState) -> str | None:
	skew = version_skew(state)
	if skew is None:
		return None
	return (
		f"camas {skew.running} does not satisfy tasks.py pin {skew.spec}; "
		f"re-run `camas mcp init --claude` after bumping"
	)


def version_skew(state: TasksState) -> VersionSkew | None:
	source = project_source(state)
	if source is None:
		return None
	req = camas_requirement_from(source)
	if req is None:
		return None
	spec = version_specifier(req)
	if spec is None:
		return None
	running = running_version()
	if running is None:
		return None
	if version_satisfies(running, spec):
		return None
	return VersionSkew(running=running, spec=spec)


def running_version() -> str | None:
	try:
		return version("camas")
	except PackageNotFoundError:
		return None


def version_satisfies(running: str, specifier: str) -> bool:
	for op in ("==", ">=", "<=", ">", "<"):
		if specifier.startswith(op):
			pinned = specifier[len(op) :].strip()
			try:
				return _op(op, _parse(running), _parse(pinned))
			except ValueError:
				return True  # can't parse — skip
	return True  # unrecognized specifier — skip


def _parse(version: str) -> tuple[int, ...]:
	for sep in ("+", "a", "b", "rc", "dev", "post"):
		idx = version.find(sep)
		if idx >= 0:
			version = version[:idx]
	return tuple(int(p) for p in version.strip().split(".") if p)


def _op(operator: str, a: tuple[int, ...], b: tuple[int, ...]) -> bool:
	n = max(len(a), len(b))
	a = a + (0,) * (n - len(a))
	b = b + (0,) * (n - len(b))
	if operator == "==":
		return a == b
	if operator == ">=":
		return a >= b
	if operator == "<=":
		return a <= b
	if operator == ">":
		return a > b
	return a < b


Snapshot: TypeAlias = dict[str, str]


def package_snapshot() -> Snapshot:
	root = camas_package_dir()
	snapshot: Snapshot = {}
	for dirpath, _, filenames in os.walk(root):
		for name in filenames:
			if not name.endswith((".py", ".so", ".pyd")):
				continue
			path = Path(dirpath) / name
			with suppress(FileNotFoundError):
				snapshot[path.relative_to(root).as_posix()] = hashlib.sha256(
					path.read_bytes()
				).hexdigest()
	return snapshot


RELOAD_EXIT_DELAY: Final = 1.0


def exit_for_reload() -> None:  # pragma: no cover  # the suite cannot survive its own process exit
	os._exit(0)


def serve_stdio(argv: list[str]) -> None:  # pragma: no cover
	os.environ["NO_COLOR"] = "1"
	base = project_base()
	session = Session(
		resolve_project_quiet(base), base, Compat(emit_structured="--plain" not in argv)
	)
	if session.version_warning is not None:
		print(session.version_warning, file=sys.stderr)
	asyncio.run(run_over_stdio(build_server(session)))


async def run_over_stdio(server: Server[object]) -> None:  # pragma: no cover
	from mcp.server.stdio import stdio_server

	options = server.create_initialization_options(NotificationOptions(tools_changed=True))
	async with stdio_server() as (read, write):
		await server.run(read, write, options)


def build_server(session: Session) -> Server[object]:
	initial = package_snapshot()
	probe_task: asyncio.Task[None] | None = None
	active_calls = 0

	def retrieve_probe_exception(task: asyncio.Task[None]) -> None:
		if not task.cancelled():
			exception = (
				task.exception()
			)  # retrieve a dead probe's failure — no unretrieved log, even idle
			if (
				exception is not None
			):  # pragma: no cover  # the suppressed surfaces cover the tested paths
				print(f"camas mcp: reload probe failed: {exception!r}", file=sys.stderr)

	async def probe_and_exit() -> None:
		await asyncio.sleep(RELOAD_EXIT_DELAY)
		if active_calls:
			return
		changed = False
		with suppress(OSError, RuntimeError):
			changed = (await asyncio.to_thread(package_snapshot)) != initial
		if changed and not active_calls:
			exit_for_reload()

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
			camas_init (the MCP mirror of `camas --init`) before authoring one by hand.
		""").strip(),
	)

	async def list_handler() -> list[types.Tool]:
		return list(tools(task_names(session.project), session.compat))

	async def call_handler(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
		nonlocal probe_task, active_calls
		try:
			active_calls += 1
			before = task_names(session.project)
			session.refresh()
			result = await call(session, name, arguments)
			if task_names(session.project) != before:
				await server.request_context.session.send_tool_list_changed()
		finally:
			active_calls -= 1
			if probe_task is not None:
				probe_task.cancel()
			probe_task = asyncio.get_running_loop().create_task(probe_and_exit())
			probe_task.add_done_callback(retrieve_probe_exception)
		return result

	server.list_tools()(list_handler)  # type: ignore[no-untyped-call]
	server.call_tool(validate_input=False)(call_handler)
	return server


async def call(session: Session, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
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
		case ToolName.FIX:
			return await fix_call(session, arguments)
		case ToolName.INIT:
			return init_call(session, arguments)
		case ToolName.GITHUB_MATRIX:
			return github_matrix_call(session, arguments)
		case _:
			known = ", ".join(repr(tool.value) for tool in ToolName)
			return error_result(f"unknown tool {name!r}; expected one of: {known}")


def parse_tool(name: str) -> ToolName | None:
	return next((tool for tool in ToolName if tool.value == name), None)


def task_names(project: TasksState) -> tuple[str, ...]:
	return tuple(sorted(project.tasks)) if isinstance(project, LoadOk) else ()


class Tools(NamedTuple):
	discovery: types.Tool
	execution: types.Tool
	validation: types.Tool
	docs: types.Tool
	gate: types.Tool
	fix: types.Tool
	init: types.Tool
	github_matrix: types.Tool


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
				working) and the github default — the exact task this project's CI runs,
				declared via Config(github_task=...) (camas is the single definition shared by
				local and CI), so running it locally before you commit or push reproduces CI
				exactly; github_default is null when the project declares no github_task (camas
				never infers it from CI workflow files). run_default names the task a no-task
				camas_run resolves (agent default, else check, else the github or default
				task). Tasks whose leaves have been timed also
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
				read-only rest in parallel (a pipe kept whole for its untimed siblings runs its
				over-budget stages too); omit task to budget the project default. For an ad-hoc
				scoped run of any task, pass paths=[…] (changed files, like the CLI --paths): each
				{paths} command is narrowed to the files it covers, and it combines with under
				(scope first, then budget). Compact failures-first summary by default; dry_run=true
				previews the fully-resolved plan without executing. Omit task entirely to run the
				project's configured default. A non-zero result means a task failed — a normal
				result, not a tool error.
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
				never drifts: the Task / Sequential / Parallel / Pipe / Config API, matrix expansion,
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
				The SA-delegation gate: first run the project's registered deterministic autofix
				(Config.agent.fix — formatters, --fix linters) over the files just changed, then scope
				THIS project's checks to them, run them, and return a binary verdict, so a residual is
				never one the autofix settles. Re-read a file before editing it after a gate: the autofix
				may have rewritten it. residual_class is
				'green' (decision 'continue') when the checks pass, or 'needs_reasoning' (decision
				'block') when a check still fails — then diagnostics carries the failing leaves. Pass
				paths=[…] (the changed files) to scope; omit to gate the whole check node, which also
				runs the autofix over the whole tree. under=<seconds>
				time-boxes the checks: leaves measured to exceed it are skipped (except the over-budget
				stages of a pipe kept whole for its untimed siblings), untimed leaves run.
			""").strip(),
			input_schema=wire.gate_input_schema(task_names),
			output_model=wire.GateResponse,
			title="SA-delegation gate",
			annotations=GATE_ANNOTATIONS,
		),
		fix=tool_def(
			compat,
			name=ToolName.FIX.value,
			description=textwrap.dedent("""\
				Run the project's registered deterministic autofix node (``Config.agent.fix``)
				scoped to the changed paths — mutating, behavior-preserving formatters and
				``--fix`` linters. It does not reason; it only fixes. Pass paths=[…] (the changed
				files) to scope; omit to run the whole fix node un-scoped. Each ``{paths}`` command
				is injected with the files it covers; a command without ``{paths}`` always runs
				unless its ``when=`` excludes the changed set.
				A no-op (exit 0, no leaves) when no fix node is registered (``Config.agent.fix`` is
				``None``).
			""").strip()
			+ f" ``jobs`` caps concurrently running leaves ({JOBS_UNIT}).",
			input_schema=wire.fix_input_schema(task_names),
			output_model=wire.RunResponse,
			title="Run deterministic autofix",
			annotations=FIX_ANNOTATIONS,
		),
		init=tool_def(
			compat,
			name=ToolName.INIT.value,
			description=textwrap.dedent("""\
				Scaffold a commented starter tasks.py in THIS project's root when it has none — the
				MCP mirror of `camas --init`, for driving camas purely over the MCP. Defaults
				(verbose=true) to the kitchen-sink template: every Task/Sequential/Parallel/Pipe/Config
				option worked and explained in place — path scoping ({paths}/when=), matrix
				expansion, agent_format structured output, and Config(agent=Claude(fix=...,
				check=..., default=...)) — with cross-platform placeholder commands, so you can read
				it once and see the whole authoring surface. Pass verbose=false for the same short
				template `camas --init` (no --verbose) writes instead. Either way this creates the
				.camas/ run-log and timing directory beside it, and returns the path plus the file's
				content so you can edit from there. Refuses to overwrite an existing tasks.py
				(returns status 'exists', file untouched). The tasks it creates are immediately
				runnable via camas_run — no restart. Then read camas_docs and validate edits with
				camas_check.
			""").strip(),
			input_schema=wire.InitRequest.model_json_schema(),
			output_model=wire.InitResponse,
			title="Scaffold a starter tasks.py",
			annotations=INIT_ANNOTATIONS,
		),
		github_matrix=tool_def(
			compat,
			name=ToolName.GITHUB_MATRIX.value,
			description=textwrap.dedent("""\
				Emit a task's fan-out as GitHub Actions strategy.matrix JSON — the MCP mirror of
				`camas --github-matrix`, for authoring or updating a CI workflow. Three shapes, from
				how tasks.py fans out: 'axes' projects a matrix= cross product as object-of-arrays,
				'variants' enumerates coupled variants= cells as GHA include: entries, and 'tasks'
				emits one job per child of a Parallel ({"task": [...]}, run as
				`camas ${{ matrix.task }}`) — the common CI shape with no version matrix at all.
				Omit shape to derive it; the response echoes the shape that was emitted plus a
				job_command template, and pinning that shape in the workflow makes a later tasks.py
				refactor fail loudly instead of silently reshaping the matrix. Pass task=<name>;
				omit to use the project default. Pin axes with matrix_overrides (like camas_run),
				e.g. {"PY": ["3.13"]} — a variants= key filters to the bundles binding it. Every
				emission is verified against the real run-set: each job must run exactly its own
				cell's leaves, and the jobs together every leaf exactly once, so a fan-out with no
				faithful projection (independent fan-outs in one tree, a plain leaf beside matrixed
				siblings) is a tool error, not a run failure. Read-only; runs nothing.
			""").strip(),
			input_schema=wire.github_matrix_input_schema(task_names),
			output_model=wire.GithubMatrixResponse,
			title="Emit GitHub Actions matrix",
			annotations=GITHUB_MATRIX_ANNOTATIONS,
		),
	)


def list_call(session: Session, arguments: dict[str, Any]) -> types.CallToolResult:
	try:
		req = wire.ListRequest.model_validate(arguments)
	except ValidationError as e:
		return error_result(f"invalid camas_list arguments:\n{e}")
	match session.project:
		case LoadErr(source=source, exception=exception):
			return error_result(load_error_hint(session, source, exception))
		case LoadOk(tasks=tasks, config=config):
			resp = to_list_response(
				tasks, config, timings.load(session.camas_dir), expand=req.expand_matrix
			)
			return success(
				with_warning(session, list_text(resp, expand_matrix=req.expand_matrix)),
				resp,
				session.compat,
			)
		case _:
			assert_never(session.project)


def check_call(session: Session) -> types.CallToolResult:
	resp = to_check_response(session.project).model_copy(
		update={"server_version": running_version()}
	)
	return success(with_warning(session, check_text(resp)), resp, session.compat)


def docs_call(session: Session) -> types.CallToolResult:
	resp = to_docs_response()
	return success(with_warning(session, docs_text(resp)), resp, session.compat)


def init_call(session: Session, arguments: dict[str, Any]) -> types.CallToolResult:
	try:
		req = wire.InitRequest.model_validate(arguments)
	except ValidationError as e:
		return error_result(f"invalid camas_init arguments:\n{e}")
	try:
		created = create_starter_tasks_py(session.base, verbose=req.verbose)
	except FileExistsError:
		resp = wire.InitResponse(status="exists", path=str(session.base / "tasks.py"))
		return success(init_text(resp), resp, session.compat)
	except OSError as e:
		return error_result(f"camas_init: could not scaffold tasks.py: {e}")
	session.refresh()
	resp = wire.InitResponse(
		status="created", path=str(created), content=starter_text(verbose=req.verbose)
	)
	return success(init_text(resp), resp, session.compat)


async def run_call(session: Session, arguments: dict[str, Any]) -> types.CallToolResult:
	match session.project:
		case LoadErr(source=source, exception=exception):
			return error_result(load_error_hint(session, source, exception))
		case LoadOk(tasks=tasks, config=config):
			return await run_for(session, tasks, config, arguments)
		case _:
			assert_never(session.project)


def leaf_color_of(config: Config | None) -> bool:
	return (config if config is not None else Config()).leaf_color


async def run_for(
	session: Session,
	tasks: Mapping[str, TaskNode],
	config: Config | None,
	arguments: dict[str, Any],
) -> types.CallToolResult:
	try:
		req = wire.RunRequest.model_validate(arguments)
	except ValidationError as e:
		return error_result(f"invalid camas_run arguments:\n{e}")
	if req.under is not None:
		return await run_budget(session, tasks, config, req, req.under)
	try:
		name, node = resolve_run_node(tasks, req, config)
	except ValueError as e:
		return error_result(str(e))
	changed = to_changed(req.paths, base_for(session))
	expanded = expand_matrix(node)
	if requested_but_unusable(req.paths, changed):
		return nothing_covered_result(session, req.paths)
	observed = timings.observed(session.camas_dir, expanded, changed)
	scoped = observed.node
	if scoped is None:
		return nothing_covered_result(session, req.paths)
	node = apply_passthrough(scoped, tuple(req.args)) if req.args else scoped
	if req.dry_run:
		return success(
			with_warning(session, dry_run_text(node)), to_plan_response(node), session.compat
		)
	result = await run(
		node,
		jobs=req.jobs,
		interactive=False,
		base=base_for(session),
		leaf_color=leaf_color_of(config),
		identities=observed.identities,
	)
	logs = write_logs(create_run_log_dir(session.camas_dir, name, session.reserve_run()), result)
	observed.record(result)
	resp = attach_logs(to_run_response(node, result, verbosity=req.verbosity), logs)
	nudge = improve_loop_nudge(
		any_truncated=resp.truncated,
		any_failing_without_agent_format=has_failing_leaf_without_agent_format(node, result),
	)
	return success(
		with_warning(session, run_text(name, resp, logs) + nudge),
		resp,
		session.compat,
		links=failing_log_links(resp, logs),
	)


def record_gate(camas_dir: Path | None, outcome: GateOutcome) -> None:
	if outcome.result is not None:
		timings.record_observed(camas_dir, timings.leaves_of(outcome.result))


def empty_run_response() -> wire.RunResponse:
	return wire.RunResponse(
		returncode=0, elapsed=0.0, passed=0, failed=0, skipped=0, interrupt_count=0, leaves=()
	)


def nothing_covered_result(session: Session, paths: list[str]) -> types.CallToolResult:
	return success(
		with_warning(session, coverage_message(paths)), empty_run_response(), session.compat
	)


def require_task(tasks: Mapping[str, TaskNode], name: str) -> TaskNode:
	if name not in tasks:
		known = ", ".join(sorted(tasks)) or "none"
		raise ValueError(f"no task named {name!r}{did_you_mean(name, tasks)} (known: {known})")
	return tasks[name]


def required_axes_message(names: tuple[str, ...]) -> str:
	plural = len(names) > 1
	overrides = ", ".join(f'"{n}": ["VALUE"]' for n in names)
	return (
		f"matrix {'axes' if plural else 'axis'} {', '.join(repr(n) for n in names)} "
		f"{'are' if plural else 'is'} required but unset — supply matrix_overrides={{{overrides}}}"
	)


def unsatisfiable_message(node: TaskNode) -> str | None:
	required = unfilled_required_axes(node)
	if required:
		return required_axes_message(required)
	empty = empty_variant_labels(node)
	if empty:
		return format_empty_variants_error(empty)
	return None


def require_filled_axes(node: TaskNode) -> None:
	blocked = unsatisfiable_message(node)
	if blocked is not None:
		raise ValueError(blocked)


def resolve_run_node(
	tasks: Mapping[str, TaskNode], req: wire.RunRequest, config: Config | None = None
) -> tuple[str, TaskNode]:
	if req.task is None:
		default = config.run_default() if config is not None else None
		if default is None:
			raise ValueError(NO_RUN_DEFAULT_MSG)
		name, node = default.name or "default task", default
	else:
		name, node = req.task, require_task(tasks, req.task)
	if req.matrix_overrides:
		node = override_matrix(node, {k: tuple(v) for k, v in req.matrix_overrides.items()})
	if req.args and not isinstance(node, Task):
		raise ValueError(f"pass-through args (--) only apply to Task, got {type(node).__name__}")
	require_filled_axes(node)
	return name, node


async def run_budget(
	session: Session,
	tasks: Mapping[str, TaskNode],
	config: Config | None,
	req: wire.RunRequest,
	budget_s: float,
) -> types.CallToolResult:
	if req.args:
		return error_result("camas_run: 'args' (passthrough) cannot be combined with 'under'")
	try:
		source = budget_source(tasks, config, req.task)
		label = req.task or source.name or "default task"
		if req.matrix_overrides:
			source = override_matrix(source, {k: tuple(v) for k, v in req.matrix_overrides.items()})
		require_filled_axes(source)
	except ValueError as e:
		return error_result(str(e))
	changed = to_changed(req.paths, base_for(session))
	if requested_but_unusable(req.paths, changed):
		return nothing_covered_result(session, req.paths)
	# Budget before scoping, as the gate does: an estimate is keyed by the label a leaf carries
	# before its {paths} are injected, so budgeting the scoped tree looks up a label nothing records.
	plan = plan_under(
		expand_matrix(source),
		budget_s,
		timings.load(session.camas_dir),
		timings.scope_of(changed),
	)
	outcome: Final = resolve_budget(plan, session.camas_dir, changed)
	match outcome:
		case NothingToRun(summary=summary, cause=cause):
			return success(
				with_warning(session, f"{headline(summary)}\n\n{cause}"),
				attach_budget(empty_run_response(), to_budget_report(summary)),
				session.compat,
			)
		case BudgetRun(keying=observed, summary=summary):
			return await execute_budgeted(session, config, req, label, observed, summary)
		case _:
			assert_never(outcome)


async def execute_budgeted(
	session: Session,
	config: Config | None,
	req: wire.RunRequest,
	label: str,
	observed: timings.Observed,
	summary: BudgetSummary,
) -> types.CallToolResult:
	run_node: Final = observed.node
	assert run_node is not None
	report = to_budget_report(summary)
	if req.dry_run:
		resp = attach_budget(to_plan_response(run_node), report)
		return success(
			with_warning(session, f"{headline(summary)}\n\n{dry_run_text(run_node)}"),
			resp,
			session.compat,
		)
	result = await run(
		run_node,
		jobs=req.jobs,
		interactive=False,
		base=base_for(session),
		leaf_color=leaf_color_of(config),
		identities=observed.identities,
	)
	logs = write_logs(create_run_log_dir(session.camas_dir, label, session.reserve_run()), result)
	observed.record(result)
	resp = attach_budget(
		attach_logs(to_run_response(run_node, result, verbosity=req.verbosity), logs), report
	)
	nudge = improve_loop_nudge(
		any_truncated=resp.truncated,
		any_failing_without_agent_format=has_failing_leaf_without_agent_format(run_node, result),
	)
	return success(
		with_warning(session, f"{headline(summary)}\n\n{run_text(label, resp, logs)}{nudge}"),
		resp,
		session.compat,
		links=failing_log_links(resp, logs),
	)


def budget_source(
	tasks: Mapping[str, TaskNode], config: Config | None, task: str | None
) -> TaskNode:
	if task is not None:
		return require_task(tasks, task)
	default = config.run_default() if config is not None else None
	if default is None:
		raise ValueError(NO_RUN_DEFAULT_MSG)
	return default


def gate_source(tasks: Mapping[str, TaskNode], config: Config | None, task: str | None) -> TaskNode:
	if task is not None:
		return require_task(tasks, task)
	check = config.gate_check(github=False) if config is not None else None
	if check is None:
		raise ValueError(
			"no task given and no check node (Config.agent.check or default_task) to gate"
		)
	return check


def to_budget_report(summary: BudgetSummary) -> wire.BudgetReport:
	return wire.BudgetReport(
		budget_s=summary.budget_s,
		selected=tuple(task_label(t) for t in summary.runnable),
		unmeasured=tuple(task_label(u.task) for u in summary.unmeasured),
		running_over_budget=tuple(_excluded_leaf(o) for o in summary.running_anyway),
		dropped_by_pipe_cut=tuple(
			wire.ExcludedLeaf(name=task_label(f.task), reason="pipe_cut", estimated_s=f.estimated_s)
			for f in summary.dropped
		),
		not_covered=tuple(
			wire.ExcludedLeaf(
				name=task_label(leaf.task), reason="not_covered", estimated_s=leaf.estimated_s
			)
			for leaf in summary.not_covered
		),
		excluded=tuple(_excluded_leaf(o) for o in summary.excluded),
	)


def _excluded_leaf(o: OverBudget) -> wire.ExcludedLeaf:
	return wire.ExcludedLeaf(
		name=task_label(o.task), reason="over_budget", estimated_s=o.estimated_s
	)


def attach_budget(resp: wire.RunResponse, report: wire.BudgetReport) -> wire.RunResponse:
	return resp.model_copy(update={"budget": report})


def census_of_report(report: wire.BudgetReport) -> BudgetCensus:
	return BudgetCensus(
		budget_s=report.budget_s,
		running=len(report.selected),
		unmeasured=tuple(report.unmeasured),
		running_anyway=tuple((e.name, e.estimated_s) for e in report.running_over_budget),
		dropped=tuple((e.name, e.estimated_s) for e in report.dropped_by_pipe_cut),
		not_covered=tuple((e.name, e.estimated_s) for e in report.not_covered),
		excluded=tuple((e.name, e.estimated_s) for e in report.excluded),
	)


def headline(summary: BudgetSummary) -> str:
	return "\n".join(summary_lines(census_of(summary)))


def create_run_log_dir(camas_dir: Path, task: str, seq: int) -> Path:
	timings.ensure_camas_dir(camas_dir)
	task_dir = camas_dir / "runs" / slug(task)
	run_dir = task_dir / str(seq)
	run_dir.mkdir(parents=True, exist_ok=True)
	prune_run_dirs(task_dir, run_dir)
	return run_dir


def _run_dir_recency(d: Path) -> tuple[int, int]:
	try:
		mtime_ns = d.stat().st_mtime_ns
	except OSError:  # pragma: no cover
		mtime_ns = 0
	return (mtime_ns, int(d.name))


def prune_run_dirs(task_dir: Path, keep: Path, *, keep_n: int = RUN_LOG_KEEP) -> tuple[Path, ...]:
	others = sorted(
		(d for d in task_dir.iterdir() if d.is_dir() and d.name.isdecimal() and d != keep),
		key=_run_dir_recency,
		reverse=True,
	)
	stale = tuple(others[keep_n - 1 :])
	for d in stale:
		shutil.rmtree(d, ignore_errors=True)
	return stale


def write_logs(run_dir: Path, result: RunResult) -> tuple[Path | None, ...]:
	return tuple(write_leaf_log(run_dir, i, leaf) for i, leaf in enumerate(result.results))


def write_leaf_log(run_dir: Path, index: int, leaf: TaskResult) -> Path | None:
	match leaf.completion:
		case Finished(output=output) | Stopped(output=output):
			if not output:
				return None
			path = run_dir / f"{index:03d}_{slug(leaf.name)}.log"
			path.write_text(decode_log(output), encoding="utf-8")
			return path
		case Skipped() | Errored():
			return None
		case _:
			assert_never(leaf.completion)


def failing_log_links(
	resp: wire.RunResponse, logs: tuple[Path | None, ...]
) -> tuple[types.ResourceLink, ...]:
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
	lines = [f"{len(resp.tasks)} task(s) defined. Call camas_run with one of these names."]
	if resp.default is not None:
		lines.append(f"default (developer's task): {resp.default}")
	if resp.github_default is not None:
		lines.append(
			f"github default (exactly what CI runs; run before committing/pushing): "
			f"{resp.github_default}. A bare `camas` runs it under GitHub Actions, so a CI "
			f"step is just `camas` — naming the task there is redundant."
		)
	if resp.run_default is not None and resp.run_default != resp.default:
		lines.append(f"no-task camas_run runs: {resp.run_default}")
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
	slowest = (
		f", slowest {task.slowest_leaf} {task.slowest_s:.2f}s"
		if task.slowest_leaf != task.name
		else ""
	)
	return f"  [~{task.estimated_s:.2f}s{slowest}, n={task.samples}]"


def run_text(task: str, resp: wire.RunResponse, logs: tuple[Path | None, ...]) -> str:
	verdict = "PASSED" if resp.returncode == 0 else "FAILED"
	lines = [
		f"camas_run {task!r}: {verdict} (returncode={resp.returncode}) in {resp.elapsed:.2f}s — "
		f"{resp.passed} passed, {resp.failed} failed, {resp.skipped} skipped",
		"",
	]
	for leaf, log in zip(resp.leaves, logs, strict=True):
		lines.extend(leaf_lines(leaf, log))
	return "\n".join(lines)


def improve_loop_nudge(*, any_truncated: bool, any_failing_without_agent_format: bool) -> str:
	if not (any_truncated or any_failing_without_agent_format):
		return ""
	gaps = (
		*(
			(
				"output was truncated (raw past its tail, or a structured payload past "
				"agent_format's limit=) — see the file/log referenced above for the rest",
			)
			if any_truncated
			else ()
		),
		*(
			("a failing leaf has no agent_format, so its output is unstructured raw text",)
			if any_failing_without_agent_format
			else ()
		),
	)
	return (
		"\n\nTip: " + "; and ".join(gaps) + ". This is a tasks.py gap, not a bug — add/adjust "
		"that leaf's agent_format=(args, kind) (limit= for a verbose format, or a {report} "
		"placeholder in args for a tool that writes its report to a file) so failures carry "
		"full, structured diagnostics next time; see camas_docs, then validate with camas_check."
	)


def dry_run_text(node: TaskNode) -> str:
	plan = "\n".join(render_tree_lines(node, show_cmd=True, color=False))
	return f"Dry run — fully-resolved plan, nothing executed:\n{plan}"


def check_text(resp: wire.CheckResponse) -> str:
	match resp.status:
		case "ok":
			how = (
				f"type-checked clean with {resp.checker}"
				if resp.checker is not None
				else "no type checker ran"
			)
			verdict = f"camas_check: OK — {resp.source} loads ({resp.task_count} task(s); {how})."
		case "type_error":
			verdict = (
				f"camas_check: TYPE ERRORS — {resp.source} loads ({resp.task_count} task(s)) but "
				f"{resp.checker} reported:\n\n{resp.diagnostics}"
			)
		case "load_error":
			verdict = (
				f"camas_check: LOAD ERROR — {resp.source} failed to evaluate:\n\n{resp.diagnostics}"
			)
		case "no_checker":
			verdict = (
				f"camas_check: {resp.source} loads ({resp.task_count} task(s)), but no type "
				f"checker is available.\n{resp.diagnostics}"
			)
		case "no_tasks":
			verdict = (
				"camas_check: no tasks.py or [tool.camas.tasks] found in this project. "
				"Call camas_init to scaffold a commented starter, or camas_docs to author one."
			)
		case _:
			assert_never(resp.status)
	version_note = (
		f"\n\ncamas server version: {resp.server_version}"
		if resp.server_version is not None
		else ""
	)
	if not resp.warnings:
		return verdict + version_note
	warnings = "\n".join(f"  - {w}" for w in resp.warnings)
	return f"{verdict}{version_note}\n\nWarnings (advisory — not failures):\n{warnings}"


def docs_text(resp: wire.DocsResponse) -> str:
	return (
		"camas authoring guide. The API source of truth is the installed camas package at:\n"
		f"  {resp.source}\n"
		"Read its v0/ submodules for exact Task/Sequential/Parallel/Pipe/Config signatures and the "
		"examples/ directory for full project layouts; validate your tasks.py with camas_check."
		f"\n\n{resp.tutorial}"
	)


def init_text(resp: wire.InitResponse) -> str:
	match resp.status:
		case "created":
			return (
				f"Scaffolded a starter tasks.py at:\n  {resp.path}\n"
				"Replace its placeholder commands with your real ones (see camas_docs), then "
				"validate with camas_check and list with camas_list. The scaffolded file:\n\n"
				f"{resp.content}"
			)
		case "exists":
			return (
				f"A tasks.py already exists at:\n  {resp.path}\nLeft untouched. "
				"Use camas_list to see its tasks, or camas_docs + camas_check to edit it."
			)
		case _:
			assert_never(resp.status)


def with_warning(session: Session, text: str) -> str:
	if session.version_warning is None:
		return text
	return session.version_warning + "\n\n" + text


def success(
	text: str,
	model: wire.ListResponse
	| wire.RunResponse
	| wire.CheckResponse
	| wire.DocsResponse
	| wire.GateResponse
	| wire.InitResponse
	| wire.GithubMatrixResponse,
	compat: Compat,
	*,
	links: tuple[types.ResourceLink, ...] = (),
) -> types.CallToolResult:
	return types.CallToolResult(
		content=[types.TextContent(type="text", text=text), *links],
		structuredContent=model.model_dump(mode="json") if compat.emit_structured else None,
		isError=False,
	)


def attach_logs(resp: wire.RunResponse, logs: tuple[Path | None, ...]) -> wire.RunResponse:
	return resp.model_copy(
		update={
			"leaves": tuple(
				leaf.model_copy(update={"log": str(log)}) if log is not None else leaf
				for leaf, log in zip(resp.leaves, logs, strict=True)
			)
		}
	)


def error_result(text: str) -> types.CallToolResult:
	return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=True)


def load_error_hint(session: Session, source: Path, exception: Exception) -> str:
	base = format_load_error_hint(source, exception)
	skew = version_skew(session.project)
	if skew is None:
		return base
	return f"{format_version_skew_hint(skew.running, skew.spec)}\n\n{base}"


def resolve_project(base: Path) -> TasksState:
	for candidate in (base, *base.parents):
		tasks_py = candidate / "tasks.py"
		if tasks_py.is_file():
			return load_py_tasks_state(tasks_py)
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
	with redirect_stdout(sys.stderr):
		return resolve_project(base)


def project_base() -> Path:
	return (Path(env) if (env := os.environ.get("CLAUDE_PROJECT_DIR")) else Path.cwd()).resolve()


def leaf_lines(leaf: wire.LeafReport, log: Path | None) -> list[str]:
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
		case wire.Errored(returncode=rc, message=message):
			return [f"ERROR  {leaf.name} (exit {rc}): {message}"]
		case wire.Planned():
			return [f"PLAN   {leaf.name}"]
		case _:
			assert_never(leaf.completion)


def indent_lines(output: list[str]) -> list[str]:
	return [f"    {line}" for line in output]


def tail_lines(truncated: bool, log: Path | None) -> list[str]:
	lines = ["    … earlier output truncated; see full log"] if truncated else []
	if log is not None:
		lines.append(f"    full log: {log}")
	return lines


def slug(name: str) -> str:
	return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:80]


def decode_log(output: Sequence[bytes]) -> str:
	return strip_ansi("".join(chunk.decode("utf-8", errors="replace") for chunk in output))


async def gate_call(session: Session, arguments: dict[str, Any]) -> types.CallToolResult:
	match session.project:
		case LoadErr(source=source, exception=exception):
			return error_result(load_error_hint(session, source, exception))
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
	try:
		req = wire.GateRequest.model_validate(arguments)
	except ValidationError as e:
		return error_result(f"invalid camas_gate arguments:\n{e}")
	try:
		node = gate_source(tasks, config, req.task)
		require_filled_axes(node)
	except ValueError as e:
		return error_result(str(e))
	changed = to_changed(req.paths, base_for(session))
	rerun = wire.GateRerun(task=req.task, paths=changed, under=req.under)
	if requested_but_unusable(req.paths, changed):
		nothing: Final = GateOutcome("green", None, None, None, coverage_message(req.paths))
		resp = to_gate_response(nothing, None, rerun)
		return success(with_warning(session, gate_text(resp)), resp, session.compat)
	settled = await settle(
		config.gate_fix() if config is not None else None,
		changed,
		camas_dir=session.camas_dir,
		jobs=req.jobs,
		base=base_for(session),
		leaf_color=leaf_color_of(config),
	)
	outcome = await run_gate(
		node,
		changed,
		under=req.under,
		jobs=req.jobs,
		base=base_for(session),
		timings=timings.load(session.camas_dir),
		leaf_color=leaf_color_of(config),
	)
	record_gate(session.camas_dir, outcome)
	budget = to_budget_report(outcome.budget) if outcome.budget is not None else None
	resp = to_gate_response(outcome, budget, rerun)
	nudge = improve_loop_nudge(
		any_truncated=any(env.truncated for env in resp.diagnostics or ()),
		any_failing_without_agent_format=(
			outcome.node is not None
			and outcome.result is not None
			and has_failing_leaf_without_agent_format(outcome.node, outcome.result)
		),
	)
	return success(
		with_warning(session, gate_text(resp) + settle_note(settled) + nudge), resp, session.compat
	)


def settle_note(settled: RunResult | None) -> str:
	failed: Final = tuple(
		f"{result.name} exited {result.completion.returncode}"
		for result in (settled.results if settled is not None else ())
		if result.completion.returncode != 0 and not isinstance(result.completion, Skipped)
	)
	if not failed:
		return ""
	return (
		f"\n\nThe autofix (Config.agent.fix) failed before the checks ran — {', '.join(failed)} —"
		" so some of its fixes may not have applied; camas_gate settles it again."
	)


async def fix_call(session: Session, arguments: dict[str, Any]) -> types.CallToolResult:
	match session.project:
		case LoadErr(source=source, exception=exception):
			return error_result(load_error_hint(session, source, exception))
		case LoadOk(tasks=tasks, config=config):
			return await fix_for(session, tasks, config, arguments)
		case _:
			assert_never(session.project)


async def fix_for(
	session: Session,
	tasks: Mapping[str, TaskNode],
	config: Config | None,
	arguments: dict[str, Any],
) -> types.CallToolResult:
	try:
		req = wire.FixRequest.model_validate(arguments)
	except ValidationError as e:
		return error_result(f"invalid camas_fix arguments:\n{e}")
	fix_node: TaskNode | None
	try:
		fix_node = (
			require_task(tasks, req.task)
			if req.task is not None
			else config.gate_fix()
			if config is not None
			else None
		)
	except ValueError as e:
		return error_result(str(e))
	# The node to run and how to observe it are bound together, since neither is meaningful without
	# the other: nothing to run means nothing to record, and a prepared bundle always has a node.
	prepared: timings.Observed | None = None
	blocked = unsatisfiable_message(fix_node) if fix_node is not None else None
	if fix_node is not None and blocked is None:
		changed = to_changed(req.paths, base_for(session))
		expanded = expand_matrix(fix_node)
		if not requested_but_unusable(req.paths, changed):
			keying = timings.observed(session.camas_dir, expanded, changed)
			if keying.node is not None:
				prepared = keying
	empty_cause: str | None
	if prepared is None:
		resp = empty_run_response()
		empty_cause = (
			"no fix node registered (Config.agent.fix is None)"
			if fix_node is None
			else blocked
			if blocked is not None
			else "no fix leaf covers the paths"
		)
	else:
		if prepared.node is None:  # pragma: no cover  # prepared is only set with a node
			return nothing_covered_result(session, req.paths)
		node: Final = prepared.node
		result = await run(
			node,
			jobs=req.jobs,
			interactive=False,
			base=base_for(session),
			leaf_color=leaf_color_of(config),
			identities=prepared.identities,
		)
		prepared.record(result)
		resp = to_run_response(node, result)
		empty_cause = None
	return success(
		with_warning(session, fix_text(resp, empty_cause=empty_cause)), resp, session.compat
	)


def fix_text(resp: wire.RunResponse, *, empty_cause: str | None = None) -> str:
	if not resp.leaves:
		return f"camas_fix: nothing to fix — {empty_cause}."
	verdict = "FIXED" if resp.returncode == 0 else "FIX FAILED"
	return (
		f"camas_fix: {verdict} (returncode={resp.returncode}) in {resp.elapsed:.2f}s — "
		f"{resp.passed} passed, {resp.failed} failed, {resp.skipped} skipped"
	)


def github_matrix_call(session: Session, arguments: dict[str, Any]) -> types.CallToolResult:
	match session.project:
		case LoadErr(source=source, exception=exception):
			return error_result(load_error_hint(session, source, exception))
		case LoadOk(tasks=tasks, config=config):
			return github_matrix_for(session, tasks, config, arguments)
		case _:
			assert_never(session.project)


def github_matrix_for(
	session: Session,
	tasks: Mapping[str, TaskNode],
	config: Config | None,
	arguments: dict[str, Any],
) -> types.CallToolResult:
	try:
		req = wire.GithubMatrixRequest.model_validate(arguments)
	except ValidationError as e:
		return error_result(f"invalid camas_github_matrix arguments:\n{e}")
	try:
		name, node = resolve_github_matrix_node(tasks, config, req)
		resp = to_github_matrix_response(name, node, tasks, req.shape)
	except ValueError as e:
		return error_result(str(e))
	return success(with_warning(session, github_matrix_text(resp)), resp, session.compat)


def resolve_github_matrix_node(
	tasks: Mapping[str, TaskNode], config: Config | None, req: wire.GithubMatrixRequest
) -> tuple[str, TaskNode]:
	if req.task is not None:
		name, node = req.task, require_task(tasks, req.task)
	else:
		default = config.bare_task(github=False) if config is not None else None
		if default is None:
			raise ValueError(
				"no task given and no project default_task to emit; pass a task whose matrix_axes "
				"is non-empty (see camas_list)"
			)
		name, node = default.name or "default task", default
	if req.matrix_overrides:
		node = override_matrix(node, {k: tuple(v) for k, v in req.matrix_overrides.items()})
	return name, node


def github_matrix_text(resp: wire.GithubMatrixResponse) -> str:
	compact = format_matrix_json(resp.matrix, pretty=False)
	pretty = format_matrix_json(resp.matrix, pretty=True)
	consume = (
		"Consume the whole object with fromJSON(...) — an include: list has no per-axis arrays "
		"to read individually."
		if resp.shape == "variants"
		else "Consume the whole object with fromJSON(...), or one axis at a time, e.g. "
		f"fromJSON(needs.discover.outputs.matrix).{next(iter(resp.matrix))}."
	)
	return (
		f"camas_github_matrix {resp.task!r}: GitHub Actions strategy.matrix "
		f"(shape={resp.shape})\n\n"
		f"{pretty}\n\n"
		f"Compact (for $GITHUB_OUTPUT): {compact}\n"
		f"Job command: {resp.job_command}\n"
		f"{consume}\n"
		f"Pin the shape to keep it stable: camas <task> --github-matrix={resp.shape} "
		f"(or shape={resp.shape!r} here)."
	)


def gate_text(resp: wire.GateResponse) -> str:
	verdict = (
		"CONTINUE — checks green; no residual needs reasoning"
		if resp.decision == "continue"
		else "BLOCK — a residual needs reasoning"
	)
	lines: list[str] = [f"camas_gate: {verdict} (residual_class={resp.residual_class})"]
	if resp.budget is not None:
		lines.extend(["", "\n".join(summary_lines(census_of_report(resp.budget)))])
	if resp.nothing_ran is not None:
		lines.extend(["", resp.nothing_ran])
	if resp.diagnostics is not None:
		lines.extend(["", "Residual (failing checks):"])
		for env in resp.diagnostics:
			lines.append(f"  {env.name} (exit {env.exit_code}, {env.output_kind})")
			lines.extend(f"    {line}" for line in env.payload.splitlines())
			if env.truncated:
				lines.append("    … earlier output truncated")
	if resp.decision == "block":
		lines.extend(
			[
				"",
				f"Re-gate this scope: {rerun_command(resp.rerun)} (read-only; camas_gate settles"
				" the autofix first)",
			]
		)
	return "\n".join(lines)


def nudge_text(resp: wire.GateResponse) -> str:
	lines: list[str] = ["camas: a residual needs reasoning — the workspace is not green."]
	if resp.diagnostics:
		lines.extend(["", "Residual (failing checks):"])
		for env in resp.diagnostics:
			lines.append(f"  {env.name} (exit {env.exit_code}, {env.output_kind})")
			lines.extend(f"    {line}" for line in env.payload.splitlines())
			if env.truncated:
				lines.append("    … earlier output truncated")
	lines.extend(
		[
			"",
			"Launch the camas fixer ladder in the background rather than reasoning about this "
			"yourself — see the gate skill: camas-lint-fixer-haiku for a lint/format residual "
			"(escalate to camas-lint-fixer-sonnet on a hand-back), camas-test-fixer for a "
			"test/coverage residual.",
		]
	)
	return "\n".join(lines)


NUDGE_MARKER_PREFIX: Final = "camas-nudge-"


def _nudge_marker(session_id: str) -> Path:
	digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
	return Path(tempfile.gettempdir()) / f"{NUDGE_MARKER_PREFIX}{digest}"


def _unlink_if_stale(path: Path, cutoff: float) -> None:
	try:
		if path.is_file() and path.stat().st_mtime < cutoff:
			path.unlink()
	except OSError:  # pragma: no cover
		pass


def prune_stale_nudge_markers(max_age_s: float = STALE_TEMP_MAX_AGE_S) -> None:
	base = Path(tempfile.gettempdir())
	cutoff = time.time() - max_age_s
	for marker in base.glob(f"{NUDGE_MARKER_PREFIX}*"):
		_unlink_if_stale(marker, cutoff)


def should_nudge(event: HookEvent) -> bool:
	if event.stop_hook_active:
		return False
	if not event.session_id or not event.prompt_id:
		return True
	try:
		return _nudge_marker(event.session_id).read_text(encoding="utf-8") != event.prompt_id
	except (OSError, ValueError):
		return True


def record_nudge(event: HookEvent) -> None:
	if not event.session_id or not event.prompt_id:
		return
	prune_stale_nudge_markers()
	with suppress(OSError):
		_nudge_marker(event.session_id).write_text(event.prompt_id, encoding="utf-8")


def rerun_command(rerun: wire.GateRerun) -> str:
	task = (rerun.task,) if rerun.task is not None else ()
	paths = tuple(arg for p in rerun.paths for arg in ("--paths", p))
	under = ("--under", str(rerun.under)) if rerun.under is not None else ()
	return shlex.join(("camas", "mcp", "gate", *task, *paths, *under))


class GateArgs(NamedTuple):
	task: str | None
	paths: tuple[str, ...]
	under: float | None
	jobs: int | None
	dry_run: bool = False
	nudge: bool = False


def parse_gate_args(argv: list[str]) -> GateArgs:
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
		metavar="PATH[,PATH...]",
		help="changed paths to scope to (repeatable, comma-separated)",
	)
	parser.add_argument(
		"--under", type=parse_duration, default=None, metavar="DURATION", help="wall-clock budget"
	)
	parser.add_argument(
		"--jobs",
		type=positive_jobs,
		default=None,
		metavar="N",
		help=f"cap concurrently running leaves at N; {JOBS_UNIT}",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		default=False,
		help="print the resolved path-scoped leaf plan without executing",
	)
	parser.add_argument(
		"--nudge",
		action="store_true",
		default=False,
		help="emit the Stop-hook async-nudge text on stderr and exit 2 when not green, else "
		"silent exit 0 — for the async Stop hook, not interactive use",
	)
	ns = parser.parse_args(argv)
	return GateArgs(
		task=ns.task,
		paths=tuple(ns.paths),
		under=ns.under,
		jobs=ns.jobs,
		dry_run=ns.dry_run,
		nudge=ns.nudge,
	)


def gate_cli_load_error(state: TasksState, source: Path, exception: Exception) -> str:
	base = f"camas mcp gate: cannot load {source}: {exception}"
	skew = version_skew(state)
	if skew is None:
		return base
	return (
		f"{base}\n"
		f"camas {skew.running} does not satisfy tasks.py pin camas{skew.spec}; run "
		"`uv run tasks.py <task>` to load the pinned version."
	)


def gate_cli(argv: list[str]) -> int:
	args = parse_gate_args(argv)
	base = project_base()
	state = resolve_project_quiet(base)
	match state:
		case LoadErr(source=source, exception=exception):
			print(gate_cli_load_error(state, source, exception), file=sys.stderr)
			return 0 if args.nudge else 2
		case LoadOk(tasks=tasks, config=config, source=source):
			return run_gate_cli(args, base_from(source, base), tasks, config)
		case _:
			assert_never(state)


def gate_dry_run_text(
	node: TaskNode, changed: tuple[str, ...], under: float | None, camas_dir: Path
) -> str:
	expanded: Final = expand_matrix(node)
	if under is None:
		scoped = scope_to_changed(expanded, changed)
		return coverage_message(changed) if scoped is None else path_scoped_plan_text(scoped)
	outcome: Final = resolve_budget(
		plan_under(expanded, under, timings.load(camas_dir), timings.scope_of(changed)),
		camas_dir,
		changed,
	)
	match outcome:
		case NothingToRun():
			return "\n".join(outcome_lines(outcome))
		case BudgetRun(keying=observed):
			run_node: Final = observed.node
			assert run_node is not None
			return "\n".join((*outcome_lines(outcome), path_scoped_plan_text(run_node)))
		case _:
			assert_never(outcome)


def path_scoped_plan_text(node: TaskNode) -> str:
	tree: Final = "\n".join(render_tree_lines(node, show_cmd=True, color=False))
	return f"Dry run — resolved path-scoped plan, nothing executed:\n{tree}"


def run_gate_cli(
	args: GateArgs, base: Path, tasks: Mapping[str, TaskNode], config: Config | None
) -> int:
	try:
		node = gate_source(tasks, config, args.task)
		require_filled_axes(node)
	except ValueError as e:
		print(f"camas mcp gate: {e}", file=sys.stderr)
		return 0 if args.nudge else 2
	event = event_from_stdin() if not args.paths else NO_EVENT
	requested = args.paths or (event.changed or ())
	changed = to_changed(requested, base)
	if requested_but_unusable(requested, changed):
		# Green either way, nudge or not: nothing the checks cover changed, so there is nothing to
		# block the turn on — and running the whole tree over an edit outside the repo is what
		# scoping was asked to prevent.
		print(coverage_message(requested))
		return 0
	camas_dir = (config if config is not None else Config()).camas_path(base)
	if args.dry_run:
		print(gate_dry_run_text(node, changed, args.under, camas_dir))
		return 0
	if args.nudge and should_nudge(event) and config is not None and config.gate_fix() is not None:
		_ = await_settled(event)
	outcome = asyncio.run(
		run_gate(
			node,
			changed,
			under=args.under,
			jobs=args.jobs,
			base=base,
			timings=timings.load(camas_dir),
			leaf_color=leaf_color_of(config),
		)
	)
	record_gate(camas_dir, outcome)
	budget = to_budget_report(outcome.budget) if outcome.budget is not None else None
	rerun = wire.GateRerun(task=args.task, paths=changed, under=args.under)
	resp = to_gate_response(outcome, budget, rerun)
	if args.nudge:
		if resp.decision != "block" or not should_nudge(event):
			return 0
		record_nudge(event)
		print(nudge_text(resp), file=sys.stderr)
		return 2
	print(resp.model_dump_json(indent=2))
	if resp.decision == "block":
		print(gate_text(resp), file=sys.stderr)
		return 2
	return 0
