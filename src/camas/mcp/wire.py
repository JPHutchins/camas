# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Pydantic wire models for the MCP tools' JSON-RPC payloads."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Finished(BaseModel):
	"""A leaf that ran to exit with a returncode."""

	type: Literal["finished"] = "finished"
	returncode: int
	elapsed: float
	output: list[str] = Field(default_factory=list)


class Skipped(BaseModel):
	"""A leaf skipped because a prior Sequential step failed; it never ran."""

	type: Literal["skipped"] = "skipped"
	returncode: int
	blocked_by: str | None = None


class Stopped(BaseModel):
	"""A leaf interrupted by a signal before it finished."""

	type: Literal["stopped"] = "stopped"
	returncode: int
	elapsed: float
	output: list[str] = Field(default_factory=list)


Completion = Annotated[Finished | Skipped | Stopped, Field(discriminator="type")]
"""Tagged union mirroring ``camas.v0.completion.Completion``."""


class LeafReport(BaseModel):
	"""One executed leaf, in DFS order, with its resolved command and outcome."""

	name: str
	command: str
	cwd: str | None = None
	completion: Completion
	truncated: bool = False
	"""True when this leaf's ``output`` is a tail excerpt; the full log is on disk."""
	log: str | None = None
	"""Filesystem path to this leaf's full output log, when one was written."""


class ExcludedLeaf(BaseModel):
	"""A leaf a time budget did not run, and why."""

	name: str
	reason: Literal["over_budget", "untimed"]
	estimated_s: float | None = None
	"""The leaf's observed estimate; null when it has never been timed."""


class BudgetReport(BaseModel):
	"""How a ``camas_run`` time budget (``under``) partitioned the task's leaves."""

	budget_s: float
	selected: tuple[str, ...]
	"""Leaves whose estimate fit the budget — the ones that were scheduled to run."""
	excluded: tuple[ExcludedLeaf, ...]


class RunResponse(BaseModel):
	"""Structured result of a run — camas's ``RunResult``, faithfully on the wire."""

	returncode: int
	elapsed: float
	passed: int
	failed: int
	skipped: int
	interrupt_count: int
	leaves: tuple[LeafReport, ...]
	truncated: bool = False
	budget: BudgetReport | None = None
	"""Present when the run was time-budgeted (``camas_run`` with ``under``)."""


class TaskInfo(BaseModel):
	"""One task discoverable in the project, as ``camas_list`` reports it."""

	name: str
	help: str | None = None
	command_preview: str
	matrix_axes: dict[str, list[str]] = Field(default_factory=dict)
	is_default: bool = False
	is_github_default: bool = False
	estimated_s: float | None = None
	"""Wall-clock seconds composed from observed leaf durations, when all are known."""
	samples: int | None = None
	"""Fewest runs informing any leaf in the estimate — its confidence."""
	slowest_leaf: str | None = None
	"""The slowest leaf in this task's subtree."""
	slowest_s: float | None = None
	"""That leaf's mean observed duration."""


class ListResponse(BaseModel):
	"""The project's task catalog: every task plus the default and CI-default names."""

	tasks: tuple[TaskInfo, ...]
	default: str | None = None
	github_default: str | None = None


class CheckResponse(BaseModel):
	"""Outcome of validating the project's tasks source — does it load, and does it type-check.

	``status`` is the verdict; the remaining fields carry the detail an agent needs to fix
	a problem. ``diagnostics`` holds the eval traceback, the type-checker output, or the
	install hint, depending on ``status``.
	"""

	status: Literal["ok", "load_error", "type_error", "no_checker", "no_tasks"]
	source: str | None = None
	task_count: int | None = None
	checker: Literal["ty", "mypy"] | None = None
	diagnostics: str | None = None


class DocsResponse(BaseModel):
	"""The camas authoring guide: the installed source path plus its tutorial, served live."""

	source: str
	"""Absolute path to the installed camas package — the API source of truth."""
	tutorial: str
	"""The package ``__init__.py`` docstring: a worked, doctested authoring tutorial."""


class ListRequest(BaseModel):
	"""Arguments to ``camas_list``."""

	model_config = ConfigDict(extra="forbid")

	expand_matrix: bool = Field(
		default=False,
		description=(
			"Expand matrix tasks into their full per-axis command tree. Off by default: the "
			"preview keeps the unexpanded template and the axes are reported under matrix_axes, "
			"which is far smaller for a matrix over many values."
		),
	)


class RunRequest(BaseModel):
	"""Arguments to ``camas_run``."""

	model_config = ConfigDict(extra="forbid")

	task: str | None = Field(
		default=None,
		description="Task name to run — one of the names returned by camas_list. May be "
		"omitted only with 'under', which then budgets the project's default task.",
	)
	under: float | None = Field(
		default=None,
		gt=0,
		description="Wall-clock budget in seconds: run only the leaves whose recorded "
		"estimate fits, mutating leaves (formatters) first then the read-only rest in "
		"parallel. Untimed leaves are skipped. Omit 'task' to budget the default task; "
		"the 'budget' field of the response reports what was selected and excluded.",
	)
	dry_run: bool = Field(
		default=False,
		description="Preview the fully-resolved task tree without executing anything.",
	)
	verbosity: Literal["summary", "failures", "full"] = Field(
		default="failures",
		description="summary: pass/fail only; failures: + output of failed leaves; full: all output.",
	)
	jobs: int | None = Field(
		default=None, ge=1, description="Max concurrent leaf subprocesses; null = unbounded."
	)
	matrix_overrides: dict[str, list[str]] = Field(
		default_factory=dict,
		description="Pin matrix axes by name, e.g. {'PY': ['3.13']}; axis names must exist.",
	)
	args: list[str] = Field(
		default_factory=list,
		description=(
			"Passthrough flags appended to a single-leaf task's command (camas's --), e.g. "
			"['tests/test_x.py::test_y', '-x']; leaf tasks only — composite tasks are rejected."
		),
	)


def run_input_schema(task_names: tuple[str, ...]) -> dict[str, Any]:
	"""``RunRequest``'s JSON Schema with the live task-name ``enum`` spliced into ``task``.

	``task`` is ``str | None`` (a budget run may omit it), so the enum lands on the
	string branch of the ``anyOf`` rather than the property itself.

	>>> schema = run_input_schema(("lint", "test"))
	>>> next(b["enum"] for b in schema["properties"]["task"]["anyOf"] if b.get("type") == "string")
	['lint', 'test']
	>>> schema["additionalProperties"]
	False
	"""
	schema: dict[str, Any] = RunRequest.model_json_schema()
	for branch in schema["properties"]["task"]["anyOf"]:
		if branch.get("type") == "string":
			branch["enum"] = list(task_names)
	return schema


class GateRequest(BaseModel):
	"""Arguments to ``camas_gate``."""

	model_config = ConfigDict(extra="forbid")

	paths: list[str] = Field(
		default_factory=list,
		description="Changed paths to scope the gate to — a FileChanged/PostToolBatch hook's "
		"edited files. Empty gates the whole task; each leaf's {paths} is injected with the "
		"files it covers and leaves covering nothing are dropped.",
	)
	task: str | None = Field(
		default=None,
		description="Task to gate — one of the names from camas_list. Omit to gate the project's "
		"default task.",
	)
	under: float | None = Field(
		default=None,
		gt=0,
		description="Wall-clock budget in seconds for the checks (the read-only leaves); the "
		"deterministic autofix always runs. Untimed leaves are skipped.",
	)
	jobs: int | None = Field(
		default=None, ge=1, description="Max concurrent leaf subprocesses; null = unbounded."
	)


class AgentEnvelope(BaseModel):
	"""One task's agent-facing result: its exit code and the tool's output tagged by the
	standard it is in (``output_kind``) and passed through verbatim — camas never parses it.
	"""

	name: str
	exit_code: int
	output_kind: Literal["sarif", "rdjson", "lsp", "junit", "tap", "raw"]
	payload: str
	truncated: bool = False


class GateResponse(BaseModel):
	"""The SA-delegation gate's verdict: how to route the batch, and the residual that decided it."""

	decision: Literal["continue", "block"]
	"""``block`` when a residual needs reasoning (a PostToolBatch hook surfaces it), else ``continue``."""
	residual_class: Literal["autofixed", "needs_reasoning"]
	diagnostics: tuple[AgentEnvelope, ...] | None = None
	"""The failing leaves as AgentJSON envelopes (failures-only); null when nothing survived the fixers."""
	budget: BudgetReport | None = None
	"""How ``under`` partitioned the checks, when a budget was given."""


def gate_input_schema(task_names: tuple[str, ...]) -> dict[str, Any]:
	"""``GateRequest``'s JSON Schema with the live task-name ``enum`` spliced into ``task``.

	>>> schema = gate_input_schema(("lint", "test"))
	>>> next(b["enum"] for b in schema["properties"]["task"]["anyOf"] if b.get("type") == "string")
	['lint', 'test']
	>>> schema["additionalProperties"]
	False
	"""
	schema: dict[str, Any] = GateRequest.model_json_schema()
	for branch in schema["properties"]["task"]["anyOf"]:
		if branch.get("type") == "string":
			branch["enum"] = list(task_names)
	return schema
