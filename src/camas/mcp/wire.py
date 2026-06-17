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


class RunResponse(BaseModel):
	"""Structured result of a run — camas's ``RunResult``, faithfully on the wire."""

	returncode: int
	elapsed: float
	passed: int
	failed: int
	skipped: int
	interrupt_count: int
	leaves: list[LeafReport]
	truncated: bool = False


class TaskInfo(BaseModel):
	"""One task discoverable in the project, as ``camas_list`` reports it."""

	name: str
	help: str | None = None
	command_preview: str
	matrix_axes: dict[str, list[str]] = Field(default_factory=dict)
	is_default: bool = False
	is_github_default: bool = False


class ListResponse(BaseModel):
	"""The project's task catalog: every task plus the default and CI-default names."""

	tasks: list[TaskInfo]
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


class RunRequest(BaseModel):
	"""Arguments to ``camas_run``."""

	model_config = ConfigDict(extra="forbid")

	task: str = Field(description="Task name to run — one of the names returned by camas_list.")
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
		description="Extra args appended to the task's command (camas's -- passthrough).",
	)


def run_input_schema(task_names: tuple[str, ...]) -> dict[str, Any]:
	"""``RunRequest``'s JSON Schema with the live task-name ``enum`` spliced into ``task``.

	>>> schema = run_input_schema(("lint", "test"))
	>>> schema["properties"]["task"]["enum"]
	['lint', 'test']
	>>> schema["additionalProperties"]
	False
	"""
	schema = RunRequest.model_json_schema()
	schema["properties"]["task"]["enum"] = list(task_names)
	return schema
