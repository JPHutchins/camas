# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import pytest
from pydantic import ValidationError

from camas.mcp.wire import (
	CheckResponse,
	DocsResponse,
	Finished,
	LeafReport,
	ListRequest,
	ListResponse,
	RunRequest,
	RunResponse,
	Skipped,
	Stopped,
	TaskInfo,
	run_input_schema,
)


def test_list_request_defaults_to_unexpanded() -> None:
	assert ListRequest.model_validate({}).expand_matrix is False


def test_list_request_rejects_unknown_field() -> None:
	with pytest.raises(ValidationError):
		ListRequest.model_validate({"bogus": 1})


def test_run_request_defaults() -> None:
	req = RunRequest.model_validate({"task": "check"})
	assert (req.task, req.dry_run, req.verbosity, req.jobs) == ("check", False, "failures", None)
	assert req.matrix_overrides == {}
	assert req.args == []


def test_run_request_rejects_unknown_field() -> None:
	with pytest.raises(ValidationError):
		RunRequest.model_validate({"task": "check", "bogus": 1})


def test_run_request_rejects_bad_verbosity() -> None:
	with pytest.raises(ValidationError):
		RunRequest.model_validate({"task": "check", "verbosity": "loud"})


def test_run_input_schema_splices_task_enum() -> None:
	schema = run_input_schema(("lint", "test", "check"))
	task_enum = next(
		b["enum"] for b in schema["properties"]["task"]["anyOf"] if b.get("type") == "string"
	)
	assert task_enum == ["lint", "test", "check"]
	assert schema["additionalProperties"] is False
	assert schema["properties"]["verbosity"]["enum"] == ["summary", "failures", "full"]


def test_leaf_report_discriminates_completion() -> None:
	report = LeafReport.model_validate(
		{
			"name": "test [PY=3.13]",
			"command": "pytest",
			"completion": {"type": "skipped", "returncode": 1, "blocked_by": "lint"},
		}
	)
	assert isinstance(report.completion, Skipped)
	assert report.completion.blocked_by == "lint"


def test_run_response_round_trips_completion_union() -> None:
	resp = RunResponse(
		returncode=1,
		elapsed=1.5,
		passed=1,
		failed=1,
		skipped=0,
		interrupt_count=0,
		leaves=(
			LeafReport(name="a", command="true", completion=Finished(returncode=0, elapsed=0.1)),
			LeafReport(name="b", command="false", completion=Stopped(returncode=130, elapsed=0.2)),
		),
	)
	dumped = resp.model_dump()
	assert dumped["leaves"][0]["completion"]["type"] == "finished"
	assert dumped["leaves"][1]["completion"]["type"] == "stopped"
	assert RunResponse.model_validate(dumped) == resp


def test_list_response_markers() -> None:
	listing = ListResponse(
		tasks=(TaskInfo(name="ci", command_preview="pytest", is_default=True),),
		default="ci",
	)
	assert listing.tasks[0].is_default is True
	assert listing.github_default is None


def test_list_response_github_default_schema_description() -> None:
	description = ListResponse.model_json_schema()["properties"]["github_default"]["description"]
	assert "Config(github_task" in description


def test_check_response_defaults_and_rejects_bad_status() -> None:
	resp = CheckResponse(status="ok", source="/x/tasks.py", task_count=3, checker="ty")
	assert (resp.diagnostics, resp.task_count, resp.warnings) == (None, 3, ())
	with pytest.raises(ValidationError):
		CheckResponse.model_validate({"status": "exploded"})


def test_check_response_warnings_round_trip() -> None:
	resp = CheckResponse(
		status="ok", source="/x/tasks.py", task_count=1, warnings=("inert paths on 'cargo'",)
	)
	assert CheckResponse.model_validate(resp.model_dump()) == resp


def test_check_response_server_version_round_trip() -> None:
	assert CheckResponse(status="ok").server_version is None
	resp = CheckResponse(status="ok", server_version="1.2.3")
	assert CheckResponse.model_validate(resp.model_dump()) == resp


def test_docs_response_round_trips() -> None:
	resp = DocsResponse(source="/site-packages/camas", tutorial="Task(...)")
	assert DocsResponse.model_validate(resp.model_dump()) == resp


def test_task_info_estimate_round_trips() -> None:
	info = TaskInfo(
		name="check",
		command_preview="Parallel(...)",
		estimated_s=32.0,
		samples=3,
		slowest_leaf="test",
		slowest_s=31.9,
	)
	dumped = info.model_dump()
	assert dumped["estimated_s"] == 32.0
	assert dumped["slowest_leaf"] == "test"
	assert TaskInfo.model_validate(dumped) == info


def test_task_info_estimate_defaults_none() -> None:
	info = TaskInfo(name="x", command_preview="Task(...)")
	assert (info.estimated_s, info.samples, info.slowest_leaf, info.slowest_s) == (
		None,
		None,
		None,
		None,
	)
