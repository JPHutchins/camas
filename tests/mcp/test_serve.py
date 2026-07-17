# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session

from camas import AgentFormat, Config, Parallel, Sequential, Task
from camas.core import timings
from camas.core.completion import RunResult, TaskResult
from camas.main.check import CheckerErr, CheckerNotFound, CheckerOk
from camas.main.state import LoadErr, LoadOk
from camas.mcp import serve, wire
from camas.mcp.serve import Compat, Session
from camas.v0.completion import Errored, Finished, Skipped, Stopped
from camas.v0.config import Claude

if TYPE_CHECKING:
	from collections.abc import Callable

	from camas.main.check import TypeCheckResult
	from camas.v0.task import TaskNode


def _session(
	tasks: dict[str, TaskNode], config: Config | None, base: Path, *, rich: bool = False
) -> Session:
	state = LoadOk(tasks=tasks, source=base / "tasks.py", scope_effects={}, config=config)
	return Session(state, base, Compat(emit_structured=rich))


def _resolved_session(base: Path, *, rich: bool = False) -> Session:
	"""A session resolved from ``base``'s tasks source, as ``call_handler`` refreshes one before
	dispatching any read-only handler."""
	return Session(serve.resolve_project(base), base, Compat(emit_structured=rich))


PASS = Task(("python", "-c", "print('ok')"), name="lint")
FAIL = Task(("python", "-c", "import sys; print('boom'); sys.exit(3)"), name="bad")


def _task_enum(input_schema: dict[str, Any]) -> list[str]:
	"""The task-name enum, spliced onto the string branch of the optional ``task`` field."""
	branches = input_schema["properties"]["task"]["anyOf"]
	enum: list[str] = next(b["enum"] for b in branches if b.get("type") == "string")
	return enum


def test_resolve_project_finds_tasks_py(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text("from camas import Task\nlint = Task('ruff check .')\n")
	state = serve.resolve_project(tmp_path)
	assert isinstance(state, LoadOk)
	assert "lint" in state.tasks


def test_resolve_project_broken_tasks_py_is_load_err(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text("raise RuntimeError('boom')\n")
	assert isinstance(serve.resolve_project(tmp_path), LoadErr)


def test_resolve_project_finds_pyproject(tmp_path: Path) -> None:
	(tmp_path / "pyproject.toml").write_text('[tool.camas.tasks]\nlint = "ruff check ."\n')
	state = serve.resolve_project(tmp_path)
	assert isinstance(state, LoadOk)
	assert "lint" in state.tasks


def test_resolve_project_bad_pyproject_is_load_err(tmp_path: Path) -> None:
	(tmp_path / "pyproject.toml").write_text("[tool.camas.tasks]\nlint = 123\n")
	assert isinstance(serve.resolve_project(tmp_path), LoadErr)


def test_resolve_project_walks_past_pyproject_without_tasks(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text("from camas import Task\ntop = Task('echo top')\n")
	sub = tmp_path / "sub"
	sub.mkdir()
	(sub / "pyproject.toml").write_text("[tool.other]\nx = 1\n")
	state = serve.resolve_project(sub)
	assert isinstance(state, LoadOk)
	assert "top" in state.tasks


def test_resolve_project_nothing_found_is_empty(tmp_path: Path) -> None:
	assert serve.task_names(serve.resolve_project(tmp_path)) == ()


def test_project_base_prefers_claude_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
	assert serve.project_base() == tmp_path.resolve()


def test_project_base_falls_back_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
	monkeypatch.chdir(tmp_path)
	assert serve.project_base() == Path.cwd().resolve()


def test_project_base_resolves_relative_env(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	(tmp_path / "sub").mkdir()
	monkeypatch.chdir(tmp_path)
	monkeypatch.setenv("CLAUDE_PROJECT_DIR", "sub")
	base = serve.project_base()
	assert base.is_absolute()
	assert base == (tmp_path / "sub").resolve()


def test_resolve_project_quiet_redirects_tasks_py_stdout(
	tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	(tmp_path / "tasks.py").write_text(
		"print('BANNER')\nfrom camas import Task\nlint = Task('x')\n"
	)
	state = serve.resolve_project_quiet(tmp_path)
	assert isinstance(state, LoadOk)
	assert "lint" in state.tasks
	captured = capsys.readouterr()
	assert "BANNER" not in captured.out
	assert "BANNER" in captured.err


def test_task_names_empty_on_load_error(tmp_path: Path) -> None:
	assert serve.task_names(LoadErr(source=tmp_path / "tasks.py", exception=RuntimeError())) == ()


def test_tools_default_omits_rich_fields() -> None:
	(tool_list, tool_run, tool_check, tool_docs, tool_gate, tool_fix, tool_init, tool_gh_matrix) = (
		serve.tools(("a", "b"), Compat(emit_structured=False))
	)
	assert (tool_list.title, tool_list.outputSchema, tool_list.annotations) == (None, None, None)
	assert _task_enum(tool_run.inputSchema) == ["a", "b"]
	assert (tool_run.outputSchema, tool_run.annotations) == (None, None)
	assert (tool_check.title, tool_check.outputSchema, tool_check.annotations) == (None, None, None)
	assert (tool_docs.title, tool_docs.outputSchema, tool_docs.annotations) == (None, None, None)
	assert tool_check.inputSchema == serve.NO_ARGS_SCHEMA
	assert tool_docs.inputSchema == serve.NO_ARGS_SCHEMA
	assert tool_list.inputSchema["properties"]["expand_matrix"]["type"] == "boolean"
	assert tool_list.inputSchema["additionalProperties"] is False
	assert (tool_gate.title, tool_gate.outputSchema, tool_gate.annotations) == (None, None, None)
	assert _task_enum(tool_gate.inputSchema) == ["a", "b"]
	assert (tool_fix.title, tool_fix.outputSchema, tool_fix.annotations) == (None, None, None)
	assert _task_enum(tool_fix.inputSchema) == ["a", "b"]
	assert (tool_init.title, tool_init.outputSchema, tool_init.annotations) == (None, None, None)
	assert tool_init.inputSchema["properties"]["verbose"]["default"] is True
	assert tool_init.inputSchema["additionalProperties"] is False
	assert (tool_gh_matrix.title, tool_gh_matrix.outputSchema, tool_gh_matrix.annotations) == (
		None,
		None,
		None,
	)
	assert _task_enum(tool_gh_matrix.inputSchema) == ["a", "b"]


def test_tools_rich_includes_title_annotations_and_schema() -> None:
	(tool_list, tool_run, tool_check, tool_docs, tool_gate, tool_fix, tool_init, tool_gh_matrix) = (
		serve.tools((), Compat(emit_structured=True))
	)
	assert tool_list.title == "List camas tasks"
	assert tool_list.annotations is not None
	assert tool_list.annotations.readOnlyHint is True
	assert tool_run.annotations is not None
	assert tool_run.annotations.destructiveHint is True
	assert tool_list.outputSchema is not None
	assert tool_run.outputSchema is not None
	assert tool_run.outputSchema["type"] == "object"
	assert tool_check.title == "Check camas tasks"
	assert tool_check.outputSchema is not None
	assert tool_check.annotations is not None
	assert tool_check.annotations.readOnlyHint is True
	assert tool_docs.title == "How to author camas tasks"
	assert tool_docs.outputSchema is not None
	assert tool_docs.annotations is not None
	assert tool_docs.annotations.readOnlyHint is True
	assert tool_gate.title == "SA-delegation gate"
	assert tool_gate.outputSchema is not None
	assert tool_gate.annotations is not None
	assert tool_gate.annotations.readOnlyHint is True
	assert tool_fix.title == "Run deterministic autofix"
	assert tool_fix.outputSchema is not None
	assert tool_fix.annotations is not None
	assert tool_fix.annotations.destructiveHint is True
	assert tool_init.title == "Scaffold a starter tasks.py"
	assert tool_init.outputSchema is not None
	assert tool_init.annotations is not None
	assert tool_init.annotations.readOnlyHint is False
	assert tool_gh_matrix.title == "Emit GitHub Actions matrix"
	assert tool_gh_matrix.outputSchema is not None
	assert tool_gh_matrix.annotations is not None
	assert tool_gh_matrix.annotations.readOnlyHint is True


def test_list_tool_description_declares_github_default() -> None:
	(tool_list, *_rest) = serve.tools((), Compat(emit_structured=False))
	assert tool_list.description is not None
	assert "Config(github_task" in tool_list.description
	assert "null" in tool_list.description


def test_list_call_text_lists_tasks_and_markers(tmp_path: Path) -> None:
	ci = Sequential(PASS, name="ci")
	session = _session(
		{"ci": ci, "lint": PASS}, Config(default_task=ci, github_task=PASS), tmp_path
	)
	result = serve.list_call(session, {})
	assert result.isError is False
	text = _text(result)
	assert "default (developer's task): ci" in text
	assert "github default" in text
	assert "no-task camas_run runs: lint" in text
	assert "lint" in text
	assert "expand_matrix=true" not in text
	assert result.structuredContent is None


def test_list_call_text_omits_run_default_when_it_matches_default(tmp_path: Path) -> None:
	ci = Sequential(PASS, name="ci")
	session = _session({"ci": ci}, Config(default_task=ci), tmp_path)
	result = serve.list_call(session, {})
	assert result.isError is False
	text = _text(result)
	assert "default (developer's task): ci" in text
	assert "no-task camas_run runs" not in text


def test_list_call_load_error(tmp_path: Path) -> None:
	session = Session(LoadErr(tmp_path / "tasks.py", RuntimeError("boom")), tmp_path, Compat())
	result = serve.list_call(session, {})
	assert result.isError is True
	assert "camas --check" in _text(result)


def test_list_call_load_error_skew_prepends_reconnect_and_cli_hint(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text(
		"# /// script\n# dependencies = [\"camas==999.0.0\"]\n# ///\nraise RuntimeError('boom')\n"
	)
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	result = serve.list_call(session, {})
	assert result.isError is True
	text = _text(result)
	assert "MCP server is running camas" in text
	assert "reconnect the MCP server" in text
	assert "uv run tasks.py" in text
	assert "Tasks unavailable" in text


def test_list_call_structured_when_rich(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path, rich=True)
	result = serve.list_call(session, {})
	assert result.structuredContent is not None
	assert result.structuredContent["tasks"][0]["name"] == "lint"


_MATRIX = Parallel(Task("test {PY}"), matrix={"PY": ("3.13", "3.14")}, name="m")


def test_list_call_default_collapses_matrix(tmp_path: Path) -> None:
	result = serve.list_call(_session({"m": _MATRIX}, None, tmp_path), {})
	text = _text(result)
	assert "test {PY}" in text
	assert "[PY=3.13]" not in text
	assert "(matrix: PY=3.13/3.14)" in text
	assert "expand_matrix=true" in text


def test_list_call_expand_matrix_inlines_leaves(tmp_path: Path) -> None:
	result = serve.list_call(_session({"m": _MATRIX}, None, tmp_path), {"expand_matrix": True})
	text = _text(result)
	assert "[PY=3.13]" in text
	assert "[PY=3.14]" in text
	assert "expand_matrix=true" not in text


def test_list_call_invalid_arguments_is_tool_error(tmp_path: Path) -> None:
	result = serve.list_call(_session({"m": _MATRIX}, None, tmp_path), {"expand_matrix": "nope"})
	assert result.isError is True
	assert "invalid camas_list arguments" in _text(result)


async def test_run_call_pass(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	result = await serve.run_call(session, {"task": "lint"})
	assert result.isError is False
	assert "PASSED" in _text(result)


async def test_run_call_failure_is_not_a_tool_error(tmp_path: Path) -> None:
	session = _session({"bad": FAIL}, None, tmp_path)
	result = await serve.run_call(session, {"task": "bad"})
	assert result.isError is False
	text = _text(result)
	assert "FAILED" in text
	assert "boom" in text
	links = [c for c in result.content if isinstance(c, types.ResourceLink)]
	assert len(links) == 1
	log = tmp_path / ".camas" / "runs" / "bad" / "1" / "000_bad.log"
	assert log.is_file()
	assert "boom" in log.read_text()


async def test_run_call_nudges_when_failing_leaf_has_no_agent_format(tmp_path: Path) -> None:
	session = _session({"bad": FAIL}, None, tmp_path)
	result = await serve.run_call(session, {"task": "bad"})
	text = _text(result)
	assert "Tip:" in text
	assert "agent_format" in text


async def test_run_call_omits_nudge_when_passing(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	result = await serve.run_call(session, {"task": "lint"})
	assert "Tip:" not in _text(result)


async def test_run_call_skipped_reports_blocker(tmp_path: Path) -> None:
	session = _session({"ci": Sequential(FAIL, PASS, name="ci")}, None, tmp_path)
	result = await serve.run_call(session, {"task": "ci"})
	text = _text(result)
	assert "SKIP" in text
	assert "blocked by 'bad'" in text


async def test_run_call_dry_run_executes_nothing(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	result = await serve.run_call(session, {"task": "lint", "dry_run": True})
	assert result.isError is False
	assert "fully-resolved plan" in _text(result)
	assert result.structuredContent is None
	assert not (tmp_path / ".camas").exists()


async def test_run_call_dry_run_structured_when_rich(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path, rich=True)
	result = await serve.run_call(session, {"task": "lint", "dry_run": True})
	assert result.structuredContent is not None
	assert result.structuredContent["returncode"] == 0
	assert result.structuredContent["skipped"] == 1
	assert result.structuredContent["leaves"][0]["name"] == "lint"


async def test_run_call_dry_run_names_anonymous_leaf_by_command(tmp_path: Path) -> None:
	node = Parallel(Task("echo anon", cwd="work"), name="grp")
	session = _session({"grp": node}, None, tmp_path, rich=True)
	result = await serve.run_call(session, {"task": "grp", "dry_run": True})
	assert result.structuredContent is not None
	assert result.structuredContent["leaves"][0]["name"] == "echo anon"
	assert result.structuredContent["leaves"][0]["cwd"] == "work"


async def test_run_call_dry_run_no_task_previews_default(tmp_path: Path) -> None:
	"""camas_run(dry_run=true) with no task resolves and previews the default task."""
	default_task = Task(("python", "-c", "print('default')"), name="ci")
	config = Config(default_task=default_task)
	session = _session({"ci": default_task}, config, tmp_path)
	result = await serve.run_call(session, {"dry_run": True})
	assert result.isError is False
	assert "fully-resolved plan" in _text(result)


async def test_run_call_dry_run_no_task_no_default_errors(tmp_path: Path) -> None:
	"""camas_run(dry_run=true) with no task and no config default still errors."""
	config = Config()  # no default_task
	session = _session({"ci": PASS}, config, tmp_path)
	result = await serve.run_call(session, {"dry_run": True})
	assert result.isError is True
	assert "has no task to run" in _text(result)


async def test_run_call_log_path_in_structured_content_when_rich(tmp_path: Path) -> None:
	session = _session({"bad": FAIL}, None, tmp_path, rich=True)
	result = await serve.run_call(session, {"task": "bad"})
	assert result.structuredContent is not None
	assert result.structuredContent["leaves"][0]["log"].endswith("000_bad.log")


async def test_run_call_records_timing(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	await serve.run_call(session, {"task": "lint"})
	cache = timings.load(session.camas_dir)
	assert cache["lint"].samples == 1
	assert cache["lint"].elapsed_s >= 0.0


async def test_run_call_dry_run_records_no_timing(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	await serve.run_call(session, {"task": "lint", "dry_run": True})
	assert timings.load(session.camas_dir) == {}


async def test_list_reflects_recorded_timing(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	await serve.run_call(session, {"task": "lint"})
	assert "n=1" in _text(serve.list_call(session, {}))


async def test_run_call_unknown_task_is_tool_error(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	result = await serve.run_call(session, {"task": "nope"})
	assert result.isError is True
	assert "no task named 'nope'" in _text(result)


async def test_run_call_validation_error_is_tool_error(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	result = await serve.run_call(session, {"task": "lint", "jobs": 0})
	assert result.isError is True
	assert "invalid camas_run arguments" in _text(result)


async def test_run_call_load_error(tmp_path: Path) -> None:
	session = Session(LoadErr(tmp_path / "tasks.py", RuntimeError("boom")), tmp_path, Compat())
	result = await serve.run_call(session, {"task": "lint"})
	assert result.isError is True


async def test_run_call_load_error_no_pin_omits_reconnect_hint(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text("raise RuntimeError('boom')\n")
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	result = await serve.run_call(session, {"task": "lint"})
	assert result.isError is True
	text = _text(result)
	assert "reconnect the MCP server" not in text
	assert "Tasks unavailable" in text


async def test_run_call_structured_when_rich(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path, rich=True)
	result = await serve.run_call(session, {"task": "lint"})
	assert result.structuredContent is not None
	assert result.structuredContent["returncode"] == 0


async def test_run_call_unknown_matrix_axis_is_tool_error(tmp_path: Path) -> None:
	node = Parallel(Task("test {PY}"), matrix={"PY": ("3.13",)}, name="m")
	session = _session({"m": node}, None, tmp_path)
	result = await serve.run_call(session, {"task": "m", "matrix_overrides": {"NOPE": ["x"]}})
	assert result.isError is True
	assert "unknown matrix axis" in _text(result)
	assert "NOPE" in _text(result)


async def test_run_call_concurrent_runs_use_distinct_log_dirs(tmp_path: Path) -> None:
	session = _session({"bad": FAIL}, None, tmp_path)
	r1, r2 = await asyncio.gather(
		serve.run_call(session, {"task": "bad"}),
		serve.run_call(session, {"task": "bad"}),
	)
	run_dirs = sorted(p.name for p in (tmp_path / ".camas" / "runs" / "bad").iterdir())
	assert run_dirs == ["1", "2"]
	uris = {str(c.uri) for r in (r1, r2) for c in r.content if isinstance(c, types.ResourceLink)}
	assert len(uris) == 2


def test_server_reports_camas_version(tmp_path: Path) -> None:
	from importlib.metadata import version

	session = _session({"lint": PASS}, None, tmp_path)
	opts = serve.build_server(session).create_initialization_options()
	assert opts.server_version == version("camas")


def test_server_advertises_instructions(tmp_path: Path) -> None:
	server = serve.build_server(_session({"lint": PASS}, None, tmp_path))
	opts = server.create_initialization_options()
	assert opts.instructions is not None
	assert "camas_run" in opts.instructions


_VALID_TASKS = "from camas import Task\nlint = Task('ruff check .')\n"


def _fixed_checker(result: TypeCheckResult) -> Callable[[Path], TypeCheckResult]:
	def checker(_source: Path) -> TypeCheckResult:
		return result

	return checker


def test_check_call_ok_typechecked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	(tmp_path / "tasks.py").write_text(_VALID_TASKS)
	monkeypatch.setattr("camas.mcp.result.run_typecheck", _fixed_checker(CheckerOk("ty")))
	result = serve.check_call(_resolved_session(tmp_path))
	assert result.isError is False
	assert "OK" in _text(result)
	assert "type-checked clean with ty" in _text(result)


def test_check_call_type_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	(tmp_path / "tasks.py").write_text(_VALID_TASKS)
	monkeypatch.setattr(
		"camas.mcp.result.run_typecheck", _fixed_checker(CheckerErr("mypy", "error: bad type"))
	)
	result = serve.check_call(_resolved_session(tmp_path))
	assert result.isError is False
	text = _text(result)
	assert "TYPE ERRORS" in text
	assert "error: bad type" in text


def test_check_call_no_checker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	(tmp_path / "tasks.py").write_text(_VALID_TASKS)
	monkeypatch.setattr("camas.mcp.result.run_typecheck", _fixed_checker(CheckerNotFound()))
	assert "no type checker is available" in _text(serve.check_call(_resolved_session(tmp_path)))


def test_check_call_load_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	(tmp_path / "tasks.py").write_text("raise RuntimeError('boom in tasks')\n")
	monkeypatch.setattr("camas.mcp.result.run_typecheck", _fixed_checker(CheckerNotFound()))
	result = serve.check_call(_resolved_session(tmp_path))
	assert result.isError is False
	text = _text(result)
	assert "LOAD ERROR" in text
	assert "boom in tasks" in text


def test_check_call_load_error_pyproject(tmp_path: Path) -> None:
	(tmp_path / "pyproject.toml").write_text("[tool.camas.tasks]\nlint = 123\n")
	text = _text(serve.check_call(_resolved_session(tmp_path)))
	assert "LOAD ERROR" in text
	assert "pyproject.toml" in text


def test_check_call_no_tasks(tmp_path: Path) -> None:
	result = serve.check_call(_resolved_session(tmp_path))
	assert result.isError is False
	assert "no tasks.py" in _text(result)


def test_check_call_pyproject_ok_skips_typecheck(tmp_path: Path) -> None:
	(tmp_path / "pyproject.toml").write_text('[tool.camas.tasks]\nlint = "ruff check ."\n')
	result = serve.check_call(_resolved_session(tmp_path))
	assert result.isError is False
	text = _text(result)
	assert "OK" in text
	assert "no type checker ran" in text


async def test_check_via_client_reflects_fresh_disk_state(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr("camas.mcp.result.run_typecheck", _fixed_checker(CheckerOk("ty")))
	tasks_py = tmp_path / "tasks.py"
	tasks_py.write_text("from camas import Task\na = Task('echo a')\n")
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		tasks_py.write_text("from camas import Task\na = Task('echo a')\nb = Task('echo b')\n")
		result = await client.call_tool("camas_check", {})
		assert result.isError is False
		assert "2 task(s)" in _text(result)


def test_check_call_structured_when_rich(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	(tmp_path / "tasks.py").write_text(_VALID_TASKS)
	monkeypatch.setattr("camas.mcp.result.run_typecheck", _fixed_checker(CheckerOk("ty")))
	result = serve.check_call(_resolved_session(tmp_path, rich=True))
	assert result.structuredContent is not None
	assert result.structuredContent["status"] == "ok"
	assert result.structuredContent["checker"] == "ty"


def test_check_call_reports_server_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	from importlib.metadata import version

	(tmp_path / "tasks.py").write_text(_VALID_TASKS)
	monkeypatch.setattr("camas.mcp.result.run_typecheck", _fixed_checker(CheckerOk("ty")))
	result = serve.check_call(_resolved_session(tmp_path, rich=True))
	assert result.structuredContent is not None
	assert result.structuredContent["server_version"] == version("camas")
	assert f"camas server version: {version('camas')}" in _text(result)


def test_check_call_server_version_degrades_when_missing(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	from importlib.metadata import PackageNotFoundError

	def _raise(name: str) -> str:
		raise PackageNotFoundError(name)

	(tmp_path / "tasks.py").write_text(_VALID_TASKS)
	monkeypatch.setattr("camas.mcp.result.run_typecheck", _fixed_checker(CheckerOk("ty")))
	monkeypatch.setattr("camas.mcp.serve.version", _raise)
	result = serve.check_call(_resolved_session(tmp_path, rich=True))
	assert result.structuredContent is not None
	assert result.structuredContent["server_version"] is None
	assert "camas server version" not in _text(result)


def test_check_call_reports_scope_warnings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	(tmp_path / "tasks.py").write_text(
		"from camas import Task\ncargo = Task('cargo build', name='cargo', paths='.')\n"
	)
	monkeypatch.setattr("camas.mcp.result.run_typecheck", _fixed_checker(CheckerOk("ty")))
	result = serve.check_call(_resolved_session(tmp_path, rich=True))
	assert result.isError is False
	text = _text(result)
	assert "camas_check: OK" in text
	assert "Warnings (advisory — not failures):" in text
	assert "cargo" in text
	assert result.structuredContent is not None
	assert result.structuredContent["status"] == "ok"
	assert len(result.structuredContent["warnings"]) == 1


def test_check_text_renders_warnings_block_and_keeps_ok_status() -> None:
	resp = wire.CheckResponse(
		status="ok",
		source="/x/tasks.py",
		task_count=1,
		checker="ty",
		warnings=("task 'cargo' sets paths= but its command has no {paths} token",),
	)
	text = serve.check_text(resp)
	assert resp.status == "ok"
	assert "camas_check: OK" in text
	assert "Warnings (advisory — not failures):" in text
	assert "  - task 'cargo' sets paths=" in text


def test_check_text_omits_warnings_block_when_clean() -> None:
	resp = wire.CheckResponse(status="ok", source="/x/tasks.py", task_count=1, checker="ty")
	assert "Warnings" not in serve.check_text(resp)


def test_check_text_omits_server_version_when_none() -> None:
	resp = wire.CheckResponse(
		status="ok", source="/x/tasks.py", task_count=1, checker="ty", server_version=None
	)
	assert "camas server version" not in serve.check_text(resp)


def test_docs_call_serves_tutorial_and_source(tmp_path: Path) -> None:
	result = serve.docs_call(_session({}, None, tmp_path))
	assert result.isError is False
	text = _text(result)
	assert "authoring guide" in text
	assert "source of truth" in text
	assert "Task(" in text
	assert result.structuredContent is None


def test_docs_call_structured_when_rich(tmp_path: Path) -> None:
	result = serve.docs_call(_session({}, None, tmp_path, rich=True))
	assert result.structuredContent is not None
	assert result.structuredContent["source"].endswith("camas")
	assert "Sequential" in result.structuredContent["tutorial"]


async def test_call_routes_docs(tmp_path: Path) -> None:
	result = await serve.call(_session({}, None, tmp_path), "camas_docs", {})
	assert "authoring guide" in _text(result)


def test_build_server_declares_list_changed(tmp_path: Path) -> None:
	from mcp.server.lowlevel.server import NotificationOptions

	server = serve.build_server(_session({"lint": PASS}, None, tmp_path))
	opts = server.create_initialization_options(NotificationOptions(tools_changed=True))
	assert opts.capabilities.tools is not None
	assert opts.capabilities.tools.listChanged is True


async def test_check_via_client_refreshes_catalog_and_notifies(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr("camas.mcp.result.run_typecheck", _fixed_checker(CheckerOk("ty")))
	(tmp_path / "tasks.py").write_text(
		"from camas import Task\na = Task(('python', '-c', 'pass'))\n"
	)
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		first = await client.list_tools()
		run_first = next(t for t in first.tools if t.name == "camas_run")
		assert _task_enum(run_first.inputSchema) == ["a"]
		(tmp_path / "tasks.py").write_text(
			"from camas import Task\n"
			"a = Task(('python', '-c', 'pass'))\n"
			"b = Task(('python', '-c', 'pass'))\n"
		)
		assert "OK" in _text(await client.call_tool("camas_check", {}))
		after = await client.list_tools()
		run_after = next(t for t in after.tools if t.name == "camas_run")
		assert set(_task_enum(run_after.inputSchema)) == {"a", "b"}
		ran = await client.call_tool("camas_run", {"task": "b"})
		assert ran.isError is False
		assert "PASSED" in _text(ran)


def test_resolve_run_node_unknown_task_raises() -> None:
	with pytest.raises(ValueError, match="no task named 'x'"):
		serve.resolve_run_node({"lint": PASS}, wire.RunRequest(task="x"))


def test_resolve_run_node_applies_overrides_and_args() -> None:
	node = Parallel(Task("test {PY}"), matrix={"PY": ("3.12", "3.13")}, name="m")
	name, resolved = serve.resolve_run_node(
		{"m": node}, wire.RunRequest(task="m", matrix_overrides={"PY": ["3.13"]})
	)
	assert name == "m"
	assert isinstance(resolved, Parallel)
	assert resolved.matrix == {"PY": ("3.13",)}
	leaf_name, leaf = serve.resolve_run_node(
		{"t": Task("pytest", name="t")}, wire.RunRequest(task="t", args=["-v"])
	)
	assert leaf_name == "t"
	assert isinstance(leaf, Task)
	assert leaf.cmd == "pytest -v"


def test_resolve_run_node_requires_task() -> None:
	with pytest.raises(ValueError, match="has no task to run"):
		serve.resolve_run_node({"lint": PASS}, wire.RunRequest())


def test_resolve_run_node_no_task_applies_args() -> None:
	name, leaf = serve.resolve_run_node(
		{}, wire.RunRequest(args=["-v"]), Config(default_task=Task("pytest", name="t"))
	)
	assert name == "t"
	assert isinstance(leaf, Task)
	assert leaf.cmd == "pytest -v"


def _record(base: Path, leaves: list[tuple[str, float]]) -> None:
	camas = base / ".camas"
	camas.mkdir(exist_ok=True)
	timings.record(camas, leaves)


_FMT = Task(("python", "-c", "print('fmt')"), name="fmt", mutates=True)
_LINT = Task(("python", "-c", "print('lint')"), name="lint")
_SLOW = Task(("python", "-c", "print('slow')"), name="slow")


async def test_run_call_under_selects_runs_and_reports(tmp_path: Path) -> None:
	default = Sequential(_FMT, Parallel(_LINT, _SLOW), name="all")
	_record(tmp_path, [("fmt", 0.1), ("lint", 0.2), ("slow", 9.0)])
	session = _session({"all": default}, Config(default_task=default), tmp_path, rich=True)
	result = await serve.run_call(session, {"under": 1.0})
	assert result.isError is False
	text = _text(result)
	assert "Time budget 1.00s — running 2 leaf(s)" in text
	assert "over budget: slow ~9.00s" in text
	assert "PASSED" in text
	assert result.structuredContent is not None
	budget = result.structuredContent["budget"]
	assert set(budget["selected"]) == {"fmt", "lint"}
	assert next(e for e in budget["excluded"] if e["name"] == "slow")["reason"] == "over_budget"


async def test_run_call_under_dry_run_previews(tmp_path: Path) -> None:
	_record(tmp_path, [("a", 0.1)])
	a = Task(("python", "-c", "print('a')"), name="a")
	session = _session({"p": Parallel(a, name="p")}, None, tmp_path, rich=True)
	result = await serve.run_call(session, {"task": "p", "under": 1.0, "dry_run": True})
	assert result.isError is False
	assert "Dry run" in _text(result)
	assert result.structuredContent is not None
	assert result.structuredContent["budget"]["budget_s"] == 1.0


async def test_run_call_under_nothing_fits(tmp_path: Path) -> None:
	_record(tmp_path, [("a", 9.0)])
	a = Task(("python", "-c", "print('a')"), name="a")
	session = _session({"p": Parallel(a, name="p")}, None, tmp_path)
	result = await serve.run_call(session, {"task": "p", "under": 0.5})
	assert result.isError is False
	assert "Nothing ran" in _text(result)


async def test_run_call_under_reports_untimed(tmp_path: Path) -> None:
	_record(tmp_path, [("a", 0.1)])
	a = Task(("python", "-c", "print('a')"), name="a")
	b = Task(("python", "-c", "print('b')"), name="b")
	session = _session({"p": Parallel(a, b, name="p")}, None, tmp_path, rich=True)
	result = await serve.run_call(session, {"task": "p", "under": 1.0})
	assert "unmeasured (running to record an estimate): b" in _text(result)
	assert result.structuredContent is not None
	assert "b" in result.structuredContent["budget"]["unmeasured"]


async def test_run_call_under_no_task_no_default_errors(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	result = await serve.run_call(session, {"under": 1.0})
	assert result.isError is True
	assert "has no task to run" in _text(result)


async def test_run_call_under_no_task_config_present_no_default_errors(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, Config(), tmp_path)
	result = await serve.run_call(session, {"under": 1.0})
	assert result.isError is True
	assert "has no task to run" in _text(result)


async def test_run_call_under_no_task_labels_by_default_name(tmp_path: Path) -> None:
	default = Sequential(_FMT, Parallel(_LINT, _SLOW), name="ci")
	_record(tmp_path, [("fmt", 0.1), ("lint", 0.2), ("slow", 9.0)])
	session = _session({"ci": default}, Config(default_task=default), tmp_path)
	result = await serve.run_call(session, {"under": 5.0})
	assert result.isError is False
	assert "camas_run 'ci'" in _text(result)


async def test_run_call_under_rejects_args(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, Config(default_task=PASS), tmp_path)
	result = await serve.run_call(session, {"under": 1.0, "args": ["-v"]})
	assert result.isError is True
	assert "'args'" in _text(result)


async def test_run_call_missing_task_errors(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	result = await serve.run_call(session, {})
	assert result.isError is True
	assert "has no task to run" in _text(result)


async def test_run_call_no_task_runs_default(tmp_path: Path) -> None:
	default = Sequential(PASS, name="ci")
	session = _session({"ci": default}, Config(default_task=default), tmp_path)
	result = await serve.run_call(session, {})
	assert result.isError is False
	assert "camas_run 'ci'" in _text(result)
	assert (tmp_path / ".camas" / "runs" / "ci").is_dir()


async def test_run_call_no_task_anonymous_default_labels_default_task(tmp_path: Path) -> None:
	anon = Task(("python", "-c", "print('ok')"))
	session = _session({}, Config(default_task=anon), tmp_path)
	result = await serve.run_call(session, {})
	assert result.isError is False
	assert "camas_run 'default task'" in _text(result)


async def test_run_call_no_task_config_present_no_default_errors(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, Config(), tmp_path)
	result = await serve.run_call(session, {})
	assert result.isError is True
	assert "has no task to run" in _text(result)


async def test_run_call_no_task_applies_matrix_overrides(tmp_path: Path) -> None:
	default = Parallel(Task("echo {PY}", name="t"), matrix={"PY": ("3.12", "3.13")}, name="m")
	session = _session({"m": default}, Config(default_task=default), tmp_path, rich=True)
	result = await serve.run_call(session, {"dry_run": True, "matrix_overrides": {"PY": ["3.13"]}})
	assert result.isError is False
	assert result.structuredContent is not None
	assert len(result.structuredContent["leaves"]) == 1
	assert "echo 3.13" in _text(result)


async def test_run_call_no_task_unknown_axis_errors(tmp_path: Path) -> None:
	default = Parallel(Task("echo {PY}", name="t"), matrix={"PY": ("3.13",)}, name="m")
	session = _session({"m": default}, Config(default_task=default), tmp_path)
	result = await serve.run_call(session, {"matrix_overrides": {"NOPE": ["x"]}})
	assert result.isError is True
	assert "unknown matrix axis" in _text(result)


async def test_run_call_under_unknown_task_errors(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	result = await serve.run_call(session, {"task": "nope", "under": 1.0})
	assert result.isError is True
	assert "no task named 'nope'" in _text(result)


async def test_run_call_under_applies_matrix_override(tmp_path: Path) -> None:
	node = Parallel(Task("echo {PY}", name="t"), matrix={"PY": ("3.12", "3.13")}, name="m")
	session = _session({"m": node}, None, tmp_path)
	result = await serve.run_call(
		session, {"task": "m", "under": 1.0, "matrix_overrides": {"PY": ["3.13"]}}
	)
	assert result.isError is False
	assert "running 1 leaf(s)" in _text(result)


async def test_run_call_under_bad_matrix_override_errors(tmp_path: Path) -> None:
	node = Parallel(Task("echo {PY}", name="t"), matrix={"PY": ("3.12",)}, name="m")
	session = _session({"m": node}, None, tmp_path)
	result = await serve.run_call(
		session, {"task": "m", "under": 1.0, "matrix_overrides": {"NOPE": ["x"]}}
	)
	assert result.isError is True
	assert "unknown matrix axis" in _text(result)


def test_create_run_log_dir_namespaces_by_task_and_is_idempotent(tmp_path: Path) -> None:
	camas_dir = tmp_path / ".camas"
	first = serve.create_run_log_dir(camas_dir, "ci", 1)
	assert (camas_dir / ".gitignore").read_text() == "*\n"
	assert first == camas_dir / "runs" / "ci" / "1"
	second = serve.create_run_log_dir(camas_dir, "ci", 2)
	assert second.is_dir()


def _mk_run_dir(task_dir: Path, name: str, *, mtime: float) -> Path:
	d = task_dir / name
	d.mkdir(parents=True)
	os.utime(d, ns=(int(mtime * 1_000_000_000), int(mtime * 1_000_000_000)))
	return d


def test_prune_run_dirs_noop_under_keep_n(tmp_path: Path) -> None:
	"""Fewer run dirs than ``keep_n`` — nothing is pruned."""
	task_dir = tmp_path / "runs" / "ci"
	dirs = [_mk_run_dir(task_dir, str(i), mtime=i) for i in range(1, 4)]
	assert serve.prune_run_dirs(task_dir, dirs[-1], keep_n=10) == ()
	assert all(d.is_dir() for d in dirs)


def test_prune_run_dirs_caps_at_keep_n(tmp_path: Path) -> None:
	task_dir = tmp_path / "runs" / "ci"
	dirs = [_mk_run_dir(task_dir, str(i), mtime=i) for i in range(1, 13)]
	removed = serve.prune_run_dirs(task_dir, dirs[-1], keep_n=10)
	assert {d.name for d in removed} == {"1", "2"}
	assert sorted((d.name for d in task_dir.iterdir()), key=int) == [str(i) for i in range(3, 13)]


def test_prune_run_dirs_keeps_newest_removes_oldest(tmp_path: Path) -> None:
	task_dir = tmp_path / "runs" / "ci"
	oldest = _mk_run_dir(task_dir, "1", mtime=1)
	newest = _mk_run_dir(task_dir, "2", mtime=100)
	removed = serve.prune_run_dirs(task_dir, newest, keep_n=1)
	assert removed == (oldest,)
	assert not oldest.exists()
	assert newest.exists()


def test_prune_run_dirs_always_keeps_the_just_created_dir(tmp_path: Path) -> None:
	"""``keep`` survives even though it is the newest and would already rank first by mtime —
	the guarantee doesn't rely on mtime ordering."""
	task_dir = tmp_path / "runs" / "ci"
	older = [_mk_run_dir(task_dir, str(i), mtime=i) for i in range(1, 11)]
	keep = _mk_run_dir(task_dir, "11", mtime=11)
	removed = serve.prune_run_dirs(task_dir, keep, keep_n=1)
	assert keep.exists()
	assert set(removed) == set(older)


def test_prune_run_dirs_ignores_non_numeric_entries(tmp_path: Path) -> None:
	task_dir = tmp_path / "runs" / "ci"
	oldest = _mk_run_dir(task_dir, "1", mtime=1)
	keep = _mk_run_dir(task_dir, "2", mtime=2)
	stray_file = task_dir / "notes.txt"
	stray_file.write_text("keep me")
	stray_dir = task_dir / "latest"
	stray_dir.mkdir()
	removed = serve.prune_run_dirs(task_dir, keep, keep_n=1)
	assert removed == (oldest,)
	assert stray_file.exists()
	assert stray_dir.exists()


def test_prune_run_dirs_ignores_non_decimal_digit_entries(tmp_path: Path) -> None:
	"""``'²'.isdigit()`` is true but ``int('²')`` raises — such an entry is foreign, not a run."""
	task_dir = tmp_path / "runs" / "ci"
	oldest = _mk_run_dir(task_dir, "1", mtime=1)
	keep = _mk_run_dir(task_dir, "2", mtime=2)
	superscript = _mk_run_dir(task_dir, "²", mtime=3)
	removed = serve.prune_run_dirs(task_dir, keep, keep_n=1)
	assert removed == (oldest,)
	assert superscript.exists()


def test_prune_run_dirs_deletion_is_best_effort(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Deletion never raises — ``shutil.rmtree`` is called with ``ignore_errors=True``."""
	task_dir = tmp_path / "runs" / "ci"
	oldest = _mk_run_dir(task_dir, "1", mtime=1)
	keep = _mk_run_dir(task_dir, "2", mtime=2)
	calls: list[tuple[Path, bool]] = []

	def _fake_rmtree(d: Path, ignore_errors: bool = False) -> None:
		calls.append((d, ignore_errors))

	monkeypatch.setattr("camas.mcp.serve.shutil.rmtree", _fake_rmtree)
	removed = serve.prune_run_dirs(task_dir, keep, keep_n=1)
	assert removed == (oldest,)
	assert calls == [(oldest, True)]
	assert oldest.exists()


def test_create_run_log_dir_prunes_to_run_log_keep(tmp_path: Path) -> None:
	"""A long session accumulating runs beyond ``RUN_LOG_KEEP`` stays capped, newest kept."""
	camas_dir = tmp_path / ".camas"
	total = serve.RUN_LOG_KEEP + 3
	created: list[Path] = []
	for i in range(1, total + 1):
		d = serve.create_run_log_dir(camas_dir, "ci", i)
		os.utime(d, ns=(i * 1_000_000_000, i * 1_000_000_000))
		created.append(d)
	remaining = sorted(int(p.name) for p in (camas_dir / "runs" / "ci").iterdir())
	assert remaining == list(range(total - serve.RUN_LOG_KEEP + 1, total + 1))
	assert created[-1].exists()


def test_write_logs_writes_output_skips_empty_and_skipped(tmp_path: Path) -> None:
	result = RunResult(
		returncode=1,
		results=(
			TaskResult("ok task", Finished(0, 0.1, (b"hello\n",))),
			TaskResult("empty", Finished(0, 0.1, ())),
			TaskResult("stopped", Stopped(130, 0.2, (b"\x1b[31mbye\x1b[0m\n",))),
			TaskResult("skip", Skipped(1, "ok task")),
		),
		elapsed=0.3,
	)
	logs = serve.write_logs(tmp_path, result)
	assert logs[0] is not None
	assert logs[1] is None
	assert logs[2] is not None
	assert logs[3] is None
	assert (tmp_path / "000_ok_task.log").read_text() == "hello\n"
	stopped = (tmp_path / "002_stopped.log").read_text()
	assert "bye" in stopped
	assert "\x1b" not in stopped


def test_write_logs_skips_errored_leaf(tmp_path: Path) -> None:
	result = RunResult(
		returncode=1,
		results=(TaskResult("ghost", Errored(127, "no such file or directory: ghost")),),
		elapsed=0.01,
	)
	logs = serve.write_logs(tmp_path, result)
	assert logs == (None,)


def test_failing_log_links_only_for_failed_leaves(tmp_path: Path) -> None:
	resp = wire.RunResponse(
		returncode=1,
		elapsed=0.1,
		passed=1,
		failed=1,
		skipped=0,
		interrupt_count=0,
		leaves=(
			wire.LeafReport(
				name="ok", command="c", completion=wire.Finished(returncode=0, elapsed=0.1)
			),
			wire.LeafReport(
				name="bad", command="c", completion=wire.Finished(returncode=1, elapsed=0.1)
			),
		),
	)
	links = serve.failing_log_links(resp, (tmp_path / "a.log", tmp_path / "b.log"))
	assert len(links) == 1
	assert links[0].name == "bad"


def test_leaf_lines_covers_errored_status() -> None:
	leaf = wire.LeafReport(
		name="ghost",
		command="does-not-exist",
		completion=wire.Errored(
			returncode=127, message="no such file or directory: does-not-exist"
		),
	)
	lines = serve.leaf_lines(leaf, None)
	assert lines == ["ERROR  ghost (exit 127): no such file or directory: does-not-exist"]


def test_run_text_covers_every_status() -> None:
	resp = wire.RunResponse(
		returncode=1,
		elapsed=1.0,
		passed=1,
		failed=2,
		skipped=2,
		interrupt_count=1,
		leaves=(
			wire.LeafReport(
				name="ok1", command="c", completion=wire.Finished(returncode=0, elapsed=0.1)
			),
			wire.LeafReport(
				name="bad",
				command="c",
				truncated=True,
				completion=wire.Finished(returncode=2, elapsed=0.2, output=["err"]),
			),
			wire.LeafReport(
				name="stp",
				command="c",
				completion=wire.Stopped(returncode=130, elapsed=0.3, output=["bye"]),
			),
			wire.LeafReport(
				name="s1", command="c", completion=wire.Skipped(returncode=2, blocked_by="bad")
			),
			wire.LeafReport(name="s2", command="c", completion=wire.Skipped(returncode=2)),
		),
	)
	ok1_log = Path("/logs/ok1.log")
	bad_log = Path("/logs/bad.log")
	text = serve.run_text("ci", resp, (ok1_log, bad_log, None, None, None))
	assert "FAILED (returncode=1)" in text
	assert "ok     ok1" in text
	assert str(ok1_log) not in text
	assert "FAIL   bad" in text
	assert "    err" in text
	assert "truncated" in text
	assert f"full log: {bad_log}" in text
	assert "STOP   stp" in text
	assert "    bye" in text
	assert "SKIP   s1 — blocked by 'bad'" in text
	assert "SKIP   s2" in text
	assert "s2 —" not in text


def test_list_text_renders_matrix_and_no_markers() -> None:
	resp = wire.ListResponse(
		tasks=(
			wire.TaskInfo(
				name="m", command_preview="test {PY}", matrix_axes={"PY": ["3.13", "3.14"]}
			),
		),
		default=None,
	)
	text = serve.list_text(resp, expand_matrix=False)
	assert "default" not in text
	assert "(matrix: PY=3.13/3.14)" in text


def test_list_text_shows_help_and_expansion_together() -> None:
	resp = wire.ListResponse(
		tasks=(
			wire.TaskInfo(
				name="lint",
				help="Lint sources",
				command_preview='Task("ruff check .", name="lint")',
			),
		),
		default=None,
	)
	text = serve.list_text(resp, expand_matrix=False)
	assert "Lint sources" in text
	assert 'Task("ruff check .", name="lint")' in text


def test_dry_run_text_shows_resolved_commands() -> None:
	text = serve.dry_run_text(Parallel(Task("a"), Task("b"), name="p"))
	assert "fully-resolved plan" in text
	assert "a" in text
	assert "b" in text


def test_list_text_shows_timing_with_slowest_leaf() -> None:
	resp = wire.ListResponse(
		tasks=(
			wire.TaskInfo(
				name="check",
				command_preview="x",
				estimated_s=32.0,
				samples=2,
				slowest_leaf="test",
				slowest_s=31.9,
			),
		),
		default=None,
	)
	assert "[~32.00s, slowest test 31.90s, n=2]" in serve.list_text(resp, expand_matrix=False)


def test_list_text_omits_slowest_when_it_is_the_task_itself() -> None:
	resp = wire.ListResponse(
		tasks=(
			wire.TaskInfo(
				name="lint",
				command_preview="x",
				estimated_s=0.2,
				samples=1,
				slowest_leaf="lint",
				slowest_s=0.2,
			),
		),
		default=None,
	)
	line = serve.list_text(resp, expand_matrix=False)
	assert "[~0.20s, n=1]" in line
	assert "slowest" not in line


def test_error_result_is_tool_error() -> None:
	err = serve.error_result("nope")
	assert err.isError is True
	assert _text(err) == "nope"


async def test_in_memory_round_trip(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text(
		"from camas import Config, Task\n"
		"lint = Task(('python', '-c', 'pass'))\n"
		"_ = Config(default_task=lint)\n"
	)
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		listed = await client.list_tools()
		assert {t.name for t in listed.tools} == {
			"camas_list",
			"camas_run",
			"camas_check",
			"camas_docs",
			"camas_gate",
			"camas_fix",
			"camas_init",
			"camas_github_matrix",
		}
		catalog = await client.call_tool("camas_list", {})
		assert "lint" in _text(catalog)
		run = await client.call_tool("camas_run", {"task": "lint"})
		assert run.isError is False
		assert "PASSED" in _text(run)
		unknown = await client.call_tool("camas_run", {"task": "lint", "dry_run": True})
		assert "fully-resolved plan" in _text(unknown)


async def test_list_via_client_expands_matrix_on_request(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text(
		"from camas import Parallel, Task\nm = Parallel(Task('test {PY}'), matrix={'PY': ('3.13', '3.14')})\n"
	)
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		collapsed = _text(await client.call_tool("camas_list", {}))
		assert "[PY=3.13]" not in collapsed
		expanded = _text(await client.call_tool("camas_list", {"expand_matrix": True}))
		assert "[PY=3.13]" in expanded
		assert "[PY=3.14]" in expanded


async def test_call_unknown_tool_is_error(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	result = await serve.call(session, "camas_nope", {})
	assert result.isError is True
	assert "unknown tool 'camas_nope'" in _text(result)


def test_session_refresh_picks_up_edits(tmp_path: Path) -> None:
	tasks_py = tmp_path / "tasks.py"
	tasks_py.write_text("from camas import Task\na = Task('x')\n")
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	assert serve.task_names(session.project) == ("a",)
	tasks_py.write_text("from camas import Task\na = Task('x')\nb = Task('y')\n")
	session.refresh()
	assert set(serve.task_names(session.project)) == {"a", "b"}


def test_session_refresh_reresolves_when_file_unchanged(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text("from camas import Task\na = Task('x')\n")
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	pinned = session.project
	session.refresh()
	assert session.project is not pinned
	assert serve.task_names(session.project) == ("a",)


def test_session_refresh_noop_without_source(tmp_path: Path) -> None:
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	session.refresh()
	assert serve.task_names(session.project) == ()


def test_session_refresh_on_deleted_source_reresolves(tmp_path: Path) -> None:
	tasks_py = tmp_path / "tasks.py"
	tasks_py.write_text("from camas import Task\na = Task('x')\n")
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	tasks_py.unlink()
	session.refresh()
	assert serve.task_names(session.project) == ()


def test_session_refresh_handles_load_error_source(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text("raise RuntimeError('boom')\n")
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	assert isinstance(session.project, LoadErr)
	session.refresh()
	assert isinstance(session.project, LoadErr)


def test_session_refresh_clears_stale_version_warning(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text(_VALID_TASKS)
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	session.version_warning = "WARNING: stale"
	session.refresh()
	assert session.version_warning is None


async def test_run_via_client_picks_up_new_task_without_check(tmp_path: Path) -> None:
	tasks_py = tmp_path / "tasks.py"
	tasks_py.write_text("from camas import Task\na = Task(('python', '-c', 'pass'))\n")
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		tasks_py.write_text(
			"from camas import Task\n"
			"a = Task(('python', '-c', 'pass'))\n"
			"b = Task(('python', '-c', 'pass'))\n"
		)
		ran = await client.call_tool("camas_run", {"task": "b"})
		assert ran.isError is False
		assert "PASSED" in _text(ran)


def _glob_check_tasks_py() -> str:
	return (
		"from pathlib import Path\n"
		"from camas import Task\n"
		"_t = [str(p) for p in sorted((Path(__file__).parent / 'targets').glob('*.txt'))]\n"
		"check = Task(('python', '-c', "
		"'import os,sys; sys.exit(0 if all(map(os.path.exists, sys.argv[1:])) else 1)', "
		"*_t), name='check')\n"
	)


def test_session_refresh_reflects_filesystem_derived_command(tmp_path: Path) -> None:
	targets = tmp_path / "targets"
	targets.mkdir()
	(targets / "a.txt").write_text("")
	(tmp_path / "tasks.py").write_text(_glob_check_tasks_py())
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	assert isinstance(session.project, LoadOk)
	before_task = session.project.tasks["check"]
	assert isinstance(before_task, Task)
	assert isinstance(before_task.cmd, tuple)
	assert before_task.cmd[-1].endswith("a.txt")
	(targets / "a.txt").rename(targets / "b.txt")
	session.refresh()
	assert isinstance(session.project, LoadOk)
	after_task = session.project.tasks["check"]
	assert isinstance(after_task, Task)
	assert isinstance(after_task.cmd, tuple)
	assert after_task.cmd[-1].endswith("b.txt")


async def test_run_via_client_reflects_filesystem_change_between_calls(tmp_path: Path) -> None:
	targets = tmp_path / "targets"
	targets.mkdir()
	(targets / "a.txt").write_text("")
	(tmp_path / "tasks.py").write_text(_glob_check_tasks_py())
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		first = await client.call_tool("camas_run", {"task": "check"})
		assert first.isError is False
		assert "PASSED" in _text(first)
		(targets / "a.txt").rename(targets / "b.txt")
		second = await client.call_tool("camas_run", {"task": "check"})
		assert second.isError is False
		assert "PASSED" in _text(second)


async def test_run_via_client_surfaces_broken_edit_between_calls(tmp_path: Path) -> None:
	tasks_py = tmp_path / "tasks.py"
	tasks_py.write_text("from camas import Task\ncheck = Task(('python', '-c', 'pass'))\n")
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		first = await client.call_tool("camas_run", {"task": "check"})
		assert first.isError is False
		assert "PASSED" in _text(first)
		tasks_py.write_text("raise RuntimeError('boom')\n")
		second = await client.call_tool("camas_run", {"task": "check"})
		assert second.isError is True
		assert "Tasks unavailable" in _text(second)


def _text(result: types.CallToolResult) -> str:
	return "\n".join(block.text for block in result.content if isinstance(block, types.TextContent))


GATE_FIX = Task(("python", "-c", "print('fixed')"), name="fmt", mutates=True)


async def test_gate_call_load_error(tmp_path: Path) -> None:
	session = Session(
		LoadErr(source=tmp_path / "tasks.py", exception=RuntimeError("boom")), tmp_path, Compat()
	)
	result = await serve.call(session, "camas_gate", {})
	assert result.isError


async def test_gate_call_invalid_args(tmp_path: Path) -> None:
	node = Parallel(GATE_FIX, PASS)
	session = _session({"all": node}, Config(default_task=node), tmp_path)
	result = await serve.call(session, "camas_gate", {"bogus": 1})
	assert result.isError
	assert "invalid camas_gate arguments" in _text(result)


async def test_gate_call_no_task_no_default_errors(tmp_path: Path) -> None:
	session = _session({"x": PASS}, None, tmp_path)
	result = await serve.call(session, "camas_gate", {})
	assert result.isError


async def test_gate_call_continue_when_checks_pass(tmp_path: Path) -> None:
	node = Parallel(GATE_FIX, PASS)
	session = _session({"all": node}, Config(default_task=node), tmp_path)
	result = await serve.call(session, "camas_gate", {})
	assert not result.isError
	assert "CONTINUE" in _text(result)


async def test_gate_call_block_when_check_fails(tmp_path: Path) -> None:
	node = Parallel(GATE_FIX, FAIL)
	session = _session({"all": node}, Config(default_task=node), tmp_path)
	result = await serve.call(session, "camas_gate", {"task": "all"})
	text = _text(result)
	assert not result.isError
	assert "BLOCK" in text
	assert "bad" in text


async def test_gate_call_nudges_when_failing_leaf_has_no_agent_format(tmp_path: Path) -> None:
	node = Parallel(GATE_FIX, FAIL)
	session = _session({"all": node}, Config(default_task=node), tmp_path)
	result = await serve.call(session, "camas_gate", {"task": "all"})
	text = _text(result)
	assert "Tip:" in text
	assert "agent_format" in text


async def test_gate_call_omits_nudge_when_failing_leaf_has_agent_format(tmp_path: Path) -> None:
	tagged_fail = Task(
		("python", "-c", "import sys; sys.exit(3)"),
		name="bad",
		agent_format=AgentFormat("--quiet", "sarif"),
	)
	node = Parallel(GATE_FIX, tagged_fail)
	session = _session({"all": node}, Config(default_task=node), tmp_path)
	result = await serve.call(session, "camas_gate", {"task": "all"})
	text = _text(result)
	assert "BLOCK" in text
	assert "Tip:" not in text


async def test_gate_call_path_mode_reads_report_file(tmp_path: Path) -> None:
	writer = Task(
		(
			"python",
			"-c",
			"import sys; sys.stderr.write('boom'); open(sys.argv[1], 'w').write('<xml/>'); sys.exit(1)",
		),
		name="writer",
		agent_format=AgentFormat("{report}", "junit"),
	)
	session = _session({"all": Parallel(writer)}, Config(default_task=Parallel(writer)), tmp_path)
	result = await serve.call(session, "camas_gate", {"task": "all"})
	text = _text(result)
	assert "<xml/>" in text
	assert "boom" not in text


async def test_gate_call_over_limit_structured_payload_points_to_file(tmp_path: Path) -> None:
	writer = Task(
		("python", "-c", "import sys; open(sys.argv[1], 'w').write('x' * 1000); sys.exit(1)"),
		name="writer",
		agent_format=AgentFormat("{report}", "junit", limit=10),
	)
	session = _session({"all": Parallel(writer)}, Config(default_task=Parallel(writer)), tmp_path)
	result = await serve.call(session, "camas_gate", {"task": "all"})
	text = _text(result)
	assert "x" * 1000 not in text
	assert "limit" in text


async def test_gate_call_with_budget_reports_partition(tmp_path: Path) -> None:
	node = Parallel(GATE_FIX, PASS)
	session = _session({"all": node}, Config(default_task=node), tmp_path)
	result = await serve.call(session, "camas_gate", {"under": 1.0})
	assert not result.isError
	assert "Time budget" in _text(result)


def test_compat_rich_by_default() -> None:
	"""``Compat()`` now defaults to ``emit_structured=True`` (rich by default)."""
	assert Compat().emit_structured is True


def test_compat_plain_in_argv() -> None:
	"""``--plain`` in argv opts out of rich (``emit_structured=False``)."""
	assert (
		Compat(emit_structured="--plain" not in ["camas", "mcp", "--plain"]).emit_structured
		is False
	)


def test_compat_bare_no_plain_is_rich() -> None:
	"""Bare ``camas mcp`` (no ``--plain``) keeps rich (``emit_structured=True``)."""
	assert Compat(emit_structured="--plain" not in ["camas", "mcp"]).emit_structured is True


FIX_TASK = Task(("python", "-c", "print('fixed')"), name="fmt", mutates=True)


async def test_fix_call_no_fix_registered_is_noop(tmp_path: Path) -> None:
	"""No fix node registered → empty RunResponse, no leaves, no error — and the text says
	*no fix node* so the agent knows to register one, not just pass different paths."""
	node = Parallel(FIX_TASK, PASS)
	session = _session({"all": node}, Config(default_task=node), tmp_path)
	result = await serve.call(session, "camas_fix", {})
	assert not result.isError
	text = _text(result)
	assert "nothing to fix" in text
	assert "no fix node registered" in text


async def test_fix_call_with_registered_fix(tmp_path: Path) -> None:
	"""A registered ``Config.agent.fix`` runs successfully."""
	cfg = Config(agent=Claude(fix=FIX_TASK))
	session = _session({"fmt": FIX_TASK}, cfg, tmp_path)
	result = await serve.call(session, "camas_fix", {})
	assert not result.isError
	assert "FIXED" in _text(result)


async def test_fix_call_with_registered_fix_and_paths(tmp_path: Path) -> None:
	"""Fix scoped to paths runs the fix node covering those paths."""
	cfg = Config(agent=Claude(fix=FIX_TASK))
	session = _session({"fmt": FIX_TASK}, cfg, tmp_path)
	result = await serve.call(session, "camas_fix", {"paths": ["src/a.py"]})
	assert not result.isError


async def test_fix_call_load_error(tmp_path: Path) -> None:
	session = Session(
		LoadErr(source=tmp_path / "tasks.py", exception=RuntimeError("boom")), tmp_path, Compat()
	)
	result = await serve.call(session, "camas_fix", {})
	assert result.isError


async def test_fix_call_invalid_args(tmp_path: Path) -> None:
	cfg = Config(agent=Claude(fix=FIX_TASK))
	session = _session({"fmt": FIX_TASK}, cfg, tmp_path)
	result = await serve.call(session, "camas_fix", {"bogus": 1})
	assert result.isError
	assert "invalid camas_fix arguments" in _text(result)


async def test_fix_call_unknown_task_errors(tmp_path: Path) -> None:
	session = _session({"fmt": FIX_TASK}, None, tmp_path)
	result = await serve.call(session, "camas_fix", {"task": "nope"})
	assert result.isError
	assert "no task named 'nope'" in _text(result)


async def test_fix_call_with_task_override(tmp_path: Path) -> None:
	"""A named task override runs that task as the fix, ignoring Config.agent.fix."""
	cfg = Config(agent=Claude(fix=FIX_TASK))
	session = _session({"fmt": FIX_TASK, "lint": PASS}, cfg, tmp_path)
	result = await serve.call(session, "camas_fix", {"task": "lint"})
	assert not result.isError
	assert "FIXED" in _text(result)


async def test_fix_call_tool_is_in_round_trip(tmp_path: Path) -> None:
	"""``camas_fix`` appears in the tool list via the in-memory server."""
	session = _session({"fmt": FIX_TASK}, None, tmp_path)
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		listed = await client.list_tools()
		names = {t.name for t in listed.tools}
		assert "camas_fix" in names


async def test_in_memory_round_trip_includes_fix_tool(tmp_path: Path) -> None:
	"""The full round trip tool list now has 8 tools including camas_github_matrix."""
	session = _session({"lint": PASS}, Config(default_task=PASS), tmp_path)
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		listed = await client.list_tools()
		assert {t.name for t in listed.tools} == {
			"camas_list",
			"camas_run",
			"camas_check",
			"camas_docs",
			"camas_gate",
			"camas_fix",
			"camas_init",
			"camas_github_matrix",
		}


def test_list_call_prepends_version_warning(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	session.version_warning = "WARNING: version mismatch"
	result = serve.list_call(session, {})
	assert result.isError is False
	text = _text(result)
	assert text.startswith("WARNING: version mismatch")


def test_docs_call_prepends_version_warning(tmp_path: Path) -> None:
	session = _session({}, None, tmp_path)
	session.version_warning = "WARNING: version mismatch"
	result = serve.docs_call(session)
	assert result.isError is False
	text = _text(result)
	assert text.startswith("WARNING: version mismatch")


def test_check_call_prepends_version_warning(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	(tmp_path / "tasks.py").write_text(
		'# /// script\n# dependencies = ["camas==999.0.0"]\n# ///\n' + _VALID_TASKS
	)
	monkeypatch.setattr("camas.mcp.result.run_typecheck", _fixed_checker(CheckerOk("ty")))
	session = Session(serve.resolve_project(tmp_path), tmp_path, Compat())
	result = serve.check_call(session)
	assert result.isError is False
	assert _text(result).startswith("camas ")
	assert "does not satisfy tasks.py pin ==999.0.0" in _text(result)


async def test_run_call_prepends_version_warning(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	session.version_warning = "WARNING: version mismatch"
	result = await serve.call(session, "camas_run", {"task": "lint"})
	assert not result.isError
	assert _text(result).startswith("WARNING: version mismatch")


async def test_gate_call_prepends_version_warning(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, Config(default_task=PASS), tmp_path)
	session.version_warning = "WARNING: version mismatch"
	result = await serve.call(session, "camas_gate", {})
	assert not result.isError
	assert _text(result).startswith("WARNING: version mismatch")


async def test_fix_call_prepends_version_warning(tmp_path: Path) -> None:
	cfg = Config(agent=Claude(fix=FIX_TASK))
	session = _session({"fmt": FIX_TASK}, cfg, tmp_path)
	session.version_warning = "WARNING: version mismatch"
	result = await serve.call(session, "camas_fix", {})
	assert not result.isError
	assert _text(result).startswith("WARNING: version mismatch")


async def test_fix_call_scoped_none_returns_empty_run(tmp_path: Path) -> None:
	"""camas_fix with paths that no fix leaf covers → empty RunResponse (no-op, not an error),
	and the text says *no fix leaf covers the paths* (distinct from no fix node registered)."""
	fix_node = Task("echo {paths}", name="fmt", mutates=True, paths="src")
	cfg = Config(agent=Claude(fix=fix_node))
	session = _session({"fmt": fix_node}, cfg, tmp_path)
	result = await serve.call(session, "camas_fix", {"paths": ["docs/readme.md"]})
	assert not result.isError
	text = _text(result)
	assert "nothing to fix" in text
	assert "no fix leaf covers the paths" in text
	assert "no fix node registered" not in text


async def test_github_matrix_call_dispatches_via_call(tmp_path: Path) -> None:
	session = _session({"m": _MATRIX}, None, tmp_path)
	result = await serve.call(session, "camas_github_matrix", {"task": "m"})
	assert result.isError is False
	assert "strategy.matrix" in _text(result)


def test_github_matrix_call_emits_compact_and_pretty(tmp_path: Path) -> None:
	session = _session({"m": _MATRIX}, None, tmp_path)
	result = serve.github_matrix_call(session, {"task": "m"})
	assert result.isError is False
	text = _text(result)
	assert '{"PY":["3.13","3.14"]}' in text
	assert "$GITHUB_OUTPUT" in text


def test_github_matrix_call_structured_when_rich(tmp_path: Path) -> None:
	session = _session({"m": _MATRIX}, None, tmp_path, rich=True)
	result = serve.github_matrix_call(session, {"task": "m"})
	assert result.structuredContent is not None
	assert result.structuredContent["task"] == "m"
	assert result.structuredContent["matrix"] == {"PY": ["3.13", "3.14"]}


def test_github_matrix_call_override_pins_axis(tmp_path: Path) -> None:
	session = _session({"m": _MATRIX}, None, tmp_path, rich=True)
	result = serve.github_matrix_call(session, {"task": "m", "matrix_overrides": {"PY": ["3.14"]}})
	assert result.structuredContent is not None
	assert result.structuredContent["matrix"] == {"PY": ["3.14"]}


def test_github_matrix_call_unknown_override_axis_is_tool_error(tmp_path: Path) -> None:
	session = _session({"m": _MATRIX}, None, tmp_path)
	result = serve.github_matrix_call(session, {"task": "m", "matrix_overrides": {"XX": ["1"]}})
	assert result.isError is True


def test_github_matrix_call_no_matrix_is_tool_error(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	result = serve.github_matrix_call(session, {"task": "lint"})
	assert result.isError is True
	assert "no matrix axes" in _text(result)


def test_github_matrix_call_unknown_task_is_tool_error(tmp_path: Path) -> None:
	session = _session({"m": _MATRIX}, None, tmp_path)
	result = serve.github_matrix_call(session, {"task": "nope"})
	assert result.isError is True
	assert "no task named 'nope'" in _text(result)


def test_github_matrix_call_no_task_uses_default(tmp_path: Path) -> None:
	session = _session({"m": _MATRIX}, Config(default_task=_MATRIX), tmp_path)
	result = serve.github_matrix_call(session, {})
	assert result.isError is False
	assert "3.13" in _text(result)


def test_github_matrix_call_no_task_no_default_is_tool_error(tmp_path: Path) -> None:
	session = _session({"m": _MATRIX}, None, tmp_path)
	result = serve.github_matrix_call(session, {})
	assert result.isError is True
	assert "no project default_task" in _text(result)


def test_github_matrix_call_validation_error_is_tool_error(tmp_path: Path) -> None:
	session = _session({"m": _MATRIX}, None, tmp_path)
	result = serve.github_matrix_call(session, {"matrix_overrides": "nope"})
	assert result.isError is True
	assert "invalid camas_github_matrix arguments" in _text(result)


def test_github_matrix_call_load_error(tmp_path: Path) -> None:
	session = Session(
		LoadErr(source=tmp_path / "tasks.py", exception=RuntimeError("boom")), tmp_path, Compat()
	)
	result = serve.github_matrix_call(session, {"task": "m"})
	assert result.isError is True


def test_github_matrix_call_prepends_version_warning(tmp_path: Path) -> None:
	session = _session({"m": _MATRIX}, None, tmp_path)
	session.version_warning = "WARNING: version mismatch"
	result = serve.github_matrix_call(session, {"task": "m"})
	assert not result.isError
	assert _text(result).startswith("WARNING: version mismatch")
