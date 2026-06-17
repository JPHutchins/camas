# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session

from camas import Config, Parallel, Sequential, Task
from camas.core.completion import RunResult, TaskResult
from camas.main.state import LoadErr, LoadOk
from camas.mcp import serve, wire
from camas.mcp.serve import Compat, Session
from camas.v0.completion import Finished, Skipped, Stopped

if TYPE_CHECKING:
	from camas.v0.task import TaskNode


def _session(
	tasks: dict[str, TaskNode], config: Config | None, base: Path, *, rich: bool = False
) -> Session:
	state = LoadOk(tasks=tasks, source=base / "tasks.py", scope_effects={}, config=config)
	return Session(state, base, Compat(emit_structured=rich))


PASS = Task(("python", "-c", "print('ok')"), name="lint")
FAIL = Task(("python", "-c", "import sys; print('boom'); sys.exit(3)"), name="bad")


# --- resolve_project / project_base / compat_from_argv ---


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


def test_compat_from_argv() -> None:
	assert serve.compat_from_argv(["--rich"]).emit_structured is True
	assert serve.compat_from_argv([]).emit_structured is False


# --- tools / task_names ---


def test_task_names_empty_on_load_error(tmp_path: Path) -> None:
	assert serve.task_names(LoadErr(source=tmp_path / "tasks.py", exception=RuntimeError())) == ()


def test_tools_default_omits_rich_fields() -> None:
	tool_list, tool_run = serve.tools(("a", "b"), Compat(emit_structured=False))
	assert (tool_list.title, tool_list.outputSchema, tool_list.annotations) == (None, None, None)
	assert tool_run.inputSchema["properties"]["task"]["enum"] == ["a", "b"]
	assert (tool_run.outputSchema, tool_run.annotations) == (None, None)


def test_tools_rich_includes_title_annotations_and_schema() -> None:
	tool_list, tool_run = serve.tools((), Compat(emit_structured=True))
	assert tool_list.title == "List camas tasks"
	assert tool_list.annotations is not None
	assert tool_list.annotations.readOnlyHint is True
	assert tool_run.annotations is not None
	assert tool_run.annotations.destructiveHint is True
	assert tool_list.outputSchema is not None
	assert tool_run.outputSchema is not None
	assert tool_run.outputSchema["type"] == "object"


# --- list_call ---


def test_list_call_text_lists_tasks_and_markers(tmp_path: Path) -> None:
	ci = Sequential(PASS, name="ci")
	session = _session(
		{"ci": ci, "lint": PASS}, Config(default_task=ci, github_task=PASS), tmp_path
	)
	result = serve.list_call(session)
	assert result.isError is False
	text = _text(result)
	assert "default (developer's task): ci" in text
	assert "github default" in text
	assert "lint" in text
	assert result.structuredContent is None


def test_list_call_load_error(tmp_path: Path) -> None:
	session = Session(LoadErr(tmp_path / "tasks.py", RuntimeError("boom")), tmp_path, Compat())
	result = serve.list_call(session)
	assert result.isError is True
	assert "camas --check" in _text(result)


def test_list_call_structured_when_rich(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path, rich=True)
	result = serve.list_call(session)
	assert result.structuredContent is not None
	assert result.structuredContent["tasks"][0]["name"] == "lint"


# --- run_call ---


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


async def test_run_call_log_path_in_structured_content_when_rich(tmp_path: Path) -> None:
	session = _session({"bad": FAIL}, None, tmp_path, rich=True)
	result = await serve.run_call(session, {"task": "bad"})
	assert result.structuredContent is not None
	assert result.structuredContent["leaves"][0]["log"].endswith("000_bad.log")


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


# --- resolve_run_node branches ---


def test_resolve_run_node_unknown_task_raises() -> None:
	with pytest.raises(ValueError, match="no task named 'x'"):
		serve.resolve_run_node({"lint": PASS}, wire.RunRequest(task="x"))


def test_resolve_run_node_applies_overrides_and_args() -> None:
	node = Parallel(Task("test {PY}"), matrix={"PY": ("3.12", "3.13")}, name="m")
	resolved = serve.resolve_run_node(
		{"m": node}, wire.RunRequest(task="m", matrix_overrides={"PY": ["3.13"]})
	)
	assert isinstance(resolved, Parallel)
	assert resolved.matrix == {"PY": ("3.13",)}
	leaf = serve.resolve_run_node(
		{"t": Task("pytest", name="t")}, wire.RunRequest(task="t", args=["-v"])
	)
	assert isinstance(leaf, Task)
	assert leaf.cmd == "pytest -v"


# --- logging ---


def test_create_run_log_dir_namespaces_by_task_and_is_idempotent(tmp_path: Path) -> None:
	first = serve.create_run_log_dir(tmp_path, "ci", 1)
	assert (tmp_path / ".camas" / ".gitignore").read_text() == "*\n"
	assert first == tmp_path / ".camas" / "runs" / "ci" / "1"
	second = serve.create_run_log_dir(tmp_path, "ci", 2)
	assert second.is_dir()


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


def test_failing_log_links_only_for_failed_leaves(tmp_path: Path) -> None:
	resp = wire.RunResponse(
		returncode=1,
		elapsed=0.1,
		passed=1,
		failed=1,
		skipped=0,
		interrupt_count=0,
		leaves=[
			wire.LeafReport(
				name="ok", command="c", completion=wire.Finished(returncode=0, elapsed=0.1)
			),
			wire.LeafReport(
				name="bad", command="c", completion=wire.Finished(returncode=1, elapsed=0.1)
			),
		],
	)
	links = serve.failing_log_links(resp, (tmp_path / "a.log", tmp_path / "b.log"))
	assert len(links) == 1
	assert links[0].name == "bad"


# --- text renderers ---


def test_run_text_covers_every_status() -> None:
	resp = wire.RunResponse(
		returncode=1,
		elapsed=1.0,
		passed=1,
		failed=2,
		skipped=2,
		interrupt_count=1,
		leaves=[
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
		],
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
		tasks=[
			wire.TaskInfo(
				name="m", command_preview="test {PY}", matrix_axes={"PY": ["3.13", "3.14"]}
			)
		],
		default=None,
	)
	text = serve.list_text(resp)
	assert "default" not in text
	assert "(matrix: PY=3.13/3.14)" in text


def test_dry_run_text_shows_resolved_commands() -> None:
	text = serve.dry_run_text(Parallel(Task("a"), Task("b"), name="p"))
	assert "fully-resolved plan" in text
	assert "a" in text
	assert "b" in text


# --- result builders ---


def test_error_result_is_tool_error() -> None:
	err = serve.error_result("nope")
	assert err.isError is True
	assert _text(err) == "nope"


# --- in-memory client integration ---


async def test_in_memory_round_trip(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, Config(default_task=PASS), tmp_path)
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		listed = await client.list_tools()
		assert {t.name for t in listed.tools} == {"camas_list", "camas_run"}
		catalog = await client.call_tool("camas_list", {})
		assert "lint" in _text(catalog)
		run = await client.call_tool("camas_run", {"task": "lint"})
		assert run.isError is False
		assert "PASSED" in _text(run)
		unknown = await client.call_tool("camas_run", {"task": "lint", "dry_run": True})
		assert "fully-resolved plan" in _text(unknown)


async def test_call_unknown_tool_is_error(tmp_path: Path) -> None:
	session = _session({"lint": PASS}, None, tmp_path)
	result = await serve.call(session, "camas_nope", {})
	assert result.isError is True
	assert "unknown tool 'camas_nope'" in _text(result)


def _text(result: types.CallToolResult) -> str:
	return "\n".join(block.text for block in result.content if isinstance(block, types.TextContent))
