# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""camas_command (#368): a Command run captured, behind its own tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp import types

from camas import Command, Config, Pipe, Task
from camas.main.state import LoadErr, LoadOk
from camas.mcp import serve
from camas.mcp.result import to_check_response
from camas.mcp.serve import Compat, Session, ToolName

if TYPE_CHECKING:
	from pathlib import Path

	from camas.v0.task import TaskNode

ECHO_STDIN = Task(("python", "-c", "import sys; print(sys.stdin.read().upper(), end='')"))


def _session(
	commands: dict[str, Command],
	base: Path,
	*,
	tasks: dict[str, TaskNode] | None = None,
	config: Config | None = None,
) -> Session:
	state = LoadOk(
		tasks=tasks or {},
		source=base / "tasks.py",
		scope_effects={},
		config=config,
		commands=commands,
	)
	return Session(state, base, Compat(emit_structured=False))


def _text(result: types.CallToolResult) -> str:
	return "\n".join(c.text for c in result.content if isinstance(c, types.TextContent))


async def test_camas_command_runs_captured_and_logs(tmp_path: Path) -> None:
	session = _session({"hi": Command(Task(("python", "-c", "print('flashed')")))}, tmp_path)
	result = await serve.call(session, ToolName.COMMAND.value, {"name": "hi"})
	assert result.isError is False
	text = _text(result)
	assert text.startswith("camas_command 'hi': PASSED")
	assert "flashed" in text
	assert (tmp_path / ".camas" / "runs" / "hi" / "1").is_dir()


async def test_camas_command_feeds_input_and_args(tmp_path: Path) -> None:
	commands = {
		"leaf": Command(ECHO_STDIN),
		"piped": Command(
			Pipe(ECHO_STDIN, Task(("python", "-c", "import sys; print(sys.stdin.read()[::-1])")))
		),
		"argv": Command(Task(("python", "-c", "import sys; print(*sys.argv[1:])"))),
	}
	session = _session(commands, tmp_path, config=Config(leaf_color=False))
	leaf = await serve.call(session, "camas_command", {"name": "leaf", "input": "y\n"})
	assert "Y" in _text(leaf)
	piped = await serve.call(session, "camas_command", {"name": "piped", "input": "ab"})
	assert "BA" in _text(piped)
	argv = await serve.call(session, "camas_command", {"name": "argv", "args": ["-q", "x"]})
	assert "-q x" in _text(argv)


async def test_camas_command_reports_the_paths_it_ran_with(tmp_path: Path) -> None:
	session = _session({"scope": Command(Task(("python", "-c", "pass", "{paths}")))}, tmp_path)
	result = await serve.call(session, "camas_command", {"name": "scope"})
	assert "ok     python -c pass . (" in _text(result)


async def test_camas_command_refusals_are_tool_errors(tmp_path: Path) -> None:
	session = _session({"seq": Command(Task("a") + Task("b"))}, tmp_path)
	cases = (
		({"name": "sqe"}, "no command named 'sqe'; did you mean 'seq'?"),
		({"name": "seq", "input": "y"}, "input applies to a single-leaf or Pipe command"),
		({"name": "seq", "args": ["-v"]}, "only apply to Task"),
		({"name": "seq", "bogus": 1}, "invalid camas_command arguments"),
	)
	for arguments, message in cases:
		result = await serve.call(session, "camas_command", arguments)
		assert result.isError is True
		assert message in _text(result)


async def test_camas_command_reports_a_load_error(tmp_path: Path) -> None:
	session = Session(LoadErr(tmp_path / "tasks.py", RuntimeError("boom")), tmp_path, Compat())
	result = await serve.call(session, "camas_command", {"name": "x"})
	assert result.isError is True
	assert serve.command_names(session.project) == ()


async def test_camas_run_points_a_command_name_to_camas_command(tmp_path: Path) -> None:
	session = _session({"flash": Command("x")}, tmp_path, tasks={"lint": Task("ruff")})
	result = await serve.run_call(session, {"task": "flash"})
	assert result.isError is True
	assert "'flash' is a Command: run it with camas_command" in _text(result)


def test_camas_list_reports_commands_apart_from_tasks(tmp_path: Path) -> None:
	session = _session(
		{
			"flash": Command("probe-rs run", help="Flash the board"),
			"quiet": Command("probe-rs reset", help=""),
		},
		tmp_path,
		tasks={"lint": Task("ruff check .")},
	)
	result = serve.list_call(session, {})
	text = _text(result)
	assert "Commands (side-effectful; run with camas_command, never camas_run):" in text
	assert "  flash  Flash the board" in text
	assert "probe-rs reset" not in text
	assert "Commands" not in _text(serve.list_call(_session({}, tmp_path), {}))


def test_camas_check_warns_on_command_placeholders_and_resolves_camas_command() -> None:
	state = LoadOk(
		tasks={"ship": Task("camas flash")},
		source=None,
		scope_effects={},
		commands={"flash": Command("probe-rs run --chip {CHIP}")},
	)
	assert to_check_response(state).warnings == (
		"command 'flash' runs {CHIP} literally: a Command has no matrix axes",
	)


def test_camas_command_tool_lists_command_names_and_is_destructive() -> None:
	tool = serve.tools((), Compat(emit_structured=True), ("flash",)).command
	assert tool.name == "camas_command"
	assert tool.inputSchema["properties"]["name"]["enum"] == ["flash"]
	assert tool.annotations is not None
	assert tool.annotations.destructiveHint is True
