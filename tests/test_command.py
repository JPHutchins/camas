# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Command (#368): attached to the terminal on the CLI, never composed into a task tree."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import pytest

from camas import Command, Config, Parallel, Pipe, Sequential, Task
from camas.core import attached
from camas.core.attached import Exited, Signaled, run_attached, run_step, spawn
from camas.main import command as command_cli
from camas.main.compose import load_py_tasks_state, state_from_scope
from camas.main.dispatch import dispatch
from camas.main.state import LoadErr, LoadOk
from camas.v0.command import body_problem
from camas.v0.task import Project

if TYPE_CHECKING:
	import subprocess
	from collections.abc import Callable, Mapping

	from camas.v0.effect import Effect
	from camas.v0.task import TaskNode


def py(code: str, *args: str) -> Task:
	return Task(("python", "-c", code, *args))


def _state(commands: Mapping[str, Command], tasks: Mapping[str, TaskNode] | None = None) -> LoadOk:
	effects: dict[str, type[Effect[Any]]] = {}
	return LoadOk(
		tasks=dict(tasks or {}), source=None, scope_effects=effects, commands=dict(commands)
	)


def test_body_coerces_and_rejects_what_cannot_share_a_terminal() -> None:
	assert Command(("cargo", "run")).body == Task(("cargo", "run"))
	assert body_problem(Command("x")) == (
		"a Command body holds Task, Pipe, and Sequential nodes, not Command"
	)
	with pytest.raises(ValueError, match="cannot hold a Parallel"):
		Command(Sequential("a", Parallel("b", "c")))
	with pytest.raises(ValueError, match="cannot fan out"):
		Command(Sequential("t {X}", variants=({"X": "1"},)))
	with pytest.raises(ValueError, match="cannot hold a Project"):
		Command(Sequential(Project("libs")))
	assert "help='Flash it'" in repr(Command("x", help="Flash it"))


COMPOSERS: tuple[Callable[[Any], object], ...] = (
	Parallel,
	lambda c: Sequential("a", c),
	lambda c: Pipe("a", c),
	lambda c: Task("a") | c,
	lambda c: Task("a") + c,
	lambda c: Task("a") > c,
	lambda c: Parallel("a").extend(c),
)


@pytest.mark.parametrize("compose", COMPOSERS)
def test_a_command_never_composes_into_a_task_tree(compose: Callable[[Any], object]) -> None:
	with pytest.raises(TypeError, match="a Command never composes into a task tree"):
		compose(Command("flash"))


def test_a_group_rejects_a_child_that_is_not_a_node() -> None:
	with pytest.raises(TypeError, match="a child is a task node or str"):
		Parallel(42)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_loader_names_commands_and_rejects_them_in_config() -> None:
	build = Task("cargo build")
	flash = Command(build + "probe-rs download")
	state = state_from_scope({"build": build, "flash": flash})
	assert isinstance(state, LoadOk)
	assert list(state.tasks) == ["build"]
	assert state.commands["flash"].name == "flash"
	refused = state_from_scope(
		{"flash": flash, "_": Config(default_task=flash)}  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
	)
	assert isinstance(refused, LoadErr)
	assert "never composes into a task tree (in the Config's default task)" in str(
		refused.exception
	)


def test_project_mounts_child_commands_anchored_to_the_child(tmp_path: Path) -> None:
	(tmp_path / "fw").mkdir()
	(tmp_path / "fw" / "tasks.py").write_text(
		dedent("""\
			from camas import Command, Config, Task
			build = Task("cargo build")
			flash = Command(build + "probe-rs download")
			_ = Config(default_task=build)
		""")
	)
	(tmp_path / "tasks.py").write_text('from camas import Project\nfw = Project("fw")\n')
	state = load_py_tasks_state(tmp_path / "tasks.py")
	assert isinstance(state, LoadOk)
	mounted = state.commands["fw.flash"]
	assert (mounted.name, mounted.cwd) == ("fw.flash", Path("fw"))


def test_project_mounts_a_child_that_defines_only_commands(tmp_path: Path) -> None:
	(tmp_path / "ops").mkdir()
	(tmp_path / "ops" / "tasks.py").write_text(
		'from camas import Command\ndeploy = Command("deploy-tool")\n'
	)
	(tmp_path / "tasks.py").write_text('from camas import Project\nops = Project("ops")\n')
	state = load_py_tasks_state(tmp_path / "tasks.py")
	assert isinstance(state, LoadOk)
	assert (list(state.tasks), list(state.commands)) == ([], ["ops.deploy"])


def test_run_attached_stops_at_the_first_failure_with_its_exact_code(
	capfd: pytest.CaptureFixture[str],
) -> None:
	outcome = run_attached(
		Sequential(py("print('one')"), py("raise SystemExit(7)"), py("print('never')")), None
	)
	assert outcome == Exited(7)
	assert capfd.readouterr().out.split() == ["one"]


def test_run_attached_pipes_stages_and_follows_pipefail(capfd: pytest.CaptureFixture[str]) -> None:
	upper = py("import sys; sys.stdout.write(sys.stdin.read().upper())")
	assert run_attached(Pipe(py("print('hi')"), upper), None) == Exited(0)
	assert capfd.readouterr().out.splitlines() == ["HI"]
	assert run_attached(Pipe(py("raise SystemExit(4)"), upper), None) == Exited(4)


def test_run_attached_gives_the_child_no_forced_color(
	capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.delenv("FORCE_COLOR", raising=False)
	command = Task(
		("python", "-c", "import os; print(os.environ.get('FORCE_COLOR'), os.environ['K'])"),
		env={"K": "declared"},
	)
	assert run_attached(command, None) == Exited(0)
	assert capfd.readouterr().out.split() == ["None", "declared"]


@pytest.mark.parametrize("missing_stage", [0, 1])
def test_an_unstartable_stage_stops_the_step_with_127(
	missing_stage: int, capfd: pytest.CaptureFixture[str]
) -> None:
	stages = [py("import time; time.sleep(30)"), py("pass")]
	stages[missing_stage] = Task("camas-no-such-program-368")
	assert run_step(Pipe(*stages), None) == Exited(127)
	assert "no such file or directory: camas-no-such-program-368" in capfd.readouterr().err


def test_a_stage_that_fails_to_spawn_takes_the_started_stages_down(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	started: list[subprocess.Popen[bytes]] = []

	def spawn_once(
		task: Task, base: Path | None, *, stdin: int | None, stdout: int | None
	) -> subprocess.Popen[bytes]:
		if started:
			raise RuntimeError("spawn failed")
		started.append(spawn(task, base, stdin=stdin, stdout=stdout))
		return started[0]

	monkeypatch.setattr(attached, "spawn", spawn_once)
	with pytest.raises(RuntimeError, match="spawn failed"):
		run_step(Pipe(py("import time; time.sleep(30)"), py("pass")), None)
	assert [proc.returncode is not None for proc in started] == [True]


def test_cli_runs_a_command_attached(capfd: pytest.CaptureFixture[str]) -> None:
	with pytest.raises(SystemExit) as exit_info:
		dispatch(
			_state({"show": Command(py("import sys; print(*sys.argv[1:])"))}),
			["show", "--", "a", "b"],
		)
	assert exit_info.value.code == 0
	assert capfd.readouterr().out.splitlines() == ["a b"]


def test_cli_exits_with_the_childs_exact_code() -> None:
	with pytest.raises(SystemExit) as exit_info:
		dispatch(_state({"fail": Command(py("raise SystemExit(5)"))}), ["fail"])
	assert exit_info.value.code == 5


@pytest.mark.parametrize(
	("argv", "flag"),
	[
		(["flash", "--jobs", "2"], "--jobs"),
		(["flash", "--effects=(Status(),)"], "--effects"),
		(["flash", "--under", "1s"], "--under"),
		(["flash", "--paths", "a.py"], "--paths"),
		(["flash", "--github-matrix"], "--github-matrix"),
		(["flash", "--matrix", "PY=3.13"], "--matrix"),
		(["flash", "--PY", "3.13"], "--PY"),
	],
)
def test_cli_refuses_tree_run_flags(
	argv: list[str], flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
	state = _state({"flash": Command("probe-rs run")})._replace(
		config=Config(default_task=Parallel(Task("t {PY}"), matrix={"PY": ("3.12",)}))
	)
	with pytest.raises(SystemExit, match="2"):
		dispatch(state, argv)
	err = capsys.readouterr().err
	assert f"error: {flag} does not apply to a Command" in err
	assert "ignoring it for this run" not in err


def test_cli_warns_that_verbose_is_ignored_and_runs_the_command(
	capfd: pytest.CaptureFixture[str],
) -> None:
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"hi": Command(py("print('ran')"))}), ["hi", "--verbose"])
	out, err = capfd.readouterr()
	assert "--verbose only applies to --init; ignoring it for this run" in err
	assert out.splitlines() == ["ran"]


def test_cli_names_a_command_composed_in_an_expression(
	capsys: pytest.CaptureFixture[str],
) -> None:
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"flash": Command("x")}, {"lint": Task("ruff")}), ["lint | flash"])
	assert (
		"error: 'flash' is a Command: a Command never composes into a task tree"
		in capsys.readouterr().err
	)


def test_cli_dry_run_and_passthrough_rules(capsys: pytest.CaptureFixture[str]) -> None:
	state = _state({"flash": Command(Task("build") + Task("probe-rs run"))})
	with pytest.raises(SystemExit, match="0"):
		dispatch(state, ["flash", "--dry-run"])
	assert "probe-rs run" in capsys.readouterr().out
	with pytest.raises(SystemExit, match="2"):
		dispatch(state, ["flash", "--", "-v"])
	assert "only apply to Task" in capsys.readouterr().err


def test_cli_help_list_and_tree_show_commands(capsys: pytest.CaptureFixture[str]) -> None:
	state = _state(
		{
			"flash": Command(Task("build") + Task("probe-rs run")),
			"run": Command("cargo run", help="Run it"),
			"flash_board": Command("probe-rs run"),
		},
		{"lint": Task("ruff check .")},
	)
	with pytest.raises(SystemExit, match="0"):
		dispatch(state, ["run", "--help"])
	assert capsys.readouterr().out.startswith(
		"usage: camas run [-h] [--dry-run] [-- ARG ...]\n\nRun it"
	)
	with pytest.raises(SystemExit, match="0"):
		dispatch(state, ["flash", "-h"])
	assert capsys.readouterr().out.startswith("usage: camas flash [-h] [--dry-run]\n")
	with pytest.raises(SystemExit, match="0"):
		dispatch(state, ["flash-board", "-h"])
	assert capsys.readouterr().out.startswith("usage: camas flash-board [-h]")
	with pytest.raises(SystemExit, match="0"):
		dispatch(state, ["--list"])
	assert (
		"\n\nCommands:\n  flash        build, probe-rs run\n  flash_board  probe-rs run\n"
		"  run          Run it"
	) in capsys.readouterr().out
	with pytest.raises(SystemExit, match="0"):
		dispatch(state, ["--tree"])
	assert "Commands:" in capsys.readouterr().out
	with pytest.raises(SystemExit, match="0"):
		dispatch(state, ["--help"])
	assert "  run          Run it" in capsys.readouterr().out


def test_cli_help_for_an_unknown_name_falls_through_to_the_parser(
	capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setenv("NO_COLOR", "1")
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"flash": Command("x")}), ["nothing", "--help"])
	assert "usage: camas" in capsys.readouterr().out


def test_cli_suggests_command_names_and_skips_the_mcp_banner_for_them(
	capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setenv("CAMAS_AGENT", "1")
	monkeypatch.delenv("CAMAS_NO_MCP_HINT", raising=False)
	state = _state({"flash": Command("x")}, {"lint": Task("ruff check")})
	with pytest.raises(SystemExit, match="2"):
		dispatch(state, ["flsah"])
	err = capsys.readouterr().err
	assert "did you mean 'flash'?" in err
	assert "prefer the camas MCP tools" in err
	with pytest.raises(SystemExit, match="0"):
		dispatch(state, ["flash", "--dry-run"])
	assert "prefer the camas MCP tools" not in capsys.readouterr().err
	with pytest.raises(SystemExit, match="0"):
		dispatch(state, ["lint", "--paths", "flash", "--dry-run"])
	assert "prefer the camas MCP tools" in capsys.readouterr().err


def test_exit_with_a_signal_dies_of_it(monkeypatch: pytest.MonkeyPatch) -> None:
	def died(signum: int) -> None:
		raise SystemExit(128 + signum)

	monkeypatch.setattr(command_cli, "die_of", died)
	with pytest.raises(SystemExit, match="143"):
		command_cli.exit_with(Signaled(15))


def test_print_command_trees_prints_nothing_without_commands(
	capsys: pytest.CaptureFixture[str],
) -> None:
	command_cli.print_command_trees({})
	assert capsys.readouterr().out == ""
