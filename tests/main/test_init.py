# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import TYPE_CHECKING, get_args

import pytest

from camas import Sequential, Task, v0
from camas.main import starter_verbose
from camas.main.check import CheckerOk, find_typechecker, run_typecheck
from camas.main.dispatch import dispatch
from camas.main.init import starter_text, write_starter_tasks_py
from camas.main.state import EMPTY_STATE, LoadErr
from camas.main.tasks import load_tasks_from_py
from camas.v0.completion import Finished
from camas.v0.task import OutputKind
from camas.v0.task_event import CompletedEvent, StartedEvent

if TYPE_CHECKING:
	from pathlib import Path

requires_checker = pytest.mark.skipif(
	find_typechecker() is None,
	reason="real-checker test requires ty or mypy on PATH (install camas[check])",
)


def _camas(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
	env = {**os.environ, "NO_COLOR": "1"}
	env.pop("GITHUB_ACTIONS", None)
	return subprocess.run(
		[sys.executable, "-m", "camas", *args],
		cwd=cwd,
		capture_output=True,
		text=True,
		encoding="utf-8",
		env=env,
		check=False,
	)


def test_write_starter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
	assert write_starter_tasks_py(tmp_path) == 0
	assert (tmp_path / "tasks.py").read_text(encoding="utf-8") == starter_text()
	assert (tmp_path / ".camas" / ".gitignore").read_text(encoding="utf-8") == "*\n"
	out = capsys.readouterr().out
	assert "Wrote" in out
	assert ".camas" in out


def test_write_starter_refuses_existing_leaves_no_camas_dir(tmp_path: Path) -> None:
	(tmp_path / "tasks.py").write_text("untouched = 1\n")
	assert write_starter_tasks_py(tmp_path) == 2
	assert not (tmp_path / ".camas").exists()


def test_write_starter_refuses_existing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
	(tmp_path / "tasks.py").write_text("untouched = 1\n")
	assert write_starter_tasks_py(tmp_path) == 2
	assert (tmp_path / "tasks.py").read_text() == "untouched = 1\n"
	assert "exists" in capsys.readouterr().err


def test_dispatch_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="0"):
		dispatch(EMPTY_STATE, ["--init"])
	assert (tmp_path / "tasks.py").exists()


def test_dispatch_init_under_load_err(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	"""``--init`` still works when a tasks file elsewhere failed to load."""
	monkeypatch.chdir(tmp_path)
	state = LoadErr(source=tmp_path / "parent" / "tasks.py", exception=ValueError("boom"))
	with pytest.raises(SystemExit, match="0"):
		dispatch(state, ["--init"])
	assert (tmp_path / "tasks.py").exists()


def test_starter_loads_with_config_default(tmp_path: Path) -> None:
	write_starter_tasks_py(tmp_path)
	loaded = load_tasks_from_py(tmp_path / "tasks.py")
	assert set(loaded.tasks) == {"hello", "greet", "ci", "autofix"}
	assert loaded.scope_effects == {}
	assert loaded.config is not None
	assert loaded.config.default_task == loaded.tasks["ci"]
	assert isinstance(loaded.config.default_task, Sequential)
	assert loaded.config.agent is not None
	assert loaded.config.agent.fix == loaded.tasks["autofix"]


def test_starter_runs_to_completion(tmp_path: Path) -> None:
	"""Bare ``camas`` in a freshly scaffolded directory runs the whole default
	tree green — the placeholder tasks must be infallible cross-platform."""
	write_starter_tasks_py(tmp_path)
	result = _camas("--effects=(Summary(show_passing=True),)", cwd=tmp_path)
	assert result.returncode == 0, result.stderr
	assert "hello from camas" in result.stdout
	assert "hello, Ada!" in result.stdout
	assert "hello, Grace!" in result.stdout
	assert "Python" in result.stdout


def test_starter_cli_init_then_list(tmp_path: Path) -> None:
	created = _camas("--init", cwd=tmp_path)
	assert created.returncode == 0, created.stderr
	again = _camas("--init", cwd=tmp_path)
	assert again.returncode == 2
	assert "exists" in again.stderr
	listed = _camas("--list", cwd=tmp_path)
	assert listed.returncode == 0, listed.stderr
	assert "say hello to everyone at once" in listed.stdout
	assert all(name in listed.stdout for name in ("hello", "greet", "ci"))


def test_starter_runs_standalone(tmp_path: Path) -> None:
	"""``python tasks.py --list`` dispatches through the scaffold's
	``run_cli(globals())`` block."""
	write_starter_tasks_py(tmp_path)
	env = {**os.environ, "NO_COLOR": "1"}
	result = subprocess.run(
		[sys.executable, str(tmp_path / "tasks.py"), "--list"],
		capture_output=True,
		text=True,
		encoding="utf-8",
		env=env,
		check=False,
	)
	assert result.returncode == 0, result.stderr
	assert all(name in result.stdout for name in ("hello", "greet", "ci"))


def test_write_starter_verbose(tmp_path: Path) -> None:
	assert write_starter_tasks_py(tmp_path, verbose=True) == 0
	written = (tmp_path / "tasks.py").read_text(encoding="utf-8")
	assert written == starter_text(verbose=True)
	assert written != starter_text()
	assert (tmp_path / ".camas" / ".gitignore").read_text(encoding="utf-8") == "*\n"


def test_dispatch_init_verbose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="0"):
		dispatch(EMPTY_STATE, ["--init", "--verbose"])
	written = (tmp_path / "tasks.py").read_text(encoding="utf-8")
	assert written == starter_text(verbose=True)


def test_dispatch_init_default_stays_minimal(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="0"):
		dispatch(EMPTY_STATE, ["--init"])
	assert (tmp_path / "tasks.py").read_text(encoding="utf-8") == starter_text()


def test_dispatch_verbose_without_init_warns(capsys: pytest.CaptureFixture[str]) -> None:
	with pytest.raises(SystemExit):
		dispatch(EMPTY_STATE, ["--verbose"])
	assert "--verbose only applies to --init" in capsys.readouterr().err


def test_init_rejects_unknown_flag_before_scaffolding(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""A typo of a flag alongside ``--init`` errors instead of silently scaffolding — the
	strict parse in the ``--init`` branch runs before the write."""
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit) as excinfo:
		dispatch(EMPTY_STATE, ["--init", "--verboes"])
	assert excinfo.value.code != 0
	assert "unrecognized arguments: --verboes" in capsys.readouterr().err
	assert not (tmp_path / "tasks.py").exists()


def test_cli_init_unknown_flag_errors(tmp_path: Path) -> None:
	result = _camas("--init", "--verboes", cwd=tmp_path)
	assert result.returncode != 0
	assert "unrecognized arguments: --verboes" in result.stderr
	assert not (tmp_path / "tasks.py").exists()


def test_verbose_without_init_warns_but_still_runs(tmp_path: Path) -> None:
	"""``--verbose`` only affects ``--init``; on a normal run it warns to stderr without
	changing the task's exit behavior."""
	write_starter_tasks_py(tmp_path)
	result = _camas("--verbose", "hello", cwd=tmp_path)
	assert result.returncode == 0, result.stderr
	assert "--verbose only applies to --init" in result.stderr


def test_normal_run_has_no_verbose_warning(tmp_path: Path) -> None:
	write_starter_tasks_py(tmp_path)
	result = _camas("hello", cwd=tmp_path)
	assert result.returncode == 0, result.stderr
	assert "--verbose" not in result.stderr


def test_verbose_starter_loads_with_config(tmp_path: Path) -> None:
	write_starter_tasks_py(tmp_path, verbose=True)
	loaded = load_tasks_from_py(tmp_path / "tasks.py")
	assert set(loaded.tasks) == {
		"hello",
		"compile_step",
		"documented",
		"native",
		"flake",
		"docs_build",
		"fmt",
		"lint",
		"web_lint",
		"checked",
		"subproject",
		"greet",
		"meet",
		"frontend",
		"versions",
		"ci",
	}
	assert set(loaded.scope_effects) == {"Announce"}
	assert loaded.config is not None
	assert loaded.config.default_task == loaded.tasks["ci"]
	assert isinstance(loaded.config.github_task, Sequential)
	assert loaded.config.camas_dir == ".camas"
	assert loaded.config.default_effects is None
	assert loaded.config.default_github_effects is None
	assert loaded.config.agent is not None
	assert loaded.config.agent.fix == loaded.tasks["fmt"]
	assert loaded.config.agent.check == loaded.tasks["ci"]
	assert loaded.config.agent.default == loaded.tasks["hello"]


def test_verbose_starter_runs_to_completion(tmp_path: Path) -> None:
	"""Bare ``camas`` in a verbose-scaffolded directory runs the whole default
	tree green — the placeholder tasks must be infallible cross-platform."""
	write_starter_tasks_py(tmp_path, verbose=True)
	result = _camas("--effects=(Summary(show_passing=True),)", cwd=tmp_path)
	assert result.returncode == 0, result.stderr
	assert "hello from the kitchen-sink starter" in result.stdout
	assert "hello, Ada!" in result.stdout
	assert "hi Jane, meet Wendy" in result.stdout
	assert "checked on 3.13" in result.stdout
	assert "subproject build" in result.stdout


@requires_checker
def test_verbose_starter_typechecks(tmp_path: Path) -> None:
	write_starter_tasks_py(tmp_path, verbose=True)
	result = run_typecheck(tmp_path / "tasks.py")
	assert isinstance(result, CheckerOk), result


def _v0_exports() -> dict[str, object]:
	eager = {
		name: val
		for name, val in vars(v0).items()
		if not name.startswith("_") and not inspect.ismodule(val)
	}
	return {**eager, "run_cli": v0.run_cli}


def test_verbose_starter_mentions_every_v0_export() -> None:
	text = starter_text(verbose=True)
	missing = [name for name in _v0_exports() if not re.search(rf"\b{re.escape(name)}\b", text)]
	assert missing == []


def _constructor_options() -> dict[str, bool]:
	"""Every parameter of every eager public v0 callable (protocols excluded), mapped to
	whether any constructor takes it as an option (keyword-only or defaulted) — those must
	appear in the verbose template as a ``name=`` usage, the rest as a word.
	"""
	constructors = {
		val
		for name, val in vars(v0).items()
		if not name.startswith("_")
		and not inspect.ismodule(val)
		and callable(val)
		and not getattr(val, "_is_protocol", False)
	}
	options: dict[str, bool] = {}
	for constructor in constructors:
		for name, param in inspect.signature(constructor).parameters.items():
			if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
				continue
			as_kwarg = param.kind is param.KEYWORD_ONLY or param.default is not param.empty
			options[name] = options.get(name, False) or as_kwarg
	return options


def test_verbose_starter_demonstrates_every_constructor_option() -> None:
	text = starter_text(verbose=True)
	missing = [
		name
		for name, as_kwarg in _constructor_options().items()
		if not re.search(
			rf"\b{re.escape(name)}\s*=" if as_kwarg else rf"\b{re.escape(name)}\b", text
		)
	]
	assert missing == []


def test_verbose_starter_names_every_output_kind() -> None:
	text = starter_text(verbose=True)
	missing = [kind for kind in get_args(OutputKind) if not re.search(rf"\b{kind}\b", text)]
	assert missing == []


def test_verbose_module_scope_callables(tmp_path: Path) -> None:
	assert starter_verbose.touches_docs(("docs/guide.md",)) is True
	assert starter_verbose.touches_docs(("src/app.py",)) is False
	assert starter_verbose.web_paths(("app.ts", "app.py")) == ("app.ts",)
	assert starter_verbose.web_paths(()) == ("web",)
	version_file = tmp_path / ".python-version"
	version_file.write_text("# pin\n\n3.12\n3.13\n", encoding="utf-8")
	assert starter_verbose.python_versions_from(version_file) == ("3.12", "3.13")
	assert starter_verbose.python_versions_from(tmp_path / "absent") == ("3.13",)


async def test_verbose_module_announce_effect(capsys: pytest.CaptureFixture[str]) -> None:
	effect = starter_verbose.Announce()
	task = Task("echo hi", name="greeter")
	now = datetime(2026, 1, 1, 12, 0, 0)
	await effect.setup(task)
	await effect.on_event(StartedEvent(task, 0, now), (), None)
	await effect.on_event(CompletedEvent(task, 0, Finished(0, 0.1, ()), now), (), None)
	await effect.teardown((None,))
	assert "announce: greeter is done" in capsys.readouterr().out
