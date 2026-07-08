# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from pathlib import Path

import pytest

from camas import Parallel, Project, Task
from camas.main.compose import load_py_state, load_scope, state_from_scope
from camas.main.state import LoadErr, LoadOk


def _write(path: Path, content: str) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(content, encoding="utf-8")
	return path


def _leaf(name: str, *, cwd: str | None = None, github: str | None = None) -> str:
	cwd_kw = f", cwd={cwd!r}" if cwd is not None else ""
	gh = f"{name}_gh = Task(('python', '-c', 'pass'), name='{name}-gh')\n" if github else ""
	gh_kw = f", github_task={name}_gh" if github else ""
	return (
		"from camas import Config, Task\n"
		f"{name} = Task(('python', '-c', 'pass'), name='{name}'{cwd_kw})\n"
		f"{gh}"
		f"_ = Config(default_task={name}{gh_kw})\n"
	)


def test_project_mounts_child_namespace(tmp_path: Path) -> None:
	_write(
		tmp_path / "tasks.py",
		"from camas import Config, Project, Task\n"
		"root = Task(('python', '-c', 'pass'), name='root')\n"
		"libs = Project('libs')\n"
		"_ = Config(default_task=root)\n",
	)
	_write(tmp_path / "libs" / "tasks.py", _leaf("build"))
	tasks = load_scope(tmp_path / "tasks.py").tasks
	assert set(tasks) == {"root", "libs", "libs.build"}
	assert tasks["libs"].name == "build"


def test_project_rebases_cwd_two_levels(tmp_path: Path) -> None:
	_write(
		tmp_path / "tasks.py",
		"from camas import Config, Project, Task\n"
		"root = Task(('python', '-c', 'pass'), name='root')\n"
		"libs = Project('libs')\n"
		"_ = Config(default_task=root)\n",
	)
	_write(
		tmp_path / "libs" / "tasks.py",
		"from camas import Config, Project, Task\n"
		"build = Task(('python', '-c', 'pass'), name='build')\n"
		"search = Project('search')\n"
		"_ = Config(default_task=build)\n",
	)
	_write(tmp_path / "libs" / "search" / "tasks.py", _leaf("lint", cwd="src"))
	tasks = load_scope(tmp_path / "tasks.py").tasks
	assert tasks["libs.search.lint"].cwd == Path("libs/search/src")
	assert tasks["libs.search"].cwd == Path("libs/search/src")


def test_gap_dir_named_by_import(tmp_path: Path) -> None:
	_write(
		tmp_path / "tasks.py",
		"from camas import Config, Project, Task\n"
		"root = Task(('python', '-c', 'pass'), name='root')\n"
		"api = Project('services/api')\n"
		"_ = Config(default_task=root)\n",
	)
	_write(tmp_path / "services" / "api" / "tasks.py", _leaf("deploy"))
	tasks = load_scope(tmp_path / "tasks.py").tasks
	assert "api" in tasks
	assert tasks["api.deploy"].cwd == Path("services/api")


def test_config_default_task_binding_resolves_once(tmp_path: Path) -> None:
	_write(
		tmp_path / "tasks.py",
		"from camas import Config, Parallel, Project\n"
		"libs = Project('libs')\n"
		"both = Parallel(libs, name='both')\n"
		"_ = Config(default_task=both)\n",
	)
	_write(tmp_path / "libs" / "tasks.py", _leaf("build"))
	loaded = load_scope(tmp_path / "tasks.py")
	assert loaded.config is not None
	assert loaded.config.default_task == loaded.tasks["both"]
	assert loaded.tasks["both"].name == "both"


def test_github_task_composite_resolves(tmp_path: Path) -> None:
	_write(
		tmp_path / "tasks.py",
		"from camas import Config, Parallel, Project, Task\n"
		"root = Task(('python', '-c', 'pass'), name='root')\n"
		"libs = Project('libs')\n"
		"_ = Config(default_task=root, github_task=Parallel(libs))\n",
	)
	_write(tmp_path / "libs" / "tasks.py", _leaf("build"))
	config = load_scope(tmp_path / "tasks.py").config
	assert config is not None
	assert isinstance(config.github_task, Parallel)
	(child,) = config.github_task.tasks
	assert child.name == "build"


def test_context_github_selects_github_task(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setenv("GITHUB_ACTIONS", "true")
	monkeypatch.delenv("CLAUDECODE", raising=False)
	monkeypatch.delenv("CAMAS_AGENT", raising=False)
	_write(
		tmp_path / "tasks.py",
		"from camas import Config, Project, Task\n"
		"root = Task(('python', '-c', 'pass'), name='root')\n"
		"web = Project('web')\n"
		"_ = Config(default_task=root)\n",
	)
	_write(tmp_path / "web" / "tasks.py", _leaf("build", github="ship"))
	tasks = load_scope(tmp_path / "tasks.py").tasks
	assert tasks["web"].name == "build-gh"


def test_context_agent_selects_run_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("CAMAS_AGENT", "1")
	monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
	_write(
		tmp_path / "tasks.py",
		"from camas import Claude, Config, Project, Task\n"
		"root = Task(('python', '-c', 'pass'), name='root')\n"
		"child = Project('child')\n"
		"_ = Config(default_task=root)\n",
	)
	_write(
		tmp_path / "child" / "tasks.py",
		"from camas import Claude, Config, Task\n"
		"work = Task(('python', '-c', 'pass'), name='work')\n"
		"agent_default = Task(('python', '-c', 'pass'), name='agent-default')\n"
		"_ = Config(default_task=work, agent=Claude(fix=work, default=agent_default))\n",
	)
	tasks = load_scope(tmp_path / "tasks.py").tasks
	assert tasks["child"].name == "agent-default"


def test_config_agent_fields_resolved(tmp_path: Path) -> None:
	_write(
		tmp_path / "tasks.py",
		"from camas import Claude, Config, Project\n"
		"child = Project('child')\n"
		"fix = child\n"
		"_ = Config(default_task=child, agent=Claude(fix=child))\n",
	)
	_write(tmp_path / "child" / "tasks.py", _leaf("build"))
	config = load_scope(tmp_path / "tasks.py").config
	assert config is not None
	assert config.agent is not None
	assert config.agent.fix.name == "build"


def test_duplicate_project_reference_shares_load(tmp_path: Path) -> None:
	_write(
		tmp_path / "tasks.py",
		"from camas import Config, Project, Task\n"
		"root = Task(('python', '-c', 'pass'), name='root')\n"
		"a = Project('libs')\n"
		"b = Project('libs')\n"
		"_ = Config(default_task=root)\n",
	)
	_write(tmp_path / "libs" / "tasks.py", _leaf("build"))
	tasks = load_scope(tmp_path / "tasks.py").tasks
	assert "a.build" in tasks
	assert "b.build" in tasks


def test_missing_project_target_raises(tmp_path: Path) -> None:
	_write(
		tmp_path / "tasks.py",
		"from camas import Config, Project, Task\n"
		"root = Task(('python', '-c', 'pass'), name='root')\n"
		"gone = Project('nope')\n"
		"_ = Config(default_task=root)\n",
	)
	with pytest.raises(ValueError, match="no tasks"):
		load_scope(tmp_path / "tasks.py")


def test_project_escaping_root_raises(tmp_path: Path) -> None:
	_write(
		tmp_path / "root" / "tasks.py",
		"from camas import Config, Project, Task\n"
		"root = Task(('python', '-c', 'pass'), name='root')\n"
		"up = Project('..')\n"
		"_ = Config(default_task=root)\n",
	)
	_write(tmp_path / "tasks.py", _leaf("outside"))
	with pytest.raises(ValueError, match="escapes the project root"):
		load_scope(tmp_path / "root" / "tasks.py")


def test_self_reference_is_a_cycle(tmp_path: Path) -> None:
	_write(
		tmp_path / "tasks.py",
		"from camas import Config, Project, Task\n"
		"root = Task(('python', '-c', 'pass'), name='root')\n"
		"loop = Project('.')\n"
		"_ = Config(default_task=root)\n",
	)
	with pytest.raises(ValueError, match="circular Project reference"):
		load_scope(tmp_path / "tasks.py")


def test_project_without_default_is_attributed_to_child(tmp_path: Path) -> None:
	_write(
		tmp_path / "tasks.py",
		"from camas import Config, Project, Task\n"
		"root = Task(('python', '-c', 'pass'), name='root')\n"
		"empty = Project('child')\n"
		"_ = Config(default_task=root)\n",
	)
	_write(tmp_path / "child" / "tasks.py", "from camas import Task\nx = Task('true', name='x')\n")
	state = load_py_state(tmp_path / "tasks.py")
	assert isinstance(state, LoadErr)
	assert state.source == tmp_path / "child" / "tasks.py"
	assert "defines no default task" in str(state.exception)


def test_broken_child_attributed_to_child(tmp_path: Path) -> None:
	_write(
		tmp_path / "tasks.py",
		"from camas import Config, Project, Task\n"
		"root = Task(('python', '-c', 'pass'), name='root')\n"
		"broken = Project('broken')\n"
		"_ = Config(default_task=root)\n",
	)
	_write(tmp_path / "broken" / "tasks.py", "raise RuntimeError('boom')\n")
	state = load_py_state(tmp_path / "tasks.py")
	assert isinstance(state, LoadErr)
	assert state.source == tmp_path / "broken" / "tasks.py"
	assert "boom" in str(state.exception)


def test_broken_grandchild_attributed_to_grandchild(tmp_path: Path) -> None:
	_write(
		tmp_path / "tasks.py",
		"from camas import Config, Project, Task\n"
		"root = Task(('python', '-c', 'pass'), name='root')\n"
		"mid = Project('mid')\n"
		"_ = Config(default_task=root)\n",
	)
	_write(
		tmp_path / "mid" / "tasks.py",
		"from camas import Config, Project, Task\n"
		"m = Task(('python', '-c', 'pass'), name='m')\n"
		"deep = Project('deep')\n"
		"_ = Config(default_task=m)\n",
	)
	_write(tmp_path / "mid" / "deep" / "tasks.py", "raise RuntimeError('deep boom')\n")
	state = load_py_state(tmp_path / "tasks.py")
	assert isinstance(state, LoadErr)
	assert state.source == tmp_path / "mid" / "deep" / "tasks.py"
	assert "deep boom" in str(state.exception)


def test_top_file_error_is_attributed_to_itself(tmp_path: Path) -> None:
	tasks_py = _write(tmp_path / "tasks.py", "raise RuntimeError('top boom')\n")
	state = load_py_state(tasks_py)
	assert isinstance(state, LoadErr)
	assert state.source == tasks_py
	assert "top boom" in str(state.exception)


def test_load_py_state_ok(tmp_path: Path) -> None:
	tasks_py = _write(tmp_path / "tasks.py", _leaf("build"))
	state = load_py_state(tasks_py)
	assert isinstance(state, LoadOk)
	assert "build" in state.tasks


def test_reserved_name_rejected(tmp_path: Path) -> None:
	tasks_py = _write(
		tmp_path / "tasks.py",
		"from camas import Task\nmcp = Task(('python', '-c', 'pass'), name='mcp')\n",
	)
	with pytest.raises(ValueError, match="reserved"):
		load_scope(tasks_py)


def test_state_from_scope_no_file_names_plain_bindings() -> None:
	state = state_from_scope({"build": Task("true", name="build")})
	assert isinstance(state, LoadOk)
	assert "build" in state.tasks
	assert state.source is None


def test_state_from_scope_no_file_with_project_errors() -> None:
	state = state_from_scope({"libs": Project("libs")})
	assert isinstance(state, LoadErr)
	assert "file-backed" in str(state.exception)


def test_state_from_scope_with_file_composes(tmp_path: Path) -> None:
	_write(tmp_path / "libs" / "tasks.py", _leaf("build"))
	state = state_from_scope({"__file__": str(tmp_path / "tasks.py"), "libs": Project("libs")})
	assert isinstance(state, LoadOk)
	assert "libs.build" in state.tasks


def test_state_from_scope_broken_child_attributed(tmp_path: Path) -> None:
	_write(tmp_path / "broken" / "tasks.py", "raise RuntimeError('scope boom')\n")
	state = state_from_scope({"__file__": str(tmp_path / "tasks.py"), "broken": Project("broken")})
	assert isinstance(state, LoadErr)
	assert state.source == tmp_path / "broken" / "tasks.py"
	assert "scope boom" in str(state.exception)


def test_state_from_scope_reserved_name_attributed_to_source(tmp_path: Path) -> None:
	state = state_from_scope(
		{"__file__": str(tmp_path / "tasks.py"), "mcp": Task("true", name="mcp")}
	)
	assert isinstance(state, LoadErr)
	assert state.source == tmp_path / "tasks.py"
	assert "reserved" in str(state.exception)
