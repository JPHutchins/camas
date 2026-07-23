# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from camas.main import dhall
from camas.main.compose import scope as compose_scope
from camas.main.dispatch import resolve_tasks_source
from camas.main.state import LoadErr, LoadOk
from camas.v0.config import Config
from camas.v0.task import AgentFormat, Parallel, ProjectRef, Sequential, Task

T = Path("tasks.dhall")


def leaf(cmd: str, **over: Any) -> dict[str, Any]:
	return {"kind": "task", "cmd": cmd, **over}


def group(kind: str, children: list[dict[str, Any]], **over: Any) -> dict[str, Any]:
	return {"kind": kind, "children": children, **over}


def scope_of(tasks: dict[str, Any], config: Any = None) -> dict[str, object]:
	return dhall.build_scope({"tasks": tasks, "config": config}, T)


def test_leaf_defaults() -> None:
	node = scope_of({"lint": leaf("ruff .")})["lint"]
	assert node == Task("ruff .", name="lint")


def test_leaf_all_fields() -> None:
	node = scope_of(
		{
			"x": leaf(
				"fmt {paths}",
				name="fmt",
				env={"K": "v"},
				cwd="sub",
				help="format",
				mutates=True,
				paths=".",
				when=["src", "include"],
				agent_format={"args": "--sarif", "kind": "sarif", "limit": 500},
			)
		}
	)["x"]
	assert isinstance(node, Task)
	assert node.name == "fmt"
	assert node.env == {"K": "v"}
	assert node.cwd == Path("sub")
	assert node.help == "format"
	assert node.mutates
	assert node.paths == "."
	assert node.when == ("src", "include")
	assert node.agent_format == AgentFormat("--sarif", "sarif", 500)


def test_agent_format_default_limit() -> None:
	node = scope_of({"x": leaf("c", agent_format={"args": "-a", "kind": "raw"})})["x"]
	assert isinstance(node, Task)
	assert node.agent_format == AgentFormat("-a", "raw")


def test_inline_nesting_and_names() -> None:
	scope = scope_of(
		{
			"typecheck": group("parallel", [leaf("mypy ."), leaf("pyright")]),
			"check": group("sequential", [group("parallel", [leaf("a")]), leaf("b")]),
		}
	)
	tc = scope["typecheck"]
	assert isinstance(tc, Parallel)
	assert tc.name == "typecheck"
	assert [child.cmd for child in tc.tasks if isinstance(child, Task)] == ["mypy .", "pyright"]
	assert tc.tasks[0].name is None
	check = scope["check"]
	assert isinstance(check, Sequential)
	assert isinstance(check.tasks[0], Parallel)


def test_explicit_inner_name() -> None:
	node = scope_of({"g": group("parallel", [leaf("m", name="mypy")])})["g"]
	assert isinstance(node, Parallel)
	assert node.tasks[0].name == "mypy"


def test_group_matrix_and_env() -> None:
	node = scope_of(
		{"m": group("sequential", [leaf("c")], env={"E": "1"}, matrix={"PY": ["3.13", "3.14"]})}
	)["m"]
	assert isinstance(node, Sequential)
	assert node.env == {"E": "1"}
	assert node.matrix == {"PY": ("3.13", "3.14")}


def test_project_binding() -> None:
	node = scope_of({"libs": {"kind": "project", "path": "./libs"}})["libs"]
	assert node == ProjectRef("./libs")


def test_config_with_agent_resolves_by_name() -> None:
	scope = scope_of(
		{"fix": leaf("fix"), "gate": leaf("gate"), "all": leaf("all")},
		{"default_task": "all", "github_task": "gate", "agent": {"fix": "fix", "check": "gate"}},
	)
	config = scope["_"]
	assert isinstance(config, Config)
	assert config.default_task is scope["all"]
	assert config.github_task is scope["gate"]
	assert config.agent is not None
	assert config.agent.fix is scope["fix"]
	assert config.agent.check is scope["gate"]
	assert config.agent.default is None
	assert config.camas_dir == ".camas"


def test_config_without_agent_and_custom_dir() -> None:
	scope = scope_of({"a": leaf("a")}, {"default_task": "a", "camas_dir": ".ci"})
	config = scope["_"]
	assert isinstance(config, Config)
	assert config.agent is None
	assert config.github_task is None
	assert config.camas_dir == ".ci"


def test_no_config_key() -> None:
	assert "_" not in scope_of({"a": leaf("a")})


@pytest.mark.parametrize(
	("data", "message"),
	[
		("nope", "expected a record"),
		({"tasks": []}, "tasks: expected a record"),
		({"tasks": {"a": 3}}, "task 'a'"),
		({"tasks": {"a": {"kind": "task", "cmd": 3}}}, "cmd: expected text"),
		({"tasks": {"a": leaf("c", mutates="x")}}, "expected a boolean"),
		({"tasks": {"a": leaf("c", env={"K": 1})}}, "expected a Map Text"),
		({"tasks": {"a": leaf("c", when=[1])}}, "expected a list of text"),
		({"tasks": {"a": {"kind": "parallel", "children": "x"}}}, "children: expected a list"),
		({"tasks": {"a": {"kind": "seq2", "children": []}}}, "unknown kind"),
		({"tasks": {"a": group("parallel", [], matrix=5)}}, "expected a Map"),
	],
)
def test_build_scope_errors(data: Any, message: str) -> None:
	with pytest.raises(ValueError, match=message):
		dhall.build_scope(data, T)


def test_agent_format_bad_kind() -> None:
	with pytest.raises(ValueError, match="kind: expected one of"):
		scope_of({"a": leaf("c", agent_format={"args": "-a", "kind": "xml"})})


@pytest.mark.parametrize("limit", [0, -3, True, "x"])
def test_agent_format_bad_limit(limit: Any) -> None:
	with pytest.raises(ValueError, match="limit: expected a positive"):
		scope_of({"a": leaf("c", agent_format={"args": "-a", "kind": "raw", "limit": limit})})


def test_config_unknown_ref() -> None:
	with pytest.raises(ValueError, match="references unknown task"):
		scope_of({"a": leaf("a")}, {"default_task": "missing"})


def test_config_agent_fix_required() -> None:
	with pytest.raises(ValueError, match="fix is required"):
		scope_of({"a": leaf("a")}, {"agent": {"fix": ""}})


def test_config_agent_not_a_record() -> None:
	with pytest.raises(ValueError, match="agent: expected a record"):
		scope_of({"a": leaf("a")}, {"agent": 5})


def test_config_not_a_record() -> None:
	with pytest.raises(ValueError, match="config: expected a record"):
		dhall.build_scope({"tasks": {}, "config": 7}, T)


def test_prelude_ships() -> None:
	prelude = dhall.prelude_path()
	assert prelude.is_file()
	assert prelude.name == "prelude.dhall"


def _fake_dhall(payload: object) -> ModuleType:
	module = ModuleType("dhall")

	def load(_handle: object) -> object:
		return payload

	cast("Any", module).load = load
	return module


def test_evaluate_dhall_parses_rendered_json(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	source = tmp_path / "t.dhall"
	source.write_text("''", encoding="utf-8")
	monkeypatch.setitem(sys.modules, "dhall", _fake_dhall('{"tasks": {}}'))
	assert dhall.evaluate_dhall(source) == {"tasks": {}}


def test_evaluate_dhall_passes_through_non_string(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	source = tmp_path / "t.dhall"
	source.write_text("{=}", encoding="utf-8")
	monkeypatch.setitem(sys.modules, "dhall", _fake_dhall({"tasks": {}}))
	assert dhall.evaluate_dhall(source) == {"tasks": {}}


def test_evaluate_dhall_missing_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setitem(sys.modules, "dhall", None)
	with pytest.raises(RuntimeError, match="camas\\[dhall\\]"):
		dhall.evaluate_dhall(tmp_path / "t.dhall")


def _write(path: Path, text: str = "unused") -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(text, encoding="utf-8")
	return path


def test_load_dhall_scope_composes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	root = _write(tmp_path / "tasks.dhall")
	data: object = {"tasks": {"a": leaf("a")}, "config": {"default_task": "a"}}

	def evaluate(_path: Path) -> object:
		return data

	monkeypatch.setattr(dhall, "evaluate_dhall", evaluate)
	loaded = compose_scope.load_dhall_scope(root)
	assert isinstance(loaded, LoadOk)
	assert loaded.tasks["a"] == Task("a", name="a")
	assert loaded.config is not None


def test_monorepo_child_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	for var in ("GITHUB_ACTIONS", "CLAUDECODE", "CAMAS_AGENT"):
		monkeypatch.delenv(var, raising=False)
	root = _write(tmp_path / "tasks.dhall")
	_write(tmp_path / "libs" / "tasks.dhall")
	root_data = {"tasks": {"libs": {"kind": "project", "path": "./libs"}}, "config": None}
	child_data = {
		"tasks": {"check": leaf("check")},
		"config": {"default_task": "check", "github_task": "check"},
	}

	def evaluate(path: Path) -> object:
		return child_data if "libs" in path.parts else root_data

	monkeypatch.setattr(dhall, "evaluate_dhall", evaluate)
	loaded = compose_scope.load_dhall_scope(root)
	assert isinstance(loaded, LoadOk)
	assert "libs.check" in loaded.tasks
	assert isinstance(loaded.tasks["libs"], Task)


def test_load_dhall_tasks_state_captures_error(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	root = _write(tmp_path / "tasks.dhall")

	def boom(_path: Path) -> object:
		raise RuntimeError("bad dhall")

	monkeypatch.setattr(dhall, "evaluate_dhall", boom)
	state = compose_scope.load_dhall_tasks_state(root)
	assert isinstance(state, LoadErr)
	assert "bad dhall" in str(state.exception)


def test_child_tasks_file(tmp_path: Path) -> None:
	assert compose_scope.child_tasks_file(tmp_path) is None
	_write(tmp_path / "tasks.dhall")
	assert compose_scope.child_tasks_file(tmp_path) == tmp_path / "tasks.dhall"
	py = _write(tmp_path / "tasks.py")
	assert compose_scope.child_tasks_file(tmp_path) == tmp_path / "tasks.py"
	assert compose_scope.child_tasks_file(py) == py


def test_resolve_source_explicit_dhall_missing_extra(tmp_path: Path) -> None:
	root = _write(tmp_path / "tasks.dhall")
	state, rest = resolve_tasks_source([str(root), "lint"])
	assert rest == ["lint"]
	assert isinstance(state, LoadErr)


def test_resolve_source_walk_finds_dhall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_write(tmp_path / "tasks.dhall")
	monkeypatch.chdir(tmp_path)

	def evaluate(_path: Path) -> object:
		return {"tasks": {"a": leaf("a")}}

	monkeypatch.setattr(dhall, "evaluate_dhall", evaluate)
	state, rest = resolve_tasks_source([])
	assert rest == []
	assert isinstance(state, LoadOk)
	assert "a" in state.tasks
