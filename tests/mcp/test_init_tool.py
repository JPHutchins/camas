# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""camas_init: scaffold a starter tasks.py over the MCP, refusing to overwrite an existing one,
and re-resolve so the new tasks are immediately live. tmp_path is auto-removed."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp import types

from camas.main.init import starter_text
from camas.main.state import LoadOk
from camas.mcp import serve
from camas.mcp.serve import Compat, Session

if TYPE_CHECKING:
	from pathlib import Path

	import pytest


def _text(result: types.CallToolResult) -> str:
	return "".join(c.text for c in result.content if isinstance(c, types.TextContent))


def _session(base: Path, *, rich: bool = False) -> Session:
	return Session(serve.resolve_project_quiet(base), base, Compat(emit_structured=rich))


def test_init_scaffolds_tasks_py_and_camas_dir(tmp_path: Path) -> None:
	result = serve.init_call(_session(tmp_path), {})
	assert result.isError is False
	assert (tmp_path / "tasks.py").exists()
	assert (tmp_path / ".camas").is_dir()
	text = _text(result)
	assert "Scaffolded" in text
	assert "from camas import" in text


def test_init_defaults_to_verbose_template(tmp_path: Path) -> None:
	serve.init_call(_session(tmp_path), {})
	written = (tmp_path / "tasks.py").read_text(encoding="utf-8")
	assert written == starter_text(verbose=True)
	assert written != starter_text()


def test_init_verbose_false_writes_minimal_template(tmp_path: Path) -> None:
	serve.init_call(_session(tmp_path), {"verbose": False})
	assert (tmp_path / "tasks.py").read_text(encoding="utf-8") == starter_text()


def test_init_invalid_arguments_is_tool_error(tmp_path: Path) -> None:
	result = serve.init_call(_session(tmp_path), {"bogus": 1})
	assert result.isError is True
	assert "invalid camas_init arguments" in _text(result)
	assert not (tmp_path / "tasks.py").exists()


def test_init_makes_tasks_immediately_runnable(tmp_path: Path) -> None:
	session = _session(tmp_path)
	serve.init_call(session, {})
	assert isinstance(session.project, LoadOk)
	assert "ci" in session.project.tasks


def test_init_refuses_to_overwrite(tmp_path: Path) -> None:
	kept = "from camas import Task\nkeep = Task('echo keep')\n"
	(tmp_path / "tasks.py").write_text(kept)
	result = serve.init_call(_session(tmp_path), {})
	assert result.isError is False
	assert "already exists" in _text(result)
	assert (tmp_path / "tasks.py").read_text() == kept


def test_init_structured_when_rich(tmp_path: Path) -> None:
	result = serve.init_call(_session(tmp_path, rich=True), {})
	assert result.structuredContent is not None
	assert result.structuredContent["status"] == "created"
	assert result.structuredContent["path"].endswith("tasks.py")


def test_init_reports_os_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	def _boom(directory: Path, *, verbose: bool = False) -> Path:
		raise PermissionError("denied")

	monkeypatch.setattr("camas.mcp.serve.create_starter_tasks_py", _boom)
	result = serve.init_call(_session(tmp_path), {})
	assert result.isError is True
	assert "could not scaffold" in _text(result)


async def test_call_routes_init(tmp_path: Path) -> None:
	result = await serve.call(_session(tmp_path), "camas_init", {})
	assert result.isError is False
	assert (tmp_path / "tasks.py").exists()


def test_init_call_repins_version_warning(tmp_path: Path) -> None:
	session = _session(tmp_path)
	session.version_warning = "WARNING: stale"
	result = serve.init_call(session, {})
	assert result.isError is False
	assert session.version_warning is None
