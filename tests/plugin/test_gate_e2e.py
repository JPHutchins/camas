# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""End-to-end gate "sandbox": a real tasks.py in a tmp project, driven through the
``camas_gate`` MCP tool (what the camas-fixer subagent runs). The gate is check-only —
it runs the project's check node over the workspace and classifies green vs needs_reasoning,
and it never mutates. tmp_path is auto-removed."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp import types

from camas.main.tasks import load_tasks_from_py
from camas.mcp import serve
from camas.mcp.serve import Compat, Session

if TYPE_CHECKING:
	from pathlib import Path

	import pytest


def _tasks_py(marker: str) -> str:
	# A read-only check that fails while `marker` is present in sample.py — the gate's check node.
	return (
		"from camas import Config, Task\n\ncheck = Task(\n"
		f'\t("python", "-c", "import pathlib, sys; sys.exit(\'{marker}\' in pathlib.Path(\'sample.py\').read_text())"),\n'
		'\tname="check",\n)\n'
		"_ = Config(default_task=check)\n"
	)


def _text(result: types.CallToolResult) -> str:
	return "".join(c.text for c in result.content if isinstance(c, types.TextContent))


def _session(tmp_path: Path, marker: str) -> Session:
	(tmp_path / "tasks.py").write_text(_tasks_py(marker))
	return Session(load_tasks_from_py(tmp_path / "tasks.py"), tmp_path, Compat())


async def test_gate_is_green_when_the_check_passes(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "sample.py").write_text("value = 1\n")
	result = await serve.call(_session(tmp_path, "FIXME"), "camas_gate", {})
	assert not result.isError
	assert "green" in _text(result)


async def test_gate_blocks_on_a_failing_check_without_mutating(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "sample.py").write_text("FIXME\nvalue = 1\n")
	result = await serve.call(_session(tmp_path, "FIXME"), "camas_gate", {})
	assert not result.isError
	assert "needs_reasoning" in _text(result)
	assert (tmp_path / "sample.py").read_text() == "FIXME\nvalue = 1\n"
