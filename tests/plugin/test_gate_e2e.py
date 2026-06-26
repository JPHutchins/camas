# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""End-to-end gate "sandbox": a real tasks.py in a tmp project, driven through the
``camas_gate`` MCP tool (what the plugin's PostToolBatch hook calls), exercising the
autofix-then-check-then-classify loop against files on disk. tmp_path is auto-removed."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp import types

from camas.main.tasks import load_tasks_from_py
from camas.mcp import serve
from camas.mcp.serve import Compat, Session

if TYPE_CHECKING:
	from pathlib import Path

	import pytest

# A fix leaf that deletes the literal "TODO" from sample.py, and a check that fails while a
# marker is still present. Same fix in both; the check's marker decides autofixed vs residual.
_FIX = (
	"fix = Task(\n"
	'\t("python", "-c", "import pathlib; p = pathlib.Path(\'sample.py\'); '
	"p.write_text(p.read_text().replace('TODO', ''))\"),\n"
	'\tname="fix",\n\tmutates=True,\n)\n'
)


def _tasks_py(marker: str) -> str:
	return (
		"from camas import Config, Sequential, Task\n\n" + _FIX + "check = Task(\n"
		f'\t("python", "-c", "import pathlib, sys; sys.exit(\'{marker}\' in pathlib.Path(\'sample.py\').read_text())"),\n'
		'\tname="check",\n)\n'
		"_ = Config(default_task=Sequential(fix, check))\n"
	)


def _text(result: types.CallToolResult) -> str:
	return "".join(c.text for c in result.content if isinstance(c, types.TextContent))


def _session(tmp_path: Path, marker: str) -> Session:
	(tmp_path / "tasks.py").write_text(_tasks_py(marker))
	return Session(load_tasks_from_py(tmp_path / "tasks.py"), tmp_path, Compat())


async def test_gate_autofix_edits_a_real_file_then_passes(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "sample.py").write_text("TODO\nvalue = 1\n")
	result = await serve.call(_session(tmp_path, "TODO"), "camas_gate", {})
	assert not result.isError
	assert "autofixed" in _text(result)
	assert "TODO" not in (tmp_path / "sample.py").read_text()


async def test_gate_blocks_on_a_residual_the_fixer_cannot_resolve(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "sample.py").write_text("FIXME\nvalue = 1\n")
	result = await serve.call(_session(tmp_path, "FIXME"), "camas_gate", {})
	assert not result.isError
	assert "needs_reasoning" in _text(result)
