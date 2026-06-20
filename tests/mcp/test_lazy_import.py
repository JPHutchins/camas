# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence
	from pathlib import Path


def test_entrypoint_mcp_branch_invokes_serve(monkeypatch: pytest.MonkeyPatch) -> None:
	from camas.main import main
	from camas.mcp import serve

	calls: list[list[str]] = []
	monkeypatch.setattr(serve, "serve_stdio", calls.append)
	monkeypatch.setattr("sys.argv", ["camas", "mcp", "--rich"])
	main()
	assert calls == [["--rich"]]


def test_entrypoint_mcp_branch_missing_extra_exits_with_feature_hint(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	from camas.main import main

	original_import = __import__

	def fake_import(
		name: str,
		globals: Mapping[str, object] | None = None,
		locals: Mapping[str, object] | None = None,
		fromlist: Sequence[str] = (),
		level: int = 0,
	) -> object:
		if name == "mcp":
			raise ModuleNotFoundError("No module named 'mcp'", name="mcp")
		return original_import(name, globals, locals, fromlist, level)

	for module in ("camas.mcp.serve", "mcp"):
		monkeypatch.delitem(sys.modules, module, raising=False)
	monkeypatch.setattr("sys.argv", ["camas", "mcp"])
	with patch("builtins.__import__", fake_import), pytest.raises(SystemExit) as exc:
		main()

	assert exc.value.code == 2
	err = capsys.readouterr().err
	assert "camas mcp: requires feature camas[mcp]" in err
	assert "pip" not in err
	assert "uv" not in err


def test_entrypoint_mcp_branch_reraises_non_mcp_missing_import(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	from camas.main import main

	original_import = __import__

	def fake_import(
		name: str,
		globals: Mapping[str, object] | None = None,
		locals: Mapping[str, object] | None = None,
		fromlist: Sequence[str] = (),
		level: int = 0,
	) -> object:
		if name == "pydantic":
			raise ModuleNotFoundError("No module named 'pydantic'", name="pydantic")
		return original_import(name, globals, locals, fromlist, level)

	for module in ("camas.mcp.serve", "pydantic"):
		monkeypatch.delitem(sys.modules, module, raising=False)
	monkeypatch.setattr("sys.argv", ["camas", "mcp"])
	with patch("builtins.__import__", fake_import), pytest.raises(ModuleNotFoundError) as exc:
		main()

	assert exc.value.name == "pydantic"


def test_camas_list_does_not_import_mcp_stack(tmp_path: Path) -> None:
	"""The MCP server's heavy deps must never load on the ``camas <task>`` hot path."""
	from textwrap import dedent

	script = dedent("""
		import sys
		sys.argv = ['camas', '--list']
		from camas.main import main
		try:
			main()
		except SystemExit:
			pass
		heavy = {'mcp', 'pydantic', 'pydantic_core', 'starlette', 'uvicorn', 'anyio'}
		leaked = sorted(m for m in sys.modules if m.split('.')[0] in heavy)
		assert not leaked, leaked
	""")
	result = subprocess.run(
		[sys.executable, "-c", script],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		check=False,
	)
	assert result.returncode == 0, result.stderr
