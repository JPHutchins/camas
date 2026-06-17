# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import Path

	import pytest


def test_entrypoint_mcp_branch_invokes_serve(monkeypatch: pytest.MonkeyPatch) -> None:
	from camas.main import main
	from camas.mcp import serve

	calls: list[list[str]] = []
	monkeypatch.setattr(serve, "serve_stdio", calls.append)
	monkeypatch.setattr("sys.argv", ["camas", "mcp", "--rich"])
	main()
	assert calls == [["--rich"]]


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
