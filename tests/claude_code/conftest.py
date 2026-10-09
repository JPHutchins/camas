# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Opt-in Claude Code integration suite: drives a real headless ``claude -p`` to pin down camas's
assumptions about Claude Code hooks (which events fire on an edit, how the changed path is
delivered) and to prove the shipped autofix hook end to end.

Skipped unless ``CAMAS_CC_E2E`` is set and ``claude`` is on PATH. The model is ``CAMAS_CC_MODEL``
(default ``sonnet`` for local runs); CI sets it to ``deepseek-v4-flash`` and points the Anthropic
backend env (``ANTHROPIC_BASE_URL``/``ANTHROPIC_AUTH_TOKEN``) at DeepSeek, exactly as the agentic
review workflow does. Excluded from coverage (see ``pyproject.toml`` ``[tool.coverage.run]``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from pathlib import Path
	from subprocess import CompletedProcess

_ENABLED = bool(os.environ.get("CAMAS_CC_E2E")) and shutil.which("claude") is not None

# The headless `claude -p` and its MCP server / Stop hooks inherit this process's env. The
# repo's .python-version (3.14) is absent on the CI runner; pin UV_PYTHON to a present interpreter
# (3.12, matching the harness workflow's own `uv run --python 3.12`) and forbid downloads, so the
# shipped `uv run camas …` launcher resolves instead of failing "No interpreter found for 3.14".
os.environ.setdefault("UV_PYTHON", "3.12")
os.environ.setdefault("UV_PYTHON_DOWNLOADS", "never")


@pytest.fixture
def run_headless() -> Callable[..., CompletedProcess[str]]:
	"""A callable that runs ``claude -p`` headless in a cwd, edits auto-approved, model from env.

	Every test in this suite requests it, so requesting it is what opts a test into the real run —
	unless ``CAMAS_CC_E2E`` is set with ``claude`` on PATH, requesting it skips the test.

	The returned callable accepts optional keyword-only overrides:

	- ``permission_mode`` (default ``acceptEdits``): ``--permission-mode`` value.
	- ``strict_mcp`` (default ``False``): load only the cwd's ``.mcp.json`` (``--mcp-config``
	  plus ``--strict-mcp-config``; the strict flag alone loads no server at all).
	- ``output_format`` ``"stream-json"`` adds the ``--verbose`` it requires in print mode.
	- ``append_system_prompt`` (default ``None``): appended via ``--append-system-prompt``.
	- ``output_format`` (default ``None``): ``--output-format`` value.
	"""
	if not _ENABLED:
		pytest.skip(
			"set CAMAS_CC_E2E=1 with `claude` on PATH to run the Claude Code integration suite"
		)

	model = os.environ.get("CAMAS_CC_MODEL", "sonnet")

	def _run(
		cwd: Path,
		prompt: str,
		*,
		permission_mode: str = "acceptEdits",
		strict_mcp: bool = False,
		append_system_prompt: str | None = None,
		output_format: str | None = None,
	) -> CompletedProcess[str]:
		argv = ["claude", "-p", prompt, "--model", model, "--permission-mode", permission_mode]
		if strict_mcp:
			argv.extend(("--mcp-config", str(cwd / ".mcp.json"), "--strict-mcp-config"))
		if append_system_prompt is not None:
			argv.extend(("--append-system-prompt", append_system_prompt))
		if output_format is not None:
			argv.extend(("--output-format", output_format))
		if output_format == "stream-json":
			argv.append("--verbose")
		return subprocess.run(
			argv,
			cwd=cwd,
			capture_output=True,
			text=True,
			timeout=300,
			check=False,
		)

	return _run


@pytest.fixture
def mcp_server_status() -> Callable[[str], dict[str, str]]:
	"""Each MCP server's status (``connected``, ``failed``, ...) from a ``stream-json`` run's init
	message — ``claude -p`` exits 0 whether or not a server started."""

	def _status(stream: str) -> dict[str, str]:
		messages = (json.loads(line) for line in stream.splitlines() if line.strip())
		init = next(
			(m for m in messages if m.get("type") == "system" and m.get("subtype") == "init"), None
		)
		assert init is not None, f"no init message in the stream-json output: {stream[:500]!r}"
		return {server["name"]: server["status"] for server in init.get("mcp_servers", [])}

	return _status
