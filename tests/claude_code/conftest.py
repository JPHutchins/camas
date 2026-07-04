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

# The headless `claude -p` and its MCP server / PostToolBatch hook inherit this process's env. The
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
	- ``strict_mcp`` (default ``False``): pass ``--strict-mcp-config``.
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
			argv.append("--strict-mcp-config")
		if append_system_prompt is not None:
			argv.extend(("--append-system-prompt", append_system_prompt))
		if output_format is not None:
			argv.extend(("--output-format", output_format))
		return subprocess.run(
			argv,
			cwd=cwd,
			capture_output=True,
			text=True,
			timeout=300,
			check=False,
		)

	return _run
