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

_ENABLED = bool(os.environ.get("CAMAS_CC_E2E")) and shutil.which("claude") is not None


@pytest.fixture
def run_headless() -> Callable[[Path, str], subprocess.CompletedProcess[str]]:
	"""A callable that runs ``claude -p`` headless in a cwd, edits auto-approved, model from env.

	Every test in this suite requests it, so requesting it is what opts a test into the real run —
	unless ``CAMAS_CC_E2E`` is set with ``claude`` on PATH, requesting it skips the test.
	"""
	if not _ENABLED:
		pytest.skip(
			"set CAMAS_CC_E2E=1 with `claude` on PATH to run the Claude Code integration suite"
		)

	def _run(cwd: Path, prompt: str) -> subprocess.CompletedProcess[str]:
		model = os.environ.get("CAMAS_CC_MODEL", "sonnet")
		return subprocess.run(
			["claude", "-p", prompt, "--model", model, "--permission-mode", "acceptEdits"],
			cwd=cwd,
			capture_output=True,
			text=True,
			timeout=300,
			check=False,
		)

	return _run
