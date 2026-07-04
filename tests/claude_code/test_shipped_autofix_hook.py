# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The actually-shipped autofix hook survives a headless edit (criterion #2).

Like ``test_autofix_e2e.py``, but the ``.claude/settings.json`` is produced by
``camas mcp init --claude`` instead of hand-crafted — the hook is the one users actually get.
The registered fixer rewrites ``BANANA`` to ``FIXED``; the edit writes ``BANANA``; the shipped
``PostToolBatch`` hook fires and leaves ``FIXED`` on disk.

Broken variant: mutate the shipped settings to move the camas hook from ``PostToolBatch`` to
``FileChanged`` (the wrong event — ``FileChanged`` does not fire on Claude's own edits, as
proved by ``test_hook_contract.py``). Re-run the edit; assert ``BANANA`` survives, proving
a wrong-event regression fails the harness.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
	from collections.abc import Callable
	from subprocess import CompletedProcess

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UV = shutil.which("uv") or "uv"
_ENV = {**os.environ, "UV_PYTHON": "3.12", "UV_PYTHON_DOWNLOADS": "never"}

_PY = shlex.quote(sys.executable)

_PYPROJECT = (
	"[project]\n"
	'name = "test-harness"\n'
	'version = "0.0.0"\n'
	'requires-python = ">=3.10"\n'
	'dependencies = ["camas"]\n'
	"\n[tool.uv.sources]\n"
	f'camas = {{ path = "{_REPO_ROOT}" }}\n'
)

_FIXER = (
	"import pathlib, sys\n"
	"for p in sys.argv[1:]:\n"
	"    fp = pathlib.Path(p)\n"
	'    fp.write_text(fp.read_text().replace("BANANA", "FIXED"))\n'
)

_TASKS = (
	"from camas import Claude, Config, Task\n"
	f'tidy = Task("{_PY} fixer.py {{paths}}", name="tidy", mutates=True, paths=".")\n'
	"_ = Config(agent=Claude(fix=tidy))\n"
)

_EDIT_PROMPT = (
	"Create a file named sample.txt containing exactly the word BANANA. "
	"Use the Write tool and nothing else."
)


def _setup_project(tmp_path: Path) -> None:
	(tmp_path / "pyproject.toml").write_text(_PYPROJECT)
	(tmp_path / "tasks.py").write_text(_TASKS)
	(tmp_path / "fixer.py").write_text(_FIXER)
	sync = subprocess.run(
		[_UV, "sync"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		timeout=180,
		check=False,
		env=_ENV,
	)
	assert sync.returncode == 0, (
		f"uv sync failed: rc={sync.returncode}\nstdout={sync.stdout}\nstderr={sync.stderr}"
	)


def _init_claude(tmp_path: Path) -> None:
	proc = subprocess.run(
		[_UV, "run", "--project", str(_REPO_ROOT), "camas", "mcp", "init", "--claude"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		timeout=180,
		check=False,
		env=_ENV,
	)
	assert proc.returncode == 0, (
		f"camas mcp init --claude failed: rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
	)


def test_shipped_post_tool_batch_hook_runs_the_autofix(
	tmp_path: Path,
	run_headless: Callable[..., CompletedProcess[str]],
) -> None:
	_setup_project(tmp_path)
	_init_claude(tmp_path)

	proc = run_headless(tmp_path, _EDIT_PROMPT)
	assert proc.returncode == 0, proc.stderr

	sample = tmp_path / "sample.txt"
	assert sample.exists(), "the edit did not happen"
	content = sample.read_text()
	assert "BANANA" not in content, (
		f"the shipped autofix did not rewrite the changed file: {content!r}"
	)
	assert "FIXED" in content, f"the shipped autofix did not rewrite the changed file: {content!r}"


def test_moved_to_filechanged_event_does_not_fire(
	tmp_path: Path,
	run_headless: Callable[..., CompletedProcess[str]],
) -> None:
	_setup_project(tmp_path)
	_init_claude(tmp_path)

	settings = cast(
		"dict[str, object]",
		json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8")),
	)
	hooks = cast("dict[str, list[dict[str, object]]]", settings["hooks"])
	ptb = hooks.pop("PostToolBatch", [])
	hooks["FileChanged"] = ptb
	settings["hooks"] = hooks
	(tmp_path / ".claude" / "settings.json").write_text(
		json.dumps(settings, indent=2) + "\n", encoding="utf-8"
	)

	proc = run_headless(tmp_path, _EDIT_PROMPT)
	assert proc.returncode == 0, proc.stderr

	sample = tmp_path / "sample.txt"
	assert sample.exists(), "the edit did not happen"
	content = sample.read_text()
	assert "FIXED" not in content, "FileChanged fired on an edit — regression in the hook contract"
	assert "BANANA" in content, "the edit should have written BANANA and it was lost"
