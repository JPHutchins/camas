# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""End-to-end proof of the shipped autofix hook: a real headless edit fires ``PostToolBatch``,
which runs the registered ``Config.agent.fix`` node over the just-changed file (path delivered on
stdin), with zero model tokens. The registered fixer rewrites ``BANANA`` to ``FIXED``; the edit
writes ``BANANA``; a green run leaves ``FIXED`` on disk.

The hook runs the *local* camas (``python -m camas``), not ``uvx`` from PyPI, so it exercises the
working tree.
"""

from __future__ import annotations

import shlex
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable
	from pathlib import Path
	from subprocess import CompletedProcess

_PY = shlex.quote(sys.executable)

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

_HOOK_SETTINGS = (
	'{ "hooks": { "PostToolBatch": [ { "hooks": [ '
	f'{{ "type": "command", "command": "{_PY} -m camas mcp fix" }}'
	" ] } ] } }"
)


def test_post_tool_batch_hook_runs_the_registered_autofix(
	tmp_path: Path, run_headless: Callable[[Path, str], CompletedProcess[str]]
) -> None:
	(tmp_path / "tasks.py").write_text(_TASKS)
	(tmp_path / "fixer.py").write_text(_FIXER)
	(tmp_path / ".claude").mkdir()
	(tmp_path / ".claude" / "settings.json").write_text(_HOOK_SETTINGS)

	proc = run_headless(
		tmp_path,
		"Create a file named sample.txt containing exactly the word BANANA. Use the Write tool and nothing else.",
	)
	assert proc.returncode == 0, proc.stderr

	sample = tmp_path / "sample.txt"
	assert sample.exists(), "the edit did not happen"
	content = sample.read_text()
	assert "BANANA" not in content, f"the autofix did not rewrite the changed file: {content!r}"
	assert "FIXED" in content, f"the autofix did not rewrite the changed file: {content!r}"
