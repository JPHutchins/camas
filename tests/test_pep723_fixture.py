# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from camas import Task
from camas.main.tasks import load_tasks_from_py

FIXTURE = Path(__file__).parent / "fixtures" / "pep723" / "tasks.py"


def test_pep723_header_is_inert_to_loader() -> None:
	"""The PEP 723 header (and the ``__main__`` block) leave the loader's view of
	the module unchanged — it reads identically to a header-less ``tasks.py``."""
	loaded = load_tasks_from_py(FIXTURE)
	assert loaded.scope_effects == {}
	assert loaded.tasks["hello"] == Task(
		("python", "-c", "print('hello from pep723')"), name="hello"
	)


def test_pep723_runs_standalone() -> None:
	"""``python tasks.py hello`` dispatches through the ``run_cli(globals())`` entry
	point using the camas in the current environment. (PEP 723 dependency
	resolution is uv's concern, not camas's, so it is not exercised here.)"""
	result = subprocess.run(
		[
			sys.executable,
			str(FIXTURE),
			"hello",
			"--effects",
			"(Summary(show_passing=True),)",
		],
		capture_output=True,
		text=True,
		check=False,
	)
	assert result.returncode == 0, result.stderr
	assert "hello from pep723" in result.stdout
