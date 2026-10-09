# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The two Stop hooks ``camas mcp init --claude`` writes, run the way Claude Code runs a Stop
event's hooks — in parallel, on the same event — as real processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import ExitStack
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import Path

_TASKS = (
	"from camas import Claude, Config, Task\n"
	"fix = Task(('python', '-c', \"import pathlib, time; time.sleep(3); "
	"pathlib.Path('sample.txt').write_text('clean')\"), name='fix', mutates=True)\n"
	"check = Task(('python', '-c', \"import pathlib, sys; "
	"sys.exit('dirty' in pathlib.Path('sample.txt').read_text())\"), name='check')\n"
	"_ = Config(default_task=check, agent=Claude(fix=fix))\n"
)


def test_the_parallel_stop_hooks_check_only_after_the_autofix_settles(tmp_path: Path) -> None:
	"""A slow autofix cleans a file the check rejects; the nudge, started alongside it, must wait
	for the fix to settle and find the workspace green — not wake the agent over a residual the
	fix was still settling."""
	(tmp_path / "tasks.py").write_text(_TASKS)
	(tmp_path / "sample.txt").write_text("dirty")
	temp = tmp_path / "tmp"
	temp.mkdir()
	env = {key: value for key, value in os.environ.items() if key != "CLAUDE_PROJECT_DIR"} | {
		"TMPDIR": str(temp),
		"TEMP": str(temp),
		"TMP": str(temp),
	}
	event = json.dumps(
		{
			"hook_event_name": "Stop",
			"session_id": "s-1",
			"prompt_id": "p-1",
			"stop_hook_active": False,
		}
	)
	hooks = (("fix",), ("gate", "--under", "5s", "--nudge"))
	logs = tuple(tmp_path / f"{args[0]}.log" for args in hooks)
	with ExitStack() as stack:
		outputs = tuple(stack.enter_context(log.open("w")) for log in logs)
		procs = tuple(
			subprocess.Popen(
				[sys.executable, "-m", "camas", "mcp", *args],
				cwd=tmp_path,
				stdin=subprocess.PIPE,
				stdout=output,
				stderr=subprocess.STDOUT,
				text=True,
				env=env,
			)
			for args, output in zip(hooks, outputs, strict=True)
		)
		for proc in procs:
			assert proc.stdin is not None
			proc.stdin.write(event)
			proc.stdin.close()
		fix, nudge = (proc.wait(timeout=120) for proc in procs)
	assert fix == 0, logs[0].read_text()
	assert nudge == 0, logs[1].read_text()
	assert (tmp_path / "sample.txt").read_text() == "clean"
