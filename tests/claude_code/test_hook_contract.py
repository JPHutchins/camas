# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The Claude Code hook contract camas relies on, verified against a real headless run:

- ``PostToolBatch`` fires on an edit and carries the changed file at
  ``tool_calls[].tool_input.file_path`` (what ``camas mcp fix`` reads from stdin).
- ``${file_path}`` is NOT interpolated into a command hook — it shell-expands to nothing, which
  is why the hook delivers the path on stdin instead of via ``--paths ${file_path}``.
- ``FileChanged`` does NOT fire on Claude's own edits (it is a disk watcher), so the autofix hook
  is on ``PostToolBatch``.

If a future Claude Code changes any of these, this suite fails and camas's hook is revisited.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
	from collections.abc import Callable
	from subprocess import CompletedProcess

	import pytest

_PROBE = Path(__file__).parent / "_probe.py"


def _settings() -> dict[str, object]:
	def probe(event: str, extra: str = "") -> dict[str, str]:
		return {"type": "command", "command": f'python3 "{_PROBE}" --event={event}{extra}'}

	return {
		"hooks": {
			"FileChanged": [{"hooks": [probe("FileChanged", " --paths ${file_path}")]}],
			"PostToolUse": [
				{
					"matcher": "Write|Edit|MultiEdit",
					"hooks": [probe("PostToolUse", " --paths ${file_path}")],
				}
			],
			"PostToolBatch": [{"hooks": [probe("PostToolBatch")]}],
		}
	}


def _get(obj: object, key: str) -> object:
	return cast("dict[str, object]", obj).get(key) if isinstance(obj, dict) else None


def _argv(record: object) -> list[str]:
	raw = _get(record, "argv")
	return (
		[a for a in cast("list[object]", raw) if isinstance(a, str)]
		if isinstance(raw, list)
		else []
	)


def _event(record: object) -> object:
	return _get(_get(record, "stdin_json"), "hook_event_name")


def _batch_file_paths(record: object) -> list[str]:
	calls = _get(_get(record, "stdin_json"), "tool_calls")
	if not isinstance(calls, list):
		return []
	paths = (_get(_get(call, "tool_input"), "file_path") for call in cast("list[object]", calls))
	return [p for p in paths if isinstance(p, str)]


def _records(log: Path) -> list[object]:
	return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def test_post_tool_batch_is_the_edit_event_and_file_path_is_not_interpolated(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	run_headless: Callable[[Path, str], CompletedProcess[str]],
) -> None:
	(tmp_path / ".claude").mkdir()
	(tmp_path / ".claude" / "settings.json").write_text(json.dumps(_settings()))
	log = tmp_path / "probe.log"
	log.touch()
	monkeypatch.setenv("CAMAS_PROBE_LOG", str(log))

	proc = run_headless(
		tmp_path,
		"Create a file named PROBE.txt containing exactly the word MANGO. "
		"Use the Write tool and nothing else.",
	)
	assert proc.returncode == 0, proc.stderr

	records = _records(log)
	assert (tmp_path / "PROBE.txt").exists(), "the edit did not happen"

	batch_paths = [p for r in records if _event(r) == "PostToolBatch" for p in _batch_file_paths(r)]
	assert batch_paths, "PostToolBatch did not fire on an edit with a changed path"
	assert any(p.endswith("PROBE.txt") for p in batch_paths), (
		f"PostToolBatch carried no path to the edited file: {batch_paths}"
	)

	interpolated = [r for r in records if "--paths" in _argv(r)]
	assert interpolated, "expected a hook whose command contained --paths ${file_path}"
	assert all(_argv(r)[-1] == "--paths" for r in interpolated), (
		f"${{file_path}} was interpolated instead of shell-expanding to nothing: "
		f"{[_argv(r) for r in interpolated]}"
	)

	assert all(_event(r) != "FileChanged" for r in records), (
		"FileChanged should not fire on Claude's edits"
	)
