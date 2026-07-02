# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Parse a Claude Code ``PostToolBatch`` hook event piped on stdin into the changed paths.

The autofix (``camas mcp fix``) and the gate (``camas mcp gate``) both read the just-edited
files from the same event, so the parsing lives here — dependency-free (``json``/``sys`` only)
so ``camas mcp fix`` works without the ``mcp`` extra.
"""

from __future__ import annotations

import json
import sys
from typing import cast


def _event_get(obj: object, key: str) -> object:
	"""``obj[key]`` for a parsed-JSON object, else ``None`` — narrows JSON's ``Any`` to ``object``."""
	return cast("dict[str, object]", obj).get(key) if isinstance(obj, dict) else None


def stdin_changed() -> tuple[str, ...] | None:
	"""The edited files in a ``PostToolBatch`` event piped on stdin (the Claude Code plugin's
	autofix/gate hook), de-duplicated in order: a (possibly empty) tuple when such an event is
	present, or ``None`` when stdin is a tty, empty, or not such an event — letting the caller
	tell "the batch changed nothing" (empty tuple) from "no event, use my default" (``None``).
	"""
	if sys.stdin.isatty():
		return None
	raw = sys.stdin.read().strip()
	if not raw:
		return None
	try:
		event: object = json.loads(raw)
	except json.JSONDecodeError:
		return None
	calls = _event_get(event, "tool_calls")
	if not isinstance(calls, list):
		return None
	edited = (
		_event_get(_event_get(call, "tool_input"), key)
		for call in cast("list[object]", calls)
		for key in ("file_path", "path", "notebook_path")
	)
	return tuple(dict.fromkeys(f for f in edited if isinstance(f, str)))


def changed_from_stdin() -> tuple[str, ...]:
	"""The edited files from a stdin ``PostToolBatch`` event, ``()`` when there is no such event
	— the gate's view, where an empty changed set falls back to the whole check node.
	"""
	return stdin_changed() or ()
