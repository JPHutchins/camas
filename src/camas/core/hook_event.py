# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Parse a Claude Code hook event (``PostToolBatch``/``Stop``) piped on stdin.

The autofix (``camas mcp fix``) and the gate (``camas mcp gate``) both read the just-edited
files from the same event, and the gate's async Stop-hook nudge reads the loop-guard fields
(``session_id``/``prompt_id``/``stop_hook_active``), so the parsing lives here —
dependency-free (``json``/``sys`` only) so ``camas mcp fix`` works without the ``mcp`` extra.
"""

from __future__ import annotations

import json
import sys
from typing import Final, NamedTuple, cast


class HookEvent(NamedTuple):
	"""The hook-relevant fields of one stdin event: the edited files (``None`` when the event
	carries no tool batch, e.g. a ``Stop`` event), and the Stop-hook nudge-guard fields.
	"""

	changed: tuple[str, ...] | None
	session_id: str | None
	prompt_id: str | None
	stop_hook_active: bool


NO_EVENT: Final = HookEvent(None, None, None, False)
"""The parse result when stdin carries no event (tty, empty, or not JSON)."""


def _event_get(obj: object, key: str) -> object:
	"""``obj[key]`` for a parsed-JSON object, else ``None`` — narrows JSON's ``Any`` to ``object``."""
	return cast("dict[str, object]", obj).get(key) if isinstance(obj, dict) else None


def _event_str(obj: object, key: str) -> str | None:
	"""``obj[key]`` when it is a string, else ``None``."""
	value = _event_get(obj, key)
	return value if isinstance(value, str) else None


def _event_changed(event: object) -> tuple[str, ...] | None:
	"""The event's edited files, de-duplicated in order; ``None`` when it has no tool batch."""
	calls = _event_get(event, "tool_calls")
	if not isinstance(calls, list):
		return None
	edited = (
		_event_get(_event_get(call, "tool_input"), key)
		for call in cast("list[object]", calls)
		for key in ("file_path", "path", "notebook_path")
	)
	return tuple(dict.fromkeys(f for f in edited if isinstance(f, str)))


def event_from_stdin() -> HookEvent:
	"""The hook event piped on stdin — one read, shared by the changed-paths extraction and the
	Stop-hook nudge guard; :data:`NO_EVENT` when stdin is a tty, empty, or not JSON.
	"""
	if sys.stdin.isatty():
		return NO_EVENT
	raw = sys.stdin.read().strip()
	if not raw:
		return NO_EVENT
	try:
		event: object = json.loads(raw)
	except json.JSONDecodeError:
		return NO_EVENT
	return HookEvent(
		_event_changed(event),
		_event_str(event, "session_id"),
		_event_str(event, "prompt_id"),
		_event_get(event, "stop_hook_active") is True,
	)


def stdin_changed() -> tuple[str, ...] | None:
	"""The edited files in a ``PostToolBatch`` event piped on stdin (the Claude Code plugin's
	autofix/gate hook), de-duplicated in order: a (possibly empty) tuple when such an event is
	present, or ``None`` when stdin is a tty, empty, or not such an event — letting the caller
	tell "the batch changed nothing" (empty tuple) from "no event, use my default" (``None``).
	"""
	return event_from_stdin().changed


def changed_from_stdin() -> tuple[str, ...]:
	"""The edited files from a stdin ``PostToolBatch`` event, ``()`` when there is no such event
	— the gate's view, where an empty changed set falls back to the whole check node.
	"""
	return stdin_changed() or ()
