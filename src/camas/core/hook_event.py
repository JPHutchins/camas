# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Parse a Claude Code hook event (``PostToolBatch``/``Stop``) piped on stdin, and mark a ``Stop``
event's prompt settled once its autofix has run.

The autofix (``camas mcp fix``) and the gate (``camas mcp gate``) both read the just-edited
files from the same event, and the gate's async Stop-hook nudge reads the loop-guard fields
(``session_id``/``prompt_id``/``stop_hook_active``) and waits on the settled marker, so both live
here — standard-library only, so ``camas mcp fix`` works without the ``mcp`` extra.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
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


SETTLED_MARKER_PREFIX: Final = "camas-settled-"
"""Prefix on the per-session markers in the machine temp dir naming the last prompt whose ``Stop``
autofix has run."""

SETTLE_WAIT_S: Final = 60.0
"""How long the async Stop-hook nudge waits for its sibling autofix before checking anyway."""


def settled_marker(session_id: str) -> Path:
	"""The session's settled marker; its content is the last prompt_id the Stop autofix ran for."""
	digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
	return Path(tempfile.gettempdir()) / f"{SETTLED_MARKER_PREFIX}{digest}"


def record_settled(event: HookEvent) -> None:
	"""Mark ``event``'s prompt settled for :func:`await_settled` — a no-op for an event that carries
	a tool batch or lacks session and prompt ids.
	"""
	if event.changed is None and event.session_id and event.prompt_id:
		with suppress(OSError):
			settled_marker(event.session_id).write_text(event.prompt_id, encoding="utf-8")


def await_settled(event: HookEvent, *, timeout: float = SETTLE_WAIT_S, poll: float = 0.1) -> bool:
	"""Wait until ``event``'s prompt is marked settled, answering whether it was within ``timeout``;
	an event without session and prompt ids answers ``True`` at once.
	"""
	if not event.session_id or not event.prompt_id:
		return True
	marker: Final = settled_marker(event.session_id)
	deadline: Final = time.monotonic() + timeout
	while True:
		with suppress(OSError):
			if marker.read_text(encoding="utf-8") == event.prompt_id:
				return True
		if time.monotonic() >= deadline:
			return False
		time.sleep(poll)
