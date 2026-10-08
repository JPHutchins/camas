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
from typing import Final, Literal, NamedTuple, cast


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


SETTLED_MARKER_PREFIX: Final = "camas-settled-"
"""Prefix on the per-session markers in the machine temp dir naming the last prompt whose ``Stop``
autofix has run."""

STALE_TEMP_MAX_AGE_S: Final = 3600.0
"""Age past which a prior run's leftovers in the system temp dir are swept — the settled and nudge
markers and the gate's report directories age out together."""

SETTLE_START_S: Final = 15.0
"""How long the async Stop-hook nudge waits for its sibling autofix to mark the prompt settling
before checking anyway — the fix hook never started."""

SETTLE_WAIT_S: Final = 600.0
"""How long the nudge waits on an autofix that marked the prompt settling before checking anyway."""

SETTLING: Final = " settling"
"""The suffix on a settled marker's prompt_id while that prompt's autofix is still running."""


def settled_marker(session_id: str) -> Path:
	"""The session's settled marker: the last prompt_id the Stop autofix ran for, suffixed with
	:data:`SETTLING` while it runs.
	"""
	digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
	return Path(tempfile.gettempdir()) / f"{SETTLED_MARKER_PREFIX}{digest}"


def _marks(event: HookEvent) -> bool:
	return event.changed is None and bool(event.session_id) and bool(event.prompt_id)


def record_settling(event: HookEvent) -> None:
	"""Mark ``event``'s prompt settling — its autofix has started — for :func:`await_settled`; a
	no-op for an event that carries a tool batch or lacks session and prompt ids.
	"""
	if _marks(event):
		with suppress(OSError):
			settled_marker(str(event.session_id)).write_text(
				f"{event.prompt_id}{SETTLING}", encoding="utf-8"
			)


def record_settled(event: HookEvent) -> None:
	"""Mark ``event``'s prompt settled for :func:`await_settled`, sweeping prior sessions' markers
	older than :data:`STALE_TEMP_MAX_AGE_S` first — a no-op for an event that carries a tool batch
	or lacks session and prompt ids.
	"""
	if _marks(event):
		cutoff: Final = time.time() - STALE_TEMP_MAX_AGE_S
		for marker in Path(tempfile.gettempdir()).glob(f"{SETTLED_MARKER_PREFIX}*"):
			with suppress(OSError):
				if marker.stat().st_mtime < cutoff:
					marker.unlink()
		with suppress(OSError):
			settled_marker(str(event.session_id)).write_text(str(event.prompt_id), encoding="utf-8")


def _settle_state(marker: Path, prompt_id: str) -> Literal["settled", "settling", "absent"]:
	"""The marker's state for ``prompt_id`` — an empty marker is one caught mid-write, so its writer
	is still running.
	"""
	try:
		content: Final = marker.read_text(encoding="utf-8")
	except (OSError, ValueError):
		return "absent"
	if content == prompt_id:
		return "settled"
	return "settling" if content in ("", f"{prompt_id}{SETTLING}") else "absent"


def await_settled(
	event: HookEvent,
	*,
	start: float = SETTLE_START_S,
	timeout: float = SETTLE_WAIT_S,
	poll: float = 0.1,
) -> bool:
	"""Wait until ``event``'s prompt is marked settled, answering whether it was: up to ``start``
	for its autofix to mark it settling, then up to ``timeout`` while it does; an event without
	session and prompt ids answers ``True`` at once.
	"""
	if not event.session_id or not event.prompt_id:
		return True
	marker: Final = settled_marker(event.session_id)
	begun: Final = time.monotonic()
	while True:
		state = _settle_state(marker, event.prompt_id)
		if state == "settled":
			return True
		if time.monotonic() - begun >= (timeout if state == "settling" else start):
			return False
		time.sleep(poll)
