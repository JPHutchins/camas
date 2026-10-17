# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

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
	changed: tuple[str, ...] | None
	session_id: str | None
	prompt_id: str | None
	stop_hook_active: bool


NO_EVENT: Final = HookEvent(None, None, None, False)


def _event_get(obj: object, key: str) -> object:
	return cast("dict[str, object]", obj).get(key) if isinstance(obj, dict) else None


def _event_str(obj: object, key: str) -> str | None:
	value = _event_get(obj, key)
	return value if isinstance(value, str) else None


def _event_changed(event: object) -> tuple[str, ...] | None:
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

STALE_TEMP_MAX_AGE_S: Final = 3600.0

SETTLE_START_S: Final = 15.0

SETTLE_WAIT_S: Final = 600.0

SETTLING: Final = " settling"


def settled_marker(session_id: str) -> Path:
	digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
	return Path(tempfile.gettempdir()) / f"{SETTLED_MARKER_PREFIX}{digest}"


def _marks(event: HookEvent) -> bool:
	return event.changed is None and bool(event.session_id) and bool(event.prompt_id)


def record_settling(event: HookEvent) -> None:
	if _marks(event):
		with suppress(OSError):
			settled_marker(str(event.session_id)).write_text(
				f"{event.prompt_id}{SETTLING}", encoding="utf-8"
			)


def record_settled(event: HookEvent) -> None:
	if _marks(event):
		cutoff: Final = time.time() - STALE_TEMP_MAX_AGE_S
		for marker in Path(tempfile.gettempdir()).glob(f"{SETTLED_MARKER_PREFIX}*"):
			with suppress(OSError):
				if marker.stat().st_mtime < cutoff:
					marker.unlink()
		with suppress(OSError):
			settled_marker(str(event.session_id)).write_text(str(event.prompt_id), encoding="utf-8")


def _settle_state(marker: Path, prompt_id: str) -> Literal["settled", "settling", "absent"]:
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
