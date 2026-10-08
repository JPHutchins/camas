# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from camas.core.hook_event import HookEvent, await_settled, record_settled, settled_marker

if TYPE_CHECKING:
	from pathlib import Path

STOP = HookEvent(changed=None, session_id="s-1", prompt_id="p-1", stop_hook_active=False)


@pytest.fixture(autouse=True)
def markers_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	monkeypatch.setattr("camas.core.hook_event.tempfile.gettempdir", lambda: str(tmp_path))
	return tmp_path


def test_record_settled_marks_a_stop_events_prompt() -> None:
	record_settled(STOP)
	assert settled_marker("s-1").read_text(encoding="utf-8") == "p-1"


@pytest.mark.parametrize(
	"event",
	[
		STOP._replace(changed=("a.py",)),
		STOP._replace(session_id=None),
		STOP._replace(prompt_id=""),
	],
)
def test_record_settled_ignores_a_tool_batch_and_an_event_without_ids(
	markers_in: Path, event: HookEvent
) -> None:
	record_settled(event)
	assert list(markers_in.iterdir()) == []


def test_await_settled_returns_once_its_prompt_is_marked() -> None:
	record_settled(STOP)
	assert await_settled(STOP, timeout=0.0)


def test_await_settled_gives_up_on_another_prompts_marker_at_the_deadline() -> None:
	record_settled(STOP._replace(prompt_id="p-0"))
	assert not await_settled(STOP, timeout=0.05, poll=0.01)


def test_await_settled_gives_up_without_a_marker_at_the_deadline() -> None:
	assert not await_settled(STOP, timeout=0.05, poll=0.01)


def test_await_settled_does_not_wait_on_an_event_without_ids() -> None:
	assert await_settled(STOP._replace(session_id=None), timeout=60.0)
