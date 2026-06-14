# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from datetime import datetime

from camas import Task
from camas.core.leaf_state import KILL_PRESSES, next_state, to_interrupting
from camas.v0.completion import Finished, Skipped, Stopped
from camas.v0.leaf_state import Completed, Interrupting, Running, Waiting
from camas.v0.task_event import CompletedEvent, OutputEvent, StartedEvent

T = Task("echo hi")
T0 = datetime(2026, 1, 1, 12, 0, 0)
T1 = datetime(2026, 1, 1, 12, 0, 1)


def test_waiting_started_becomes_running() -> None:
	assert next_state(Waiting(T), StartedEvent(T, 0, T0)) == Running(T, T0, b"")


def test_waiting_completed_becomes_completed() -> None:
	assert next_state(Waiting(T), CompletedEvent(T, 0, Skipped(1), T0)) == Completed(T, Skipped(1))


def test_running_output_stays_running() -> None:
	assert next_state(Running(T, T0, b""), OutputEvent(T, 0, b"hi", T1)) == Running(T, T0, b"hi")


def test_running_completed_becomes_completed() -> None:
	done = Finished(0, 0.5, (b"done",))
	assert next_state(Running(T, T0, b""), CompletedEvent(T, 0, done, T1)) == Completed(T, done)


def test_interrupting_output_updates_last_line_keeping_press_count() -> None:
	assert next_state(Interrupting(T, T0, b"", 2), OutputEvent(T, 0, b"hi", T1)) == Interrupting(
		T, T0, b"hi", 2
	)


def test_interrupting_completed_keeps_reported_completion() -> None:
	stopped = Stopped(130, 0.5, ())
	assert next_state(Interrupting(T, T0, b"", 1), CompletedEvent(T, 0, stopped, T1)) == Completed(
		T, stopped
	)


def test_to_interrupting_running_records_press_count() -> None:
	assert to_interrupting(Running(T, T0, b"x"), 1) == Interrupting(T, T0, b"x", 1)


def test_to_interrupting_interrupting_updates_press_count() -> None:
	assert to_interrupting(Interrupting(T, T0, b"x", 1), KILL_PRESSES) == Interrupting(
		T, T0, b"x", KILL_PRESSES
	)


def test_to_interrupting_waiting_passes_through() -> None:
	assert to_interrupting(Waiting(T), 1) == Waiting(T)


def test_to_interrupting_completed_passes_through() -> None:
	done = Completed(T, Finished(0, 0.1, ()))
	assert to_interrupting(done, 2) == done
