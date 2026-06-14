# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from datetime import datetime

from camas import Task
from camas.core.leaf_state import KILL_PRESSES, next_state
from camas.v0.completion import INTERRUPT_RC, Finished, Skipped, Stopped
from camas.v0.leaf_state import Completed, Interrupting, Running, Waiting
from camas.v0.task_event import (
	AbortedEvent,
	CompletedEvent,
	InterruptedEvent,
	OutputEvent,
	StartedEvent,
)

T = Task("echo hi")
T0 = datetime(2026, 1, 1, 12, 0, 0)
T1 = datetime(2026, 1, 1, 12, 0, 1)


def test_waiting_started_becomes_running() -> None:
	assert next_state(Waiting(T), StartedEvent(T, 0, T0)) == Running(T, T0, b"")


def test_waiting_completed_becomes_completed() -> None:
	assert next_state(Waiting(T), CompletedEvent(T, 0, Skipped(1), T0)) == Completed(T, Skipped(1))


def test_waiting_interrupted_resolves_stopped() -> None:
	assert next_state(Waiting(T), InterruptedEvent(T, 0, T0)) == Completed(
		T, Stopped(INTERRUPT_RC, 0.0, ())
	)


def test_waiting_aborted_resolves_stopped() -> None:
	assert next_state(Waiting(T), AbortedEvent(T, 0, T0)) == Completed(
		T, Stopped(INTERRUPT_RC, 0.0, ())
	)


def test_running_interrupted_becomes_interrupting_at_one_press() -> None:
	assert next_state(Running(T, T0, b"x"), InterruptedEvent(T, 0, T1)) == Interrupting(
		T, T0, b"x", 1
	)


def test_running_output_stays_running() -> None:
	assert next_state(Running(T, T0, b""), OutputEvent(T, 0, b"hi", T1)) == Running(T, T0, b"hi")


def test_running_completed_becomes_completed() -> None:
	done = Finished(0, 0.5, (b"done",))
	assert next_state(Running(T, T0, b""), CompletedEvent(T, 0, done, T1)) == Completed(T, done)


def test_running_aborted_becomes_interrupting_at_kill_level() -> None:
	assert next_state(Running(T, T0, b"x"), AbortedEvent(T, 0, T1)) == Interrupting(
		T, T0, b"x", KILL_PRESSES
	)


def test_interrupting_interrupted_increments_press_count() -> None:
	assert next_state(Interrupting(T, T0, b"x", 1), InterruptedEvent(T, 0, T1)) == Interrupting(
		T, T0, b"x", 2
	)


def test_interrupting_output_updates_last_line_keeping_press_count() -> None:
	assert next_state(Interrupting(T, T0, b"", 2), OutputEvent(T, 0, b"hi", T1)) == Interrupting(
		T, T0, b"hi", 2
	)


def test_interrupting_completed_keeps_reported_completion() -> None:
	stopped = Stopped(130, 0.5, ())
	assert next_state(Interrupting(T, T0, b"", 1), CompletedEvent(T, 0, stopped, T1)) == Completed(
		T, stopped
	)


def test_interrupting_aborted_jumps_to_kill_level() -> None:
	assert next_state(Interrupting(T, T0, b"x", 1), AbortedEvent(T, 0, T1)) == Interrupting(
		T, T0, b"x", KILL_PRESSES
	)
