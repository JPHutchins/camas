# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar

from camas import Parallel, Sequential, Task
from camas.core import timings
from camas.effect.timings import Timings
from camas.v0.completion import Finished, Skipped
from camas.v0.leaf_state import LeafState, Waiting
from camas.v0.task_event import CompletedEvent, StartedEvent, TaskEvent

if TYPE_CHECKING:
	from pathlib import Path

	from camas.v0.effect import Effect

TS = datetime(2026, 5, 21, 14, 30, 0)
T = TypeVar("T")


def _task(name: str) -> Task:
	return Task(("python", "-c", "pass"), name=name)


async def drive(
	effect: Effect[T], task: Task | Sequential | Parallel, events: list[TaskEvent]
) -> None:
	from camas.core.leaf_state import next_state
	from camas.core.traversal import flatten_leaves

	leaves = flatten_leaves(task)
	states: list[LeafState] = [Waiting(info.task) for info in leaves]
	initial = await effect.setup(task)
	ctxs: list[T] = [initial for _ in leaves]
	try:
		for event in events:
			states[event.leaf_index] = next_state(states[event.leaf_index], event)
			ctxs[event.leaf_index] = await effect.on_event(event, states, ctxs[event.leaf_index])
	finally:
		await effect.teardown(tuple(ctxs))


def test_records_each_leaf(tmp_path: Path) -> None:
	a, b = _task("fast"), _task("slow")
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		StartedEvent(b, 1, TS),
		CompletedEvent(a, 0, Finished(0, 0.1, ()), TS),
		CompletedEvent(b, 1, Finished(0, 0.5, ()), TS),
	]
	asyncio.run(drive(Timings(camas_dir=tmp_path), Parallel(a, b, name="quick"), events))
	cache = timings.load(tmp_path)
	assert cache["fast"].elapsed_s == 0.1
	assert cache["slow"].elapsed_s == 0.5


def test_anonymous_run_records_its_leaves(tmp_path: Path) -> None:
	a = _task("solo")
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		CompletedEvent(a, 0, Finished(0, 0.1, ()), TS),
	]
	asyncio.run(drive(Timings(camas_dir=tmp_path), Parallel(a), events))
	assert timings.load(tmp_path)["solo"].elapsed_s == 0.1


def test_anonymous_leaves_named_by_command(tmp_path: Path) -> None:
	s, t = Task("echo hi"), Task(("python", "-c", "pass"))
	events: list[TaskEvent] = [
		StartedEvent(s, 0, TS),
		StartedEvent(t, 1, TS),
		CompletedEvent(s, 0, Finished(0, 0.5, ()), TS),
		CompletedEvent(t, 1, Finished(0, 0.1, ()), TS),
	]
	asyncio.run(drive(Timings(camas_dir=tmp_path), Parallel(s, t, name="grp"), events))
	cache = timings.load(tmp_path)
	assert cache["echo hi"].elapsed_s == 0.5
	assert cache["python -c pass"].elapsed_s == 0.1


def test_unfinished_leaf_excluded(tmp_path: Path) -> None:
	a, b = _task("done"), _task("never")
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		StartedEvent(b, 1, TS),
		CompletedEvent(a, 0, Finished(0, 0.2, ()), TS),
	]
	asyncio.run(drive(Timings(camas_dir=tmp_path), Parallel(a, b, name="grp"), events))
	cache = timings.load(tmp_path)
	assert "done" in cache
	assert "never" not in cache


def test_skipped_leaf_excluded(tmp_path: Path) -> None:
	a, b = _task("fail"), _task("skip")
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		CompletedEvent(a, 0, Finished(2, 0.3, ()), TS),
		CompletedEvent(b, 1, Skipped(2), TS),
	]
	asyncio.run(drive(Timings(camas_dir=tmp_path), Sequential(a, b, name="seq"), events))
	cache = timings.load(tmp_path)
	assert "fail" in cache
	assert "skip" not in cache
