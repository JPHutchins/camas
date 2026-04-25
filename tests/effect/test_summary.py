# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

import pytest

from camas import (
	CompletedEvent,
	Effect,
	Finished,
	LeafState,
	Parallel,
	Sequential,
	Skipped,
	StartedEvent,
	Task,
	TaskEvent,
	Waiting,
)
from camas.effect.summary import Fixed, Summary, SummaryOptions

T = TypeVar("T")


def make_task(name: str) -> Task:
	return Task(("python", "-c", "pass"), name=name)


async def drive(
	effect: Effect[T],
	task: Task | Sequential | Parallel,
	events: list[TaskEvent],
) -> None:
	"""Feed an effect through a full setup/on_event*/teardown lifecycle."""
	from camas import flatten_leaves, next_state

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


def test_summary_renders_only_at_teardown(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("alpha")
	b = make_task("beta")
	task = Parallel(a, b)
	events: list[TaskEvent] = [
		StartedEvent(0, 100.0),
		StartedEvent(1, 100.0),
		CompletedEvent(0, Finished(0, 0.1, (b"clean\n",))),
		CompletedEvent(1, Finished(0, 0.2, (b"ok\n",))),
	]

	async def run_and_capture() -> tuple[str, str]:
		effect = Summary(SummaryOptions())
		from camas import flatten_leaves, next_state

		leaves = flatten_leaves(task)
		states: list[LeafState] = [Waiting(info.task) for info in leaves]
		ctx = await effect.setup(task)
		ctxs = [ctx for _ in leaves]
		for event in events:
			states[event.leaf_index] = next_state(states[event.leaf_index], event)
			ctxs[event.leaf_index] = await effect.on_event(event, states, ctxs[event.leaf_index])
		before = capsys.readouterr().out
		await effect.teardown(tuple(ctxs))
		after = capsys.readouterr().out
		return before, after

	before, after = asyncio.run(run_and_capture())
	assert before == ""
	assert "alpha" in after
	assert "beta" in after
	assert "PASS" in after


def test_summary_failure_prints_details(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("boom")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(0, 100.0),
		CompletedEvent(0, Finished(1, 0.1, (b"error details\n",))),
	]
	asyncio.run(drive(Summary(SummaryOptions()), task, events))
	out = capsys.readouterr().out
	assert "FAIL" in out
	assert "FAILED: boom" in out
	assert "error details" in out


def test_summary_sequential_skipped(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("first")
	b = make_task("second")
	task = Sequential(a, b, name="pipeline")
	events: list[TaskEvent] = [
		StartedEvent(0, 100.0),
		CompletedEvent(0, Finished(1, 0.1, (b"failed\n",))),
		CompletedEvent(1, Skipped(1)),
	]
	asyncio.run(drive(Summary(SummaryOptions()), task, events))
	out = capsys.readouterr().out
	assert "pipeline" in out
	assert "SKIP" in out
	assert "FAIL" in out


def test_summary_fixed_width_overrides_terminal_detection() -> None:
	async def capture_width() -> int:
		effect = Summary(SummaryOptions(term_width=Fixed(160)))
		ctx = await effect.setup(make_task("solo"))
		return ctx.term_width

	assert asyncio.run(capture_width()) == 160


def test_summary_show_passing_prints_passed_output(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("alpha")
	b = make_task("beta")
	task = Parallel(a, b)
	events: list[TaskEvent] = [
		StartedEvent(0, 100.0),
		StartedEvent(1, 100.0),
		CompletedEvent(0, Finished(0, 0.1, (b"alpha output\n",))),
		CompletedEvent(1, Finished(1, 0.2, (b"beta error\n",))),
	]
	asyncio.run(drive(Summary(SummaryOptions(show_passing=True)), task, events))
	out = capsys.readouterr().out
	assert "FAILED: beta" in out
	assert "beta error" in out
	assert "PASSED: alpha" in out
	assert "alpha output" in out


def test_summary_show_passing_defaults_to_false(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("alpha")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(0, 100.0),
		CompletedEvent(0, Finished(0, 0.1, (b"alpha output\n",))),
	]
	asyncio.run(drive(Summary(SummaryOptions()), task, events))
	out = capsys.readouterr().out
	assert "PASSED:" not in out


def test_summary_creates_no_background_tasks() -> None:
	task = make_task("solo")
	events: list[TaskEvent] = [
		StartedEvent(0, 100.0),
		CompletedEvent(0, Finished(0, 0.05, (b"done\n",))),
	]

	async def count_tasks_after_teardown() -> int:
		before: Callable[[], set[asyncio.Task[object]]] = asyncio.all_tasks
		baseline = before()
		await drive(Summary(SummaryOptions()), task, events)
		return len(asyncio.all_tasks() - baseline)

	assert asyncio.run(count_tasks_after_teardown()) == 0
