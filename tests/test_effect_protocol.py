from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from camas import (
	CompletedEvent,
	Parallel,
	Sequential,
	StartedEvent,
	Task,
	TaskEvent,
	TaskNode,
	run,
)


async def record(events: AsyncIterator[TaskEvent], sink: list[TaskEvent]) -> None:
	async for event in events:
		sink.append(event)


def test_silent_run_returns_results() -> None:
	result = asyncio.run(run(Task(("python", "-c", "pass"))))
	assert result.returncode == 0
	assert len(result.results) == 1


def test_effect_receives_events_in_order() -> None:
	collected: list[TaskEvent] = []

	async def collector(task: TaskNode, events: AsyncIterator[TaskEvent]) -> None:
		await record(events, collected)

	task = Parallel(
		tasks=(
			Task(("python", "-c", "pass"), name="a"),
			Task(("python", "-c", "pass"), name="b"),
		)
	)
	asyncio.run(run(task, effects=(collector,)))

	started = [e for e in collected if isinstance(e, StartedEvent)]
	completed = [e for e in collected if isinstance(e, CompletedEvent)]
	assert {e.leaf_index for e in started} == {0, 1}
	assert {e.leaf_index for e in completed} == {0, 1}
	for s, c in zip(
		sorted(started, key=lambda e: e.leaf_index),
		sorted(completed, key=lambda e: e.leaf_index),
	):
		assert s.timestamp <= s.timestamp + c.elapsed


def test_multi_effect_receives_identical_streams() -> None:
	stream_a: list[TaskEvent] = []
	stream_b: list[TaskEvent] = []

	async def eff_a(task: TaskNode, events: AsyncIterator[TaskEvent]) -> None:
		await record(events, stream_a)

	async def eff_b(task: TaskNode, events: AsyncIterator[TaskEvent]) -> None:
		await record(events, stream_b)

	asyncio.run(
		run(
			Sequential(
				tasks=(
					Task(("python", "-c", "pass"), name="one"),
					Task(("python", "-c", "pass"), name="two"),
				)
			),
			effects=(eff_a, eff_b),
		)
	)

	assert stream_a == stream_b
	assert any(isinstance(e, StartedEvent) for e in stream_a)
	assert any(isinstance(e, CompletedEvent) for e in stream_a)


def test_sequential_skip_emits_skipped_completed_event() -> None:
	collected: list[TaskEvent] = []

	async def collector(task: TaskNode, events: AsyncIterator[TaskEvent]) -> None:
		await record(events, collected)

	task = Sequential(
		tasks=(
			Task(("python", "-c", "raise SystemExit(1)"), name="fail"),
			Task(("python", "-c", "pass"), name="skipped"),
		)
	)
	asyncio.run(run(task, effects=(collector,)))

	skip_completions = [
		e for e in collected if isinstance(e, CompletedEvent) and e.returncode == -1
	]
	assert len(skip_completions) == 1
	assert skip_completions[0].leaf_index == 1


def test_effect_exception_surfaces() -> None:
	class Boom(Exception):
		pass

	async def failing(task: TaskNode, events: AsyncIterator[TaskEvent]) -> None:
		await anext(events)
		raise Boom("effect crashed")

	with pytest.raises(ExceptionGroup) as info:
		asyncio.run(run(Task(("python", "-c", "pass")), effects=(failing,)))

	assert any(isinstance(e, Boom) for e in info.value.exceptions)
