# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from typing import NamedTuple

import pytest

if sys.version_info >= (3, 11):
	from builtins import BaseExceptionGroup
else:  # pragma: no cover
	from exceptiongroup import BaseExceptionGroup

from camas import (
	CompletedEvent,
	Done,
	LeafState,
	Parallel,
	Sequential,
	StartedEvent,
	Task,
	TaskEvent,
	TaskNode,
	run,
)


class RecorderCtx(NamedTuple):
	setup_task: TaskNode
	events: tuple[TaskEvent, ...]


class Recorder:
	def __init__(self) -> None:
		self.final: tuple[RecorderCtx, ...] | None = None
		self.torn_down: bool = False

	async def setup(self, task: TaskNode) -> RecorderCtx:
		return RecorderCtx(setup_task=task, events=())

	async def on_event(
		self, event: TaskEvent, states: Sequence[LeafState], ctx: RecorderCtx
	) -> RecorderCtx:
		return ctx._replace(events=(*ctx.events, event))

	async def teardown(self, ctxs: tuple[RecorderCtx, ...]) -> None:
		self.final = ctxs
		self.torn_down = True


def test_silent_run_returns_results() -> None:
	result = asyncio.run(run(Task(("python", "-c", "pass"))))
	assert result.returncode == 0
	assert len(result.results) == 1


def test_effect_lifecycle_threads_context_through() -> None:
	recorder = Recorder()
	task = Parallel(
		tasks=(
			Task(("python", "-c", "pass"), name="a"),
			Task(("python", "-c", "pass"), name="b"),
		)
	)
	asyncio.run(run(task, effects=(recorder,)))

	assert recorder.torn_down is True
	assert recorder.final is not None
	assert all(ctx.setup_task is not None for ctx in recorder.final)
	all_events = [e for ctx in recorder.final for e in ctx.events]
	started = [e for e in all_events if isinstance(e, StartedEvent)]
	completed = [e for e in all_events if isinstance(e, CompletedEvent)]
	assert {e.leaf_index for e in started} == {0, 1}
	assert {e.leaf_index for e in completed} == {0, 1}


def test_each_leaf_sees_only_its_own_events() -> None:
	recorder = Recorder()
	task = Parallel(
		tasks=(
			Task(("python", "-c", "pass"), name="a"),
			Task(("python", "-c", "pass"), name="b"),
		)
	)
	asyncio.run(run(task, effects=(recorder,)))

	assert recorder.final is not None
	for leaf_idx, ctx in enumerate(recorder.final):
		assert all(e.leaf_index == leaf_idx for e in ctx.events)


def test_multi_effect_receives_identical_streams() -> None:
	a = Recorder()
	b = Recorder()

	asyncio.run(
		run(
			Sequential(
				tasks=(
					Task(("python", "-c", "pass"), name="one"),
					Task(("python", "-c", "pass"), name="two"),
				)
			),
			effects=(a, b),
		)
	)

	assert a.final is not None and b.final is not None
	assert a.final == b.final
	all_events = [e for ctx in a.final for e in ctx.events]
	assert any(isinstance(e, StartedEvent) for e in all_events)
	assert any(isinstance(e, CompletedEvent) for e in all_events)


def test_effect_receives_post_reduction_leaf_state() -> None:
	class WatcherCtx(NamedTuple):
		last_state: LeafState | None

	class Watcher:
		async def setup(self, task: TaskNode) -> WatcherCtx:
			return WatcherCtx(last_state=None)

		async def on_event(
			self, event: TaskEvent, states: Sequence[LeafState], ctx: WatcherCtx
		) -> WatcherCtx:
			return ctx._replace(last_state=states[event.leaf_index])

		async def teardown(self, ctxs: tuple[WatcherCtx, ...]) -> None:
			captured.append(ctxs)

	captured: list[tuple[WatcherCtx, ...]] = []
	asyncio.run(run(Task(("python", "-c", "pass"), name="solo"), effects=(Watcher(),)))

	final_ctxs = captured[0]
	assert len(final_ctxs) == 1
	assert isinstance(final_ctxs[0].last_state, Done)


def test_sequential_skip_emits_skipped_completed_event() -> None:
	recorder = Recorder()
	task = Sequential(
		tasks=(
			Task(("python", "-c", "raise SystemExit(1)"), name="fail"),
			Task(("python", "-c", "pass"), name="skipped"),
		)
	)
	asyncio.run(run(task, effects=(recorder,)))

	assert recorder.final is not None
	all_events = [e for ctx in recorder.final for e in ctx.events]
	skip_completions = [
		e for e in all_events if isinstance(e, CompletedEvent) and e.returncode == -1
	]
	assert len(skip_completions) == 1
	assert skip_completions[0].leaf_index == 1


def test_on_event_exception_surfaces_and_teardown_still_runs() -> None:
	class Boom(Exception):
		pass

	class FailCtx(NamedTuple):
		torn_down: bool

	class FailingOnEvent:
		def __init__(self) -> None:
			self.torn_down = False

		async def setup(self, task: TaskNode) -> FailCtx:
			return FailCtx(torn_down=False)

		async def on_event(
			self, event: TaskEvent, states: Sequence[LeafState], ctx: FailCtx
		) -> FailCtx:
			raise Boom("effect crashed")

		async def teardown(self, ctxs: tuple[FailCtx, ...]) -> None:
			self.torn_down = True

	failing = FailingOnEvent()
	with pytest.raises(Boom):
		asyncio.run(run(Task(("python", "-c", "pass")), effects=(failing,)))
	assert failing.torn_down is True


def test_setup_errors_collected_into_group_and_torn_down_effects_cleaned() -> None:
	class ACtx(NamedTuple):
		value: int = 0

	class A:
		def __init__(self) -> None:
			self.torn_down = False

		async def setup(self, task: TaskNode) -> ACtx:
			return ACtx()

		async def on_event(self, event: TaskEvent, states: Sequence[LeafState], ctx: ACtx) -> ACtx:
			return ctx  # pragma: no cover

		async def teardown(self, ctxs: tuple[ACtx, ...]) -> None:
			self.torn_down = True

	class FailingSetup:
		async def setup(self, task: TaskNode) -> ACtx:
			raise RuntimeError("setup failed")

		async def on_event(self, event: TaskEvent, states: Sequence[LeafState], ctx: ACtx) -> ACtx:
			return ctx  # pragma: no cover

		async def teardown(self, ctxs: tuple[ACtx, ...]) -> None:
			pass  # pragma: no cover

	a = A()
	with pytest.raises(BaseExceptionGroup) as info:
		asyncio.run(run(Task(("python", "-c", "pass")), effects=(a, FailingSetup())))
	assert any(isinstance(e, RuntimeError) for e in info.value.exceptions)
	assert a.torn_down is True


def test_teardown_errors_collected_into_group() -> None:
	class NoneCtx(NamedTuple):
		pass

	class A:
		async def setup(self, task: TaskNode) -> NoneCtx:
			return NoneCtx()

		async def on_event(
			self, event: TaskEvent, states: Sequence[LeafState], ctx: NoneCtx
		) -> NoneCtx:
			return ctx

		async def teardown(self, ctxs: tuple[NoneCtx, ...]) -> None:
			raise ValueError("a failed")

	class B:
		async def setup(self, task: TaskNode) -> NoneCtx:
			return NoneCtx()

		async def on_event(
			self, event: TaskEvent, states: Sequence[LeafState], ctx: NoneCtx
		) -> NoneCtx:
			return ctx

		async def teardown(self, ctxs: tuple[NoneCtx, ...]) -> None:
			raise RuntimeError("b failed")

	with pytest.raises(BaseExceptionGroup) as info:
		asyncio.run(run(Task(("python", "-c", "pass")), effects=(A(), B())))

	messages = {str(e) for e in info.value.exceptions}
	assert messages == {"a failed", "b failed"}


def test_parallel_on_event_runs_concurrently() -> None:
	"""Per-leaf slots let async-IO on_event callbacks for concurrent leaves overlap."""

	class SlowCtx(NamedTuple):
		pass

	class Slow:
		async def setup(self, task: TaskNode) -> SlowCtx:
			return SlowCtx()

		async def on_event(
			self, event: TaskEvent, states: Sequence[LeafState], ctx: SlowCtx
		) -> SlowCtx:
			if isinstance(event, CompletedEvent):
				await asyncio.sleep(1.0)
			return ctx

		async def teardown(self, ctxs: tuple[SlowCtx, ...]) -> None:
			pass

	task = Parallel(
		tasks=(
			Task(("python", "-c", "pass"), name="a"),
			Task(("python", "-c", "pass"), name="b"),
		)
	)
	result = asyncio.run(run(task, effects=(Slow(),)))
	assert result.returncode == 0
	# Concurrent: ~1.0s + spawn overhead. Sequential would add another 1.0s.
	assert result.elapsed < 2.5
