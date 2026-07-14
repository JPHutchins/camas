# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar

import pytest

from camas import Parallel, Sequential, Task
from camas.core.leaf_state import next_state
from camas.core.traversal import flatten_leaves
from camas.effect.status import (
	Active,
	Done,
	Idle,
	Status,
)
from camas.v0.completion import Errored, Finished, Skipped, Stopped
from camas.v0.leaf_state import LeafState, Waiting
from camas.v0.task_event import CompletedEvent, OutputEvent, StartedEvent, TaskEvent

if TYPE_CHECKING:
	from camas.v0.effect import Effect

TS = datetime(2026, 5, 21, 14, 30, 0, 750_000)

T = TypeVar("T")


def make_task(name: str) -> Task:
	return Task(("python", "-c", "pass"), name=name)


async def drive(
	effect: Effect[T],
	task: Task | Sequential | Parallel,
	events: list[TaskEvent],
) -> tuple[T, ...]:
	"""Feed an effect through its full setup/on_event*/teardown lifecycle."""
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
	return tuple(ctxs)


def test_default_options_pass_failure_through(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("alpha")
	b = make_task("boom")
	task = Parallel(a, b)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		StartedEvent(b, 1, TS),
		OutputEvent(a, 0, b"a-out\n", TS),
		OutputEvent(b, 1, b"\x1b[31mb-error\x1b[0m\n", TS),
		CompletedEvent(a, 0, Finished(0, 0.1, ()), TS),
		CompletedEvent(b, 1, Finished(2, 0.2, ()), TS),
	]
	asyncio.run(drive(Status(), task, events))
	out = capsys.readouterr().out
	assert "▶ [alpha] started" in out
	assert "▶ [boom] started" in out
	assert "✓ [alpha] success" in out
	assert "(0.100s)" in out
	assert "✗ [boom] error" in out
	assert "exit=2 (0.200s)" in out
	assert "\x1b[31mb-error\x1b[0m" in out
	assert "a-out" not in out


def test_stopped_completion_renders_stopped_line(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("lint")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		CompletedEvent(a, 0, Stopped(130, 0.5, ()), TS),
	]
	asyncio.run(drive(Status(), task, events))
	out = capsys.readouterr().out
	assert "■ [lint] stopped" in out
	assert "exit=130 (0.500s)" in out


def test_errored_completion_renders_errored_line(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("ghost")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		CompletedEvent(a, 0, Errored(127, "no such file or directory: ghost"), TS),
	]
	asyncio.run(drive(Status(), task, events))
	out = capsys.readouterr().out
	assert "⚠ [ghost] errored" in out
	assert "no such file or directory: ghost" in out


def test_empty_errored_template_suppresses_line(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("ghost")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		CompletedEvent(a, 0, Errored(127, "no such file or directory: ghost"), TS),
	]
	asyncio.run(drive(Status(errored_fmt=""), task, events))
	assert "errored" not in capsys.readouterr().out


def test_empty_stopped_template_suppresses_line(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("lint")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		CompletedEvent(a, 0, Stopped(130, 0.5, ()), TS),
	]
	asyncio.run(drive(Status(stopped_fmt=""), task, events))
	assert "stopped" not in capsys.readouterr().out


def test_all_mode_dumps_every_block(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("alpha")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		OutputEvent(a, 0, b"hello\n", TS),
		CompletedEvent(a, 0, Finished(0, 0.1, ()), TS),
	]
	asyncio.run(drive(Status(output_mode="all"), task, events))
	out = capsys.readouterr().out
	assert "✓ [alpha] success" in out
	assert "hello" in out


def test_quiet_mode_drops_output(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("alpha")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		OutputEvent(a, 0, b"shhh\n", TS),
		CompletedEvent(a, 0, Finished(0, 0.1, ()), TS),
	]
	ctxs = asyncio.run(drive(Status(output_mode="quiet"), task, events))
	out = capsys.readouterr().out
	assert "▶ [alpha] started" in out
	assert "✓ [alpha] success" in out
	assert "shhh" not in out
	assert ctxs == (Done(),)


def test_stream_mode_emits_lines_live_and_strips_ansi(
	capsys: pytest.CaptureFixture[str],
) -> None:
	a = make_task("lint")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		OutputEvent(a, 0, b"\x1b[31mred line\x1b[0m\n", TS),
		OutputEvent(a, 0, b"plain\n", TS),
		CompletedEvent(a, 0, Finished(0, 0.1, ()), TS),
	]
	asyncio.run(drive(Status(output_mode="stream"), task, events))
	out = capsys.readouterr().out
	assert "[lint]\x1b[0m red line" in out
	assert "[lint]\x1b[0m plain" in out
	assert out.count("· [lint]\x1b[0m ") == 2


def test_stream_mode_interleaves_across_parallel_tasks(
	capsys: pytest.CaptureFixture[str],
) -> None:
	"""Stream mode emits in arrival order, not grouped by task."""
	a = make_task("fast")
	b = make_task("slow")
	task = Parallel(a, b)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		StartedEvent(b, 1, TS),
		OutputEvent(a, 0, b"a1\n", TS),
		OutputEvent(b, 1, b"b1\n", TS),
		OutputEvent(a, 0, b"a2\n", TS),
		OutputEvent(b, 1, b"b2\n", TS),
		CompletedEvent(a, 0, Finished(0, 0.1, ()), TS),
		CompletedEvent(b, 1, Finished(0, 0.2, ()), TS),
	]
	asyncio.run(drive(Status(output_mode="stream"), task, events))
	out = capsys.readouterr().out
	a1 = out.index("[fast]\x1b[0m a1")
	b1 = out.index("[slow]\x1b[0m b1")
	a2 = out.index("[fast]\x1b[0m a2")
	b2 = out.index("[slow]\x1b[0m b2")
	assert a1 < b1 < a2 < b2


def test_github_mode_wraps_blocks_with_workflow_commands(
	capsys: pytest.CaptureFixture[str],
) -> None:
	a = make_task("test")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		OutputEvent(a, 0, b"PASSED 7 tests\n", TS),
		CompletedEvent(a, 0, Finished(0, 0.4, ()), TS),
	]
	asyncio.run(drive(Status(output_mode="github"), task, events))
	out = capsys.readouterr().out
	assert "✓ [test] success" in out
	assert "::group::test\n" in out
	assert "PASSED 7 tests" in out
	assert "::endgroup::\n" in out


def test_skipped_emits_skip_line_only(capsys: pytest.CaptureFixture[str]) -> None:
	first = make_task("first")
	second = make_task("second")
	task = Sequential(first, second, name="pipeline")
	events: list[TaskEvent] = [
		StartedEvent(first, 0, TS),
		CompletedEvent(first, 0, Finished(1, 0.1, ()), TS),
		CompletedEvent(second, 1, Skipped(1), TS),
	]
	asyncio.run(drive(Status(output_mode="all"), task, events))
	out = capsys.readouterr().out
	assert "✗ [first] error" in out
	assert "exit=1" in out
	assert "⏭ [second] skipped" in out
	assert "(prior rc=1)" in out
	assert out.count("::") == 0


def test_empty_template_suppresses_line(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("alpha")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		CompletedEvent(a, 0, Finished(0, 0.1, ()), TS),
	]
	asyncio.run(
		drive(
			Status(started_fmt="", finished_fmt="✓ {name}"),
			task,
			events,
		)
	)
	out = capsys.readouterr().out
	assert "▶" not in out
	assert "✓ alpha" in out


def test_custom_format_string_substitutes_all_fields(
	capsys: pytest.CaptureFixture[str],
) -> None:
	a = Task("ruff check .", name="lint")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		CompletedEvent(a, 0, Finished(0, 1.234, ()), TS),
	]
	asyncio.run(
		drive(
			Status(
				started_fmt="{timestamp:%H:%M:%S}.{ms:03d} START {name} [{cmd}]",
				finished_fmt="OK {name} {elapsed:.3f}s rc={rc}",
			),
			task,
			events,
		)
	)
	out = capsys.readouterr().out
	assert "14:30:00.750 START lint [ruff check .]" in out
	assert "OK lint 1.234s rc=0" in out


def test_missing_field_in_template_raises_keyerror() -> None:
	a = make_task("alpha")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
	]
	with pytest.raises(KeyError):
		asyncio.run(
			drive(
				Status(started_fmt="{name} {elapsed:.2f}"),
				task,
				events,
			)
		)


def test_initial_ctx_is_idle() -> None:
	async def get_initial() -> object:
		return await Status().setup(make_task("solo"))

	assert asyncio.run(get_initial()) == Idle()


def test_block_modes_buffer_into_active_ctx(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("alpha")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		OutputEvent(a, 0, b"one\n", TS),
		OutputEvent(a, 0, b"two\n", TS),
	]
	ctxs = asyncio.run(drive(Status(output_mode="all"), task, events))
	assert ctxs == (Active(b"one\ntwo\n"),)
	out = capsys.readouterr().out
	assert "▶ [alpha] started" in out
	assert "one" not in out


def test_stale_event_combinations_pass_through() -> None:
	"""Idle+Output, Active+Started, Done+anything — all should be no-ops on ctx.

	We can't easily drive these through the normal sequencer (which respects the
	state machine) so we call on_event directly.
	"""
	a = make_task("a")
	effect = Status()

	async def hit_unreachable_branches() -> tuple[object, ...]:
		states: list[LeafState] = [Waiting(a)]
		r1 = await effect.on_event(OutputEvent(a, 0, b"x", TS), states, Idle())
		r2 = await effect.on_event(StartedEvent(a, 0, TS), states, Active(b""))
		r3 = await effect.on_event(StartedEvent(a, 0, TS), states, Done())
		return (r1, r2, r3)

	r1, r2, r3 = asyncio.run(hit_unreachable_branches())
	assert r1 == Idle()
	assert r2 == Active(b"")
	assert r3 == Done()


def test_creates_no_background_tasks() -> None:
	task = make_task("solo")
	events: list[TaskEvent] = [
		StartedEvent(task, 0, TS),
		CompletedEvent(task, 0, Finished(0, 0.05, ()), TS),
	]

	async def count_tasks() -> int:
		baseline = asyncio.all_tasks()
		await drive(Status(), task, events)
		return len(asyncio.all_tasks() - baseline)

	assert asyncio.run(count_tasks()) == 0


def test_block_appends_newline_when_output_lacks_one(
	capsys: pytest.CaptureFixture[str],
) -> None:
	a = make_task("alpha")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		OutputEvent(a, 0, b"no-trailing-newline", TS),
		CompletedEvent(a, 0, Finished(2, 0.1, ()), TS),
	]
	asyncio.run(drive(Status(output_mode="errors"), task, events))
	out = capsys.readouterr().out
	assert "no-trailing-newline\n" in out


def test_errors_mode_skips_block_on_pass(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("ok")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		OutputEvent(a, 0, b"all good\n", TS),
		CompletedEvent(a, 0, Finished(0, 0.1, ()), TS),
	]
	asyncio.run(drive(Status(output_mode="errors"), task, events))
	out = capsys.readouterr().out
	assert "✓ [ok] success" in out
	assert "all good" not in out


def test_default_includes_iso_timestamp_with_milliseconds(
	capsys: pytest.CaptureFixture[str],
) -> None:
	a = make_task("solo")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		CompletedEvent(a, 0, Finished(0, 0.1, ()), TS),
	]
	asyncio.run(drive(Status(), task, events))
	out = capsys.readouterr().out
	assert "[2026-05-21 14:30:00.750]" in out


def test_discoverable_via_parse_effects() -> None:
	"""``Status`` should be reachable through the --effects mini-language."""
	from camas.main.effects import parse_effects

	effects = parse_effects("(Status(),)")
	assert len(effects) == 1
	assert type(effects[0]).__name__ == "Status"
	effects2 = parse_effects("(Status(output_mode='github'),)")
	assert type(effects2[0]).__name__ == "Status"
