# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import TypeVar

import pytest

from camas import (
	CompletedEvent,
	Done,
	Effect,
	LeafState,
	OutputEvent,
	Parallel,
	Running,
	Sequential,
	Skipped,
	StartedEvent,
	Task,
	TaskEvent,
	TaskResult,
	Waiting,
)
from camas.effect.termtree import (
	STATUS_COL_WIDTH,
	Termtree,
	TermtreeOptions,
	flatten_rows,
	print_failures,
	print_passes,
	render_frame,
	render_lines,
	strip_ansi,
)

ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def visible_width(line: str) -> int:
	return len(ANSI_ESCAPE_PATTERN.sub("", line).lstrip("\r"))


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


def test_render_frame_all_waiting_shows_group_header_and_wait_cells(
	capsys: pytest.CaptureFixture[str],
) -> None:
	tree = Parallel(tasks=(make_task("a"), make_task("b")), name="root")
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = (Waiting(make_task("a")), Waiting(make_task("b")))
	frame = render_frame(rows, states, term_width=80, display_width=60, now=100.0, wall_start=100.0)
	assert "root" in frame
	assert "WAIT" in frame


def test_render_frame_mixed_states() -> None:
	a = make_task("a")
	b = make_task("b")
	c = make_task("c")
	tree = Parallel(tasks=(a, b, c))
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = (
		Done(a, TaskResult("a", 0, 0.1, (b"all clean\n",))),
		Running(b, 100.0, b"working..."),
		Skipped(c),
	)
	frame = render_frame(rows, states, term_width=80, display_width=60, now=100.5, wall_start=100.0)
	assert "PASS" in frame
	assert "SKIP" in frame
	assert "all clean" in frame


def test_render_frame_failure_summary() -> None:
	a = make_task("a")
	tree = Parallel(tasks=(a,))
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = (Done(a, TaskResult("a", 1, 0.1, (b"boom\n",))),)
	frame = render_frame(rows, states, term_width=80, display_width=60, now=100.5, wall_start=100.0)
	assert "FAIL" in frame


def test_print_failures_outputs_failed_task(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("a")
	b = make_task("b")
	states: tuple[LeafState, ...] = (
		Done(a, TaskResult("a", 0, 0.1, (b"ok\n",))),
		Done(b, TaskResult("b", 1, 0.2, (b"error details\n",))),
	)
	print_failures(states)
	captured = capsys.readouterr()
	assert "FAILED: b" in captured.out
	assert "error details" in captured.out
	assert "FAILED: a" not in captured.out


def test_print_passes_outputs_passed_task(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("a")
	b = make_task("b")
	states: tuple[LeafState, ...] = (
		Done(a, TaskResult("a", 0, 0.1, (b"clean output\n",))),
		Done(b, TaskResult("b", 1, 0.2, (b"error details\n",))),
	)
	print_passes(states)
	captured = capsys.readouterr()
	assert "PASSED: a" in captured.out
	assert "clean output" in captured.out
	assert "PASSED: b" not in captured.out


def test_termtree_show_passing_prints_passed_output(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = Parallel(tasks=(make_task("a"), make_task("b")))
	events: list[TaskEvent] = [
		StartedEvent(0, 100.0),
		StartedEvent(1, 100.0),
		CompletedEvent(0, 0, 0.1, (b"first output\n",)),
		CompletedEvent(1, 0, 0.2, (b"second output\n",)),
	]
	asyncio.run(
		drive(
			Termtree(TermtreeOptions(frame_interval_ms=50, show_passing=True)),
			task,
			events,
		)
	)
	captured = capsys.readouterr()
	assert "PASSED: a" in captured.out
	assert "first output" in captured.out
	assert "PASSED: b" in captured.out
	assert "second output" in captured.out


def test_termtree_show_passing_defaults_to_false(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = Parallel(tasks=(make_task("a"),))
	events: list[TaskEvent] = [
		StartedEvent(0, 100.0),
		CompletedEvent(0, 0, 0.1, (b"quiet\n",)),
	]
	asyncio.run(drive(Termtree(TermtreeOptions(frame_interval_ms=50)), task, events))
	captured = capsys.readouterr()
	assert "PASSED:" not in captured.out


def test_termtree_effect_consumes_events_and_renders(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = Parallel(tasks=(make_task("a"), make_task("b")))
	events: list[TaskEvent] = [
		StartedEvent(0, 100.0),
		StartedEvent(1, 100.0),
		OutputEvent(0, b"line from a\n", 100.05),
		CompletedEvent(0, 0, 0.1, (b"line from a\n",)),
		CompletedEvent(1, 1, 0.2, (b"boom\n",)),
	]
	asyncio.run(drive(Termtree(TermtreeOptions(frame_interval_ms=50)), task, events))
	captured = capsys.readouterr()
	assert "FAILED: b" in captured.out


def test_termtree_frame_tick_keeps_spinner_alive_between_events() -> None:
	"""setup spawns an asyncio.Task that redraws on an interval; teardown cancels it."""
	task = make_task("solo")

	async def run_effect() -> bool:
		effect = Termtree(TermtreeOptions(frame_interval_ms=20))
		ctx = await effect.setup(task)
		# Idle for long enough that several ticks fire while nothing is happening.
		await asyncio.sleep(0.08)
		ctx = await effect.on_event(
			CompletedEvent(0, 0, 0.08, (b"done\n",)),
			(Done(task, TaskResult("solo", 0, 0.08, (b"done\n",))),),
			ctx,
		)
		assert ctx.state.tick_task is not None
		was_running = not ctx.state.tick_task.done()
		await effect.teardown((ctx,))
		assert ctx.state.tick_task.done()
		return was_running

	assert asyncio.run(run_effect()) is True


def test_termtree_effect_handles_groups_and_skipped(
	capsys: pytest.CaptureFixture[str],
) -> None:
	a = make_task("a")
	b = make_task("b")
	task = Sequential(tasks=(a, b), name="pipeline")
	events: list[TaskEvent] = [
		StartedEvent(0, 100.0),
		CompletedEvent(0, 1, 0.1, (b"failed\n",)),
		CompletedEvent(1, -1, 0.0, ()),
	]
	asyncio.run(drive(Termtree(TermtreeOptions(frame_interval_ms=50)), task, events))
	captured = capsys.readouterr()
	assert "pipeline" in captured.out
	assert "SKIP" in captured.out


StateFactory = Callable[[Task], LeafState]


def _waiting(t: Task) -> LeafState:
	return Waiting(t)


def _running(t: Task) -> LeafState:
	return Running(t, 100.0, b"working on it")


def _done(t: Task) -> LeafState:
	return Done(t, TaskResult("x", 0, 0.1, (b"built\n",)))


def _skipped(t: Task) -> LeafState:
	return Skipped(t)


@pytest.mark.parametrize("term_width", [60, 77, 80, 120])
@pytest.mark.parametrize(
	"state_factory",
	[_waiting, _running, _done, _skipped],
	ids=["waiting", "running", "done", "skipped"],
)
def test_render_lines_never_exceed_term_width_for_long_matrix_names(
	term_width: int,
	state_factory: StateFactory,
) -> None:
	long_task = Task(
		("python", "-c", "pass"),
		name="build postgres/release [DB=postgres, OPT=release]",
	)
	tree = Sequential(tasks=(long_task,), name="ci [DB=postgres, OPT=release]")
	rows = flatten_rows(tree)
	display_width = term_width - STATUS_COL_WIDTH - 1
	states: tuple[LeafState, ...] = (state_factory(long_task),)
	lines = render_lines(rows, states, term_width, display_width, 100.5, 100.0)
	for line in lines:
		assert visible_width(line) < term_width, (
			f"line {visible_width(line)} chars exceeds term_width {term_width}: {line!r}"
		)


def test_strip_ansi_removes_color_codes() -> None:
	assert strip_ansi("\x1b[38;5;214mwarning\x1b[0m") == "warning"


def test_strip_ansi_removes_control_chars_and_ansi() -> None:
	assert strip_ansi("\r\x1b[2Kprogress") == "progress"


def test_render_lines_strips_ansi_from_done_tail() -> None:
	task = make_task("a")
	rows = flatten_rows(Parallel(tasks=(task,)))
	states: tuple[LeafState, ...] = (
		Done(task, TaskResult("a", 0, 0.1, (b"\x1b[38;5;214mcolored\x1b[0m\n",))),
	)
	lines = render_lines(
		rows, states, term_width=120, display_width=100, now=100.5, wall_start=100.0
	)
	combined = "".join(lines)
	assert "\x1b[38;5;214m" not in combined
	assert "colored" in ANSI_ESCAPE_PATTERN.sub("", combined)


def test_render_lines_strips_ansi_from_running_tail() -> None:
	task = make_task("a")
	rows = flatten_rows(Parallel(tasks=(task,)))
	states: tuple[LeafState, ...] = (Running(task, 100.0, b"\x1b[38;5;214mbuilding\x1b[0m"),)
	lines = render_lines(
		rows, states, term_width=120, display_width=100, now=100.5, wall_start=100.0
	)
	combined = "".join(lines)
	assert "\x1b[38;5;214m" not in combined
	assert "building" in ANSI_ESCAPE_PATTERN.sub("", combined)


def test_render_lines_preserves_full_name_when_it_fits() -> None:
	"""Name expands fully; the live stream fills only the leftover gap."""
	task = Task(("python", "-c", "pass"), name="uv run ruff check .")
	tree = Parallel(tasks=(task,))
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = (
		Done(task, TaskResult("x", 0, 0.1, (b"All checks passed!\n",))),
	)
	lines = render_lines(
		rows, states, term_width=120, display_width=100, now=100.5, wall_start=100.0
	)
	leaf_line = next(ln for ln in lines if "PASS" in ln)
	plain = ANSI_ESCAPE_PATTERN.sub("", leaf_line).lstrip("\r")
	assert "uv run ruff check ." in plain
	assert "uv run ruff ch..." not in plain
	assert "All checks passed!" in plain


def test_render_lines_truncates_name_only_when_it_cannot_fit() -> None:
	"""Middle-truncation kicks in only when the command itself overflows the leaf column."""
	long_task = Task(
		("python", "-c", "pass"),
		name="uv run pytest --doctest-modules -v -m 'not slow' --really-long",
	)
	tree = Parallel(tasks=(long_task,))
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = (Done(long_task, TaskResult("x", 0, 0.1, (b"built\n",))),)
	lines = render_lines(rows, states, term_width=60, display_width=41, now=100.5, wall_start=100.0)
	leaf_line = next(ln for ln in lines if "PASS" in ln)
	plain = ANSI_ESCAPE_PATTERN.sub("", leaf_line).lstrip("\r")
	assert "..." in plain


def test_render_lines_stream_uses_only_leftover_space() -> None:
	"""When the full command consumes the column, no stream text is rendered."""
	task = Task(("python", "-c", "pass"), name="x" * 40)
	tree = Parallel(tasks=(task,))
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = (Done(task, TaskResult("x", 0, 0.1, (b"shouldnt appear\n",))),)
	lines = render_lines(rows, states, term_width=60, display_width=41, now=100.5, wall_start=100.0)
	leaf_line = next(ln for ln in lines if "PASS" in ln)
	plain = ANSI_ESCAPE_PATTERN.sub("", leaf_line).lstrip("\r")
	assert "shouldnt appear" not in plain
	assert "shouldnt" not in plain
