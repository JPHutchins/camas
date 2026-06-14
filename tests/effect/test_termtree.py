# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar

import pytest

from camas import Parallel, Sequential, Task
from camas.core.leaf_state import KILL_PRESSES
from camas.core.render import flatten_rows, strip_ansi
from camas.effect.termtree import (
	STATUS_COL_WIDTH,
	Termtree,
	print_failures,
	print_passes,
	render_frame,
	render_lines,
)
from camas.v0.completion import INTERRUPT_RC, Finished, Skipped, Stopped
from camas.v0.leaf_state import Completed, Interrupting, LeafState, Running, Waiting
from camas.v0.task_event import (
	CompletedEvent,
	OutputEvent,
	StartedEvent,
	TaskEvent,
)

if TYPE_CHECKING:
	from camas.v0.effect import Effect

TS = datetime(2026, 5, 21, 14, 30, 0)

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


def test_render_frame_all_waiting_shows_group_header_and_wait_cells(
	capsys: pytest.CaptureFixture[str],
) -> None:
	tree = Parallel(make_task("a"), make_task("b"), name="root")
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = (Waiting(make_task("a")), Waiting(make_task("b")))
	frame = render_frame(
		rows, states, term_width=80, term_height=24, now=TS, wall_elapsed=0.0, prev_visible=0
	)
	assert "root" in frame.text
	assert "WAIT" in frame.text


def test_render_frame_mixed_states() -> None:
	a = make_task("a")
	b = make_task("b")
	c = make_task("c")
	tree = Parallel(a, b, c)
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = (
		Completed(a, Finished(0, 0.1, (b"all clean\n",))),
		Running(b, TS, b"working..."),
		Completed(c, Skipped(1)),
	)
	frame = render_frame(
		rows, states, term_width=80, term_height=24, now=TS, wall_elapsed=0.0, prev_visible=0
	)
	assert "PASS" in frame.text
	assert "SKIP" in frame.text
	assert "all clean" in frame.text


def test_render_frame_failure_summary() -> None:
	a = make_task("a")
	tree = Parallel(a)
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = (Completed(a, Finished(1, 0.1, (b"boom\n",))),)
	frame = render_frame(
		rows, states, term_width=80, term_height=24, now=TS, wall_elapsed=0.0, prev_visible=0
	)
	assert "FAIL" in frame.text


def test_render_frame_interrupting_shows_press_count_labels() -> None:
	a, b, c = make_task("a"), make_task("b"), make_task("c")
	rows = flatten_rows(Parallel(a, b, c))
	states: tuple[LeafState, ...] = (
		Interrupting(a, TS, b"working...", 1),
		Interrupting(b, TS, b"", 2),
		Interrupting(c, TS, b"", KILL_PRESSES),
	)
	frame = render_frame(
		rows, states, term_width=80, term_height=24, now=TS, wall_elapsed=0.0, prev_visible=0
	)
	assert "^C^C" in frame.text  # second press
	assert "KILL" in frame.text  # third press, also the summary (max presses)
	assert "working" in frame.text
	assert "PASS" not in frame.text


def test_render_frame_stopped_shows_stop_and_summary() -> None:
	a, b = make_task("a"), make_task("b")
	rows = flatten_rows(Parallel(a, b))
	states: tuple[LeafState, ...] = (
		Completed(a, Stopped(130, 0.2, (b"bye\n",))),
		Completed(b, Stopped(INTERRUPT_RC, 0.0, ())),
	)
	frame = render_frame(
		rows, states, term_width=80, term_height=24, now=TS, wall_elapsed=0.0, prev_visible=0
	)
	assert "STOP" in frame.text
	assert "bye" in frame.text


_STOPPED_EVENTS: list[TaskEvent] = [
	StartedEvent(make_task("a"), 0, TS),
	CompletedEvent(
		make_task("a"), 0, Stopped(INTERRUPT_RC, 0.1, (b"line one\n", b"line two\n")), TS
	),
]


def test_termtree_output_ctrl_c_dumps_stopped_output(capsys: pytest.CaptureFixture[str]) -> None:
	asyncio.run(
		drive(Termtree(output_ctrl_c=True), Parallel(make_task("a")), list(_STOPPED_EVENTS))
	)
	out = capsys.readouterr().out
	assert "STOPPED" in out
	assert "line one" in out  # full dump, not just the inline last line


def test_termtree_default_does_not_dump_stopped_output(capsys: pytest.CaptureFixture[str]) -> None:
	asyncio.run(drive(Termtree(), Parallel(make_task("a")), list(_STOPPED_EVENTS)))
	out = capsys.readouterr().out
	assert "STOPPED" not in out
	assert "line one" not in out  # only the inline "line two" survives in the row


def _wide_tree(leaves: int) -> Parallel:
	return Parallel(*(make_task(f"t{i}") for i in range(leaves)), name="root")


def test_render_frame_clamps_to_height_and_keeps_summary() -> None:
	"""A frame taller than the window collapses to the height, summary always last."""
	tree = _wide_tree(40)
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = tuple(Waiting(make_task(f"t{i}")) for i in range(40))
	frame = render_frame(
		rows, states, term_width=80, term_height=10, now=TS, wall_elapsed=0.0, prev_visible=0
	)
	body = frame.text.split("\n")
	assert frame.visible == 10
	assert len(body) == 10
	assert "more" in frame.text  # elision marker
	assert "result" in body[-1]  # summary survives the clamp


def test_render_frame_does_not_clamp_when_it_fits() -> None:
	tree = _wide_tree(3)
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = tuple(Waiting(make_task(f"t{i}")) for i in range(3))
	frame = render_frame(
		rows, states, term_width=80, term_height=24, now=TS, wall_elapsed=0.0, prev_visible=0
	)
	assert frame.visible == len(rows) + 1
	assert "more" not in frame.text


def test_render_frame_repositions_with_previous_height() -> None:
	"""Cursor moves up by the *previous* frame's height, not the current one."""
	tree = _wide_tree(2)
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = (Waiting(make_task("t0")), Waiting(make_task("t1")))
	first = render_frame(
		rows, states, term_width=80, term_height=24, now=TS, wall_elapsed=0.0, prev_visible=0
	)
	assert not first.text.startswith("\x1b[")  # no reposition on the first frame
	repaint = render_frame(
		rows,
		states,
		term_width=80,
		term_height=24,
		now=TS,
		wall_elapsed=0.0,
		prev_visible=first.visible,
	)
	assert repaint.text.startswith(f"\x1b[{first.visible - 1}F")


def test_render_frame_erases_leftover_rows_when_shrinking() -> None:
	"""A now-shorter frame erases the rows the previous (taller) one left behind."""
	tree = _wide_tree(40)
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = tuple(Waiting(make_task(f"t{i}")) for i in range(40))
	frame = render_frame(
		rows, states, term_width=80, term_height=10, now=TS, wall_elapsed=0.0, prev_visible=24
	)
	assert frame.visible == 10
	assert frame.text.endswith("\x1b[0J")


def test_render_frame_final_emits_full_tree_unclamped() -> None:
	"""The teardown frame skips the clamp so the whole tree lands in the scrollback."""
	tree = _wide_tree(40)
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = tuple(Waiting(make_task(f"t{i}")) for i in range(40))
	frame = render_frame(
		rows,
		states,
		term_width=80,
		term_height=10,
		now=TS,
		wall_elapsed=0.0,
		prev_visible=10,
		final=True,
	)
	assert frame.visible == len(rows) + 1  # all rows, not clamped to 10
	assert "more" not in frame.text


def test_print_failures_outputs_failed_task(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("a")
	b = make_task("b")
	states: tuple[LeafState, ...] = (
		Completed(a, Finished(0, 0.1, (b"ok\n",))),
		Completed(b, Finished(1, 0.2, (b"error details\n",))),
	)
	print_failures(states, term_width=80)
	captured = capsys.readouterr()
	assert "FAILED: b" in captured.out
	assert "error details" in captured.out
	assert "FAILED: a" not in captured.out


def test_print_passes_outputs_passed_task(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("a")
	b = make_task("b")
	states: tuple[LeafState, ...] = (
		Completed(a, Finished(0, 0.1, (b"clean output\n",))),
		Completed(b, Finished(1, 0.2, (b"error details\n",))),
	)
	print_passes(states, term_width=80)
	captured = capsys.readouterr()
	assert "PASSED: a" in captured.out
	assert "clean output" in captured.out
	assert "PASSED: b" not in captured.out


def test_termtree_show_passing_prints_passed_output(
	capsys: pytest.CaptureFixture[str],
) -> None:
	a = make_task("a")
	b = make_task("b")
	task = Parallel(a, b)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		StartedEvent(b, 1, TS),
		CompletedEvent(a, 0, Finished(0, 0.1, (b"first output\n",)), TS),
		CompletedEvent(b, 1, Finished(0, 0.2, (b"second output\n",)), TS),
	]
	asyncio.run(
		drive(
			Termtree(frame_interval_ms=50, show_passing=True),
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
	a = make_task("a")
	task = Parallel(a)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		CompletedEvent(a, 0, Finished(0, 0.1, (b"quiet\n",)), TS),
	]
	asyncio.run(drive(Termtree(frame_interval_ms=50), task, events))
	captured = capsys.readouterr()
	assert "PASSED:" not in captured.out


def test_termtree_effect_consumes_events_and_renders(
	capsys: pytest.CaptureFixture[str],
) -> None:
	a = make_task("a")
	b = make_task("b")
	task = Parallel(a, b)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		StartedEvent(b, 1, TS),
		OutputEvent(a, 0, b"line from a\n", TS),
		CompletedEvent(a, 0, Finished(0, 0.1, (b"line from a\n",)), TS),
		CompletedEvent(b, 1, Finished(1, 0.2, (b"boom\n",)), TS),
	]
	asyncio.run(drive(Termtree(frame_interval_ms=50), task, events))
	captured = capsys.readouterr()
	assert "FAILED: b" in captured.out


def test_termtree_frame_tick_keeps_spinner_alive_between_events() -> None:
	"""setup spawns an asyncio.Task that redraws on an interval; teardown cancels it."""
	task = make_task("solo")

	async def run_effect() -> bool:
		effect = Termtree(frame_interval_ms=20)
		ctx = await effect.setup(task)
		# Idle for long enough that several ticks fire while nothing is happening.
		await asyncio.sleep(0.08)
		ctx = await effect.on_event(
			CompletedEvent(task, 0, Finished(0, 0.08, (b"done\n",)), TS),
			(Completed(task, Finished(0, 0.08, (b"done\n",))),),
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
	task = Sequential(a, b, name="pipeline")
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		CompletedEvent(a, 0, Finished(1, 0.1, (b"failed\n",)), TS),
		CompletedEvent(b, 1, Skipped(1), TS),
	]
	asyncio.run(drive(Termtree(frame_interval_ms=50), task, events))
	captured = capsys.readouterr()
	assert "pipeline" in captured.out
	assert "SKIP" in captured.out


StateFactory = Callable[[Task], LeafState]


def _waiting(t: Task) -> LeafState:
	return Waiting(t)


def _running(t: Task) -> LeafState:
	return Running(t, TS, b"working on it")


def _done(t: Task) -> LeafState:
	return Completed(t, Finished(0, 0.1, (b"built\n",)))


def _skipped(t: Task) -> LeafState:
	return Completed(t, Skipped(1))


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
	tree = Sequential(long_task, name="ci [DB=postgres, OPT=release]")
	rows = flatten_rows(tree)
	display_width = term_width - STATUS_COL_WIDTH - 1
	states: tuple[LeafState, ...] = (state_factory(long_task),)
	lines = render_lines(rows, states, term_width, display_width, TS, 0.0)
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
	rows = flatten_rows(Parallel(task))
	states: tuple[LeafState, ...] = (
		Completed(task, Finished(0, 0.1, (b"\x1b[38;5;214mcolored\x1b[0m\n",))),
	)
	lines = render_lines(rows, states, term_width=120, display_width=100, now=TS, wall_elapsed=0.0)
	combined = "".join(lines)
	assert "\x1b[38;5;214m" not in combined
	assert "colored" in ANSI_ESCAPE_PATTERN.sub("", combined)


def test_render_lines_strips_ansi_from_running_tail() -> None:
	task = make_task("a")
	rows = flatten_rows(Parallel(task))
	states: tuple[LeafState, ...] = (Running(task, TS, b"\x1b[38;5;214mbuilding\x1b[0m"),)
	lines = render_lines(rows, states, term_width=120, display_width=100, now=TS, wall_elapsed=0.0)
	combined = "".join(lines)
	assert "\x1b[38;5;214m" not in combined
	assert "building" in ANSI_ESCAPE_PATTERN.sub("", combined)


def test_render_lines_preserves_full_name_when_it_fits() -> None:
	"""Name expands fully; the live stream fills only the leftover gap."""
	task = Task(("python", "-c", "pass"), name="uv run ruff check .")
	tree = Parallel(task)
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = (Completed(task, Finished(0, 0.1, (b"All checks passed!\n",))),)
	lines = render_lines(rows, states, term_width=120, display_width=100, now=TS, wall_elapsed=0.0)
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
	tree = Parallel(long_task)
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = (Completed(long_task, Finished(0, 0.1, (b"built\n",))),)
	lines = render_lines(rows, states, term_width=60, display_width=41, now=TS, wall_elapsed=0.0)
	leaf_line = next(ln for ln in lines if "PASS" in ln)
	plain = ANSI_ESCAPE_PATTERN.sub("", leaf_line).lstrip("\r")
	assert "..." in plain


def test_render_lines_stream_uses_only_leftover_space() -> None:
	"""When the full command consumes the column, no stream text is rendered."""
	task = Task(("python", "-c", "pass"), name="x" * 40)
	tree = Parallel(task)
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = (Completed(task, Finished(0, 0.1, (b"shouldnt appear\n",))),)
	lines = render_lines(rows, states, term_width=60, display_width=41, now=TS, wall_elapsed=0.0)
	leaf_line = next(ln for ln in lines if "PASS" in ln)
	plain = ANSI_ESCAPE_PATTERN.sub("", leaf_line).lstrip("\r")
	assert "shouldnt appear" not in plain
	assert "shouldnt" not in plain
