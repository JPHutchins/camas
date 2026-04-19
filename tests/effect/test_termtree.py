from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import pytest

from camas import (
	CompletedEvent,
	Done,
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
	TermtreeOptions,
	flatten_rows,
	print_failures,
	render_frame,
	render_lines,
	termtree,
)

ANSI_ESCAPE_PATTERN = __import__("re").compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def visible_width(line: str) -> int:
	return len(ANSI_ESCAPE_PATTERN.sub("", line).lstrip("\r"))


def make_task(name: str) -> Task:
	return Task(("python", "-c", "pass"), name=name)


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


async def events_from(seq: list[TaskEvent]) -> AsyncIterator[TaskEvent]:
	for ev in seq:
		yield ev


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
	asyncio.run(termtree(TermtreeOptions(frame_interval_ms=50))(task, events_from(events)))
	captured = capsys.readouterr()
	assert "FAILED: b" in captured.out


def test_termtree_effect_handles_timeout_tick(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = make_task("solo")

	async def slow_events() -> AsyncIterator[TaskEvent]:
		await asyncio.sleep(0.15)
		yield StartedEvent(0, 100.0)
		yield CompletedEvent(0, 0, 0.15, (b"done\n",))

	asyncio.run(termtree(TermtreeOptions(frame_interval_ms=50))(task, slow_events()))
	captured = capsys.readouterr()
	assert "solo" in captured.out
	assert "PASS" in captured.out


def test_termtree_effect_skips_animation_when_tree_exceeds_terminal_height(
	capsys: pytest.CaptureFixture[str],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	import os

	monkeypatch.setattr("shutil.get_terminal_size", lambda: os.terminal_size((80, 3)))
	a = make_task("a")
	b = make_task("b")
	task = Parallel(tasks=(a, b))
	events: list[TaskEvent] = [
		StartedEvent(0, 100.0),
		CompletedEvent(0, 0, 0.1, (b"ok a\n",)),
		StartedEvent(1, 100.0),
		CompletedEvent(1, 0, 0.2, (b"ok b\n",)),
	]
	asyncio.run(termtree(TermtreeOptions(frame_interval_ms=50))(task, events_from(events)))
	captured = capsys.readouterr()
	assert "PASS" in captured.out
	assert "\x1b[2F" not in captured.out


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
	asyncio.run(termtree(TermtreeOptions(frame_interval_ms=50))(task, events_from(events)))
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


def test_render_lines_truncates_name_when_detail_needs_room() -> None:
	"""Long names get middle-truncated to leave MIN_DETAIL_WIDTH for output tails."""
	long_task = Task(
		("python", "-c", "pass"),
		name="build postgres/release [DB=postgres, OPT=release]",
	)
	tree = Parallel(tasks=(long_task,))
	rows = flatten_rows(tree)
	states: tuple[LeafState, ...] = (Done(long_task, TaskResult("x", 0, 0.1, (b"built\n",))),)
	lines = render_lines(rows, states, term_width=77, display_width=58, now=100.5, wall_start=100.0)
	leaf_line = next(ln for ln in lines if "PASS" in ln)
	assert "..." in leaf_line
	assert "built" in leaf_line or ".." in leaf_line
