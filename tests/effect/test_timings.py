# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar, cast

import pytest

from camas import Parallel, Sequential, Task
from camas.core import timings
from camas.effect.timings import Timings
from camas.v0.completion import Finished, Skipped
from camas.v0.leaf_state import LeafState, Waiting
from camas.v0.task_event import CompletedEvent, StartedEvent, TaskEvent

if TYPE_CHECKING:
	from pathlib import Path

	from camas.v0.effect import Effect


def cache_key(label: str, scope: int = 0) -> timings.CacheKey:
	"""The cache key for ``label`` at ``scope`` — whole-tree unless a test says otherwise."""
	return timings.CacheKey(label, scope)


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
		await effect.teardown(tuple(ctxs) or (initial,))


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
	assert cache[cache_key("fast")].elapsed_s == 0.1
	assert cache[cache_key("slow")].elapsed_s == 0.5


def test_anonymous_run_records_its_leaves(tmp_path: Path) -> None:
	a = _task("solo")
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		CompletedEvent(a, 0, Finished(0, 0.1, ()), TS),
	]
	asyncio.run(drive(Timings(camas_dir=tmp_path), Parallel(a), events))
	assert timings.load(tmp_path)[cache_key("solo")].elapsed_s == 0.1


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
	assert cache[cache_key("echo hi")].elapsed_s == 0.5
	assert cache[cache_key("python -c pass")].elapsed_s == 0.1


def test_unfinished_leaf_excluded(tmp_path: Path) -> None:
	a, b = _task("done"), _task("never")
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		StartedEvent(b, 1, TS),
		CompletedEvent(a, 0, Finished(0, 0.2, ()), TS),
	]
	asyncio.run(drive(Timings(camas_dir=tmp_path), Parallel(a, b, name="grp"), events))
	cache = timings.load(tmp_path)
	assert cache_key("done") in cache
	assert cache_key("never") not in cache


def test_zero_leaf_run_records_nothing(tmp_path: Path) -> None:
	asyncio.run(drive(Timings(camas_dir=tmp_path), Parallel(), []))
	assert timings.load(tmp_path) == {}


def test_skipped_leaf_excluded(tmp_path: Path) -> None:
	a, b = _task("fail"), _task("skip")
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		CompletedEvent(a, 0, Finished(2, 0.3, ()), TS),
		CompletedEvent(b, 1, Skipped(2), TS),
	]
	asyncio.run(drive(Timings(camas_dir=tmp_path), Sequential(a, b, name="seq"), events))
	cache = timings.load(tmp_path)
	assert cache_key("fail") in cache
	assert cache_key("skip") not in cache


def test_for_run_keys_a_configured_effect_to_the_scope_and_identities(tmp_path: Path) -> None:
	"""A ``Timings`` written out by hand — in ``--effects`` or a ``Config`` — cannot know what the run
	it lands in is scoped to, so it records whole-tree by default. ``for_run`` is how the caller that
	does know keys it, and without that a path-scoped run would be recorded as a whole-tree
	observation: #224, for explicitly configured effects.
	"""
	scoped = Task(("python", "-c", "pass"), name="pylint a.py")
	events: list[TaskEvent] = [
		StartedEvent(scoped, 0, TS),
		CompletedEvent(scoped, 0, Finished(0, 0.4, ()), TS),
	]
	asyncio.run(
		drive(
			Timings(camas_dir=tmp_path).for_run(2, (cache_key("pylint .", 2),)),
			Parallel(scoped, name="check"),
			events,
		)
	)
	assert timings.load(tmp_path) == {cache_key("pylint .", 2): timings.TaskTiming(0.4, 1)}


def test_setup_rejects_identities_not_parallel_to_the_leaves(tmp_path: Path) -> None:
	"""A ``Timings`` keyed with identities is validated against the tree it is set up on, so a
	mismatch fails loudly here rather than silently mis-keying records at teardown."""
	a, b = _task("one"), _task("two")
	with pytest.raises(ValueError, match="identities must be parallel"):
		asyncio.run(
			drive(Timings(camas_dir=tmp_path, identities=(cache_key("one"),)), Parallel(a, b), [])
		)


def test_constructor_rejects_the_pre_289_keys_mapping_shape(tmp_path: Path) -> None:
	"""The third slot carries identities now; a dict — the label→key mapping the slot took before
	#289 — fails loudly naming the change instead of mis-parsing as a short identities tuple."""
	with pytest.raises(TypeError, match="keys="):
		Timings(camas_dir=tmp_path, identities=cast("Any", {"lint": cache_key("lint")}))


def test_identities_key_by_leaf_position_not_completion_order(tmp_path: Path) -> None:
	"""A carried identity is the leaf's position among the run's leaves, so an earlier leaf that
	never completed must not shift the later ones' keys."""
	a, b = _task("first"), _task("second")
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		StartedEvent(b, 1, TS),
		CompletedEvent(b, 1, Finished(0, 0.5, ()), TS),
	]
	identities = (cache_key("first"), cache_key("second"))
	asyncio.run(drive(Timings(camas_dir=tmp_path, identities=identities), Parallel(a, b), events))
	assert timings.load(tmp_path) == {cache_key("second"): timings.TaskTiming(0.5, 1)}


def test_a_leaf_named_empty_records_nothing_rather_than_something_unreadable(
	tmp_path: Path,
) -> None:
	"""A leaf named ``""`` has no label a cache row can carry — the label is a row's first
	whitespace-separated field, so it would be written and then discarded on every load. It is
	dropped on the way in instead, and the run still records everything else.
	"""
	blank, named = Task(("python", "-c", "pass"), name=""), _task("real")
	events: list[TaskEvent] = [
		StartedEvent(blank, 0, TS),
		StartedEvent(named, 1, TS),
		CompletedEvent(blank, 0, Finished(0, 0.2, ()), TS),
		CompletedEvent(named, 1, Finished(0, 0.3, ()), TS),
	]
	asyncio.run(drive(Timings(camas_dir=tmp_path), Parallel(blank, named, name="grp"), events))
	assert timings.load(tmp_path) == {cache_key("real"): timings.TaskTiming(0.3, 1)}
