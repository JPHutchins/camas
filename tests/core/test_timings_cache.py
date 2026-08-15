# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from camas import Parallel, Sequential, Task
from camas.core import timings
from camas.core.completion import RunResult, TaskResult
from camas.core.scope import scoped_leaves
from camas.core.traversal import flatten_leaves
from camas.v0.completion import Errored, Finished, Skipped, Stopped

if TYPE_CHECKING:
	from pathlib import Path


def cache_key(label: str, scope: int = 0) -> timings.CacheKey:
	"""The cache key for ``label`` at ``scope`` — whole-tree unless a test says otherwise."""
	return timings.CacheKey(label, scope)


def _result(*leaves: TaskResult, elapsed: float) -> RunResult:
	return RunResult(returncode=0, results=leaves, elapsed=elapsed)


def test_load_missing_is_empty(tmp_path: Path) -> None:
	assert timings.load(tmp_path) == {}


def test_ensure_camas_dir_creates_dir_and_gitignore(tmp_path: Path) -> None:
	camas = tmp_path / ".camas"
	timings.ensure_camas_dir(camas)
	assert camas.is_dir()
	assert (camas / ".gitignore").read_text(encoding="utf-8") == "*\n"
	timings.ensure_camas_dir(camas)
	assert (camas / ".gitignore").read_text(encoding="utf-8") == "*\n"


def test_record_run_writes_each_leaf(tmp_path: Path) -> None:
	timings.Observed(tmp_path, 0, {}).record(
		_result(
			TaskResult("lint", Finished(0, 0.1, ())),
			TaskResult("test", Finished(0, 2.0, ())),
			elapsed=2.1,
		),
	)
	cache = timings.load(tmp_path)
	assert cache[cache_key("lint")] == timings.TaskTiming(elapsed_s=0.1, samples=1)
	assert cache[cache_key("test")] == timings.TaskTiming(elapsed_s=2.0, samples=1)


def test_record_averages_repeated_runs(tmp_path: Path) -> None:
	timings.record(tmp_path, [(cache_key("a"), 1.0)])
	timings.record(tmp_path, [(cache_key("a"), 3.0)])
	assert timings.load(tmp_path)[cache_key("a")] == timings.TaskTiming(elapsed_s=2.0, samples=2)


def test_record_counts_stopped_leaf(tmp_path: Path) -> None:
	timings.Observed(tmp_path, 0, {}).record(
		_result(TaskResult("x", Stopped(130, 0.3, ())), elapsed=0.3)
	)
	assert timings.load(tmp_path)[cache_key("x")].elapsed_s == 0.3


def test_record_skips_run_with_no_timed_leaf(tmp_path: Path) -> None:
	timings.Observed(tmp_path, 0, {}).record(
		_result(TaskResult("s", Skipped(1, "blk")), elapsed=0.0)
	)
	assert timings.load(tmp_path) == {}
	assert not (tmp_path / timings.CACHE_NAME).exists()


def test_record_skips_errored_leaf(tmp_path: Path) -> None:
	timings.Observed(tmp_path, 0, {}).record(
		_result(TaskResult("ghost", Errored(127, "no such file or directory: ghost")), elapsed=0.0),
	)
	assert timings.load(tmp_path) == {}
	assert not (tmp_path / timings.CACHE_NAME).exists()


def test_elapsed_of_errored_is_none() -> None:
	assert timings.elapsed_of(Errored(127, "no such file or directory: x")) is None


def test_load_skips_malformed_lines(tmp_path: Path) -> None:
	(tmp_path / timings.CACHE_NAME).write_text(
		"0\ngood 1.0 2\ngarbage\nbad x 2\n", encoding="utf-8"
	)
	cache = timings.load(tmp_path)
	assert set(cache) == {cache_key("good")}
	assert cache[cache_key("good")] == timings.TaskTiming(1.0, 2)


def test_load_ignores_unversioned_cache(tmp_path: Path) -> None:
	(tmp_path / timings.CACHE_NAME).write_text("lint 1.0 2\n", encoding="utf-8")
	assert timings.load(tmp_path) == {}


def test_leaf_name_with_spaces_round_trips(tmp_path: Path) -> None:
	timings.record(
		tmp_path, [(cache_key("test [PY=3.13]"), 3.0), (cache_key("test [PY=3.12]"), 1.0)]
	)
	cache = timings.load(tmp_path)
	assert cache[cache_key("test [PY=3.13]")].elapsed_s == 3.0
	assert cache[cache_key("test [PY=3.12]")].elapsed_s == 1.0


def test_record_is_versioned(tmp_path: Path) -> None:
	timings.record(tmp_path, [(cache_key("lint"), 0.5)])
	assert (tmp_path / timings.CACHE_NAME).read_text(encoding="utf-8").startswith("1\n")


def test_estimate_leaf_uses_its_own_timing() -> None:
	cache = {cache_key("lint"): timings.TaskTiming(0.2, 1)}
	assert timings.estimate(Task("ruff", name="lint"), cache) == timings.Estimate(
		0.2, 1, "lint", 0.2
	)


def test_estimate_sequential_sums_children() -> None:
	cache = {cache_key("a"): timings.TaskTiming(1.0, 3), cache_key("b"): timings.TaskTiming(2.0, 1)}
	est = timings.estimate(Sequential(Task("x", name="a"), Task("y", name="b")), cache)
	assert est == timings.Estimate(elapsed_s=3.0, samples=1, slowest_leaf="b", slowest_s=2.0)


def test_estimate_parallel_takes_max() -> None:
	cache = {cache_key("a"): timings.TaskTiming(1.0, 2), cache_key("b"): timings.TaskTiming(2.0, 2)}
	est = timings.estimate(Parallel(Task("x", name="a"), Task("y", name="b")), cache)
	assert est == timings.Estimate(elapsed_s=2.0, samples=2, slowest_leaf="b", slowest_s=2.0)


def test_estimate_is_none_when_a_leaf_was_never_timed() -> None:
	cache = {cache_key("a"): timings.TaskTiming(1.0, 1)}
	assert timings.estimate(Sequential(Task("x", name="a"), Task("y", name="b")), cache) is None


def test_v1_malformed_rows_are_ignored(tmp_path: Path) -> None:
	(tmp_path / timings.CACHE_NAME).write_text(
		"1\ngood 1.0 2 0\ngarbage\nbad x 2 0\nbad 1.0 2 x\n", encoding="utf-8"
	)
	assert timings.load(tmp_path) == {cache_key("good"): timings.TaskTiming(1.0, 2)}


def test_v0_cache_reads_as_whole_tree_observations(tmp_path: Path) -> None:
	"""Upgrading keeps the estimates ``camas --list`` shows: every V0 row was, by construction, an
	unscoped run.
	"""
	(tmp_path / timings.CACHE_NAME).write_text("0\nlint 0.5 2\n", encoding="utf-8")
	assert timings.load(tmp_path) == {cache_key("lint"): timings.TaskTiming(0.5, 2)}


def test_observations_at_different_scopes_do_not_mix(tmp_path: Path) -> None:
	"""#224's poisoning: one whole-tree run used to drag the running mean the scoped gate read."""
	timings.record(tmp_path, [(cache_key("lint", 0), 100.0), (cache_key("lint", 1), 0.5)])
	cache = timings.load(tmp_path)
	assert cache[cache_key("lint", 0)] == timings.TaskTiming(100.0, 1)
	assert cache[cache_key("lint", 1)] == timings.TaskTiming(0.5, 1)


def test_estimate_at_one_scope_ignores_anothers_observation() -> None:
	"""Only for a leaf whose cost can vary with the change — one that takes the paths."""
	scoped = Task("ruff {paths}", name="lint", paths=".")
	cache = {cache_key("lint", 0): timings.TaskTiming(100.0, 1)}
	assert timings.estimate(scoped, cache, 1) is None
	assert timings.estimate(scoped, cache, 0) is not None


def test_estimate_of_a_leaf_that_ignores_the_paths_reads_any_scope() -> None:
	"""#259 follow-through: a command with no ``{paths}`` runs identically whatever changed, so its
	one observation answers every scope instead of being re-learned per bucket.
	"""
	cache = {cache_key("lint", 0): timings.TaskTiming(100.0, 1)}
	assert timings.estimate(Task("ruff src", name="lint"), cache, 4) is not None


def test_record_observed_without_a_camas_dir_is_a_noop(tmp_path: Path) -> None:
	timings.record_observed(None, [(cache_key("lint"), 0.5)])
	timings.record_observed(tmp_path / "absent", [(cache_key("lint"), 0.5)])
	assert not (tmp_path / "absent").exists()


def test_record_observed_swallows_a_write_failure(tmp_path: Path) -> None:
	"""The run these durations describe has already finished, so a cache that cannot be written must
	not raise into the ``Stop`` hook that most often produced them.
	"""
	(tmp_path / timings.CACHE_NAME).mkdir()
	timings.record_observed(tmp_path, [(cache_key("lint"), 0.5)])


def test_load_drops_a_corrupted_non_finite_row(tmp_path: Path) -> None:
	"""A ``nan`` would stick permanently — it propagates through the running mean and compares false
	against every budget, so the leaf is over budget forever and never runs to correct itself.
	"""
	(tmp_path / timings.CACHE_NAME).write_text(
		"1\ngood 1.0 2 0\npoisoned nan 1 0\nendless inf 1 0\nunsampled 1.0 0 0\n", encoding="utf-8"
	)
	assert timings.load(tmp_path) == {cache_key("good"): timings.TaskTiming(1.0, 2)}


def test_load_treats_invalid_utf8_as_an_empty_cache(tmp_path: Path) -> None:
	"""``load`` promises an unreadable file is an empty cache; invalid bytes raise
	``UnicodeDecodeError``, which is a ``ValueError`` and not an ``OSError``.
	"""
	(tmp_path / timings.CACHE_NAME).write_bytes(b"1\nlint 0.5 2 0\n\xff\xfe not utf-8\n")
	assert timings.load(tmp_path) == {}


def test_a_label_a_row_cannot_carry_is_dropped_rather_than_misread(tmp_path: Path) -> None:
	"""A newline in a label fragments its row, and the tail re-parses as a genuine-looking
	observation under a truncated label — so it is refused at the write instead.
	"""
	timings.record(
		tmp_path,
		[
			(cache_key("real"), 0.5),
			(cache_key("two\nlines 9.0 9 0"), 1.0),
			(cache_key("trailing "), 2.0),
			(cache_key(" "), 3.0),
		],
	)
	assert timings.load(tmp_path) == {cache_key("real"): timings.TaskTiming(0.5, 1)}


def test_observed_identities_align_with_the_scoped_run_order() -> None:
	"""The identities tuple is parallel to the leaves the scoped tree runs, in the same order —
	the invariant every caller that threads identities into run() relies on."""
	tree = Parallel(
		Sequential(Task("ruff check {paths}", paths="."), Task("mypy .", name="types")),
		Task("pytest", name="test"),
	)
	changed = ("src/app.py",)
	keying = timings.observed(None, tree, changed)
	assert keying.node is not None
	assert [info.task.cmd for info in flatten_leaves(keying.node)] == [
		s.cmd for _original, s in scoped_leaves(tree, changed)
	]
	assert len(keying.identities) == len(list(flatten_leaves(keying.node)))


def test_leaves_of_prefers_the_carried_identity_over_the_label_mapping() -> None:
	"""A rewrite the mapping does not know (agent_format, passthrough) changes the reported
	label; the carried identity keeps the observation under the key a later budget reads."""
	key: Final = cache_key("ruff check .")
	result = _result(
		TaskResult(
			"claude-code-agent -- ruff check src/app.py",
			Finished(0, 1.5, ()),
			key,
		),
		elapsed=1.5,
	)
	keys: Final = {"ruff check src/app.py": cache_key("wrong")}
	assert timings.leaves_of(result, 0, keys) == [(key, 1.5)]
