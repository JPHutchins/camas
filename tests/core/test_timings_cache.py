# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING

from camas import Parallel, Sequential, Task
from camas.core import timings
from camas.core.completion import RunResult, TaskResult
from camas.v0.completion import Errored, Finished, Skipped, Stopped

if TYPE_CHECKING:
	from pathlib import Path


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
	timings.record_run(
		tmp_path,
		_result(
			TaskResult("lint", Finished(0, 0.1, ())),
			TaskResult("test", Finished(0, 2.0, ())),
			elapsed=2.1,
		),
	)
	cache = timings.load(tmp_path)
	assert cache["lint"] == timings.TaskTiming(elapsed_s=0.1, samples=1)
	assert cache["test"] == timings.TaskTiming(elapsed_s=2.0, samples=1)


def test_record_averages_repeated_runs(tmp_path: Path) -> None:
	timings.record(tmp_path, [("a", 1.0)])
	timings.record(tmp_path, [("a", 3.0)])
	assert timings.load(tmp_path)["a"] == timings.TaskTiming(elapsed_s=2.0, samples=2)


def test_record_counts_stopped_leaf(tmp_path: Path) -> None:
	timings.record_run(tmp_path, _result(TaskResult("x", Stopped(130, 0.3, ())), elapsed=0.3))
	assert timings.load(tmp_path)["x"].elapsed_s == 0.3


def test_record_skips_run_with_no_timed_leaf(tmp_path: Path) -> None:
	timings.record_run(tmp_path, _result(TaskResult("s", Skipped(1, "blk")), elapsed=0.0))
	assert timings.load(tmp_path) == {}
	assert not (tmp_path / timings.CACHE_NAME).exists()


def test_record_skips_errored_leaf(tmp_path: Path) -> None:
	timings.record_run(
		tmp_path,
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
	assert set(cache) == {"good"}
	assert cache["good"] == timings.TaskTiming(1.0, 2)


def test_load_ignores_unversioned_cache(tmp_path: Path) -> None:
	(tmp_path / timings.CACHE_NAME).write_text("lint 1.0 2\n", encoding="utf-8")
	assert timings.load(tmp_path) == {}


def test_leaf_name_with_spaces_round_trips(tmp_path: Path) -> None:
	timings.record(tmp_path, [("test [PY=3.13]", 3.0), ("test [PY=3.12]", 1.0)])
	cache = timings.load(tmp_path)
	assert cache["test [PY=3.13]"].elapsed_s == 3.0
	assert cache["test [PY=3.12]"].elapsed_s == 1.0


def test_record_is_versioned(tmp_path: Path) -> None:
	timings.record(tmp_path, [("lint", 0.5)])
	assert (tmp_path / timings.CACHE_NAME).read_text(encoding="utf-8").startswith("0\n")


def test_estimate_leaf_uses_its_own_timing() -> None:
	cache = {"lint": timings.TaskTiming(0.2, 1)}
	assert timings.estimate(Task("ruff", name="lint"), cache) == timings.Estimate(
		0.2, 1, "lint", 0.2
	)


def test_estimate_sequential_sums_children() -> None:
	cache = {"a": timings.TaskTiming(1.0, 3), "b": timings.TaskTiming(2.0, 1)}
	est = timings.estimate(Sequential(Task("x", name="a"), Task("y", name="b")), cache)
	assert est == timings.Estimate(elapsed_s=3.0, samples=1, slowest_leaf="b", slowest_s=2.0)


def test_estimate_parallel_takes_max() -> None:
	cache = {"a": timings.TaskTiming(1.0, 2), "b": timings.TaskTiming(2.0, 2)}
	est = timings.estimate(Parallel(Task("x", name="a"), Task("y", name="b")), cache)
	assert est == timings.Estimate(elapsed_s=2.0, samples=2, slowest_leaf="b", slowest_s=2.0)


def test_estimate_is_none_when_a_leaf_was_never_timed() -> None:
	cache = {"a": timings.TaskTiming(1.0, 1)}
	assert timings.estimate(Sequential(Task("x", name="a"), Task("y", name="b")), cache) is None
