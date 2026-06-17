# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING

from camas.core import timings
from camas.core.completion import RunResult, TaskResult
from camas.v0.completion import Finished, Skipped, Stopped

if TYPE_CHECKING:
	from pathlib import Path


def _result(*leaves: TaskResult, elapsed: float) -> RunResult:
	return RunResult(returncode=0, results=leaves, elapsed=elapsed)


def test_load_missing_is_empty(tmp_path: Path) -> None:
	assert timings.load(tmp_path) == {}


def test_record_is_noop_without_camas_dir(tmp_path: Path) -> None:
	timings.record_run(tmp_path, "x", _result(TaskResult("x", Finished(0, 0.5, ())), elapsed=0.5))
	assert timings.load(tmp_path) == {}
	assert not (tmp_path / ".camas").exists()


def test_record_run_writes_slowest_leaf_when_camas_exists(tmp_path: Path) -> None:
	(tmp_path / ".camas").mkdir()
	timings.record_run(
		tmp_path,
		"check",
		_result(
			TaskResult("lint", Finished(0, 0.1, ())),
			TaskResult("test", Finished(0, 2.0, ())),
			elapsed=2.1,
		),
	)
	entry = timings.load(tmp_path)["check"]
	assert entry == timings.TaskTiming(
		elapsed_s=2.1, samples=1, slowest_leaf="test", slowest_elapsed_s=2.0
	)


def test_record_increments_samples(tmp_path: Path) -> None:
	(tmp_path / ".camas").mkdir()
	run = _result(TaskResult("a", Finished(0, 0.5, ())), elapsed=0.5)
	timings.record_run(tmp_path, "a", run)
	timings.record_run(tmp_path, "a", run)
	assert timings.load(tmp_path)["a"].samples == 2


def test_record_counts_stopped_leaf(tmp_path: Path) -> None:
	(tmp_path / ".camas").mkdir()
	timings.record_run(tmp_path, "t", _result(TaskResult("x", Stopped(130, 0.3, ())), elapsed=0.3))
	assert timings.load(tmp_path)["t"].slowest_leaf == "x"


def test_record_skips_run_with_no_timed_leaf(tmp_path: Path) -> None:
	(tmp_path / ".camas").mkdir()
	timings.record_run(tmp_path, "t", _result(TaskResult("s", Skipped(1, "blk")), elapsed=0.0))
	assert timings.load(tmp_path) == {}


def test_load_skips_malformed_lines(tmp_path: Path) -> None:
	(tmp_path / ".camas").mkdir()
	(tmp_path / ".camas" / "timings").write_text(
		"good 1.0 2 0.9 leaf\ngarbage\nbad x 2 0.9 leaf\n", encoding="utf-8"
	)
	cache = timings.load(tmp_path)
	assert set(cache) == {"good"}
	assert cache["good"] == timings.TaskTiming(1.0, 2, "leaf", 0.9)


def test_slowest_leaf_name_with_spaces_round_trips(tmp_path: Path) -> None:
	(tmp_path / ".camas").mkdir()
	timings.record(tmp_path, "m", 3.0, [("test [PY=3.13]", 3.0), ("test [PY=3.12]", 1.0)])
	assert timings.load(tmp_path)["m"].slowest_leaf == "test [PY=3.13]"
