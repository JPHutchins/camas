# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import time

import pytest

from camas import Parallel, Sequential, Task, run

SLEEP = "import time; time.sleep({t})"


def _s(t: float, name: str) -> Task:
	return Task(("python", "-c", SLEEP.format(t=t)), name=name)


@pytest.mark.slow
@pytest.mark.parametrize(
	("task", "min_time", "max_time"),
	[
		(
			Parallel(_s(0.3, "a"), _s(0.3, "b"), _s(0.3, "c")),
			0.3,
			0.8,
		),
		(
			Sequential(_s(0.15, "a"), _s(0.15, "b")),
			0.3,
			0.8,
		),
		(
			Sequential(Parallel(_s(0.2, "p1"), _s(0.2, "p2")), _s(0.2, "after")),
			0.4,
			1.0,
		),
		(
			Parallel(Sequential(_s(0.15, "s1"), _s(0.15, "s2")), _s(0.15, "par")),
			0.3,
			0.8,
		),
		(
			Sequential(
				Parallel(_s(0.15, "a"), _s(0.15, "b")), Parallel(_s(0.15, "c"), _s(0.15, "d"))
			),
			0.3,
			0.8,
		),
		(
			Parallel(
				Sequential(_s(0.1, "s1a"), _s(0.1, "s1b")),
				Sequential(_s(0.1, "s2a"), _s(0.1, "s2b")),
				Sequential(_s(0.1, "s3a"), _s(0.1, "s3b")),
			),
			0.2,
			0.7,
		),
		(
			Sequential(
				Parallel(Sequential(_s(0.1, "deep1"), _s(0.1, "deep2")), _s(0.1, "shallow")),
				_s(0.1, "final"),
			),
			0.3,
			1.2,
		),
		(
			Parallel(
				Sequential(Parallel(_s(0.1, "a"), _s(0.1, "b")), _s(0.1, "c")),
				Sequential(_s(0.1, "d"), Parallel(_s(0.1, "e"), _s(0.1, "f"))),
			),
			0.2,
			0.7,
		),
	],
	ids=[
		"3_parallel",
		"2_sequential",
		"parallel_then_sequential",
		"sequential_alongside_parallel",
		"seq_of_two_parallel_groups",
		"3_parallel_sequences",
		"deep_nested_seq_par_seq",
		"mirror_nested_par_seq_par",
	],
)
def test_timing(task: Task | Parallel | Sequential, min_time: float, max_time: float) -> None:
	start = time.perf_counter()
	asyncio.run(run(task))
	elapsed = time.perf_counter() - start
	assert elapsed >= min_time, f"too fast: {elapsed:.3f}s < {min_time:.3f}s"
	assert elapsed <= max_time, f"too slow: {elapsed:.3f}s > {max_time:.3f}s"
