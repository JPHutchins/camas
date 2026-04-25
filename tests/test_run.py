# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import time

import pytest

from camas import Finished, Parallel, Sequential, Task, run


def test_force_color_injected_in_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("NO_COLOR", raising=False)
	task = Task(
		("python", "-c", "import os,sys; sys.exit(0 if os.environ.get('FORCE_COLOR')=='1' else 1)")
	)
	assert asyncio.run(run(task)).returncode == 0


def test_no_color_suppresses_force_color(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("NO_COLOR", "1")
	task = Task(
		("python", "-c", "import os,sys; sys.exit(0 if 'FORCE_COLOR' not in os.environ else 1)")
	)
	assert asyncio.run(run(task)).returncode == 0


def test_single_cmd_success() -> None:
	assert asyncio.run(run(Task(("python", "-c", "pass")))).returncode == 0


def test_single_cmd_failure() -> None:
	assert asyncio.run(run(Task(("python", "-c", "raise SystemExit(1)")))).returncode == 1


def test_cmd_string_form() -> None:
	assert asyncio.run(run(Task("python -c pass"))).returncode == 0


def test_cmd_env_passed() -> None:
	task = Task(
		(
			"python",
			"-c",
			"import os; raise SystemExit(0 if os.environ.get('MY_VAR')=='42' else 1)",
		),
		env={"MY_VAR": "42"},
	)
	assert asyncio.run(run(task)).returncode == 0


@pytest.mark.parametrize(
	("returncodes", "expected_exit"),
	[
		((0, 0, 0), 0),
		((0, 1, 0), 1),
		((1, 1, 1), 1),
	],
)
def test_parallel_returncodes(returncodes: tuple[int, ...], expected_exit: int) -> None:
	task = Parallel(
		*(
			Task(("python", "-c", f"raise SystemExit({rc})"), name=f"t{i}")
			for i, rc in enumerate(returncodes)
		)
	)
	assert asyncio.run(run(task)).returncode == expected_exit


@pytest.mark.parametrize(
	("returncodes", "expected_exit"),
	[
		((0, 0, 0), 0),
		((1, 0, 0), 1),
		((0, 2, 0), 1),
		((0, 0, 1), 1),
	],
)
def test_sequential_returncodes(returncodes: tuple[int, ...], expected_exit: int) -> None:
	task = Sequential(
		*(
			Task(("python", "-c", f"raise SystemExit({rc})"), name=f"t{i}")
			for i, rc in enumerate(returncodes)
		)
	)
	assert asyncio.run(run(task)).returncode == expected_exit


def test_nested_parallel_in_sequential() -> None:
	task = Sequential(
		Parallel(
			Task(("python", "-c", "pass"), name="p1"), Task(("python", "-c", "pass"), name="p2")
		),
		Task(("python", "-c", "pass"), name="after"),
	)
	assert asyncio.run(run(task)).returncode == 0


def test_nested_sequential_in_parallel() -> None:
	task = Parallel(
		Sequential(
			Task(("python", "-c", "pass"), name="s1"), Task(("python", "-c", "pass"), name="s2")
		),
		Task(("python", "-c", "pass"), name="par"),
	)
	assert asyncio.run(run(task)).returncode == 0


def test_deeply_nested() -> None:
	task = Sequential(
		Parallel(
			Sequential(
				Task(("python", "-c", "pass"), name="deep1"),
				Task(("python", "-c", "pass"), name="deep2"),
			),
			Task(("python", "-c", "pass"), name="shallow"),
		),
		Task(("python", "-c", "pass"), name="final"),
	)
	assert asyncio.run(run(task)).returncode == 0


def test_matrix_env_reaches_subprocess() -> None:
	task = Parallel(
		Task(
			(
				"python",
				"-c",
				"import os; raise SystemExit(0 if os.environ.get('X')=='1' else 1)",
			),
		),
		matrix={"X": ("1",)},
	)
	assert asyncio.run(run(task)).returncode == 0


def test_matrix_substitution_in_cmd() -> None:
	task = Parallel(
		Task(
			(
				"python",
				"-c",
				"import sys; raise SystemExit(0 if '{V}' == 'hello' else 1)",
			),
		),
		matrix={"V": ("hello",)},
	)
	assert asyncio.run(run(task)).returncode == 0


def test_parallel_concurrency() -> None:
	task = Parallel(
		Task(("python", "-c", "import time; time.sleep(1.0)"), name="a"),
		Task(("python", "-c", "import time; time.sleep(1.0)"), name="b"),
		Task(("python", "-c", "import time; time.sleep(1.0)"), name="c"),
	)
	start = time.perf_counter()
	asyncio.run(run(task))
	elapsed = time.perf_counter() - start
	# Concurrent: ~1.0s + spawn overhead. Sequential would be ≥3.0s.
	assert elapsed < 2.5


def test_sequential_ordering() -> None:
	task = Sequential(
		Task(("python", "-c", "import time; time.sleep(0.1)"), name="a"),
		Task(("python", "-c", "import time; time.sleep(0.1)"), name="b"),
	)
	start = time.perf_counter()
	asyncio.run(run(task))
	elapsed = time.perf_counter() - start
	assert elapsed >= 0.2


def test_task_with_stdout_output() -> None:
	task = Task(("python", "-c", "print('hello world')"), name="printer")
	result = asyncio.run(run(task))
	assert result.returncode == 0
	assert any(
		b"hello world" in line
		for r in result.results
		if isinstance(r.completion, Finished)
		for line in r.completion.output
	)


def test_sequential_skip_nested_group() -> None:
	task = Sequential(
		Task(("python", "-c", "raise SystemExit(1)"), name="fail"),
		Parallel(
			Task(("python", "-c", "pass"), name="skipped1"),
			Task(("python", "-c", "pass"), name="skipped2"),
		),
	)
	result = asyncio.run(run(task))
	assert result.returncode == 1


def test_matrix_nested_sequential_in_parallel() -> None:
	task = Parallel(
		Sequential(
			Task(("python", "-c", "pass"), name="a"), Task(("python", "-c", "pass"), name="b")
		),
		matrix={"X": ("1", "2")},
	)
	result = asyncio.run(run(task))
	assert result.returncode == 0
	assert len(result.results) == 4


def test_run_result_has_structured_results() -> None:
	task = Parallel(
		Task(("python", "-c", "pass"), name="a"),
		Task(("python", "-c", "raise SystemExit(1)"), name="b"),
	)
	result = asyncio.run(run(task))
	assert result.returncode == 1
	assert len(result.results) == 2
	assert result.elapsed > 0
	names = {r.name for r in result.results}
	assert names == {"a", "b"}
