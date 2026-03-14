from __future__ import annotations

import time

import pytest

from camas import Parallel, Sequential, Task, run

# --- Single command ---


def test_single_cmd_success() -> None:
	assert run(Task(("python", "-c", "pass"))) == 0


def test_single_cmd_failure() -> None:
	assert run(Task(("python", "-c", "raise SystemExit(1)"))) == 1


def test_cmd_string_form() -> None:
	assert run(Task("python -c pass")) == 0


def test_cmd_env_passed() -> None:
	task = Task(
		(
			"python",
			"-c",
			"import os; raise SystemExit(0 if os.environ.get('MY_VAR')=='42' else 1)",
		),
		env={"MY_VAR": "42"},
	)
	assert run(task) == 0


# --- Parallel ---


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
		tasks=tuple(
			Task(("python", "-c", f"raise SystemExit({rc})"), name=f"t{i}")
			for i, rc in enumerate(returncodes)
		)
	)
	assert run(task) == expected_exit


# --- Sequential ---


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
		tasks=tuple(
			Task(("python", "-c", f"raise SystemExit({rc})"), name=f"t{i}")
			for i, rc in enumerate(returncodes)
		)
	)
	assert run(task) == expected_exit


# --- Nesting ---


def test_nested_parallel_in_sequential() -> None:
	task = Sequential(
		tasks=(
			Parallel(
				tasks=(
					Task(("python", "-c", "pass"), name="p1"),
					Task(("python", "-c", "pass"), name="p2"),
				)
			),
			Task(("python", "-c", "pass"), name="after"),
		)
	)
	assert run(task) == 0


def test_nested_sequential_in_parallel() -> None:
	task = Parallel(
		tasks=(
			Sequential(
				tasks=(
					Task(("python", "-c", "pass"), name="s1"),
					Task(("python", "-c", "pass"), name="s2"),
				)
			),
			Task(("python", "-c", "pass"), name="par"),
		)
	)
	assert run(task) == 0


def test_deeply_nested() -> None:
	task = Sequential(
		tasks=(
			Parallel(
				tasks=(
					Sequential(
						tasks=(
							Task(("python", "-c", "pass"), name="deep1"),
							Task(("python", "-c", "pass"), name="deep2"),
						)
					),
					Task(("python", "-c", "pass"), name="shallow"),
				)
			),
			Task(("python", "-c", "pass"), name="final"),
		)
	)
	assert run(task) == 0


# --- Matrix integration ---


def test_matrix_env_reaches_subprocess() -> None:
	task = Parallel(
		tasks=(
			Task(
				(
					"python",
					"-c",
					"import os; raise SystemExit(0 if os.environ.get('X')=='1' else 1)",
				),
			),
		),
		matrix={"X": ("1",)},
	)
	assert run(task) == 0


def test_matrix_substitution_in_cmd() -> None:
	task = Parallel(
		tasks=(
			Task(
				(
					"python",
					"-c",
					"import sys; raise SystemExit(0 if '{V}' == 'hello' else 1)",
				),
			),
		),
		matrix={"V": ("hello",)},
	)
	assert run(task) == 0


# --- Timing ---


def test_parallel_concurrency() -> None:
	task = Parallel(
		tasks=(
			Task(("python", "-c", "import time; time.sleep(0.2)"), name="a"),
			Task(("python", "-c", "import time; time.sleep(0.2)"), name="b"),
			Task(("python", "-c", "import time; time.sleep(0.2)"), name="c"),
		)
	)
	start = time.perf_counter()
	run(task)
	elapsed = time.perf_counter() - start
	assert elapsed < 1.0


def test_sequential_ordering() -> None:
	task = Sequential(
		tasks=(
			Task(("python", "-c", "import time; time.sleep(0.1)"), name="a"),
			Task(("python", "-c", "import time; time.sleep(0.1)"), name="b"),
		)
	)
	start = time.perf_counter()
	run(task)
	elapsed = time.perf_counter() - start
	assert elapsed >= 0.2
