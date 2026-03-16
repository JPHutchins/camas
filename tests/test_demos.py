from __future__ import annotations

import asyncio

import cyclopts
import pytest

from camas import Parallel, Sequential, Task, run

app = cyclopts.App()

SLEEP = "import time; time.sleep({t})"
LOREM = "import time; time.sleep({t}); print('{msg}')"
FAIL = "import time; time.sleep({t}); raise SystemExit(1)"


def _sleep(t: float, name: str) -> Task:
	return Task(("python", "-c", SLEEP.format(t=t)), name=name)


def _lorem(t: float, name: str, msg: str) -> Task:
	return Task(("python", "-c", LOREM.format(t=t, msg=msg)), name=name)


def _fail(t: float, name: str) -> Task:
	return Task(("python", "-c", FAIL.format(t=t)), name=name)


@app.command
@pytest.mark.slow
def test_demo_parallel_5_tasks() -> None:
	task = Parallel(
		tasks=(
			_sleep(0.5, "format"),
			_lorem(0.8, "lint", "All clean!"),
			_sleep(1.2, "mypy"),
			_lorem(1.0, "pyright", "0 errors, 0 warnings"),
			_sleep(1.5, "test"),
		)
	)
	assert asyncio.run(run(task)).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_parallel_with_failure() -> None:
	task = Parallel(
		tasks=(
			_sleep(0.5, "format"),
			_fail(0.8, "lint"),
			_sleep(1.2, "mypy"),
			_sleep(0.3, "pyright"),
			_sleep(1.0, "test"),
		)
	)
	assert asyncio.run(run(task)).returncode == 1


@app.command
@pytest.mark.slow
def test_demo_sequential_3_steps() -> None:
	task = Sequential(
		tasks=(
			_lorem(0.5, "build", "compiled 42 modules"),
			_lorem(0.8, "test", "12 passed"),
			_lorem(0.3, "deploy", "deployed to staging"),
		)
	)
	assert asyncio.run(run(task)).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_sequential_fail_skips() -> None:
	task = Sequential(
		tasks=(
			_sleep(0.3, "build"),
			_fail(0.5, "test"),
			_sleep(0.3, "deploy"),
			_sleep(0.3, "notify"),
		)
	)
	assert asyncio.run(run(task)).returncode == 1


@app.command
@pytest.mark.slow
def test_demo_nested_parallel_in_sequential() -> None:
	task = Sequential(
		tasks=(
			Parallel(
				tasks=(
					_sleep(0.5, "format"),
					_sleep(0.6, "lint"),
				)
			),
			Parallel(
				tasks=(
					_sleep(0.8, "mypy"),
					_sleep(0.7, "pyright"),
				)
			),
			_sleep(1.0, "test"),
		)
	)
	assert asyncio.run(run(task)).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_nested_sequential_in_parallel() -> None:
	task = Parallel(
		tasks=(
			Sequential(
				tasks=(
					_lorem(0.4, "build-lib", "lib built"),
					_lorem(0.6, "test-lib", "lib tests passed"),
				)
			),
			Sequential(
				tasks=(
					_lorem(0.3, "build-app", "app built"),
					_lorem(0.8, "test-app", "app tests passed"),
				)
			),
			_lorem(0.5, "lint", "no issues"),
		)
	)
	assert asyncio.run(run(task)).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_matrix_python_versions() -> None:
	task = Parallel(
		tasks=(_sleep(0.8, "test {PY}"),),
		matrix={"PY": ("3.12", "3.13", "3.14")},
	)
	assert asyncio.run(run(task)).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_matrix_2d() -> None:
	task = Parallel(
		tasks=(_sleep(0.6, "{DB}/{OPT}"),),
		matrix={"DB": ("sqlite", "postgres"), "OPT": ("debug", "release")},
	)
	assert asyncio.run(run(task)).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_sequential_matrix_clones() -> None:
	task = Sequential(
		tasks=(
			_lorem(0.3, "build {PY}", "built for {PY}"),
			_lorem(0.5, "test {PY}", "tested on {PY}"),
		),
		name="ci",
		matrix={"PY": ("3.12", "3.13", "3.14")},
	)
	assert asyncio.run(run(task)).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_deeply_nested_with_output() -> None:
	task = Sequential(
		tasks=(
			Parallel(
				tasks=(
					Sequential(
						tasks=(
							_lorem(0.3, "compile", "compiling..."),
							_lorem(0.4, "link", "linking..."),
						)
					),
					_lorem(0.5, "lint", "checking style..."),
				)
			),
			Parallel(
				tasks=(
					_lorem(0.6, "unit-tests", "42 tests passed"),
					_lorem(0.8, "integration", "7 scenarios ok"),
					_lorem(0.4, "e2e", "3 flows verified"),
				)
			),
			_lorem(0.2, "package", "artifact created"),
		)
	)
	assert asyncio.run(run(task)).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_deeply_nested_with_output_named() -> None:
	task = Sequential(
		name="My App",
		tasks=(
			Parallel(
				name="Build",
				tasks=(
					Sequential(
						tasks=(
							_lorem(0.3, "compile", "compiling..."),
							_lorem(0.4, "link", "linking..."),
						)
					),
					_lorem(0.5, "lint", "checking style..."),
				),
			),
			Parallel(
				name="Test",
				tasks=(
					_lorem(0.6, "unit-tests", "42 tests passed"),
					_lorem(0.8, "integration", "7 scenarios ok"),
					_lorem(0.4, "e2e", "3 flows verified"),
				),
			),
			_lorem(0.2, "package", "artifact created"),
		),
	)
	assert asyncio.run(run(task)).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_matrix_full_ci_pipeline() -> None:
	task = Sequential(
		tasks=(
			_lorem(0.2, "format {PY}", "formatted"),
			_lorem(0.3, "lint {PY}", "no issues"),
			Parallel(
				tasks=(
					_lorem(0.4, "mypy {PY}", "0 errors"),
					_lorem(0.5, "pyright {PY}", "0 errors"),
				)
			),
			_lorem(0.6, "test {PY}", "42 passed"),
		),
		name="check",
		matrix={"PY": ("3.13", "3.14")},
	)
	assert asyncio.run(run(task)).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_matrix_2d_with_nesting() -> None:
	task = Sequential(
		tasks=(
			_lorem(0.2, "build {DB}/{OPT}", "built"),
			Parallel(
				tasks=(
					_lorem(0.3, "unit {DB}/{OPT}", "passed"),
					_lorem(0.4, "integ {DB}/{OPT}", "passed"),
				)
			),
		),
		name="ci",
		matrix={"DB": ("sqlite", "postgres"), "OPT": ("debug", "release")},
	)
	assert asyncio.run(run(task)).returncode == 0


if __name__ == "__main__":
	app()
