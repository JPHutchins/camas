from __future__ import annotations

import asyncio

import cyclopts
import pytest

from camas import Parallel, Sequential, Task, run
from camas.effect.termtree import Termtree, TermtreeOptions

app = cyclopts.App()

SLEEP = "import time; time.sleep({t})"
LOREM = (
	"import time\n"
	"lines = {lines!r}\n"
	"per_line = max({t} / len(lines), 0.01)\n"
	"for line in lines:\n"
	"    print(line, flush=True)\n"
	"    time.sleep(per_line)\n"
)
FAIL = "import time; time.sleep({t}); raise SystemExit(1)"


def sleep_task(t: float, name: str) -> Task:
	return Task(("python", "-c", SLEEP.format(t=t)), name=name)


def lorem_task(t: float, name: str, lines: tuple[str, ...]) -> Task:
	return Task(("python", "-c", LOREM.format(t=t, lines=lines)), name=name)


def fail_task(t: float, name: str) -> Task:
	return Task(("python", "-c", FAIL.format(t=t)), name=name)


@app.command
@pytest.mark.slow
def test_demo_parallel_5_tasks() -> None:
	task = Parallel(
		tasks=(
			sleep_task(0.5, "format"),
			lorem_task(0.8, "lint", ("scanning src/", "scanning tests/", "All clean!")),
			sleep_task(1.2, "mypy"),
			lorem_task(
				1.0,
				"pyright",
				(
					"analysing 42 files",
					"type-check pass 1",
					"type-check pass 2",
					"0 errors, 0 warnings",
				),
			),
			sleep_task(1.5, "test"),
		)
	)
	assert asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),))).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_parallel_with_failure() -> None:
	task = Parallel(
		tasks=(
			sleep_task(0.5, "format"),
			fail_task(0.8, "lint"),
			sleep_task(1.2, "mypy"),
			sleep_task(0.3, "pyright"),
			sleep_task(1.0, "test"),
		)
	)
	assert asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),))).returncode == 1


@app.command
@pytest.mark.slow
def test_demo_sequential_3_steps() -> None:
	task = Sequential(
		tasks=(
			lorem_task(
				0.5, "build", ("resolving deps", "compiling 42 modules", "compiled 42 modules")
			),
			lorem_task(
				0.8,
				"test",
				("collecting tests", "running unit tests", "running integration", "12 passed"),
			),
			lorem_task(
				0.3, "deploy", ("uploading artifact", "restarting service", "deployed to staging")
			),
		)
	)
	assert asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),))).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_sequential_fail_skips() -> None:
	task = Sequential(
		tasks=(
			sleep_task(0.3, "build"),
			fail_task(0.5, "test"),
			sleep_task(0.3, "deploy"),
			sleep_task(0.3, "notify"),
		)
	)
	assert asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),))).returncode == 1


@app.command
@pytest.mark.slow
def test_demo_nested_parallel_in_sequential() -> None:
	task = Sequential(
		tasks=(
			Parallel(
				tasks=(
					sleep_task(0.5, "format"),
					sleep_task(0.6, "lint"),
				)
			),
			Parallel(
				tasks=(
					sleep_task(0.8, "mypy"),
					sleep_task(0.7, "pyright"),
				)
			),
			sleep_task(1.0, "test"),
		)
	)
	assert asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),))).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_nested_sequential_in_parallel() -> None:
	task = Parallel(
		tasks=(
			Sequential(
				tasks=(
					lorem_task(0.4, "build-lib", ("resolving deps", "compiling lib", "lib built")),
					lorem_task(
						0.6, "test-lib", ("running tests", "42 tests run", "lib tests passed")
					),
				)
			),
			Sequential(
				tasks=(
					lorem_task(0.3, "build-app", ("compiling app", "linking", "app built")),
					lorem_task(
						0.8,
						"test-app",
						("unit tests", "integration tests", "e2e tests", "app tests passed"),
					),
				)
			),
			lorem_task(0.5, "lint", ("scanning", "checking imports", "no issues")),
		)
	)
	assert asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),))).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_matrix_python_versions() -> None:
	task = Parallel(
		tasks=(sleep_task(0.8, "test {PY}"),),
		matrix={"PY": ("3.12", "3.13", "3.14")},
	)
	assert asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),))).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_matrix_2d() -> None:
	task = Parallel(
		tasks=(sleep_task(0.6, "{DB}/{OPT}"),),
		matrix={"DB": ("sqlite", "postgres"), "OPT": ("debug", "release")},
	)
	assert asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),))).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_sequential_matrix_clones() -> None:
	task = Sequential(
		tasks=(
			lorem_task(0.3, "build {PY}", ("fetching {PY}", "compiling", "built for {PY}")),
			lorem_task(0.5, "test {PY}", ("collecting", "running", "tested on {PY}")),
		),
		name="ci",
		matrix={"PY": ("3.12", "3.13", "3.14")},
	)
	assert asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),))).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_deeply_nested_with_output() -> None:
	task = Sequential(
		tasks=(
			Parallel(
				tasks=(
					Sequential(
						tasks=(
							lorem_task(0.3, "compile", ("parsing", "typechecking", "compiling...")),
							lorem_task(
								0.4, "link", ("resolving symbols", "merging objects", "linking...")
							),
						)
					),
					lorem_task(0.5, "lint", ("reading config", "walking AST", "checking style...")),
				)
			),
			Parallel(
				tasks=(
					lorem_task(
						0.6, "unit-tests", ("collecting", "running 42 tests", "42 tests passed")
					),
					lorem_task(
						0.8,
						"integration",
						("booting fixture", "scenario 1", "scenario 4", "7 scenarios ok"),
					),
					lorem_task(0.4, "e2e", ("spinning browser", "flow 1", "3 flows verified")),
				)
			),
			lorem_task(0.2, "package", ("tarring", "artifact created")),
		)
	)
	assert asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),))).returncode == 0


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
							lorem_task(0.3, "compile", ("parsing", "typechecking", "compiling...")),
							lorem_task(
								0.4, "link", ("resolving symbols", "merging objects", "linking...")
							),
						)
					),
					lorem_task(0.5, "lint", ("reading config", "walking AST", "checking style...")),
				),
			),
			Parallel(
				name="Test",
				tasks=(
					lorem_task(
						0.6, "unit-tests", ("collecting", "running 42 tests", "42 tests passed")
					),
					lorem_task(
						0.8,
						"integration",
						("booting fixture", "scenario 1", "scenario 4", "7 scenarios ok"),
					),
					lorem_task(0.4, "e2e", ("spinning browser", "flow 1", "3 flows verified")),
				),
			),
			lorem_task(0.2, "package", ("tarring", "artifact created")),
		),
	)
	assert asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),))).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_matrix_full_ci_pipeline() -> None:
	task = Sequential(
		tasks=(
			lorem_task(0.2, "format {PY}", ("scanning", "formatted")),
			lorem_task(0.3, "lint {PY}", ("parsing", "checking", "no issues")),
			Parallel(
				tasks=(
					lorem_task(0.4, "mypy {PY}", ("building index", "checking types", "0 errors")),
					lorem_task(0.5, "pyright {PY}", ("analysing", "resolving", "0 errors")),
				)
			),
			lorem_task(0.6, "test {PY}", ("collecting", "running", "42 passed")),
		),
		name="check",
		matrix={"PY": ("3.13", "3.14")},
	)
	assert asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),))).returncode == 0


@app.command
@pytest.mark.slow
def test_demo_matrix_2d_with_nesting() -> None:
	task = Sequential(
		tasks=(
			lorem_task(0.2, "build {DB}/{OPT}", ("prepping", "compiling", "built")),
			Parallel(
				tasks=(
					lorem_task(0.3, "unit {DB}/{OPT}", ("setup", "running", "passed")),
					lorem_task(
						0.4, "integ {DB}/{OPT}", ("setup", "scenario 1", "scenario 2", "passed")
					),
				)
			),
		),
		name="ci",
		matrix={"DB": ("sqlite", "postgres"), "OPT": ("debug", "release")},
	)
	assert asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),))).returncode == 0


if __name__ == "__main__":
	app()
