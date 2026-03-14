# Copyright (c) 2025 JP Hutchins
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from subprocess import STDOUT
from typing import NamedTuple


class Task(NamedTuple):
	name: str
	cmd: tuple[str, ...]
	extra_env: dict[str, str] | None = None


class OutputLine(NamedTuple):
	name: str
	line: str


class TaskDone(NamedTuple):
	name: str
	returncode: int
	elapsed: float
	output: str


type TaskEvent = OutputLine | TaskDone

CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
	("format", ("ruff", "format", "--check", "--verbose", ".")),
	("lint", ("ruff", "check", "--verbose", ".")),
	("mypy", ("mypy", "--verbose", ".")),
	("pyright", ("pyright", "src", "examples", "tests")),
	("test", ("pytest", "-v", "--doctest-modules", "--cov", "--cov-report=term-missing")),
)

PYTHON_VERSIONS = ("3.12", "3.13", "3.14")

VERSION_INDEPENDENT = frozenset({"format", "lint"})

BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
CLEAR_LINE = "\033[K"
SPINNER = (
	" ▄    ",
	"  ▄   ",
	"   ▄  ",
	"    ▄ ",
	"    ▐ ",
	"    ▀ ",
	"   ▀  ",
	"  ▀   ",
	" ▀    ",
	" ▌    ",
)


def _build_tasks(matrix: bool) -> tuple[Task, ...]:
	if not matrix:
		return tuple(Task(name, cmd) for name, cmd in CHECKS)
	tasks: list[Task] = []
	for name, cmd in CHECKS:
		if name in VERSION_INDEPENDENT:
			tasks.append(Task(name, cmd))
			continue
		for version in PYTHON_VERSIONS:
			version_cmd = ("uv", "run", "--all-packages", "--python", version, *cmd)
			if name == "test":
				version_cmd = (
					"uv",
					"run",
					"--all-packages",
					"--python",
					version,
					"pytest",
					"-v",
					"--doctest-modules",
				)
			tasks.append(
				Task(
					f"{name} {version}",
					version_cmd,
					extra_env={"UV_PROJECT_ENVIRONMENT": f".venv-{version}", "VIRTUAL_ENV": ""},
				)
			)
	return tuple(tasks)


async def _run_task(task: Task, queue: asyncio.Queue[TaskEvent]) -> None:
	start = time.monotonic()
	env = {**os.environ, **task.extra_env} if task.extra_env else None
	proc = await asyncio.create_subprocess_exec(
		*task.cmd,
		stdout=asyncio.subprocess.PIPE,
		stderr=STDOUT,
		env=env,
	)
	output_lines: list[str] = []
	assert proc.stdout is not None
	async for raw in proc.stdout:
		line = raw.decode(errors="replace").rstrip()
		output_lines.append(line)
		if line.strip():
			await queue.put(OutputLine(task.name, line))
	await proc.wait()
	await queue.put(
		TaskDone(
			task.name,
			proc.returncode or 0,
			time.monotonic() - start,
			"\n".join(output_lines),
		)
	)


def _truncate_middle(text: str, max_width: int) -> str:
	if len(text) <= max_width:
		return text
	side = (max_width - 3) // 2
	return text[:side] + "..." + text[len(text) - (max_width - 3 - side) :]


def _render_frame(
	tasks: tuple[Task, ...],
	last_lines: dict[str, str],
	start_times: dict[str, float],
	completed: dict[str, TaskDone],
	term_width: int,
	name_width: int,
	now: float,
) -> str:
	n = len(tasks)
	prefix_width = name_width + 21
	lines: list[str] = []
	for task in tasks:
		name_col = f"{BOLD}{task.name:>{name_width}}{RESET}"
		if task.name in completed:
			done = completed[task.name]
			color = GREEN if done.returncode == 0 else RED
			label = " PASS " if done.returncode == 0 else " FAIL "
			lines.append(f"\r{name_col} [{color}{label}{RESET}] {done.elapsed:5.1f}s{CLEAR_LINE}")
		else:
			elapsed = now - start_times[task.name]
			spin = SPINNER[int(elapsed * 10) % len(SPINNER)]
			detail = _truncate_middle(last_lines.get(task.name, ""), term_width - prefix_width)
			lines.append(
				f"\r{name_col} [{YELLOW}{spin}{RESET}] {elapsed:5.1f}s  {detail}{CLEAR_LINE}"
			)
	return f"\033[{n}F" + "\n".join(lines) + "\n"


def _process_event(
	event: TaskEvent,
	last_lines: dict[str, str],
	completed: dict[str, TaskDone],
) -> None:
	match event:
		case OutputLine(name, line):
			last_lines[name] = line
		case TaskDone() as done:
			completed[done.name] = done


async def _render_loop(
	tasks: tuple[Task, ...],
	queue: asyncio.Queue[TaskEvent],
) -> dict[str, TaskDone]:
	term_width = shutil.get_terminal_size().columns
	name_width = max(len(t.name) for t in tasks)
	last_lines: dict[str, str] = {}
	start_times: dict[str, float] = {t.name: time.monotonic() for t in tasks}
	completed: dict[str, TaskDone] = {}

	sys.stdout.write("\n" * len(tasks))
	sys.stdout.flush()

	while len(completed) < len(tasks):
		try:
			_process_event(await asyncio.wait_for(queue.get(), timeout=0.1), last_lines, completed)
		except TimeoutError:
			pass
		while not queue.empty():
			_process_event(queue.get_nowait(), last_lines, completed)
		sys.stdout.write(
			_render_frame(
				tasks, last_lines, start_times, completed, term_width, name_width, time.monotonic()
			)
		)
		sys.stdout.flush()

	return completed


def _print_failures(tasks: tuple[Task, ...], completed: dict[str, TaskDone]) -> None:
	for task in tasks:
		done = completed[task.name]
		if done.returncode != 0:
			sys.stdout.write(f"\n{RED}{'=' * 60}{RESET}\n")
			sys.stdout.write(f"{RED}{BOLD} FAILED: {done.name} {RESET}\n")
			sys.stdout.write(f"{RED}{'=' * 60}{RESET}\n")
			sys.stdout.write(done.output)
			sys.stdout.write("\n")


async def _run(tasks: tuple[Task, ...]) -> int:
	queue: asyncio.Queue[TaskEvent] = asyncio.Queue()
	async with asyncio.TaskGroup() as tg:
		for task in tasks:
			tg.create_task(_run_task(task, queue))
		completed = await _render_loop(tasks, queue)
	_print_failures(tasks, completed)
	return 1 if any(d.returncode != 0 for d in completed.values()) else 0


def main() -> None:
	tasks = _build_tasks(matrix="--matrix" in sys.argv)
	sys.exit(asyncio.run(_run(tasks)))


if __name__ == "__main__":
	main()
