# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``.camas/timings`` cache of per-task run durations, read back by the CLI and MCP."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Final, NamedTuple

from ..v0.completion import Finished, Skipped, Stopped

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

if TYPE_CHECKING:
	from collections.abc import Sequence
	from pathlib import Path

	from ..v0.completion import Completion
	from .completion import RunResult


CAMAS_DIR: Final = ".camas"
CACHE_NAME: Final = "timings"


class TaskTiming(NamedTuple):
	"""A task's last observed run duration, its slowest leaf, and how many runs informed it."""

	elapsed_s: float
	samples: int
	slowest_leaf: str
	slowest_elapsed_s: float


def load(base: Path) -> dict[str, TaskTiming]:
	"""Read the cache under ``base``; an absent or unreadable file is an empty cache."""
	try:
		text = (base / CAMAS_DIR / CACHE_NAME).read_text(encoding="utf-8")
	except OSError:
		return {}
	return dict(filter(None, (_parse(line) for line in text.splitlines())))


def record(base: Path, task: str, elapsed_s: float, leaves: Sequence[tuple[str, float]]) -> None:
	"""Merge ``task``'s duration into the cache, but only where ``.camas`` already exists.

	A run with no leaf timed, or a project that has not opted in by having a ``.camas``
	directory, records nothing — this never creates ``.camas``.
	"""
	directory = base / CAMAS_DIR
	if not leaves or not directory.is_dir():
		return
	slowest_leaf, slowest_elapsed_s = max(leaves, key=lambda leaf: leaf[1])
	cache = load(base)
	previous = cache.get(task)
	cache[task] = TaskTiming(
		elapsed_s=elapsed_s,
		samples=previous.samples + 1 if previous is not None else 1,
		slowest_leaf=slowest_leaf,
		slowest_elapsed_s=slowest_elapsed_s,
	)
	(directory / CACHE_NAME).write_text(
		"".join(_format(name, timing) for name, timing in sorted(cache.items())), encoding="utf-8"
	)


def record_run(base: Path, task: str, result: RunResult) -> None:
	"""Record ``task`` from a finished run's per-leaf durations."""
	record(base, task, result.elapsed, _leaves(result))


def elapsed_of(completion: Completion) -> float | None:
	"""A completion's wall-clock seconds, or ``None`` when the leaf never ran."""
	match completion:
		case Finished(elapsed=elapsed) | Stopped(elapsed=elapsed):
			return elapsed
		case Skipped():
			return None
		case _:
			assert_never(completion)


def _leaves(result: RunResult) -> list[tuple[str, float]]:
	return [(r.name, e) for r in result.results if (e := elapsed_of(r.completion)) is not None]


def _format(task: str, timing: TaskTiming) -> str:
	return (
		f"{task} {timing.elapsed_s} {timing.samples} "
		f"{timing.slowest_elapsed_s} {timing.slowest_leaf}\n"
	)


def _parse(line: str) -> tuple[str, TaskTiming] | None:
	parts = line.split(maxsplit=4)
	if len(parts) != 5:
		return None
	task, elapsed_s, samples, slowest_elapsed_s, slowest_leaf = parts
	try:
		return task, TaskTiming(
			elapsed_s=float(elapsed_s),
			samples=int(samples),
			slowest_leaf=slowest_leaf,
			slowest_elapsed_s=float(slowest_elapsed_s),
		)
	except ValueError:
		return None
