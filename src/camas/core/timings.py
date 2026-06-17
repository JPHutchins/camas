# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``.camas/timings`` cache of observed per-leaf durations, composed into task estimates."""

from __future__ import annotations

import os
import sys
from typing import IO, TYPE_CHECKING, Final, NamedTuple

from ..v0.completion import Finished, Skipped, Stopped
from ..v0.task import Parallel, Sequential, Task
from .task import task_label

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence
	from pathlib import Path

	from ..v0.completion import Completion
	from ..v0.task import TaskNode
	from .completion import RunResult


CAMAS_DIR: Final = ".camas"
CACHE_NAME: Final = "timings"


class TaskTiming(NamedTuple):
	"""A leaf's mean observed duration and the number of runs that informed it."""

	elapsed_s: float
	samples: int


class Estimate(NamedTuple):
	"""A task's duration composed from the per-leaf cache, with its slowest leaf."""

	elapsed_s: float
	samples: int
	slowest_leaf: str
	slowest_s: float


def load(base: Path) -> dict[str, TaskTiming]:
	"""Read the cache under ``base``; an absent or unreadable file is an empty cache."""
	try:
		with (base / CAMAS_DIR / CACHE_NAME).open("r", encoding="utf-8") as handle:
			_flock(handle, exclusive=False)
			text = handle.read()
	except OSError:
		return {}
	return _parse_text(text)


def estimate(node: TaskNode, timings: Mapping[str, TaskTiming]) -> Estimate | None:
	"""Compose ``node``'s estimate from observed leaf durations: a leaf is its own timing,
	a Sequential the sum of its children, a Parallel their max. ``None`` when any leaf in
	the subtree has never been timed.
	"""
	match node:
		case Task():
			label = task_label(node)
			timing = timings.get(label)
			if timing is None:
				return None
			return Estimate(timing.elapsed_s, timing.samples, label, timing.elapsed_s)
		case Sequential(tasks=children):
			return _combine(children, timings, parallel=False)
		case Parallel(tasks=children):
			return _combine(children, timings, parallel=True)
		case _:
			assert_never(node)


def record(base: Path, leaves: Sequence[tuple[str, float]]) -> None:
	"""Fold a run's per-leaf durations into the cache, but only where ``.camas`` already
	exists; this never creates it.

	The read-modify-write runs under an exclusive lock on a single open handle, so the
	CLI and MCP (which share this path) can run concurrently without losing samples or
	reading a half-written cache. A run with no timed leaf records nothing.
	"""
	directory: Final = base / CAMAS_DIR
	if not leaves or not directory.is_dir():
		return
	observed: Final = dict(leaves)
	with _open_rw(directory / CACHE_NAME) as handle:
		_flock(handle, exclusive=True)
		cache = _parse_text(handle.read())
		merged = {**cache, **{n: _fold(cache.get(n), s) for n, s in observed.items()}}
		handle.seek(0)
		handle.truncate()
		handle.write("".join(_format(n, t) for n, t in sorted(merged.items())))


def record_run(base: Path, result: RunResult) -> None:
	"""Record a finished run's per-leaf durations."""
	record(base, _leaves(result))


def elapsed_of(completion: Completion) -> float | None:
	"""A completion's wall-clock seconds, or ``None`` when the leaf never ran."""
	match completion:
		case Finished(elapsed=elapsed) | Stopped(elapsed=elapsed):
			return elapsed
		case Skipped():
			return None
		case _:
			assert_never(completion)


def _combine(
	children: Sequence[TaskNode], timings: Mapping[str, TaskTiming], *, parallel: bool
) -> Estimate | None:
	"""Fold children's estimates: max for a Parallel, sum for a Sequential; ``None`` if any
	child is unknown. The slowest leaf is the worst across the whole subtree.
	"""
	parts: Final = [e for child in children if (e := estimate(child, timings)) is not None]
	if len(parts) != len(children):
		return None
	elapsed: Final = (
		max(p.elapsed_s for p in parts) if parallel else sum(p.elapsed_s for p in parts)
	)
	slowest: Final = max(parts, key=lambda p: p.slowest_s)
	return Estimate(elapsed, min(p.samples for p in parts), slowest.slowest_leaf, slowest.slowest_s)


def _open_rw(path: Path) -> IO[str]:
	"""Open ``path`` read-write for the locked update, creating it without truncating."""
	return os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o644), "r+", encoding="utf-8")


if sys.platform != "win32":
	import fcntl

	def _flock(handle: IO[str], *, exclusive: bool) -> None:
		"""Take an advisory ``flock`` on ``handle`` (POSIX)."""
		fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

else:  # pragma: no cover

	def _flock(handle: IO[str], *, exclusive: bool) -> None:
		"""Advisory file locking is POSIX-only; a no-op on Windows."""


def _fold(previous: TaskTiming | None, elapsed_s: float) -> TaskTiming:
	"""Fold a fresh sample into a leaf's running mean."""
	if previous is None:
		return TaskTiming(elapsed_s=elapsed_s, samples=1)
	samples: Final = previous.samples + 1
	return TaskTiming((previous.elapsed_s * previous.samples + elapsed_s) / samples, samples)


def _leaves(result: RunResult) -> list[tuple[str, float]]:
	return [(r.name, e) for r in result.results if (e := elapsed_of(r.completion)) is not None]


def _format(task: str, timing: TaskTiming) -> str:
	return f"{task} {timing.elapsed_s} {timing.samples}\n"


def _parse_text(text: str) -> dict[str, TaskTiming]:
	return dict(filter(None, (_parse(line) for line in text.splitlines())))


def _parse(line: str) -> tuple[str, TaskTiming] | None:
	parts = line.rsplit(maxsplit=2)
	if len(parts) != 3:
		return None
	task, elapsed_s, samples = parts
	try:
		return task, TaskTiming(elapsed_s=float(elapsed_s), samples=int(samples))
	except ValueError:
		return None
