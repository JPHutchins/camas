# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``timings.txt`` cache of observed per-leaf durations, composed into task estimates."""

from __future__ import annotations

import os
import sys
from enum import IntEnum
from typing import IO, TYPE_CHECKING, Final, NamedTuple, TypeAlias

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


CACHE_NAME: Final = "timings.txt"


class CacheVersion(IntEnum):
	"""Versions of the ``timings.txt`` format; the file's first line is the writer's version."""

	V0 = 0


TaskLabel: TypeAlias = str
"""A leaf's :func:`task_label` — its ``name`` or its joined command — the cache key."""


class TaskTiming(NamedTuple):
	"""A leaf's mean observed duration and the number of runs that informed it."""

	elapsed_s: float
	samples: int

	def fold(self, elapsed_s: float) -> TaskTiming:
		"""This leaf's running mean after one more observation."""
		samples = self.samples + 1
		return TaskTiming((self.elapsed_s * self.samples + elapsed_s) / samples, samples)


class Estimate(NamedTuple):
	"""A task's duration composed from the per-leaf cache, with its slowest leaf."""

	elapsed_s: float
	samples: int
	slowest_leaf: TaskLabel
	slowest_s: float


def load(camas_dir: Path) -> dict[TaskLabel, TaskTiming]:
	"""Read the cache in ``camas_dir``; an absent or unreadable file is an empty cache."""
	try:
		with (camas_dir / CACHE_NAME).open("r", encoding="utf-8") as handle:
			lock(handle, exclusive=False)
			text = handle.read()
	except OSError:
		return {}
	return parse(text)


def estimate(node: TaskNode, timings: Mapping[TaskLabel, TaskTiming]) -> Estimate | None:
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
			parts = child_estimates(children, timings)
			return rolled_up(parts, sum(p.elapsed_s for p in parts)) if parts else None
		case Parallel(tasks=children):
			parts = child_estimates(children, timings)
			return rolled_up(parts, max(p.elapsed_s for p in parts)) if parts else None
		case _:
			assert_never(node)


def record(camas_dir: Path, leaves: Sequence[tuple[TaskLabel, float]]) -> None:
	"""Fold a run's observed per-leaf durations into the cache under an exclusive lock.

	``camas_dir`` must already exist.
	"""
	if not leaves:
		return
	observed: Final = dict(leaves)
	with open_for_update(camas_dir / CACHE_NAME) as handle:
		lock(handle, exclusive=True)
		cache = parse(handle.read())
		merged = {
			**cache,
			**{
				label: cache[label].fold(s) if label in cache else TaskTiming(s, 1)
				for label, s in observed.items()
			},
		}
		handle.seek(0)
		handle.truncate()
		handle.write(serialize(merged))


def record_run(camas_dir: Path, result: RunResult) -> None:
	"""Record a finished run's per-leaf durations."""
	record(camas_dir, leaves_of(result))


def ensure_camas_dir(camas_dir: Path) -> None:
	"""Create ``camas_dir`` and its catch-all ``.gitignore`` if either is absent."""
	camas_dir.mkdir(exist_ok=True)
	gitignore = camas_dir / ".gitignore"
	if not gitignore.exists():
		gitignore.write_text("*\n", encoding="utf-8")


def elapsed_of(completion: Completion) -> float | None:
	"""A completion's wall-clock seconds, or ``None`` when the leaf never ran."""
	match completion:
		case Finished(elapsed=elapsed) | Stopped(elapsed=elapsed):
			return elapsed
		case Skipped():
			return None
		case _:
			assert_never(completion)


def serialize(timings: Mapping[TaskLabel, TaskTiming]) -> str:
	r"""Render the cache: a version line, then ``<label> <mean_seconds> <samples>`` per leaf.

	>>> serialize({"lint": TaskTiming(0.5, 2), "ruff check .": TaskTiming(1.25, 1)})
	'0\nlint 0.5 2\nruff check . 1.25 1\n'
	"""
	rows = (f"{label} {t.elapsed_s} {t.samples}" for label, t in sorted(timings.items()))
	return "\n".join((str(CacheVersion.V0.value), *rows)) + "\n"


def parse(text: str) -> dict[TaskLabel, TaskTiming]:
	r"""Parse a cache; a missing or unknown version line yields an empty cache.

	>>> parse("0\nlint 0.5 2\nruff check . 1.25 1\n") == {
	...     "lint": TaskTiming(0.5, 2), "ruff check .": TaskTiming(1.25, 1)
	... }
	True
	>>> parse("999\nlint 0.5 2\n")
	{}
	>>> parse("")
	{}
	"""
	lines = text.splitlines()
	if not lines or lines[0] != str(CacheVersion.V0.value):
		return {}
	return dict(filter(None, (parse_line(line) for line in lines[1:])))


def child_estimates(
	children: Sequence[TaskNode], timings: Mapping[TaskLabel, TaskTiming]
) -> list[Estimate] | None:
	"""Every child's estimate, or ``None`` if any child has an un-timed leaf."""
	parts = [e for child in children if (e := estimate(child, timings)) is not None]
	return parts if len(parts) == len(children) else None


def rolled_up(parts: list[Estimate], elapsed_s: float) -> Estimate:
	"""An ``elapsed_s`` estimate carrying the subtree's least-sampled count and slowest leaf."""
	slowest = max(parts, key=lambda p: p.slowest_s)
	return Estimate(
		elapsed_s, min(p.samples for p in parts), slowest.slowest_leaf, slowest.slowest_s
	)


def open_for_update(path: Path) -> IO[str]:
	"""Open ``path`` read-write for the locked update, creating it without truncating."""
	return os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o644), "r+", encoding="utf-8")


if sys.platform != "win32":
	import fcntl

	def lock(handle: IO[str], *, exclusive: bool) -> None:
		"""Take an advisory ``flock`` on ``handle`` (POSIX)."""
		fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

else:  # pragma: no cover

	def lock(handle: IO[str], *, exclusive: bool) -> None:
		"""Advisory file locking is POSIX-only; a no-op on Windows."""


def leaves_of(result: RunResult) -> list[tuple[TaskLabel, float]]:
	return [(r.name, e) for r in result.results if (e := elapsed_of(r.completion)) is not None]


def parse_line(line: str) -> tuple[TaskLabel, TaskTiming] | None:
	parts = line.rsplit(maxsplit=2)
	if len(parts) != 3:
		return None
	label, elapsed_s, samples = parts
	try:
		return label, TaskTiming(float(elapsed_s), int(samples))
	except ValueError:
		return None
