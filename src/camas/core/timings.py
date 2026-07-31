# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``timings.txt`` cache of observed per-leaf durations, composed into task estimates."""

from __future__ import annotations

import os
import sys
from contextlib import suppress
from enum import IntEnum
from typing import IO, TYPE_CHECKING, Final, NamedTuple, TypeAlias

from ..v0.completion import Errored, Finished, Skipped, Stopped
from ..v0.task import Parallel, Sequential, Task
from .scope import resolve_default_leaf
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
	V1 = 1
	"""Adds the scope column, so an observation is keyed by the size of the change it was made on
	and a whole-tree run stops deciding a scoped gate's budget. A V0 file still reads, its rows
	taken as the unscoped observations they were."""


TaskLabel: TypeAlias = str
"""A leaf's :func:`task_label` — its ``name`` or its joined command."""


class CacheKey(NamedTuple):
	"""What one cached duration is an observation *of*: a leaf, run over a change of some size."""

	label: TaskLabel
	scope: int
	"""How many changed paths the run was scoped to, rounded up to a power of two — ``0`` for an
	unscoped whole-tree run. A leaf costs what its input costs, so two observations are only
	comparable at the same scale: scoping ``pylint`` to one file is not the run that took 206s."""


def scope_of(changed: Sequence[str]) -> int:
	"""The :attr:`CacheKey.scope` for a run over ``changed`` — ``0`` when nothing scopes it, else
	the count rounded up to a power of two, so neighbouring change sizes share an observation
	instead of each having to learn its own.

	>>> scope_of(()), scope_of(("a",)), scope_of(("a", "b")), scope_of(("a", "b", "c"))
	(0, 1, 2, 4)
	>>> scope_of(tuple("abcde")), scope_of(tuple("abcdefghi"))
	(8, 16)
	"""
	return 0 if not changed else 1 << (len(changed) - 1).bit_length()


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


def load(camas_dir: Path) -> dict[CacheKey, TaskTiming]:
	"""Read the cache in ``camas_dir``; an absent or unreadable file is an empty cache."""
	try:
		with (camas_dir / CACHE_NAME).open("r", encoding="utf-8") as handle:
			lock(handle, exclusive=False)
			text = handle.read()
	except OSError:
		return {}
	return parse(text)


def estimate(
	node: TaskNode, timings: Mapping[CacheKey, TaskTiming], scope: int = 0
) -> Estimate | None:
	"""Compose ``node``'s estimate at ``scope`` from observed leaf durations: a leaf is its own
	timing, a Sequential the sum of its children, a Parallel their max. ``None`` when any leaf in
	the subtree has never been timed *at that scope* — an observation from another scale is not an
	estimate of this run, and treating it as one is what excluded a one-file ``pylint`` for costing
	206s over the whole tree. An unmeasured leaf runs and is thereby measured, so each scope a
	project actually gates at converges after one run.
	"""
	match node:
		case Task():
			label = task_label(resolve_default_leaf(node))
			timing = timings.get(CacheKey(label, scope))
			if timing is None:
				return None
			return Estimate(timing.elapsed_s, timing.samples, label, timing.elapsed_s)
		case Sequential(tasks=children):
			parts = child_estimates(children, timings, scope)
			return rolled_up(parts, sum(p.elapsed_s for p in parts)) if parts else None
		case Parallel(tasks=children):
			parts = child_estimates(children, timings, scope)
			return rolled_up(parts, max(p.elapsed_s for p in parts)) if parts else None
		case _:
			assert_never(node)


def record(camas_dir: Path, leaves: Sequence[tuple[CacheKey, float]]) -> None:
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
				key: cache[key].fold(s) if key in cache else TaskTiming(s, 1)
				for key, s in observed.items()
			},
		}
		handle.seek(0)
		handle.truncate()
		handle.write(serialize(merged))


def record_observed(camas_dir: Path | None, leaves: Sequence[tuple[CacheKey, float]]) -> None:
	"""The one place that decides whether a finished run is observed, so that no path which runs
	leaves can quietly decline to measure them: three of them did, and a gate whose own runs are
	never observed can never budget — its leaves stay unmeasured, so ``--under`` runs every one of
	them on every turn, including the one that always fails.

	No camas directory is the documented opt-out. A write that cannot happen is swallowed rather
	than raised: the run these durations describe has already finished, so failing here would turn a
	completed check into an error over a cache — and :func:`load` already treats an unreadable cache
	as an empty one, which is the same judgement on the reading side.
	"""
	if camas_dir is None or not camas_dir.is_dir():
		return
	with suppress(OSError):
		record(camas_dir, leaves)


def record_run(camas_dir: Path | None, result: RunResult, scope: int = 0) -> None:
	"""Record a finished run's per-leaf durations as observations at ``scope``."""
	record_observed(camas_dir, leaves_of(result, scope))


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
		case Skipped() | Errored():
			return None
		case _:
			assert_never(completion)


def serialize(timings: Mapping[CacheKey, TaskTiming]) -> str:
	r"""Render the cache: a version line, then ``<label> <mean_seconds> <samples> <scope>`` per
	observation. The scope goes last so a label may still contain spaces.

	>>> print(serialize({CacheKey("lint", 0): TaskTiming(0.5, 2), CacheKey("lint", 1): TaskTiming(0.1, 1)}))
	1
	lint 0.5 2 0
	lint 0.1 1 1
	<BLANKLINE>
	"""
	rows = (
		f"{key.label} {t.elapsed_s} {t.samples} {key.scope}" for key, t in sorted(timings.items())
	)
	return "\n".join((str(CacheVersion.V1.value), *rows)) + "\n"


def parse(text: str) -> dict[CacheKey, TaskTiming]:
	r"""Parse a cache; a missing or unknown version line yields an empty cache. A V0 file — written
	before observations were keyed by scope — reads as the unscoped observations its rows were, so
	upgrading keeps the whole-tree estimates ``camas --list`` shows.

	>>> parse("1\nlint 0.5 2 0\nruff check . 1.25 1 4\n") == {
	...     CacheKey("lint", 0): TaskTiming(0.5, 2),
	...     CacheKey("ruff check .", 4): TaskTiming(1.25, 1),
	... }
	True
	>>> parse("0\nlint 0.5 2\n") == {CacheKey("lint", 0): TaskTiming(0.5, 2)}
	True
	>>> parse("999\nlint 0.5 2\n")
	{}
	>>> parse("")
	{}
	"""
	lines = text.splitlines()
	if not lines:
		return {}
	if lines[0] == str(CacheVersion.V0.value):
		return dict(filter(None, (parse_v0_line(line) for line in lines[1:])))
	if lines[0] == str(CacheVersion.V1.value):
		return dict(filter(None, (parse_line(line) for line in lines[1:])))
	return {}


def child_estimates(
	children: Sequence[TaskNode], timings: Mapping[CacheKey, TaskTiming], scope: int
) -> list[Estimate] | None:
	"""Every child's estimate, or ``None`` if any child has an un-timed leaf."""
	parts = [e for child in children if (e := estimate(child, timings, scope)) is not None]
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


def leaves_of(result: RunResult, scope: int = 0) -> list[tuple[CacheKey, float]]:
	return [
		(CacheKey(r.name, scope), e)
		for r in result.results
		if (e := elapsed_of(r.completion)) is not None
	]


def parse_line(line: str) -> tuple[CacheKey, TaskTiming] | None:
	parts = line.rsplit(maxsplit=3)
	if len(parts) != 4:
		return None
	label, elapsed_s, samples, scope = parts
	try:
		return CacheKey(label, int(scope)), TaskTiming(float(elapsed_s), int(samples))
	except ValueError:
		return None


def parse_v0_line(line: str) -> tuple[CacheKey, TaskTiming] | None:
	parts = line.rsplit(maxsplit=2)
	if len(parts) != 3:
		return None
	label, elapsed_s, samples = parts
	try:
		return CacheKey(label, 0), TaskTiming(float(elapsed_s), int(samples))
	except ValueError:
		return None
