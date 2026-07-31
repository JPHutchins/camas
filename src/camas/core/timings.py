# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``timings.txt`` cache of observed per-leaf durations, composed into task estimates."""

from __future__ import annotations

import os
import sys
from contextlib import suppress
from enum import IntEnum
from functools import reduce
from itertools import groupby
from math import isfinite
from typing import IO, TYPE_CHECKING, Final, NamedTuple, TypeAlias

from ..v0.completion import Errored, Finished, Skipped, Stopped
from ..v0.task import Parallel, Sequential, Task
from .scope import PATHS_TOKEN, resolve_default_leaf
from .task import task_label

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping, Sequence
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


def leaf_scope(task: Task, scope: int) -> int:
	"""The scope one leaf's observation belongs to: the run's, when its command takes the changed
	paths, and ``0`` when it does not.

	A command with no ``{paths}`` runs identically whatever changed, so its cost does not vary with
	the change size and bucketing it by that would only make it re-run once per bucket to learn the
	same number — which for a slow unscopable leaf is #218's blocked turn again, once per bucket the
	project happens to gate at.

	>>> leaf_scope(Task("mypy src"), 4), leaf_scope(Task("pylint {paths}", paths="."), 4)
	(0, 4)
	"""
	return scope if PATHS_TOKEN in task.cmd else 0


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
			timing = timings.get(CacheKey(label, leaf_scope(node, scope)))
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

	An observation with an empty label is dropped, because a row cannot carry one: the label is the
	first field of a whitespace-separated line, so ``" 0.5 1 0"`` reads back as three fields and the
	row is discarded on load. Dropping it on the way in says so, rather than writing something that
	silently never returns. Only a leaf explicitly named ``""`` produces one.
	"""
	if not leaves:
		return
	with open_for_update(camas_dir / CACHE_NAME) as handle:
		lock(handle, exclusive=True)
		merged = folded(parse(handle.read()), [o for o in leaves if o[0].label])
		handle.seek(0)
		handle.truncate()
		handle.write(serialize(merged))


NO_OBSERVATIONS: Final = TaskTiming(0.0, 0)
"""The neutral value for :meth:`TaskTiming.fold` — folding the first duration into it yields that
duration with one sample, so a key absent from the cache needs no separate case."""


def folded(
	cache: Mapping[CacheKey, TaskTiming], leaves: Sequence[tuple[CacheKey, float]]
) -> dict[CacheKey, TaskTiming]:
	"""``cache`` with every observation in ``leaves`` folded in.

	Grouped by key and folded per group, so two leaves reporting the same label in one run — two
	``Task`` objects sharing a name, or a command — both count rather than the later replacing the
	earlier, and the cache is copied once rather than once per observation.

	>>> folded({}, [(CacheKey("a", 0), 1.0), (CacheKey("a", 0), 3.0)])
	{CacheKey(label='a', scope=0): TaskTiming(elapsed_s=2.0, samples=2)}
	>>> folded({CacheKey("a", 0): TaskTiming(2.0, 2)}, [(CacheKey("a", 0), 8.0)])
	{CacheKey(label='a', scope=0): TaskTiming(elapsed_s=4.0, samples=3)}
	"""
	return {
		**cache,
		**{
			key: reduce(
				lambda timing, elapsed: timing.fold(elapsed),
				(elapsed for _, elapsed in group),
				cache.get(key, NO_OBSERVATIONS),
			)
			for key, group in groupby(sorted(leaves), key=lambda observation: observation[0])
		},
	}


def record_observed(camas_dir: Path | None, leaves: Sequence[tuple[CacheKey, float]]) -> None:
	"""The one place that decides whether a finished run is observed, so that no path which runs
	leaves can quietly decline to measure them: three of them did, and a gate whose own runs are
	never observed can never budget — its leaves stay unmeasured, so ``--under`` runs every one of
	them on every turn, including the one that always fails.

	No camas directory is the documented opt-out. A write that cannot happen is swallowed rather
	than raised: the run these durations describe has already finished, so failing here would turn a
	completed check into an error over a cache — and :func:`load` already treats an unreadable cache
	as an empty one, which is the same judgement on the reading side.

	``ValueError`` alongside ``OSError`` because recording reads the existing cache first, and a file
	holding invalid UTF-8 raises ``UnicodeDecodeError`` — a ``ValueError``, not an ``OSError``. The
	same pair, for the same reason, guards the ``mcp init`` warning (:func:`camas.mcp.gitignore.
	warn_uncommittable`, #271).
	"""
	if camas_dir is None or not camas_dir.is_dir():
		return
	with suppress(OSError, ValueError):
		record(camas_dir, leaves)


class Observed(NamedTuple):
	"""How one run's durations are to be recorded: where, keyed to what size of change, and under
	which labels.

	Built once where a run is set up, and carried to wherever the run finishes, so that no path can
	run leaves and get part of this right. Deriving the three by hand per call site is what let a gate
	record under labels its own budget could not read — in three separate places, each found only
	after the previous one was fixed.
	"""

	camas_dir: Path | None
	scope: int
	keys: Mapping[TaskLabel, CacheKey]
	"""Where each leaf's observation goes, by the label it reports — see
	:func:`observation_keys`. A label with no entry is keyed by itself at this run's scope."""

	def record(self, result: RunResult) -> None:
		"""Record ``result``'s timed leaves."""
		record_observed(self.camas_dir, leaves_of(result, self.scope, self.keys))


def observation_keys(
	expanded: TaskNode, changed: tuple[str, ...], scope: int
) -> dict[TaskLabel, CacheKey]:
	"""The key each surviving leaf's observation belongs under, by the label that leaf will report.

	Two things differ from the reported label, and both have to be undone here or the observation
	lands where no budget looks: scoping rewrote a nameless leaf's command, and a leaf that ignores
	the changed paths belongs at scope ``0`` rather than this run's.
	"""
	from .scope import scoped_leaves

	return {
		task_label(scoped): CacheKey(
			task_label(resolve_default_leaf(task)), leaf_scope(task, scope)
		)
		for task, scoped in scoped_leaves(expanded, changed)
	}


def observed(camas_dir: Path | None, expanded: TaskNode, changed: Sequence[str]) -> Observed:
	"""How to record a run of ``expanded`` scoped to ``changed`` — the one derivation of the three
	things that keying needs.
	"""
	return Observed(
		camas_dir,
		scope_of(changed),
		observation_keys(expanded, tuple(changed), scope_of(changed)),
	)


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


def observations(
	reported: Iterable[tuple[TaskLabel, Completion]],
	scope: int,
	keys: Mapping[TaskLabel, CacheKey],
) -> list[tuple[CacheKey, float]]:
	"""The recordable observations among ``reported`` — each leaf's label as it ran, paired with how
	it completed.

	A leaf reports the label it ran under, which for a scoped ``{paths}`` leaf with no ``name`` is the
	command with those paths already in it. ``keys`` — from :func:`observation_keys` — says where that
	belongs instead, so what a scoped run records is what a later budget can read. The one place that
	mapping happens, for both the run result and the effect that watches leaf states.
	"""
	return [
		(keys.get(label, CacheKey(label, scope)), elapsed)
		for label, completion in reported
		if (elapsed := elapsed_of(completion)) is not None
	]


def leaves_of(
	result: RunResult, scope: int = 0, keys: Mapping[TaskLabel, CacheKey] = {}
) -> list[tuple[CacheKey, float]]:
	"""``result``'s timed leaves as observations at ``scope``."""
	return observations(((r.name, r.completion) for r in result.results), scope, keys)


def observation(elapsed_s: str, samples: str) -> TaskTiming | None:
	r"""One row's timing, or ``None`` when the row cannot be one.

	``float`` accepts ``nan`` and ``inf``, which :func:`serialize` can never write and a corrupted
	cache can still hold. Either would stick permanently: :meth:`TaskTiming.fold` keeps propagating
	it through the running mean, and every ``elapsed_s <= budget_s`` comparison against a ``nan`` is
	false, so the leaf is over budget forever and never runs to correct itself. A sample count below
	one is the same kind of impossibility — the mean it weights is of no observations — and so is a
	negative duration, which would drag the mean down and make the leaf look ever cheaper than it is.

	>>> observation("0.5", "2")
	TaskTiming(elapsed_s=0.5, samples=2)
	>>> print(observation("nan", "1"), observation("inf", "1"), observation("0.5", "0"))
	None None None
	>>> print(observation("-0.5", "1"))
	None
	>>> print(observation("x", "1"), observation("0.5", "x"))
	None None
	"""
	try:
		elapsed, count = float(elapsed_s), int(samples)
	except ValueError:
		return None
	if not isfinite(elapsed) or elapsed < 0 or count < 1:
		return None
	return TaskTiming(elapsed, count)


def parse_line(line: str) -> tuple[CacheKey, TaskTiming] | None:
	parts = line.rsplit(maxsplit=3)
	if len(parts) != 4:
		return None
	label, elapsed_s, samples, scope = parts
	timing = observation(elapsed_s, samples)
	if timing is None:
		return None
	try:
		return CacheKey(label, int(scope)), timing
	except ValueError:
		return None


def parse_v0_line(line: str) -> tuple[CacheKey, TaskTiming] | None:
	parts = line.rsplit(maxsplit=2)
	if len(parts) != 3:
		return None
	label, elapsed_s, samples = parts
	timing = observation(elapsed_s, samples)
	return None if timing is None else (CacheKey(label, 0), timing)
