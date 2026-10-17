# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import os
import sys
from contextlib import suppress
from enum import IntEnum
from functools import reduce
from itertools import groupby
from math import isfinite
from typing import IO, TYPE_CHECKING, Final, NamedTuple, NoReturn, TypeAlias, cast

from ..v0.completion import Errored, Finished, Skipped, Stopped
from ..v0.task import Parallel, Pipe, Sequential, Task
from .scope import PATHS_TOKEN, resolve_default_leaf
from .task import task_label
from .traversal import flatten_leaves

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
	from .scope import Pruned


CACHE_NAME: Final = "timings.txt"


class CacheVersion(IntEnum):
	V0 = 0
	V1 = 1


TaskLabel: TypeAlias = str


class CacheKey(NamedTuple):
	label: TaskLabel
	scope: int


def leaf_scope(task: Task, scope: int) -> int:
	"""
	>>> leaf_scope(Task("mypy src"), 4), leaf_scope(Task("pylint {paths}", paths="."), 4)
	(0, 4)
	"""
	return scope if PATHS_TOKEN in task.cmd else 0


def scope_of(changed: Sequence[str]) -> int:
	"""
	>>> scope_of(()), scope_of(("a",)), scope_of(("a", "b")), scope_of(("a", "b", "c"))
	(0, 1, 2, 4)
	>>> scope_of(tuple("abcde")), scope_of(tuple("abcdefghi"))
	(8, 16)
	"""
	return 0 if not changed else 1 << (len(changed) - 1).bit_length()


class TaskTiming(NamedTuple):
	elapsed_s: float
	samples: int

	def fold(self, elapsed_s: float) -> TaskTiming:
		samples = self.samples + 1
		return TaskTiming((self.elapsed_s * self.samples + elapsed_s) / samples, samples)


class Estimate(NamedTuple):
	elapsed_s: float
	samples: int
	slowest_leaf: TaskLabel
	slowest_s: float


def load(camas_dir: Path) -> dict[CacheKey, TaskTiming]:
	try:
		with (camas_dir / CACHE_NAME).open("r", encoding="utf-8") as handle:
			lock(handle, exclusive=False)
			text = handle.read()
	except (OSError, ValueError):
		return {}
	return parse(text)


def estimate(
	node: TaskNode, timings: Mapping[CacheKey, TaskTiming], scope: int = 0
) -> Estimate | None:
	match node:
		case Task():
			key = leaf_key(node, scope)
			timing = timings.get(key)
			if timing is None:
				return None
			return Estimate(timing.elapsed_s, timing.samples, key.label, timing.elapsed_s)
		case Sequential(tasks=children):
			parts = child_estimates(children, timings, scope)
			return rolled_up(parts, sum(p.elapsed_s for p in parts)) if parts else None
		case Parallel(tasks=children) | Pipe(tasks=children):
			parts = child_estimates(children, timings, scope)
			return rolled_up(parts, max(p.elapsed_s for p in parts)) if parts else None
		case _:
			assert_never(node)


def record(camas_dir: Path, leaves: Sequence[tuple[CacheKey, float]]) -> None:
	if not leaves:
		return
	with open_for_update(camas_dir / CACHE_NAME) as handle:
		lock(handle, exclusive=True)
		merged = folded(parse(handle.read()), [o for o in leaves if recordable_label(o[0].label)])
		handle.seek(0)
		handle.truncate()
		handle.write(serialize(merged))


NO_OBSERVATIONS: Final = TaskTiming(0.0, 0)


def folded(
	cache: Mapping[CacheKey, TaskTiming], leaves: Sequence[tuple[CacheKey, float]]
) -> dict[CacheKey, TaskTiming]:
	"""
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
	if camas_dir is None:
		return
	with suppress(OSError, ValueError):
		if camas_dir.is_dir():
			record(camas_dir, leaves)


class Observed(NamedTuple):
	camas_dir: Path | None
	scope: int
	identities: tuple[CacheKey, ...] | None = None
	node: TaskNode | None = None
	pairs: tuple[tuple[Task, Task], ...] = ()
	pipe_cut: tuple[Task, ...] = ()

	def record(self, result: RunResult) -> None:
		record_observed(self.camas_dir, leaves_of(result, self.scope))

	def original_for(self) -> dict[int, Task]:
		return pairs_index(self.pairs)


def pairs_index(pairs: Iterable[tuple[Task, Task]]) -> dict[int, Task]:
	return {id(scoped): original for original, scoped in pairs}


def leaf_key(task: Task, scope: int) -> CacheKey:
	"""
	>>> leaf_key(Task("ruff check {paths}", paths="."), 0)
	CacheKey(label='ruff check .', scope=0)
	"""
	return CacheKey(task_label(resolve_default_leaf(task)), leaf_scope(task, scope))


def raise_identities_mismatch(identities_count: int, leaves_count: int) -> NoReturn:
	raise ValueError(
		f"identities must be parallel to the run's leaves: "
		f"{identities_count} keys for {leaves_count} leaves"
	)


def reject_non_tuple_identities(value: object) -> None:
	if value is None:
		return
	if not isinstance(value, tuple):
		raise ValueError(
			f"identities must be a tuple of per-leaf cache keys, got {type(value).__name__}"
		)
	if not all(isinstance(item, CacheKey) for item in cast("tuple[object, ...]", value)):
		raise ValueError(
			"identities must be a tuple of per-leaf cache keys, got a tuple of other things"
		)


def observed(camas_dir: Path | None, expanded: TaskNode, changed: Sequence[str]) -> Observed:
	from .scope import scoped_walk

	changed_t = tuple(changed)
	path_scoped: Final = bool(changed_t)
	scope = scope_of(changed)

	def keyed(original: Task, scoped: Task) -> CacheKey:
		return leaf_key(original, scope) if path_scoped else CacheKey(task_label(scoped), 0)

	walk = scoped_walk(expanded, changed_t)
	identities = tuple(keyed(original, scoped) for original, scoped in walk.pairs)
	return Observed(camas_dir, scope, identities, walk.node, walk.pairs, walk.pipe_cut)


def narrowed(keying: Observed, dropped: Pruned) -> Observed:
	if keying.node is None or dropped.node is keying.node:
		return keying
	assert keying.identities is not None
	original_for: Final = keying.original_for()
	keying_order: Final = {
		id(original_for.get(id(info.task), info.task)): i
		for i, info in enumerate(flatten_leaves(keying.node))
	}
	identities: Final = (
		tuple(
			keying.identities[keying_order[id(original_for.get(id(info.task), info.task))]]
			for info in flatten_leaves(dropped.node)
		)
		if dropped.node is not None
		else ()
	)
	return Observed(
		keying.camas_dir,
		keying.scope,
		identities,
		dropped.node,
		keying.pairs,
		keying.pipe_cut + dropped.pipe_cut,
	)


def ensure_camas_dir(camas_dir: Path) -> None:
	camas_dir.mkdir(exist_ok=True)
	gitignore = camas_dir / ".gitignore"
	if not gitignore.exists():
		gitignore.write_text("*\n", encoding="utf-8")


def elapsed_of(completion: Completion) -> float | None:
	match completion:
		case Finished(elapsed=elapsed) | Stopped(elapsed=elapsed):
			return elapsed
		case Skipped() | Errored():
			return None
		case _:
			assert_never(completion)


def serialize(timings: Mapping[CacheKey, TaskTiming]) -> str:
	r"""
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
	r"""
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
	parts = [e for child in children if (e := estimate(child, timings, scope)) is not None]
	return parts if len(parts) == len(children) else None


def rolled_up(parts: list[Estimate], elapsed_s: float) -> Estimate:
	slowest = max(parts, key=lambda p: p.slowest_s)
	return Estimate(
		elapsed_s, min(p.samples for p in parts), slowest.slowest_leaf, slowest.slowest_s
	)


def open_for_update(path: Path) -> IO[str]:
	return os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o644), "r+", encoding="utf-8")


if sys.platform != "win32":
	import fcntl

	def lock(handle: IO[str], *, exclusive: bool) -> None:
		fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

else:  # pragma: no cover

	def lock(handle: IO[str], *, exclusive: bool) -> None: ...


def observations(
	reported: Iterable[tuple[TaskLabel, Completion]],
	scope: int,
) -> list[tuple[CacheKey, float]]:
	return [
		(CacheKey(label, scope), elapsed)
		for label, completion in reported
		if (elapsed := elapsed_of(completion)) is not None
	]


def leaves_of(result: RunResult, scope: int = 0) -> list[tuple[CacheKey, float]]:
	return [
		(r.identity if r.identity is not None else CacheKey(r.name, scope), elapsed)
		for r in result.results
		if (elapsed := elapsed_of(r.completion)) is not None
	]


def observation(elapsed_s: str, samples: str) -> TaskTiming | None:
	r"""
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


def recordable_scope(scope: str) -> int | None:
	"""
	>>> recordable_scope("0"), recordable_scope("4"), recordable_scope("16")
	(0, 4, 16)
	>>> print(recordable_scope("-1"), recordable_scope("3"), recordable_scope("x"))
	None None None
	"""
	try:
		value = int(scope)
	except ValueError:
		return None
	return value if value == 0 or (value > 0 and value & (value - 1) == 0) else None


def recordable_label(label: TaskLabel) -> bool:
	r"""
	>>> [recordable_label(x) for x in ("lint", "ruff check .", "test [PY=3.13]", "  lead")]
	[True, True, True, True]
	>>> [recordable_label(x) for x in ("", " ", "trail ", "two\nlines")]
	[False, False, False, False]
	"""
	lines = f"{label} 0.0 1 0".splitlines()
	return len(lines) == 1 and parse_line(lines[0]) == (CacheKey(label, 0), TaskTiming(0.0, 1))


def parse_line(line: str) -> tuple[CacheKey, TaskTiming] | None:
	parts = line.rsplit(maxsplit=3)
	if len(parts) != 4:
		return None
	label, elapsed_s, samples, scope = parts
	timing = observation(elapsed_s, samples)
	value = recordable_scope(scope)
	if timing is None or value is None:
		return None
	return CacheKey(label, value), timing


def parse_v0_line(line: str) -> tuple[CacheKey, TaskTiming] | None:
	parts = line.rsplit(maxsplit=2)
	if len(parts) != 3:
		return None
	label, elapsed_s, samples = parts
	timing = observation(elapsed_s, samples)
	return None if timing is None else (CacheKey(label, 0), timing)
