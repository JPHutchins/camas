# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import sys
from contextlib import suppress
from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn, Protocol, TypeVar

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

UNWIND_TIMEOUT_S: Final = 5.0

_T = TypeVar("_T")


class Reapable(Protocol):
	@property
	def returncode(self) -> int | None: ...

	def kill(self) -> None: ...

	async def wait(self) -> int: ...


class Settled(NamedTuple):
	finished: bool
	cancelled: bool


async def settle(future: asyncio.Future[_T], deadline: float) -> Settled:
	loop: Final = asyncio.get_running_loop()
	cancelled = False
	while not future.done() and (remaining := deadline - loop.time()) > 0:
		try:
			await asyncio.wait({future}, timeout=remaining)
		except asyncio.CancelledError:  # noqa: PERF203  # absorbing each cancel is the loop's purpose
			cancelled = True
	if future.done() and not future.cancelled():
		future.exception()
	return Settled(future.done(), cancelled)


async def _settle_each(
	futures: tuple[asyncio.Future[_T], ...], deadline: float
) -> tuple[Settled, ...]:
	outcomes: Final[list[Settled]] = []
	for future in futures:
		outcomes.append(await settle(future, deadline))
	return tuple(outcomes)


def _reaped(reap: asyncio.Future[int]) -> bool:
	return reap.done() and not reap.cancelled() and reap.exception() is None


class _TornDown(NamedTuple):
	unreaped: frozenset[int]
	cancelled: bool


async def _teardown(
	children: Mapping[int, Reapable], tasks: Sequence[asyncio.Future[None]], timeout_s: float
) -> _TornDown:
	deadline: Final = asyncio.get_running_loop().time() + timeout_s
	for child in children.values():
		if child.returncode is None:
			with suppress(ProcessLookupError, OSError):
				child.kill()
	reaps: Final = {key: asyncio.ensure_future(child.wait()) for key, child in children.items()}
	reap_outcomes: Final = await _settle_each(tuple(reaps.values()), deadline)
	task_outcomes: Final = await _settle_each(tuple(tasks), deadline)
	unreaped: Final = frozenset(key for key, reap in reaps.items() if not _reaped(reap))
	lingering: Final = tuple(
		task for task, outcome in zip(tasks, task_outcomes, strict=True) if not outcome.finished
	)
	for key in unreaped:
		reaps[key].cancel()
	for task in lingering:
		task.cancel()
	if unreaped:
		print(
			f"camas: {len(unreaped)} killed child(ren) not reaped within {timeout_s}s; continuing "
			"the unwind — they may still be alive",
			file=sys.stderr,
		)
	if lingering:
		print(
			f"camas: {len(lingering)} output reader(s) still open after {timeout_s}s (a "
			"grandchild holding the pipe?); dropping their unread tail",
			file=sys.stderr,
		)
	return _TornDown(
		unreaped, any(outcome.cancelled for outcome in (*reap_outcomes, *task_outcomes))
	)


async def unwind(
	children: Mapping[int, Reapable],
	tasks: Sequence[asyncio.Future[None]],
	timeout_s: float = UNWIND_TIMEOUT_S,
) -> frozenset[int]:
	torn_down: Final = await _teardown(children, tasks, timeout_s)
	if torn_down.cancelled:
		raise asyncio.CancelledError
	return torn_down.unreaped


async def unwind_failure(
	cause: BaseException,
	children: Mapping[int, Reapable],
	tasks: Sequence[asyncio.Future[None]],
	timeout_s: float = UNWIND_TIMEOUT_S,
) -> NoReturn:
	torn_down: Final = await _teardown(children, tasks, timeout_s)
	if torn_down.cancelled and isinstance(cause, Exception):
		raise asyncio.CancelledError from cause
	raise cause
