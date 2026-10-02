# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Cancel-safe teardown of spawned children: kill them, reap them, and drain their readers
within one deadline, absorbing cancels without ever turning one into a success.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import suppress
from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn, Protocol, TypeVar

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

UNWIND_TIMEOUT_S: Final = 5.0
"""How long one unwind waits, in total, for its killed children to reap and their readers to
drain — a child wedged in uninterruptible sleep, or a grandchild holding a pipe open, must not
make the run unkillable."""

_T = TypeVar("_T")


class Reapable(Protocol):
	"""A child the unwind can kill and reap."""

	@property
	def returncode(self) -> int | None: ...

	def kill(self) -> None: ...

	async def wait(self) -> int: ...


class Settled(NamedTuple):
	"""How waiting on one future ended."""

	finished: bool
	"""The future completed before the deadline."""
	cancelled: bool
	"""A cancel landed while waiting and was absorbed — the caller re-raises it once its own
	cleanup is done."""


class Unwound(NamedTuple):
	"""How an unwind ended."""

	unreaped: frozenset[int]
	"""The keys of the children killed but not reaped by the deadline."""
	cancelled: bool
	"""A cancel landed during the unwind and was absorbed."""


async def settle(
	future: asyncio.Future[_T], deadline: float, *, cancelled: bool = False
) -> Settled:
	"""Wait for ``future`` until the loop-time ``deadline`` through any number of cancels.
	A finished future's exception is retrieved so a faulted one never logs as unretrieved.
	"""
	remaining: Final = deadline - asyncio.get_running_loop().time()
	if future.done() or remaining <= 0:
		if future.done() and not future.cancelled():
			future.exception()
		return Settled(future.done(), cancelled)
	try:
		await asyncio.wait({future}, timeout=remaining)
	except asyncio.CancelledError:
		return await settle(future, deadline, cancelled=True)
	return await settle(future, deadline, cancelled=cancelled)


async def _settle_each(
	futures: tuple[asyncio.Future[_T], ...], deadline: float
) -> tuple[Settled, ...]:
	"""Each future settled in turn against the one deadline."""
	if not futures:
		return ()
	return (await settle(futures[0], deadline), *await _settle_each(futures[1:], deadline))


async def unwind(
	children: Mapping[int, Reapable],
	tasks: Sequence[asyncio.Future[None]],
	timeout_s: float = UNWIND_TIMEOUT_S,
) -> Unwound:
	"""Kill every live child, then reap them and drain ``tasks`` against one deadline
	``timeout_s`` away, so a wedge costs one bound, not one per child.
	"""
	deadline: Final = asyncio.get_running_loop().time() + timeout_s
	for child in children.values():
		if child.returncode is None:
			with suppress(ProcessLookupError, OSError):
				child.kill()
	reaps: Final = {key: asyncio.ensure_future(child.wait()) for key, child in children.items()}
	reaped: Final = tuple(
		zip(reaps, await _settle_each(tuple(reaps.values()), deadline), strict=True)
	)
	drained: Final = tuple(zip(tasks, await _settle_each(tuple(tasks), deadline), strict=True))
	unreaped: Final = frozenset(key for key, outcome in reaped if not outcome.finished)
	lingering: Final = tuple(task for task, outcome in drained if not outcome.finished)
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
	return Unwound(
		unreaped,
		any(outcome.cancelled for _, outcome in reaped)
		or any(outcome.cancelled for _, outcome in drained),
	)


def reraise(unwound: Unwound, exc: BaseException) -> NoReturn:
	"""Propagate the failure that triggered the unwind — unless the unwind absorbed a cancel
	and ``exc`` is an ordinary failure: the cancel wins, so a caller's timeout still fires. A
	``KeyboardInterrupt`` or ``SystemExit`` is never converted.

	Raises:
		asyncio.CancelledError: when the unwind absorbed one and ``exc`` is an ``Exception``;
			``exc`` itself is re-raised otherwise.
	"""
	if unwound.cancelled and isinstance(exc, Exception):
		raise asyncio.CancelledError from exc
	raise exc


def rethrow_cancel(unwound: Unwound) -> None:
	"""On a path that returns normally after an unwind, re-raise a cancel the unwind absorbed —
	a cancellation never becomes a success.

	>>> rethrow_cancel(Unwound(frozenset(), False))

	Raises:
		asyncio.CancelledError: when the unwind absorbed one.
	"""
	if unwound.cancelled:
		raise asyncio.CancelledError
