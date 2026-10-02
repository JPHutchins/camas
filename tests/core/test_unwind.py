# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import gc
import sys
import time
from typing import TYPE_CHECKING, Any

import pytest

from camas.core.unwind import Settled, Unwound, reraise, rethrow_cancel, settle, unwind

if TYPE_CHECKING:
	from collections.abc import Iterable


class Wedged:
	"""A killed child whose reap never completes — uninterruptible sleep after SIGKILL."""

	@property
	def returncode(self) -> int | None:
		return None

	def kill(self) -> None:
		raise ProcessLookupError

	async def wait(self) -> int:
		await asyncio.Event().wait()
		raise AssertionError("unreachable")


async def _forever() -> None:
	await asyncio.Event().wait()


def _deadline(seconds: float) -> float:
	return asyncio.get_running_loop().time() + seconds


async def test_settle_reports_a_finished_future() -> None:
	done: asyncio.Future[int] = asyncio.get_running_loop().create_future()
	done.set_result(1)
	assert await settle(done, _deadline(1)) == Settled(finished=True, cancelled=False)


async def test_settle_gives_up_at_the_deadline_without_cancelling_the_future() -> None:
	pending = asyncio.ensure_future(asyncio.Event().wait())
	assert await settle(pending, _deadline(0.05)) == Settled(finished=False, cancelled=False)
	assert not pending.cancelled()
	pending.cancel()


async def test_settle_absorbs_repeated_cancels_and_still_waits_for_the_future() -> None:
	release = asyncio.Event()
	gated = asyncio.ensure_future(release.wait())
	settling = asyncio.ensure_future(settle(gated, _deadline(5)))
	for _ in range(3):
		await asyncio.sleep(0)
		assert settling.cancel()
	release.set()
	assert await settling == Settled(finished=True, cancelled=True)


async def test_settle_propagates_a_failure_of_the_wait_itself(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Only a cancel is absorbed — a wait that fails some other way must not spin forever."""

	async def broken_wait(fs: Iterable[asyncio.Future[Any]], *, timeout: float | None) -> Any:
		raise TypeError("not a future")

	monkeypatch.setattr(asyncio, "wait", broken_wait)
	pending = asyncio.ensure_future(asyncio.Event().wait())
	with pytest.raises(TypeError, match="not a future"):
		await settle(pending, _deadline(1))
	pending.cancel()


async def test_settle_retrieves_a_failed_futures_exception() -> None:
	loop = asyncio.get_running_loop()
	reported: list[dict[str, Any]] = []
	previous = loop.get_exception_handler()
	loop.set_exception_handler(lambda _, context: reported.append(context))
	try:
		failed: asyncio.Future[None] = loop.create_future()
		failed.set_exception(RuntimeError("boom"))
		assert await settle(failed, _deadline(1)) == Settled(finished=True, cancelled=False)
		del failed
		gc.collect()
	finally:
		loop.set_exception_handler(previous)
	assert reported == []


async def test_unwind_kills_and_reaps_real_children() -> None:
	children = {
		i: await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(60)")
		for i in range(2)
	}
	assert await unwind(children, ()) == Unwound(unreaped=frozenset(), cancelled=False)
	assert all(child.returncode is not None for child in children.values())


async def test_unwind_bounds_every_wedged_child_by_one_shared_deadline(
	capsys: pytest.CaptureFixture[str],
) -> None:
	started = time.monotonic()
	unwound = await unwind({i: Wedged() for i in range(3)}, (), timeout_s=0.2)
	assert time.monotonic() - started < 0.5
	assert unwound == Unwound(unreaped=frozenset({0, 1, 2}), cancelled=False)
	assert "3 killed child(ren) not reaped within 0.2s" in capsys.readouterr().err


async def test_unwind_cancels_and_reports_a_reader_that_never_drains(
	capsys: pytest.CaptureFixture[str],
) -> None:
	reader = asyncio.ensure_future(_forever())
	assert await unwind({}, (reader,), timeout_s=0.1) == Unwound(frozenset(), cancelled=False)
	await asyncio.sleep(0)
	assert reader.cancelled()
	assert "1 output reader(s) still open after 0.1s" in capsys.readouterr().err


async def test_unwind_reports_a_cancel_it_absorbed(capsys: pytest.CaptureFixture[str]) -> None:
	unwinding = asyncio.ensure_future(unwind({0: Wedged()}, (), timeout_s=0.2))
	await asyncio.sleep(0)
	assert unwinding.cancel()
	assert await unwinding == Unwound(unreaped=frozenset({0}), cancelled=True)
	capsys.readouterr()


def test_reraise_lets_an_absorbed_cancel_win_over_an_ordinary_failure() -> None:
	failure = ValueError("stage overflowed")
	with pytest.raises(asyncio.CancelledError) as info:
		reraise(Unwound(frozenset(), cancelled=True), failure)
	assert info.value.__cause__ is failure


def test_reraise_propagates_the_failure_itself_when_no_cancel_was_absorbed() -> None:
	failure = ValueError("stage overflowed")
	with pytest.raises(ValueError, match="stage overflowed") as info:
		reraise(Unwound(frozenset(), cancelled=False), failure)
	assert info.value is failure


def test_reraise_never_converts_a_keyboard_interrupt() -> None:
	with pytest.raises(KeyboardInterrupt):
		reraise(Unwound(frozenset(), cancelled=True), KeyboardInterrupt())


def test_rethrow_cancel_raises_a_cancel_the_unwind_absorbed() -> None:
	with pytest.raises(asyncio.CancelledError):
		rethrow_cancel(Unwound(frozenset(), cancelled=True))
