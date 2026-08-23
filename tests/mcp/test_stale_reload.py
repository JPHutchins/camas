# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The live reload contract of the camas MCP server, driven over real stdio (#58)."""

from __future__ import annotations

import asyncio
import functools
import os
import queue
import select
import subprocess
import sys
import threading
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

import camas
from camas import Config, Task
from camas.main.state import LoadOk
from camas.mcp import serve
from camas.mcp.serve import Compat, Session
from camas.mcp.serve import call as _dispatch

if TYPE_CHECKING:
	from pathlib import Path
	from typing import IO

	from camas.v0.task import TaskNode


PASS = Task(("python", "-c", "pass"), name="pass")


async def _slow_call(
	counter: list[int], session: Session, name: str, arguments: dict[str, Any]
) -> Any:
	"""A call that takes 0.3s the first time and 0.6s after, so a concurrently-dispatched call
	is still in flight when an earlier stale call's exit timer fires."""
	counter[0] += 1
	await asyncio.sleep(0.3 if counter[0] == 1 else 0.6)
	return await _dispatch(session, name, arguments)


async def _raising_call(session: Session, name: str, arguments: dict[str, Any]) -> Any:
	raise RuntimeError("boom")


def _session(tasks: dict[str, TaskNode], config: Config | None, base: Path) -> Session:
	state = LoadOk(tasks=tasks, source=base / "tasks.py", scope_effects={}, config=config)
	return Session(state, base, Compat())


@pytest.fixture(autouse=True)
def reload_exits(monkeypatch: pytest.MonkeyPatch) -> list[int]:
	"""Record scheduled reload exits instead of letting one kill the test process."""
	exits: list[int] = []
	monkeypatch.setattr(serve, "exit_for_reload", lambda: exits.append(1))
	return exits


def _fake_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	"""Point ``camas.__file__`` at a scratch package so snapshot tests don't touch the real one."""
	pkg = tmp_path / "pkg"
	pkg.mkdir()
	(pkg / "__init__.py").write_text("")
	(pkg / "a.py").write_text("x = 1\n")
	monkeypatch.setattr(camas, "__file__", str(pkg / "__init__.py"))
	return pkg


async def test_reverted_package_keeps_the_server_up(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reload_exits: list[int], wait_until: Any
) -> None:
	"""The exit re-checks the snapshot when it fires: a package reverted before the timer ran
	keeps the server up instead of exiting on a stale observation."""
	monkeypatch.setattr(serve, "RELOAD_EXIT_DELAY", 1.0)
	pkg = _fake_package(tmp_path, monkeypatch)
	session = _session({"lint": PASS}, None, tmp_path)
	real_snapshot = serve.package_snapshot
	walks = 0

	def counting_snapshot() -> dict[str, str]:
		nonlocal walks
		result = real_snapshot()
		walks += 1
		return result

	monkeypatch.setattr(serve, "package_snapshot", counting_snapshot)
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		(pkg / "a.py").write_text("y = 2\n")
		assert not (await client.call_tool("camas_list", {})).isError
		(pkg / "a.py").write_text("x = 1\n")
		await wait_until(lambda: walks >= 2)  # the fire-time walk completed before the assert
	assert reload_exits == []


async def test_probe_skips_the_walk_while_a_call_is_in_flight(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reload_exits: list[int], wait_until: Any
) -> None:
	"""The pre-walk idle check: a probe firing while a call runs returns without walking the
	package — the walk count stays at the build-time one."""
	monkeypatch.setattr(serve, "RELOAD_EXIT_DELAY", 0.5)
	_fake_package(tmp_path, monkeypatch)
	session = _session({"lint": PASS}, None, tmp_path)
	initial = serve.package_snapshot()
	walks = 0

	def counting_snapshot() -> dict[str, str]:
		nonlocal walks
		walks += 1
		return {"changed": "1"} if walks > 1 else initial

	monkeypatch.setattr(serve, "package_snapshot", counting_snapshot)
	calls = 0

	async def timed_call(session: Session, name: str, arguments: dict[str, Any]) -> Any:
		nonlocal calls
		calls += 1
		await asyncio.sleep(0.3 if calls == 1 else serve.RELOAD_EXIT_DELAY + 0.2)
		return await _dispatch(session, name, arguments)

	monkeypatch.setattr(serve, "call", timed_call)
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		assert not (await client.call_tool("camas_list", {})).isError
		assert walks == 1
		second = asyncio.create_task(client.call_tool("camas_list", {}))
		assert not (await second).isError
		assert walks == 1
		await wait_until(lambda: bool(reload_exits))
	assert reload_exits == [1]


async def test_mid_call_change_arms_the_exit(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reload_exits: list[int], wait_until: Any
) -> None:
	"""A package change landing before the fire-time probe runs is caught — the call-end probe
	is armed unconditionally, so the flip needs no call-start sample."""
	monkeypatch.setattr(serve, "RELOAD_EXIT_DELAY", 0.05)
	_fake_package(tmp_path, monkeypatch)
	session = _session({"lint": PASS}, None, tmp_path)
	initial = serve.package_snapshot()
	changed = False

	def snapshot() -> dict[str, str]:
		return {"changed": "1"} if changed else initial

	async def mid_call_change(session: Session, name: str, arguments: dict[str, Any]) -> Any:
		nonlocal changed
		changed = True
		return await _dispatch(session, name, arguments)

	monkeypatch.setattr(serve, "package_snapshot", snapshot)
	monkeypatch.setattr(serve, "call", mid_call_change)
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		assert not (await client.call_tool("camas_list", {})).isError
		await wait_until(lambda: bool(reload_exits))
	assert reload_exits == [1]


async def test_call_arriving_during_the_probe_walk_survives(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reload_exits: list[int], wait_until: Any
) -> None:
	"""The idle re-check happens after the off-loop walk: a call in flight when the walk
	returns changed is never the one the exit kills — the exit waits for the next idle gap."""
	monkeypatch.setattr(serve, "RELOAD_EXIT_DELAY", 0.05)
	_fake_package(tmp_path, monkeypatch)
	session = _session({"lint": PASS}, None, tmp_path)
	initial = serve.package_snapshot()
	entered = 0

	def slow_snapshot() -> dict[str, str]:
		nonlocal entered
		time.sleep(0.3)
		entered += 1
		return {"changed": "1"} if entered > 1 else initial

	monkeypatch.setattr(serve, "package_snapshot", slow_snapshot)
	monkeypatch.setattr(serve, "call", functools.partial(_slow_call, [0]))
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		first = asyncio.create_task(client.call_tool("camas_list", {}))
		await asyncio.sleep(0.4)
		second = asyncio.create_task(client.call_tool("camas_list", {}))
		assert not (await first).isError
		await asyncio.sleep(0.3)  # the walk returned; the slow call is still in flight
		assert reload_exits == []
		assert not (await second).isError
		await wait_until(lambda: bool(reload_exits))
	assert reload_exits == [1]


async def test_call_completing_during_the_probe_walk_keeps_its_flush_window(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reload_exits: list[int], wait_until: Any
) -> None:
	"""The flush-window contract, pinned directly: the exit fires no sooner than
	``RELOAD_EXIT_DELAY`` after the last call completes — a probe spawned before a call never
	decides the exit for a session that has seen a call since."""
	monkeypatch.setattr(serve, "RELOAD_EXIT_DELAY", 1.0)
	_fake_package(tmp_path, monkeypatch)
	session = _session({"lint": PASS}, None, tmp_path)
	initial = serve.package_snapshot()
	entered = 0
	exit_times: list[float] = []
	loop = asyncio.get_running_loop()

	def slow_snapshot() -> dict[str, str]:
		nonlocal entered
		time.sleep(0.5)
		entered += 1
		return {"changed": "1"} if entered > 1 else initial

	monkeypatch.setattr(serve, "package_snapshot", slow_snapshot)

	def record_exit() -> None:
		reload_exits.append(1)
		exit_times.append(loop.time())

	monkeypatch.setattr(serve, "exit_for_reload", record_exit)
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		assert not (await client.call_tool("camas_list", {})).isError
		await asyncio.sleep(1.05)  # the first call's timer fired at 1.0; its walk runs
		second = asyncio.create_task(client.call_tool("camas_list", {}))
		assert not (await second).isError
		call_end = loop.time()
		await wait_until(lambda: bool(reload_exits), 4.0)
	assert reload_exits == [1]
	assert exit_times[0] - call_end >= serve.RELOAD_EXIT_DELAY


async def test_exit_arms_at_call_end_not_call_entry(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reload_exits: list[int], wait_until: Any
) -> None:
	"""The probe must arm from the call's finally — an arm-at-entry timer counts the call's
	own duration against the flush window and the gap assert fails."""
	monkeypatch.setattr(serve, "RELOAD_EXIT_DELAY", 1.0)
	_fake_package(tmp_path, monkeypatch)
	session = _session({"lint": PASS}, None, tmp_path)
	initial = serve.package_snapshot()
	exit_times: list[float] = []
	loop = asyncio.get_running_loop()
	changed = False

	def slow_snapshot() -> dict[str, str]:
		time.sleep(0.5)
		return {"changed": "1"} if changed else initial

	calls = 0

	async def slow_second(session: Session, name: str, arguments: dict[str, Any]) -> Any:
		nonlocal calls, changed
		calls += 1
		if calls == 2:
			changed = True
			await asyncio.sleep(0.8)
		return await _dispatch(session, name, arguments)

	def record_exit() -> None:
		reload_exits.append(1)
		exit_times.append(loop.time())

	monkeypatch.setattr(serve, "package_snapshot", slow_snapshot)
	monkeypatch.setattr(serve, "call", slow_second)
	monkeypatch.setattr(serve, "exit_for_reload", record_exit)
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		assert not (await client.call_tool("camas_list", {})).isError
		await asyncio.sleep(1.6)  # the first probe (1.5-2.0) saw an unchanged package
		second = asyncio.create_task(client.call_tool("camas_list", {}))
		assert not (await second).isError
		call_end = loop.time()
		await wait_until(lambda: bool(reload_exits), 4.0)
	assert reload_exits == [1]
	assert exit_times[0] - call_end >= serve.RELOAD_EXIT_DELAY


@pytest.mark.parametrize("exc", [OSError("locked"), RuntimeError("no thread")])
async def test_unreadable_package_stays_up_without_an_unretrieved_exception(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	reload_exits: list[int],
	wait_until: Any,
	exc: OSError | RuntimeError,
) -> None:
	"""A probe whose walk raises stays up — the suppression behind the docstring's "unreadable
	stays up" promise — and the next call re-probes cleanly."""
	monkeypatch.setattr(serve, "RELOAD_EXIT_DELAY", 0.05)
	_fake_package(tmp_path, monkeypatch)
	session = _session({"lint": PASS}, None, tmp_path)
	initial = serve.package_snapshot()
	raises: list[BaseException] = []
	walking = False

	def flaky_snapshot() -> dict[str, str]:
		if walking:
			raises.append(exc)
			raise exc
		return initial

	monkeypatch.setattr(serve, "package_snapshot", flaky_snapshot)
	server = serve.build_server(session)
	walking = True
	async with create_connected_server_and_client_session(server) as client:
		assert not (await client.call_tool("camas_list", {})).isError
		await wait_until(lambda: bool(raises), 4.0)
		walking = False
		assert raises == [exc]
		assert reload_exits == []
		assert not (await client.call_tool("camas_list", {})).isError
	assert reload_exits == []


async def test_stale_package_schedules_reload_even_when_the_call_raises(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reload_exits: list[int], wait_until: Any
) -> None:
	"""A raising handler cannot strand the server on stale code — the exit arms in the
	``finally``, and the client gets the tool error before the reconnect."""
	monkeypatch.setattr(serve, "RELOAD_EXIT_DELAY", 0.05)
	monkeypatch.setattr(serve, "call", _raising_call)
	pkg = _fake_package(tmp_path, monkeypatch)
	session = _session({"lint": PASS}, None, tmp_path)
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		(pkg / "a.py").write_text("y = 2\n")
		assert (await client.call_tool("camas_list", {})).isError
		await wait_until(lambda: bool(reload_exits))
	assert reload_exits == [1]


async def test_concurrent_calls_exit_after_the_last_finishes(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reload_exits: list[int], wait_until: Any
) -> None:
	"""A timer firing while a call is in flight is dropped — the slow concurrent call is never
	killed, and the exit fires only after the last call arms a fresh probe and the client is
	idle."""
	monkeypatch.setattr(serve, "RELOAD_EXIT_DELAY", 0.05)
	monkeypatch.setattr(serve, "call", functools.partial(_slow_call, [0]))
	pkg = _fake_package(tmp_path, monkeypatch)
	session = _session({"lint": PASS}, None, tmp_path)
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		(pkg / "a.py").write_text("y = 2\n")
		await asyncio.gather(
			client.call_tool("camas_list", {}),
			client.call_tool("camas_list", {}),
		)
		await wait_until(lambda: bool(reload_exits))
	assert reload_exits == [1]


@pytest.mark.skipif(sys.platform == "win32", reason="select on pipes is POSIX-only")
def test_stderr_prefers_the_drain_buffer() -> None:
	"""With a drain thread present, the failure report reads its buffer — not the pipe, which
	the drain may already have consumed."""
	assert _stderr(None, [b"oops\n"]) == "oops\n"


@pytest.mark.skipif(sys.platform == "win32", reason="select on pipes is POSIX-only")
def test_readline_raises_on_early_eof() -> None:
	"""A server that dies before writing fails with the EOF report, not a hang."""
	server = subprocess.Popen(
		[sys.executable, "-c", "pass"],
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		bufsize=1,
	)
	try:
		with pytest.raises(AssertionError, match="server closed stdout"):
			_readline(server)
	finally:
		server.wait()
		cast("IO[str]", server.stdin).close()
		cast("IO[str]", server.stdout).close()
		cast("IO[str]", server.stderr).close()


@pytest.mark.skipif(sys.platform == "win32", reason="select on pipes is POSIX-only")
def test_readline_times_out_with_stderr() -> None:
	"""A server that never writes fails at the deadline with its stderr, not a hang."""
	server = subprocess.Popen(
		[
			sys.executable,
			"-c",
			"import sys, time; print('oops', file=sys.stderr); sys.stderr.flush(); time.sleep(5)",
		],
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		bufsize=1,
	)
	try:
		with pytest.raises(AssertionError, match="timed out waiting for the server"):
			_readline(server, timeout=0.5)
	finally:
		server.kill()
		server.wait()
		cast("IO[str]", server.stdin).close()
		cast("IO[str]", server.stdout).close()
		cast("IO[str]", server.stderr).close()


class _ServerEofError(AssertionError):
	"""The live server died or stalled — distinguishable from a bad answer by the e2e guard."""


def _readline(
	server: subprocess.Popen[str],
	timeout: float = 15.0,
	stderr_chunks: list[bytes] | None = None,
) -> str:
	"""One full JSON-RPC line from the live server, bounded: a reader thread feeds a queue so a
	server wedged mid-line fails at the deadline instead of hanging the suite. ``stderr_chunks``
	is the drain thread's buffer when one exists — the failure report reads it instead of the
	pipe, so the two readers never race.

	Raises:
		_ServerEofError: on the deadline or an early EOF, with the server's stderr when it has
			written any.
	"""
	assert server.stdout is not None
	stdout: IO[str] = server.stdout
	lines: queue.Queue[str] = queue.Queue()

	def reader() -> None:
		lines.put(stdout.readline())

	threading.Thread(target=reader, daemon=True).start()
	try:
		line = lines.get(timeout=timeout)
	except queue.Empty:
		raise _ServerEofError(
			f"timed out waiting for the server; stderr: {_stderr(server, stderr_chunks)!r}"
		) from None
	if not line:
		raise _ServerEofError(
			f"server closed stdout before responding; stderr: {_stderr(server, stderr_chunks)!r}"
		)
	return line


_MAX_DIAGNOSTIC_BYTES = 65536


def _stderr(  # pragma: no cover  # behavior pinned by tests; coverage misreports the dispatch arc
	server: subprocess.Popen[str] | None, chunks: list[bytes] | None = None
) -> str:
	"""The server's stderr for a failure report: the drain thread's buffer when one exists (so
	the two readers never race), else a bounded direct read — capped, because a server flooding
	stderr must not make the diagnostic grow or block without end."""
	if chunks is not None:
		return b"".join(chunks)[:_MAX_DIAGNOSTIC_BYTES].decode(errors="replace")
	if (
		server is None or server.stderr is None
	):  # pragma: no cover  # only the unit test passes None
		return ""
	fd = server.stderr.fileno()
	data = b""
	while (
		len(data) < _MAX_DIAGNOSTIC_BYTES
	):  # pragma: no cover  # only a flooding server exits by the cap
		ready, _, _ = select.select([fd], [], [], 0.05)
		if not ready:
			break
		chunk = os.read(fd, 4096)
		if not chunk:  # pragma: no cover  # EOF on a dead server's pipe
			break
		data += chunk
	return data.decode(errors="replace")


def _send_rpc(server: subprocess.Popen[str], call_id: int) -> None:
	"""Write one ``camas_list`` call over the live server's stdio."""
	assert server.stdin is not None
	server.stdin.write(
		f'{{"jsonrpc":"2.0","id":{call_id},"method":"tools/call",'
		'"params":{"name":"camas_list","arguments":{}}}\n'
	)
	server.stdin.flush()


def _read_rpc(
	server: subprocess.Popen[str], call_id: int, stderr_chunks: list[bytes] | None = None
) -> None:
	"""Read one ``camas_list`` response and assert it answers ``call_id`` without an error."""
	line = _readline(server, stderr_chunks=stderr_chunks)
	assert f'"id":{call_id}' in line
	assert '"error"' not in line


@pytest.mark.skipif(sys.platform == "win32", reason="select on pipes is POSIX-only")
def test_live_server_answers_then_dies_when_the_package_changes(tmp_path: Path) -> None:
	"""Drive the real stdio server over JSON-RPC: a stale package is answered, then the process
	exits so the client sees EOF — the two contract halves the in-memory transport cannot
	exercise (its exit is faked and it has no ``to_thread`` writer hop)."""
	(tmp_path / "tasks.py").write_text("from camas import Task\nlint = Task('echo hi')\n")
	# The server boots through a bootstrap that points camas.__file__ at a scratch directory this
	# test owns, so the staleness probe lands there — the package itself imports normally (the
	# mypyc layouts that broke the private-copy approach are untouched) and no shared tree is
	# ever written.
	scratch = tmp_path / "scratch"
	scratch.mkdir()
	bootstrap = (
		"import camas, pathlib, sys; "
		"scratch = pathlib.Path(sys.argv[1]); "
		"(scratch / '__init__.py').write_text(''); "
		"camas.__file__ = str(scratch / '__init__.py'); "
		"from camas.mcp.serve import serve_stdio; serve_stdio(sys.argv[2:])"
	)
	server = subprocess.Popen(
		[sys.executable, "-c", bootstrap, str(scratch), "--plain"],
		cwd=tmp_path,
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		bufsize=1,
	)
	assert server.stdin is not None
	assert server.stdout is not None
	# Drain stderr continuously so a chatty server cannot deadlock itself on a full pipe.
	assert server.stderr is not None
	stderr_chunks: list[bytes] = []
	stderr_fd = server.stderr.fileno()

	def drain() -> None:
		for chunk in iter(lambda: os.read(stderr_fd, 4096), b""):
			stderr_chunks.append(chunk)  # pragma: no cover  # noqa: PERF402

	threading.Thread(target=drain, daemon=True).start()
	try:
		server.stdin.write(
			'{"jsonrpc":"2.0","id":0,"method":"initialize","params":'
			'{"protocolVersion":"2024-11-05","capabilities":{},'
			'"clientInfo":{"name":"t","version":"1"}}}\n'
		)
		server.stdin.flush()
		assert '"id":0' in _readline(server, stderr_chunks=stderr_chunks)
		for call_id in (1, 2):
			_send_rpc(server, call_id)
			_read_rpc(server, call_id, stderr_chunks)
		probe = scratch / "_stale_probe.py"
		# Call 3's finally cancels the pending call-2 probe and arms a fresh one, so the exit
		# runs only after call 3 is answered; sending call 3 before the file exists additionally
		# keeps an early fire (parent stalled between send and write) probing an unchanged
		# package. Call 3's read is the first whose flush can race the exit, so it is guarded
		# like call 4's; a stall long enough for the exit to fire before a call's answer skips
		# that call's answer check — the wait and returncode assert below still verify the exit
		# contract.
		_send_rpc(server, 3)
		probe.write_text("")
		try:
			with suppress(_ServerEofError):  # pragma: no cover  # the exit raced call 3's flush
				_read_rpc(server, 3, stderr_chunks)
			try:
				_send_rpc(server, 4)
				_read_rpc(server, 4, stderr_chunks)
			except (BrokenPipeError, _ServerEofError):  # pragma: no cover  # the exit already fired
				pass
			server.wait(timeout=15)
		finally:
			probe.unlink(missing_ok=True)
	finally:
		if server.poll() is None:
			server.kill()  # pragma: no cover  # only a wedged server takes this
		server.wait()
		server.stdin.close()
		server.stdout.close()
		server.stderr.close()
	assert server.returncode == 0
