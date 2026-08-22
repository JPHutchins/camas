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
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reload_exits: list[int]
) -> None:
	"""The exit re-checks the snapshot when it fires: a package reverted before the timer ran
	keeps the server up instead of exiting on a stale observation."""
	monkeypatch.setattr(serve, "RELOAD_EXIT_DELAY", 1.0)
	pkg = _fake_package(tmp_path, monkeypatch)
	session = _session({"lint": PASS}, None, tmp_path)
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		(pkg / "a.py").write_text("y = 2\n")
		assert not (await client.call_tool("camas_list", {})).isError
		(pkg / "a.py").write_text("x = 1\n")
		await asyncio.sleep(1.1)
	assert reload_exits == []


async def test_mid_call_change_arms_the_exit(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reload_exits: list[int]
) -> None:
	"""A package change landing while the call runs is caught by the exit-time re-sample — the
	call-start sample alone would leave the old import serving one extra call."""
	monkeypatch.setattr(serve, "RELOAD_EXIT_DELAY", 0.05)
	_fake_package(tmp_path, monkeypatch)
	session = _session({"lint": PASS}, None, tmp_path)
	initial = serve.package_snapshot()
	calls = 0

	def snapshot() -> dict[str, str]:
		nonlocal calls
		calls += 1
		return {"changed": "1"} if calls >= 3 else initial

	monkeypatch.setattr(serve, "package_snapshot", snapshot)
	async with create_connected_server_and_client_session(serve.build_server(session)) as client:
		assert not (await client.call_tool("camas_list", {})).isError
		await asyncio.sleep(serve.RELOAD_EXIT_DELAY + 0.05)
	assert reload_exits == [1]


async def test_stale_package_schedules_reload_even_when_the_call_raises(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reload_exits: list[int]
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
		await asyncio.sleep(serve.RELOAD_EXIT_DELAY)
	assert reload_exits == [1]


async def test_slow_concurrent_calls_rearm_until_idle(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reload_exits: list[int]
) -> None:
	"""The exit re-arms while a call is in flight — a slow concurrent call is never killed, and
	the exit fires exactly once the server is idle."""
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
		await asyncio.sleep(0.8)
	assert reload_exits == [1]


@pytest.mark.skipif(sys.platform == "win32", reason="select on pipes is POSIX-only")
def test_stderr_prefers_the_drain_buffer() -> None:
	"""With a drain thread present, the failure report reads its buffer — not the pipe, which
	the drain may already have consumed."""
	assert _stderr(None, [b"oops\n"]) == "oops\n"


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
		AssertionError: on the deadline or an early EOF, with the server's stderr when it has
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
		raise AssertionError(
			f"timed out waiting for the server; stderr: {_stderr(server, stderr_chunks)!r}"
		) from None
	if not line:
		raise AssertionError(
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


def _rpc(
	server: subprocess.Popen[str], call_id: int, stderr_chunks: list[bytes] | None = None
) -> str:
	"""Send one ``camas_list`` call over the live server's stdio and return its response line."""
	assert server.stdin is not None
	server.stdin.write(
		f'{{"jsonrpc":"2.0","id":{call_id},"method":"tools/call",'
		'"params":{"name":"camas_list","arguments":{}}}\n'
	)
	server.stdin.flush()
	line = _readline(server, stderr_chunks=stderr_chunks)
	assert f'"id":{call_id}' in line
	assert '"error"' not in line
	return line


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
			_rpc(server, call_id, stderr_chunks)
		probe = scratch / "_stale_probe.py"
		probe.write_text("")
		try:
			_rpc(server, 3, stderr_chunks)
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
