# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Command (#368): POSIX signals reach an attached child the way a shell delivers them."""

from __future__ import annotations

import sys

if sys.platform != "win32":  # pragma: no branch
	import os
	import signal
	import subprocess
	import threading
	import time
	from contextlib import suppress
	from textwrap import dedent
	from typing import TYPE_CHECKING

	import pytest

	from camas import Sequential, Task
	from camas.core.attached import (
		Exited,
		Signaled,
		in_foreground,
		outcome_of,
		run_attached,
		run_step,
		spawn_stages,
	)

	if TYPE_CHECKING:
		from pathlib import Path

	def py(code: str, *args: str) -> Task:
		return Task(("python", "-c", code, *args))

	def touch_then_sleep(seconds: float) -> str:
		return (
			f"import pathlib, sys, time; pathlib.Path(sys.argv[1]).touch(); time.sleep({seconds})"
		)

	def foreground_of(_fd: int) -> int:
		return os.getpgrp()

	def signal_main_thread_when(ready: Path, signum: int) -> threading.Thread:
		def deliver() -> None:
			deadline = time.monotonic() + 10
			while not ready.exists() and time.monotonic() < deadline:
				time.sleep(0.01)
			if ready.exists():  # pragma: no branch  # a child that never readies sends nothing
				signal.pthread_kill(threading.main_thread().ident or 0, signum)

		thread = threading.Thread(target=deliver, daemon=True)
		thread.start()
		return thread

	def test_a_forwarded_signal_kills_the_child_and_is_reraised(tmp_path: Path) -> None:
		ready = tmp_path / "ready"
		signal_main_thread_when(ready, signal.SIGTERM)
		assert run_step(py(touch_then_sleep(30), str(ready)), None) == Signaled(signal.SIGTERM)

	def test_a_forwarded_signal_ends_the_sequence_even_when_the_child_exits_cleanly(
		tmp_path: Path,
	) -> None:
		ready = tmp_path / "ready"
		flashed = tmp_path / "flashed"
		graceful = py(
			"import pathlib, signal, sys, time; "
			"signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
			"pathlib.Path(sys.argv[1]).touch(); time.sleep(30)",
			str(ready),
		)
		flash = py("import pathlib, sys; pathlib.Path(sys.argv[1]).touch()", str(flashed))
		signal_main_thread_when(ready, signal.SIGTERM)
		assert run_attached(Sequential(graceful, flash), None) == Signaled(signal.SIGTERM)
		assert not flashed.exists()

	def test_a_terminal_ctrl_c_is_left_to_the_child(
		tmp_path: Path, monkeypatch: pytest.MonkeyPatch
	) -> None:
		monkeypatch.setattr(os, "ctermid", lambda: os.devnull)
		monkeypatch.setattr(os, "tcgetpgrp", foreground_of)
		ready = tmp_path / "ready"
		signal_main_thread_when(ready, signal.SIGINT)
		assert run_step(py(touch_then_sleep(3), str(ready)), None) == Exited(0)

	def test_a_ctrl_c_without_a_controlling_terminal_is_forwarded(
		tmp_path: Path, monkeypatch: pytest.MonkeyPatch
	) -> None:
		monkeypatch.setattr(os, "ctermid", lambda: str(tmp_path / "no-terminal"))
		assert not in_foreground()
		ready = tmp_path / "ready"
		signal_main_thread_when(ready, signal.SIGINT)
		assert run_step(py(touch_then_sleep(30), str(ready)), None) == Signaled(signal.SIGINT)

	def test_a_signal_received_before_a_stage_starts_still_reaches_it() -> None:
		procs: list[subprocess.Popen[bytes]] = []
		sleeper = py("import time; time.sleep(30)")
		spawn_stages(procs, (sleeper, sleeper), None, (signal.SIGTERM, signal.SIGUSR1))
		assert [proc.wait() for proc in procs] == [-signal.SIGTERM, -signal.SIGTERM]

	IN_TERMINAL = dedent("""\
		import fcntl, os, sys, termios
		os.setsid()
		terminal = os.open(sys.argv[1], os.O_RDWR | os.O_NOCTTY)
		fcntl.ioctl(terminal, termios.TIOCSCTTY, 0)
		stdio = terminal if sys.argv[2] == "terminal" else os.open(os.devnull, os.O_RDWR)
		for fd in (0, 1, 2):
		    os.dup2(stdio, fd)
		os.execv(sys.executable, [sys.executable, *sys.argv[3:]])
	""")

	FOREGROUND_PROBE = dedent("""\
		import os, sys
		from camas.core.attached import in_foreground
		def report():
		    with open(sys.argv[1], "a") as file:
		        file.write(f"{in_foreground()}\\n")
		report()
		if os.fork() == 0:
		    os.setpgid(0, 0)
		    report()
		    os._exit(0)
		os.wait()
	""")

	@pytest.mark.parametrize("stdio", ["terminal", "redirected"])
	def test_camas_is_in_the_terminals_foreground_whatever_its_stdio(
		tmp_path: Path, stdio: str
	) -> None:
		import pty

		main, sub = pty.openpty()
		report = tmp_path / "report"
		try:
			subprocess.run(
				[
					*(sys.executable, "-c", IN_TERMINAL, os.ttyname(sub), stdio),
					*("-c", FOREGROUND_PROBE, str(report)),
				],
				timeout=30,
				check=True,
			)
		finally:
			os.close(sub)
			os.close(main)
		assert report.read_text().split() == ["True", "False"]

	@pytest.mark.parametrize("stdio", ["terminal", "redirected"])
	def test_ctrl_c_on_the_terminal_reaches_the_child_once_and_camas_dies_of_it(
		tmp_path: Path, stdio: str
	) -> None:
		import pty

		(tmp_path / "tasks.py").write_text(
			dedent("""\
				from camas import Command
				counter = Command(("python", "-c", '''
				import pathlib, signal, time
				sigints = pathlib.Path("sigints")
				def count(signum, frame):
				    sigints.write_text(sigints.read_text() + "INT")
				sigints.write_text("")
				signal.signal(signal.SIGINT, count)
				pathlib.Path("ready").touch()
				deadline = time.monotonic() + 30
				while not sigints.read_text() and time.monotonic() < deadline:
				    time.sleep(0.01)
				time.sleep(0.5)
				signal.signal(signal.SIGINT, signal.SIG_DFL)
				signal.raise_signal(signal.SIGINT)
				'''))
			""")
		)
		main, sub = pty.openpty()
		proc = subprocess.Popen(
			[sys.executable, "-c", IN_TERMINAL, os.ttyname(sub), stdio, "-m", "camas", "counter"],
			cwd=tmp_path,
			env={**os.environ, "CLAUDECODE": "", "CAMAS_AGENT": ""},
		)
		os.close(sub)
		try:
			deadline = time.monotonic() + 10
			while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
				time.sleep(0.01)
			assert (tmp_path / "ready").exists()
			os.write(main, b"\x03")
			try:
				returncode = proc.wait(timeout=10)
			except (
				subprocess.TimeoutExpired
			):  # pragma: no cover  # diagnoses a hang seen once in CI
				pytest.fail(outlived_sigint(proc.pid, main, tmp_path / "sigints"))
			assert returncode == -signal.SIGINT
			assert (tmp_path / "sigints").read_text() == "INT"
		finally:
			os.close(main)
			with suppress(ProcessLookupError):
				os.killpg(proc.pid, signal.SIGKILL)
			proc.wait()

	def outlived_sigint(  # pragma: no cover  # diagnoses a hang seen once in CI
		pgid: int, fd: int, sigints: Path
	) -> str:
		table = subprocess.run(
			["ps", "-A", "-o", "pid=,pgid=,stat=,args="],
			capture_output=True,
			text=True,
			check=False,
		).stdout
		group = [row for row in table.splitlines() if row.split()[1:2] == [str(pgid)]]
		late = read_until(fd, b"\0", deadline=time.monotonic() + 1)
		return (
			f"camas outlived SIGINT; sigints={sigints.read_text()!r}; "
			f"pty={late!r}; group:\n" + "\n".join(group)
		)

	def read_until(  # pragma: no cover  # diagnoses a hang seen once in CI
		fd: int, marker: bytes, *, deadline: float
	) -> bytes:
		import select

		seen = b""
		while (
			marker not in seen
			and time.monotonic() < deadline
			and select.select([fd], [], [], max(0.0, deadline - time.monotonic()))[0]
		):
			seen += os.read(fd, 1024)
		return seen

	def test_a_terminating_signal_overrides_a_clean_exit() -> None:
		assert outcome_of(0, (signal.SIGINT,)) == Exited(0)
		assert outcome_of(0, (signal.SIGINT, signal.SIGTERM)) == Signaled(signal.SIGTERM)
		assert outcome_of(1, (signal.SIGTERM,)) == Exited(1)
