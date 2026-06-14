# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Little subprocesses that respond to SIGINT differently, for exercising camas's
four-press Ctrl-C escalation (issue #16) by hand.

Driven by the sibling ``tasks.py``; pick a behavior with ``sigint_app.py <mode>``:

    polite    exits 130 on the 1st SIGINT (well-behaved CLI)        -> STOP at press 1
    cleanup   traps SIGINT, prints, sleeps ~2s, exits 0             -> SIGINT then STOP
    stubborn  ignores the 1st SIGINT, exits on the 2nd              -> STOP at press 2
    ignore    ignores SIGINT entirely; only SIGKILL stops it        -> STOP at press 3 (kill)
    leaker    ignores SIGINT and leaks a child holding stdout open  -> camas hangs after the
              kill (the pipe never EOFs), so the 4th press cancels the run

POSIX-oriented: this models the platform where camas runs the full escalation.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time


def heartbeat(label: str) -> None:
	print(f"{label}: working...", flush=True)


def polite() -> None:
	try:
		while True:
			heartbeat("polite")
			time.sleep(1)
	except KeyboardInterrupt:
		print("polite: SIGINT -> exiting 130", flush=True)
		sys.exit(130)


def cleanup() -> None:
	def on_sigint(_signum: int, _frame: object) -> None:
		print("cleanup: SIGINT -> cleaning up (~2s)", flush=True)
		time.sleep(2)
		print("cleanup: done, exiting 0", flush=True)
		sys.exit(0)

	signal.signal(signal.SIGINT, on_sigint)
	while True:
		heartbeat("cleanup")
		time.sleep(1)


def stubborn() -> None:
	hits = 0

	def on_sigint(_signum: int, _frame: object) -> None:
		nonlocal hits
		hits += 1
		if hits >= 2:
			print("stubborn: 2nd SIGINT -> exiting 130", flush=True)
			sys.exit(130)
		print("stubborn: ignoring 1st SIGINT", flush=True)

	signal.signal(signal.SIGINT, on_sigint)
	while True:
		heartbeat("stubborn")
		time.sleep(1)


def ignore() -> None:
	signal.signal(signal.SIGINT, signal.SIG_IGN)
	while True:
		heartbeat("ignore (immune to SIGINT — needs SIGKILL)")
		time.sleep(1)


def leaker() -> None:
	signal.signal(signal.SIGINT, signal.SIG_IGN)
	# A detached grandchild inherits our stdout (the pipe camas reads). It outlives
	# our SIGKILL and keeps the write end open, so camas's read never EOFs and the
	# run hangs after the 3rd press — the 4th press cancels it. The grandchild
	# self-reaps in ~60s.
	subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
	while True:
		heartbeat("leaker (leaks a child holding stdout open)")
		time.sleep(1)


BEHAVIORS = {
	"polite": polite,
	"cleanup": cleanup,
	"stubborn": stubborn,
	"ignore": ignore,
	"leaker": leaker,
}


if __name__ == "__main__":
	if len(sys.argv) != 2 or sys.argv[1] not in BEHAVIORS:
		print(f"usage: sigint_app.py {{{'|'.join(BEHAVIORS)}}}", file=sys.stderr)
		sys.exit(2)
	BEHAVIORS[sys.argv[1]]()
