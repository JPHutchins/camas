# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Fixtures shared across the suite."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
	from collections.abc import Awaitable, Callable
	from pathlib import Path

	from camas import Task


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
	"""A committed one-file repository the drift-gate pins run inside; the identity and
	signing configs are pinned so ambient git config cannot fail the commit. Skips when git
	is absent — the nix build's hermetic test phase has none on PATH."""
	if shutil.which("git") is None:  # pragma: no cover — only hermetic builds lack git
		pytest.skip("the pins run the drift check against a real git repository")
	subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
	(tmp_path / "tracked.txt").write_text("original\n", encoding="utf-8")
	subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
	subprocess.run(
		[
			"git",
			"-c",
			"user.name=t",
			"-c",
			"user.email=t@t",
			"-c",
			"commit.gpgSign=false",
			"commit",
			"-qm",
			"init",
		],
		cwd=tmp_path,
		check=True,
	)
	return tmp_path


@pytest.fixture
def unforced_color(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Clear the color environment, so a test reads camas's own decision rather than whatever the
	developer's shell or the CI runner exported into it.
	"""
	for name in ("NO_COLOR", "FORCE_COLOR", "CLICOLOR_FORCE"):
		monkeypatch.delenv(name, raising=False)


@pytest.fixture
def forked(monkeypatch: pytest.MonkeyPatch) -> list[subprocess.Popen[bytes]]:
	"""Every child the event loop's unix subprocess transport forks during the test: the
	handle for checking a child the run never handed back as a ``Process`` (a spawn cancelled
	mid-flight). The Windows transport forks through its own ``Popen``, so a test using this
	skips on win32.
	"""
	popens: list[subprocess.Popen[bytes]] = []

	class Recording(subprocess.Popen[bytes]):
		def __init__(self, *args: Any, **kwargs: Any) -> None:
			super().__init__(*args, **kwargs)
			popens.append(self)

	monkeypatch.setattr(subprocess, "Popen", Recording)
	return popens


@pytest.fixture
def cancel_inside_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Cancel the spawning task while the real ``create_subprocess_exec`` awaits its transport
	— after the fork, before a ``Process`` comes back."""
	from camas.core import execution as execution_module

	real_spawn = execution_module._spawn_stage  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # the monkeypatched seam, kept for pass-through

	async def cancelled_mid_spawn(task: Task, **kwargs: Any) -> asyncio.subprocess.Process:
		current = asyncio.current_task()
		assert current is not None
		asyncio.get_running_loop().call_soon(
			current.cancel
		)  # zuban: ignore[call-arg] # zuban drops Task.cancel's optional msg
		return await real_spawn(task, **kwargs)

	monkeypatch.setattr(execution_module, "_spawn_stage", cancelled_mid_spawn)


@pytest.fixture
def wait_until() -> Callable[[Callable[[], bool], float], Awaitable[None]]:
	"""Polls a condition on the loop until it holds or the deadline passes — the
	mechanism-awaiting stand-in for racing sleep margins against a thread hop or a child's
	startup."""

	async def poll(condition: Callable[[], bool], timeout: float = 2.0) -> None:
		deadline = asyncio.get_running_loop().time() + timeout
		while not condition() and asyncio.get_running_loop().time() < deadline:  # noqa: ASYNC110
			await asyncio.sleep(0.05)
		if not condition():
			raise AssertionError(f"condition not met within {timeout}s")

	return poll
