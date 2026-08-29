# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``Clean`` drift gate: run a mutating generator, fail if the tree is not exactly as
it was (#233)."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Final

import pytest
from typing_extensions import assert_type

from camas import GIT_PORCELAIN, Clean, Sequential, Task
from camas.core.execution import run
from camas.v0.completion import Finished, Skipped

if TYPE_CHECKING:
	from pathlib import Path

	from camas.core.completion import RunResult


def test_clean_returns_a_sequential_typed() -> None:
	assert_type(Clean(Task("make gen", mutates=True)), Sequential)


def test_clean_expands_to_before_mutater_after() -> None:
	node = Clean(Task("make gen", mutates=True))
	assert [task.name for task in node.tasks] == ["clean-before", None, "clean-after"]
	match node.tasks:
		case (Task(cmd=before_cmd), _, Task(cmd=after_cmd)):
			assert before_cmd == GIT_PORCELAIN.cmd
			assert after_cmd == GIT_PORCELAIN.cmd
		case _:
			raise AssertionError(f"unexpected shape: {node.tasks!r}")


def test_clean_before_false_drops_the_precheck() -> None:
	node = Clean(Task("make gen", mutates=True), before=False)
	assert [task.name for task in node.tasks] == [None, "clean-after"]


def test_clean_uses_a_custom_check() -> None:
	check = Task("python -c pass", name="my-clean")
	node = Clean(Task("make gen", mutates=True), check=check)
	match node.tasks:
		case (Task(cmd=before_cmd), _, Task(env=after_env)):
			assert before_cmd == "python -c pass"
			assert after_env is check.env
		case _:
			raise AssertionError(f"unexpected shape: {node.tasks!r}")


def test_clean_rejects_a_non_mutating_generator() -> None:
	with pytest.raises(ValueError, match="mutates=True"):
		Clean(Task("make gen"))


_HAS_GIT: Final = shutil.which("git") is not None
"""The functional pins shell out to a real git repository; hermetic environments (the nix
build's test phase) have no git on PATH."""


def _git_repo(tmp_path: Path) -> None:
	"""A committed one-file repository the functional pins run inside."""
	subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
	(tmp_path / "tracked.txt").write_text("original\n", encoding="utf-8")
	subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
	subprocess.run(
		["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"],
		cwd=tmp_path,
		check=True,
	)


@pytest.mark.skipif(
	sys.platform == "win32" or not _HAS_GIT,
	reason="the default check is POSIX shell and the pin needs a real git repository",
)
def test_clean_passes_a_clean_tree(tmp_path: Path) -> None:
	"""A generator that writes nothing leaves the tree clean; the run succeeds."""
	_git_repo(tmp_path)
	node = Clean(Task("python -c pass", mutates=True))

	async def scenario() -> RunResult:
		return await run(node, base=tmp_path)

	result = asyncio.run(scenario())
	assert result.returncode == 0
	assert all(isinstance(r.completion, Finished) for r in result.results)


@pytest.mark.skipif(
	sys.platform == "win32" or not _HAS_GIT,
	reason="the default check is POSIX shell and the pin needs a real git repository",
)
def test_clean_fails_when_the_generator_dirties_the_tree(tmp_path: Path) -> None:
	"""The after-check leaf fails and its output is the drift diagnostic."""
	_git_repo(tmp_path)
	node = Clean(Task(("python", "-c", "open('tracked.txt', 'a').write('drift\\n')"), mutates=True))

	async def scenario() -> RunResult:
		return await run(node, base=tmp_path)

	result = asyncio.run(scenario())
	assert result.returncode == 1
	after = result.results[2].completion
	assert isinstance(after, Finished)
	assert after.returncode == 1
	assert b"tracked.txt" in b"".join(after.output)


@pytest.mark.skipif(
	sys.platform == "win32" or not _HAS_GIT,
	reason="the default check is POSIX shell and the pin needs a real git repository",
)
def test_clean_fails_fast_on_a_dirty_start(tmp_path: Path) -> None:
	"""A dirty tree fails clean-before first and the generator is skipped, its writes never
	landing."""
	_git_repo(tmp_path)
	(tmp_path / "tracked.txt").write_text("preexisting dirt\n", encoding="utf-8")
	node = Clean(
		Task(
			("python", "-c", "open('tracked.txt', 'a').write('generator ran\\n')"),
			mutates=True,
		)
	)

	async def scenario() -> RunResult:
		return await run(node, base=tmp_path)

	result = asyncio.run(scenario())
	assert result.returncode == 1
	assert isinstance(result.results[1].completion, Skipped)
	assert (tmp_path / "tracked.txt").read_text(encoding="utf-8") == "preexisting dirt\n"
