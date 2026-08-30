# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``Clean`` drift gate: run a mutating generator, fail if the tree is not exactly as
it was (#233)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

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
	assert [task.name for task in node.tasks] == ["make gen-before", None, "make gen-after"]
	match node.tasks:
		case (Task(cmd=before_cmd), _, Task(cmd=after_cmd)):
			assert before_cmd == GIT_PORCELAIN().cmd
			assert after_cmd == GIT_PORCELAIN().cmd
		case _:
			raise AssertionError(f"unexpected shape: {node.tasks!r}")


def test_clean_before_false_drops_the_precheck() -> None:
	node = Clean(Task("make gen", mutates=True), before=False)
	assert [task.name for task in node.tasks] == [None, "make gen-after"]


def test_clean_uses_a_custom_check() -> None:
	check = Task("python -c pass", name="my-clean")
	node = Clean(Task("make gen", mutates=True), check=check)
	match node.tasks:
		case (Task(cmd=before_cmd), _, Task(cmd=after_cmd, env=after_env)):
			assert before_cmd == "python -c pass"
			assert after_cmd == "python -c pass"
			assert after_env is check.env
		case _:
			raise AssertionError(f"unexpected shape: {node.tasks!r}")


def test_clean_rejects_a_non_mutating_generator() -> None:
	with pytest.raises(ValueError, match="mutates=True"):
		Clean(Task("make gen"))
	with pytest.raises(ValueError, match="mutates=True"):
		Clean("make gen")
	with pytest.raises(ValueError, match="mutates=True"):
		Clean(cast("Task", Sequential(Task("gen a", mutates=True), Task("gen b", mutates=True))))
	with pytest.raises(ValueError, match="check must be a Task"):
		Clean(
			Task("make gen", mutates=True), check=cast("Task", Sequential(Task("c1"), Task("c2")))
		)


def test_clean_rejects_a_scoped_check() -> None:
	"""A check whose ``when``/``cwd`` scoping could prune it while the mutator still runs
	would silently green a dirtied tree on scoped runs."""
	with pytest.raises(ValueError, match="when="):
		Clean(Task("make gen", mutates=True), check=Task("python -c pass", when="src"))
	with pytest.raises(ValueError, match="cwd"):
		Clean(Task("make gen", mutates=True), check=Task("python -c pass", cwd="sub"))


def test_clean_check_leaves_keep_the_checks_fields() -> None:
	check = Task(
		"python -c pass",
		name="verify",
		paths="src",
		help="verify generated output",
		agent_format=("--format json", "raw"),
	)
	node = Clean(Task("make gen", mutates=True), check=check)
	for leaf in (node.tasks[0], node.tasks[2]):
		assert isinstance(leaf, Task)
		assert leaf.paths == "src"
		assert leaf.help == "verify generated output"
		assert leaf.agent_format == check.agent_format


def test_clean_named_gate_prefixes_its_check_leaves() -> None:
	node = Clean(Task("make gen", mutates=True), name="openapi")
	assert [task.name for task in node.tasks] == ["openapi-before", None, "openapi-after"]


def test_clean_coerces_a_string_check() -> None:
	node = Clean(Task("make gen", mutates=True), check="git status --porcelain")
	assert [task.name for task in node.tasks] == ["make gen-before", None, "make gen-after"]
	match node.tasks:
		case (Task(cmd=before_cmd), _, Task(cmd=after_cmd)):
			assert before_cmd == "git status --porcelain"
			assert after_cmd == "git status --porcelain"
		case _:
			raise AssertionError(f"unexpected shape: {node.tasks!r}")


def test_clean_passes_a_clean_tree(git_repo: Path) -> None:
	"""A generator that writes nothing leaves the tree clean; the run succeeds."""
	node = Clean(Task("python -c pass", mutates=True))

	async def scenario() -> RunResult:
		return await run(node, base=git_repo)

	result = asyncio.run(scenario())
	assert result.returncode == 0
	assert all(isinstance(r.completion, Finished) for r in result.results)


def test_clean_fails_when_the_generator_dirties_the_tree(git_repo: Path) -> None:
	"""The after-check leaf fails and its output is the drift diagnostic."""
	node = Clean(Task(("python", "-c", "open('tracked.txt', 'a').write('drift\\n')"), mutates=True))

	async def scenario() -> RunResult:
		return await run(node, base=git_repo)

	result = asyncio.run(scenario())
	assert result.returncode == 1
	after = result.results[2].completion
	assert isinstance(after, Finished)
	assert after.returncode == 1
	assert b"tracked.txt" in b"".join(after.output)


def test_clean_fails_fast_on_a_dirty_start(git_repo: Path) -> None:
	"""A dirty tree fails the before-check first and the generator is skipped, its writes
	never landing."""
	(git_repo / "tracked.txt").write_text("preexisting dirt\n", encoding="utf-8")
	node = Clean(
		Task(
			("python", "-c", "open('tracked.txt', 'a').write('generator ran\\n')"),
			mutates=True,
		)
	)

	async def scenario() -> RunResult:
		return await run(node, base=git_repo)

	result = asyncio.run(scenario())
	assert result.returncode == 1
	assert isinstance(result.results[1].completion, Skipped)
	assert (git_repo / "tracked.txt").read_text(encoding="utf-8") == "preexisting dirt\n"


def test_clean_fails_in_a_non_git_directory(git_repo: Path) -> None:
	"""git's own failure (no repository) must fail the gate, not silently green it. The
	directory sits beside the repo — inside it, git would walk up and find the parent."""
	non_repo = git_repo.parent / "no-repo"
	non_repo.mkdir()
	node = Clean(Task("python -c pass", mutates=True))

	async def scenario() -> RunResult:
		return await run(node, base=non_repo)

	result = asyncio.run(scenario())
	assert result.returncode == 1
