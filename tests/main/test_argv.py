# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from pathlib import Path

import pytest

from camas import Parallel, Sequential, Task
from camas.main.argv import apply_passthrough, split_passthrough


def test_split_passthrough_no_separator() -> None:
	split = split_passthrough(["a", "b"])
	assert split.head == ("a", "b")
	assert split.passthrough == ()


def test_split_passthrough_with_separator() -> None:
	split = split_passthrough(["mytask", "--", "-v", "-k", "x"])
	assert split.head == ("mytask",)
	assert split.passthrough == ("-v", "-k", "x")


def test_split_passthrough_empty_passthrough() -> None:
	split = split_passthrough(["mytask", "--"])
	assert split.head == ("mytask",)
	assert split.passthrough == ()


def test_split_passthrough_only_first_separator() -> None:
	split = split_passthrough(["mytask", "--", "--", "x"])
	assert split.passthrough == ("--", "x")


def test_apply_passthrough_str_cmd_stays_string() -> None:
	assert apply_passthrough(Task("pytest"), ("-v",)) == Task("pytest -v")


def test_apply_passthrough_str_cmd_preserves_quoting() -> None:
	"""String cmds keep their original quoting; passthrough args are shell-joined."""
	assert apply_passthrough(Task("git commit -m 'big msg'"), ("--no-verify",)) == Task(
		"git commit -m 'big msg' --no-verify"
	)


def test_apply_passthrough_str_cmd_quotes_passthrough_with_spaces() -> None:
	assert apply_passthrough(Task("pytest"), ("-k", "a b")) == Task("pytest -k 'a b'")


def test_apply_passthrough_tuple_cmd_preserves_name_env_cwd() -> None:
	original = Task(("pytest",), name="test", env={"X": "1"}, cwd=Path("/tmp"))
	assert apply_passthrough(original, ("-v",)) == Task(
		("pytest", "-v"), name="test", env={"X": "1"}, cwd=Path("/tmp")
	)


def test_apply_passthrough_str_cmd_preserves_name_env_cwd() -> None:
	original = Task("pytest", name="test", env={"X": "1"}, cwd=Path("/tmp"))
	assert apply_passthrough(original, ("-v",)) == Task(
		"pytest -v", name="test", env={"X": "1"}, cwd=Path("/tmp")
	)


def test_apply_passthrough_rejects_sequential() -> None:
	with pytest.raises(ValueError, match="only apply to Task, got Sequential"):
		apply_passthrough(Sequential(Task("a"), Task("b")), ("-v",))


def test_apply_passthrough_rejects_parallel() -> None:
	with pytest.raises(ValueError, match="only apply to Task, got Parallel"):
		apply_passthrough(Parallel(Task("a"), Task("b")), ("-v",))
