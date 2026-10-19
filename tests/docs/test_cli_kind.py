# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The `cli <dir> <args>` directive: a verbatim camas session, pinned so it renders the same everywhere."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from docs.readme import main

if TYPE_CHECKING:
	from collections.abc import Callable
	from pathlib import Path

TASKS = 'from camas import Task\nhello = Task(("python", "-c", "print(\'hi\')"))\n'


def test_help_in_a_scratch_directory_drops_the_machine_specific_source_line(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	session = generated("cli - --help")
	assert session[:3] == ("```console", "$ camas --help", session[2])
	assert session[2].startswith("usage: camas")
	assert session[-1] == "```"
	assert not any("Source (docs live here)" in line for line in session)


def test_a_project_directory_shows_as_a_relative_path(
	project: Path, generated: Callable[[str], tuple[str, ...]]
) -> None:
	(project / "tasks.py").write_text(TASKS, encoding="utf-8")
	session = generated("cli . --list")
	assert session[:3] == (
		"```console",
		"$ camas --list",
		f"Available tasks from .../{project.resolve().name}/tasks.py:",
	)


def test_quoted_arguments_are_shown_as_typed(
	project: Path, generated: Callable[[str], tuple[str, ...]]
) -> None:
	(project / "tasks.py").write_text(TASKS, encoding="utf-8")
	assert (
		generated("cli . '--effects=(Summary(),)' hello")[1]
		== "$ camas '--effects=(Summary(),)' hello"
	)


@pytest.mark.parametrize(
	("directive", "error"),
	[
		("cli nope --list", "docs/README.in.md:1: no directory 'nope'"),
		("cli . nosuchtask", "docs/README.in.md:1: `camas nosuchtask` exited 2"),
		("cli . 'unclosed", "docs/README.in.md:1: unparsable directive"),
	],
)
def test_a_failing_session_is_an_error(
	project: Path,
	template: Callable[[str], None],
	capsys: pytest.CaptureFixture[str],
	directive: str,
	error: str,
) -> None:
	(project / "tasks.py").write_text(TASKS, encoding="utf-8")
	template(f"<!-- GENERATED: {directive} -->\n")
	assert main(["--check"]) == 2
	assert capsys.readouterr().err.startswith(error)
