# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The `example <module> <binding>` directive: a doctested tasks.py and its camas commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from docs.readme import main

if TYPE_CHECKING:
	from collections.abc import Callable


def test_an_example_shows_its_tasks_py_and_commands(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	assert generated("example demo.examples hello") == (
		"```python",
		"from camas import Task",
		'hello = Task("echo hi")',
		"```",
		"",
		"```console",
		"$ camas '--effects=(Summary(),)' hello",
		"$ camas --list",
		"```",
	)


def test_an_example_without_camas_calls_shows_only_its_tasks_py(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	assert generated("example demo.examples Loud") == (
		"```python",
		"from camas import Task",
		"class Loud: ...",
		'again = Task("echo again")',
		"```",
	)


@pytest.mark.parametrize(
	("directive", "error"),
	[
		(
			"example demo.examples again",
			"docs/README.in.md:1: 2 examples in src/demo/examples.py's docstring bind 'again'",
		),
		(
			"example demo.examples nope",
			"docs/README.in.md:1: no example in src/demo/examples.py's docstring binds 'nope'",
		),
		("example demo.nope hello", "docs/README.in.md:1: no module 'demo.nope' under src/"),
	],
)
def test_a_missing_or_ambiguous_example_is_an_error(
	template: Callable[[str], None], capsys: pytest.CaptureFixture[str], directive: str, error: str
) -> None:
	template(f"<!-- GENERATED: {directive} -->\n")
	assert main(["--check"]) == 2
	assert capsys.readouterr().err.startswith(error)
