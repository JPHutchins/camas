# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import pytest

from camas import Config, Task

DEFAULT = Task("default", name="default")
GH = Task("gh", name="gh")


@pytest.mark.parametrize(
	("config", "github", "expected"),
	[
		(Config(), False, None),
		(Config(), True, None),
		(Config(default_task=DEFAULT), False, DEFAULT),
		(Config(default_task=DEFAULT), True, DEFAULT),
		(Config(default_task=DEFAULT, github_task=GH), False, DEFAULT),
		(Config(default_task=DEFAULT, github_task=GH), True, GH),
		(Config(github_task=GH), False, None),
		(Config(github_task=GH), True, GH),
	],
)
def test_bare_task_resolution(config: Config, github: bool, expected: Task | None) -> None:
	"""``github_task`` wins under GitHub Actions, falling back to ``default_task``."""
	assert config.bare_task(github=github) is expected
