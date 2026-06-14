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


@pytest.mark.parametrize("github", [False, True])
def test_effects_default_is_none_deferring_to_engine(github: bool) -> None:
	"""Unset effects return ``None`` — the signal for the engine to substitute its
	environment default, keeping the concrete effects out of the type layer.
	"""
	assert Config().effects(github=github) is None


def test_effects_returns_per_environment_override() -> None:
	from camas.effect.summary import Summary

	local = (Summary(),)
	gh = (Summary(show_passing=True),)
	config = Config(default_effects=local, default_github_effects=gh)
	assert config.effects(github=False) is local
	assert config.effects(github=True) is gh
