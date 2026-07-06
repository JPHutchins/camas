# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import pytest

from camas import Claude, Config, Task

DEFAULT = Task("default", name="default")
GH = Task("gh", name="gh")
FIX = Task("fix", name="fix", mutates=True)
CHECK = Task("check", name="check")
RUNDEF = Task("rundef", name="rundef")


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


def test_gate_check_prefers_agent_check_else_bare_task() -> None:
	assert (
		Config(default_task=DEFAULT, agent=Claude(fix=FIX, check=CHECK)).gate_check(github=False)
		is CHECK
	)
	assert Config(default_task=DEFAULT, agent=Claude(fix=FIX)).gate_check(github=False) is DEFAULT
	assert Config(default_task=DEFAULT).gate_check(github=False) is DEFAULT


def test_gate_fix_is_the_agent_fix_node_or_none() -> None:
	assert Config(agent=Claude(fix=FIX)).gate_fix() is FIX
	assert Config(default_task=DEFAULT).gate_fix() is None


@pytest.mark.parametrize(
	("config", "expected"),
	[
		(Config(), None),
		(Config(default_task=DEFAULT), DEFAULT),
		(Config(github_task=GH), GH),
		(Config(default_task=DEFAULT, github_task=GH), GH),
		(Config(default_task=DEFAULT, agent=Claude(fix=FIX)), DEFAULT),
		(Config(default_task=DEFAULT, agent=Claude(fix=FIX, check=CHECK)), CHECK),
		(Config(default_task=DEFAULT, agent=Claude(fix=FIX, default=RUNDEF)), RUNDEF),
		(
			Config(
				default_task=DEFAULT,
				github_task=GH,
				agent=Claude(fix=FIX, check=CHECK, default=RUNDEF),
			),
			RUNDEF,
		),
	],
)
def test_run_default_resolves_the_chain(config: Config, expected: Task | None) -> None:
	"""``run_default`` prefers the agent's ``default``, then ``check``, then the
	github or default task.
	"""
	assert config.run_default() is expected
