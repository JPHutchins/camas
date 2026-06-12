# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from camas import Parallel, Sequential, Task
from camas.main.tasks import load_tasks_from_py

FIXTURE = Path(__file__).parent / "fixtures" / "config" / "tasks.py"
SUMMARY = "--effects=(Summary(SummaryOptions(show_passing=True)),)"


def _camas(*args: str, github: bool) -> subprocess.CompletedProcess[str]:
	env = {**os.environ, "NO_COLOR": "1"}
	env.pop("GITHUB_ACTIONS", None)
	if github:
		env["GITHUB_ACTIONS"] = "true"
	return subprocess.run(
		[sys.executable, "-m", "camas", *args],
		cwd=FIXTURE.parent,
		capture_output=True,
		text=True,
		encoding="utf-8",
		env=env,
		check=False,
	)


def test_config_resolves_promoted_default_and_github_tasks() -> None:
	"""The loader exposes the project ``Config`` with both task fields resolved to
	their promoted (named) bindings."""
	_tasks, _effects, config = load_tasks_from_py(FIXTURE)
	assert config is not None
	assert config.default_task == Sequential(
		Task(("python", "-c", "print('lint ran')"), name="lint"),
		Task(("python", "-c", "print('test ran')"), name="test"),
		name="ci",
	)
	assert isinstance(config.github_task, Parallel)
	assert config.github_task.name == "ci_full"


def test_bare_camas_runs_default_task() -> None:
	"""Bare ``camas`` (no GitHub Actions) runs ``default_task`` (``ci``)."""
	result = _camas(SUMMARY, github=False)
	assert result.returncode == 0, result.stderr
	assert "lint ran" in result.stdout
	assert "test ran" in result.stdout
	assert "cov ran" not in result.stdout


def test_bare_camas_runs_github_task_under_actions() -> None:
	"""Bare ``camas`` under ``GITHUB_ACTIONS=true`` runs ``github_task`` (``ci_full``)."""
	result = _camas(SUMMARY, github=True)
	assert result.returncode == 0, result.stderr
	assert "cov ran" in result.stdout
