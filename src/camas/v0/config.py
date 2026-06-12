# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Project-wide configuration discovered from a ``tasks.py`` scope by type."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
	from .task import TaskNode


class Config(NamedTuple):
	"""Project-wide camas configuration, bound at module scope in ``tasks.py``
	(conventionally ``camas_config = Config(...)``) and discovered by type.

	``default_task`` is what bare ``camas`` runs; ``github_task`` is its
	GitHub Actions counterpart, falling back to ``default_task`` when unset.

	>>> from camas.v0.task import Task
	>>> Config().bare_task(github=False) is None
	True
	>>> Config(default_task=Task("pytest")).bare_task(github=True)
	Task(cmd='pytest', name=None, env={}, cwd=None)
	>>> ci = Config(default_task=Task("pytest"), github_task=Task("pytest --cov"))
	>>> ci.bare_task(github=True)
	Task(cmd='pytest --cov', name=None, env={}, cwd=None)
	"""

	default_task: TaskNode | None = None
	github_task: TaskNode | None = None

	def bare_task(self, *, github: bool) -> TaskNode | None:
		"""The task bare ``camas`` runs: ``github_task`` under GitHub Actions
		(falling back to ``default_task``), else ``default_task``.
		"""
		if github and self.github_task is not None:
			return self.github_task
		return self.default_task
