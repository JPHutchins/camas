# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Project-wide configuration discovered from a ``tasks.py`` scope by type."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
	from .task import TaskNode


class Config(NamedTuple):
	"""Project configuration, bound under any name (conventionally ``_``) at
	module scope in ``tasks.py``.
	"""

	default_task: TaskNode | None = None
	github_task: TaskNode | None = None

	def bare_task(self, *, github: bool) -> TaskNode | None:
		"""The task a bare ``camas`` invocation runs."""
		if github and self.github_task is not None:
			return self.github_task
		return self.default_task
