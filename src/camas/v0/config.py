# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Project-wide configuration discovered from a ``tasks.py`` scope by type."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
	from typing import Any

	from .effect import Effect
	from .task import TaskNode


class Config(NamedTuple):
	"""Project configuration, bound under any name (conventionally ``_``) at
	module scope in ``tasks.py``.
	"""

	default_task: TaskNode | None = None
	github_task: TaskNode | None = None
	default_effects: tuple[Effect[Any], ...] | None = None
	"""``None`` defers to the engine's default; ``()`` is an explicit no-effects."""
	default_github_effects: tuple[Effect[Any], ...] | None = None
	"""``None`` defers to the engine's default; ``()`` is an explicit no-effects."""

	def bare_task(self, *, github: bool) -> TaskNode | None:
		"""The task a bare ``camas`` invocation runs."""
		if github and self.github_task is not None:
			return self.github_task
		return self.default_task

	def effects(self, *, github: bool) -> tuple[Effect[Any], ...] | None:
		"""The configured default effects for the environment, or ``None`` to defer
		to the engine's built-in default.
		"""
		if github:
			return self.default_github_effects
		return self.default_effects
