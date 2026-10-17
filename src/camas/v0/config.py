# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Project-wide configuration discovered from a ``tasks.py`` scope by type."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple, TypeAlias

if TYPE_CHECKING:
	from pathlib import Path
	from typing import Any

	from .effect import Effect
	from .task import TaskNode


DEFAULT_CAMAS_DIR: Final = ".camas"


class Claude(NamedTuple):
	"""The Claude Code agent integration."""

	fix: TaskNode
	"""The deterministic, behavior-preserving autofix node."""
	check: TaskNode | None = None
	"""The node the gate checks; ``None`` defers to the project's bare task."""
	default: TaskNode | None = None
	"""The task a no-task ``camas_run`` runs; ``None`` defers to the check node."""


Agent: TypeAlias = Claude


class Config(NamedTuple):
	"""Project configuration for a ``tasks.py``."""

	default_task: TaskNode | None = None
	github_task: TaskNode | None = None
	"""The bare task under GitHub Actions; ``None`` defers to ``default_task``."""
	default_effects: tuple[Effect[Any], ...] | None = None
	"""``None`` keeps the engine's default; ``()`` runs none."""
	default_github_effects: tuple[Effect[Any], ...] | None = None
	"""``None`` keeps the engine's default; ``()`` runs none."""
	camas_dir: str = DEFAULT_CAMAS_DIR
	"""Subdirectory for run logs and timings; delete it to opt out."""
	agent: Agent | None = None
	leaf_color: bool = True
	"""Whether camas forces color on in leaf subprocesses."""

	def camas_path(self, base: Path) -> Path:
		"""The resolved camas directory under ``base``."""
		return base / self.camas_dir

	def bare_task(self, *, github: bool) -> TaskNode | None:
		"""The task a bare ``camas`` invocation runs."""
		if github and self.github_task is not None:
			return self.github_task
		return self.default_task

	def effects(self, *, github: bool) -> tuple[Effect[Any], ...] | None:
		"""The configured default effects for the environment."""
		if github:
			return self.default_github_effects
		return self.default_effects

	def gate_check(self, *, github: bool) -> TaskNode | None:
		"""The node the gate checks."""
		if self.agent is not None and self.agent.check is not None:
			return self.agent.check
		return self.bare_task(github=github)

	def gate_fix(self) -> TaskNode | None:
		"""The declared autofix node."""
		return self.agent.fix if self.agent is not None else None

	def run_default(self) -> TaskNode | None:
		"""The task a no-task ``camas_run`` runs."""
		if self.agent is not None and self.agent.default is not None:
			return self.agent.default
		return self.gate_check(github=True)
