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
"""The default project subdirectory for camas's run logs and timing cache."""


class Claude(NamedTuple):
	"""The Claude Code agent integration: the explicitly declared fix and check nodes."""

	fix: TaskNode
	"""The deterministic, behavior-preserving autofix node the FileChanged hook runs (scoped,
	zero tokens). Declared, not derived from ``mutates`` — a mutating leaf may be codegen or a
	compiler, which is not a fixer.
	"""
	check: TaskNode | None = None
	"""The node the gate checks; ``None`` defers to the default (or github) task, scoped by
	``--paths`` and time-boxed by ``--under``.
	"""


Agent: TypeAlias = Claude
"""The agent integration backend — a union as backends land (e.g. an ``InternalModel`` CI backend)."""


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
	camas_dir: str = DEFAULT_CAMAS_DIR
	"""Project subdirectory for run logs and the timing cache; delete it to opt out."""
	agent: Agent | None = None
	"""The agent integration (the gate's fix and check nodes); ``None`` when unconfigured."""

	def camas_path(self, base: Path) -> Path:
		"""The resolved camas directory under ``base``."""
		return base / self.camas_dir

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

	def gate_check(self, *, github: bool) -> TaskNode | None:
		"""The node the gate checks: the agent's explicit ``check`` override, else the bare task."""
		if self.agent is not None and self.agent.check is not None:
			return self.agent.check
		return self.bare_task(github=github)

	def gate_fix(self) -> TaskNode | None:
		"""The declared deterministic autofix node, or ``None`` when no agent is configured."""
		return self.agent.fix if self.agent is not None else None
