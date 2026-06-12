# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins
"""Tasks-source resolution outcome as a sum type.

A run of ``resolve_tasks_source`` ends in exactly one of:

* :class:`LoadOk` — tasks loaded cleanly (possibly empty, when no tasks file
  exists anywhere up the tree).
* :class:`LoadErr` — a ``tasks.py`` was found but evaluating it raised.

Lives in its own module so :mod:`parser` and :mod:`dispatch` can both import
the types without a cycle.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, NamedTuple, TypeAlias

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from ..v0.config import Config
	from ..v0.effect import Effect
	from ..v0.task import TaskNode


class LoadOk(NamedTuple):
	"""Tasks loaded successfully (or no tasks file found at all).

	``tasks`` and ``scope_effects`` are read-only ``Mapping``s so the shared
	:data:`EMPTY_STATE` constant can't be mutated through accidental
	``state.tasks[...] = ...`` aliasing.
	"""

	tasks: Mapping[str, TaskNode]
	source: Path | None
	"""``None`` only when no tasks file exists anywhere up the tree."""
	scope_effects: Mapping[str, type[Effect[Any]]]
	config: Config | None = None
	"""The project :class:`Config`, if one was defined."""


class LoadErr(NamedTuple):
	"""A ``tasks.py`` was found but raised during evaluation."""

	source: Path
	exception: Exception


TasksState: TypeAlias = LoadOk | LoadErr


EMPTY_STATE: Final = LoadOk(
	tasks=MappingProxyType({}),
	source=None,
	scope_effects=MappingProxyType({}),
)
"""Default :class:`LoadOk` used by parser/dispatch when no state is supplied —
behaves like ``camas`` invoked in a directory with no tasks file. Backed by
``MappingProxyType`` so it stays immutable across calls."""
