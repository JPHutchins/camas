# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins
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
	tasks: Mapping[str, TaskNode]
	source: Path | None
	scope_effects: Mapping[str, type[Effect[Any]]]
	config: Config | None = None
	naming_warnings: tuple[str, ...] = ()


class LoadErr(NamedTuple):
	source: Path
	exception: Exception


TasksState: TypeAlias = LoadOk | LoadErr


EMPTY_STATE: Final = LoadOk(
	tasks=MappingProxyType({}),
	source=None,
	scope_effects=MappingProxyType({}),
)
