# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``Effect`` protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
	from collections.abc import Sequence

	from .leaf_state import LeafState
	from .task import TaskNode
	from .task_event import TaskEvent

T = TypeVar("T")


@runtime_checkable
class Effect(Protocol[T]):
	"""An observer of a run's events, holding one context per leaf."""

	async def setup(self, task: TaskNode) -> T:
		"""Return the context every leaf starts with."""
		...

	async def on_event(self, event: TaskEvent, states: Sequence[LeafState], ctx: T) -> T:
		"""Return the event's leaf's next context; calls for different leaves may overlap."""
		...

	async def teardown(self, ctxs: tuple[T, ...]) -> None:
		"""Receive every leaf's final context, in leaf-index order."""
		...
