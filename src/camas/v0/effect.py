# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``Effect`` protocol: observers over a run's event stream with per-leaf contexts."""

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
	"""Observer over a run's event stream, with a per-leaf context of type T.

	Each leaf owns an independent chain: setup seeds every leaf's slot,
	on_event advances the slot for the affected leaf, and teardown receives
	one final T per leaf in leaf-index order. Different leaves' on_event
	calls may run concurrently.
	"""

	async def setup(self, task: TaskNode) -> T:
		"""Return the initial per-leaf context."""
		...

	async def on_event(self, event: TaskEvent, states: Sequence[LeafState], ctx: T) -> T:
		"""Return the next context for the leaf identified by event.leaf_index."""
		...

	async def teardown(self, ctxs: tuple[T, ...]) -> None:
		"""Receive every leaf's final context, in leaf-index order."""
		...
