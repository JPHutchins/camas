# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Depth-first tree walks and leaf-index bookkeeping."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Final

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..v0.task import Parallel, Sequential, Task, TaskNode
from .leaf_state import ChainLink, LeafInfo

if TYPE_CHECKING:
	from collections.abc import Iterator


def iter_leaves(
	node: TaskNode,
	depth: int,
	is_last_chain: tuple[ChainLink, ...],
) -> Iterator[LeafInfo]:
	"""Walk a task tree depth-first, yielding LeafInfo for each leaf."""
	match node:
		case Task():
			yield LeafInfo(node, depth, is_last_chain)
		case Sequential(tasks=children) | Parallel(tasks=children):
			parent_is_par: Final = isinstance(node, Parallel)
			last_i: Final = len(children) - 1
			for i, child in enumerate(children):
				link = ChainLink(is_last=i == last_i, parent_is_parallel=parent_is_par)
				yield from iter_leaves(child, depth + 1, (*is_last_chain, link))
		case _:
			assert_never(node)


def flatten_leaves(task: TaskNode) -> tuple[LeafInfo, ...]:
	"""Flatten a task tree into a tuple of LeafInfo in depth-first order."""
	return tuple(iter_leaves(task, depth=0, is_last_chain=()))


def subtree_leaf_indices(task: TaskNode, index_map: dict[int, int]) -> tuple[int, ...]:
	"""Collect all leaf indices within a subtree, in depth-first order."""
	match task:
		case Task():
			return (index_map[id(task)],)
		case Sequential(tasks=tasks) | Parallel(tasks=tasks):
			return tuple(i for child in tasks for i in subtree_leaf_indices(child, index_map))
		case _:
			assert_never(task)
