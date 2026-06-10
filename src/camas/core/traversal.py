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

from .leaf_state import ChainLink, LeafInfo
from .task import Parallel, Sequential, Task, TaskNode

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
	"""Flatten a task tree into a tuple of LeafInfo in depth-first order.

	>>> [info.task.cmd for info in flatten_leaves(Parallel(Task("a"), Task("b")))]
	['a', 'b']
	>>> flatten_leaves(Task("echo hi"))[0].depth
	0
	>>> flatten_leaves(Parallel(Task("a"), Task("b")))[0].is_last_chain
	(ChainLink(is_last=False, parent_is_parallel=True),)
	>>> flatten_leaves(Parallel(Task("a"), Task("b")))[1].is_last_chain
	(ChainLink(is_last=True, parent_is_parallel=True),)
	"""
	return tuple(iter_leaves(task, depth=0, is_last_chain=()))


def build_leaf_index_map(task: TaskNode) -> dict[int, int]:
	"""Map `id(Task)` to leaf index (depth-first position) for the whole tree.

	>>> t1, t2 = Task("a"), Task("b")
	>>> m = build_leaf_index_map(Parallel(t1, t2))
	>>> m[id(t1)], m[id(t2)]
	(0, 1)
	"""
	return {id(info.task): i for i, info in enumerate(flatten_leaves(task))}


def subtree_leaf_indices(task: TaskNode, index_map: dict[int, int]) -> tuple[int, ...]:
	"""Collect all leaf indices within a subtree.

	>>> t1, t2 = Task("a"), Task("b")
	>>> tree = Parallel(t1, t2)
	>>> subtree_leaf_indices(tree, build_leaf_index_map(tree))
	(0, 1)
	"""
	match task:
		case Task():
			return (index_map[id(task)],)
		case Sequential(tasks=tasks) | Parallel(tasks=tasks):
			return tuple(i for child in tasks for i in subtree_leaf_indices(child, index_map))
		case _:
			assert_never(task)
