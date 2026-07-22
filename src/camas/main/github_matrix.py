# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""``--github-matrix``: emit a task's matrix as GitHub Actions ``strategy.matrix`` JSON.

The axis values come from the task's *actual* run-set — the leaves
:func:`camas.core.matrix.expand_matrix` produces — so the emitted object expands, under
GHA's cross-product semantics, to exactly the jobs camas runs::

    {"PY": ["3.10", "3.11"], "PROFILE": ["debug", "release"]}

Consume the whole object as the matrix::

    matrix: ${{ fromJSON(needs.discover.outputs.matrix) }}

or one axis at a time, composed with the YAML-side axes a shell command can't set from
inside a job (a runner's ``os``)::

    matrix:
      os: [ubuntu-latest, macos-latest]
      PY: ${{ fromJSON(needs.discover.outputs.matrix).PY }}

CLI overrides (``--PY 3.13``) flow through the same ``override_matrix`` pipeline the runner
uses, so the emitted JSON reflects exactly what the run would have executed.

Object-of-arrays is GHA's cross-product form; a run-set that isn't a clean cross-product
(heterogeneous nested matrices, or independent fan-outs in one tree) has no faithful
object-of-arrays and is rejected rather than silently widened or narrowed.

Output is TTY-aware: indented for interactive preview, compact one-line for pipes — so
``camas matrix --github-matrix`` reads cleanly in a shell *and*
``$(camas matrix --github-matrix)`` works directly with ``$GITHUB_OUTPUT``.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Final

from ..core.matrix import expand_matrix, matrix_axes
from ..core.task import task_label
from ..core.traversal import flatten_leaves

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..core.leaf_state import LeafInfo
	from ..v0.task import TaskNode


def matrix_combinations(
	task: TaskNode, axes: tuple[str, ...] | None = None
) -> tuple[dict[str, str], ...]:
	"""The distinct matrix bindings ``task`` would actually run, in first-seen order.

	Sourced from :func:`camas.core.matrix.expand_matrix` — the run-set SSOT — so the result is
	what camas runs, not a re-derived guess. A leaf under no matrix contributes nothing. ``axes``
	reuses an already-computed axis list (it must equal :func:`camas.core.matrix.matrix_axes`);
	it defaults to deriving them from ``task``.

	>>> from camas import Parallel, Task
	>>> matrix_combinations(Parallel(Task("t"), matrix={"PY": ("3.12", "3.13")}))
	({'PY': '3.12'}, {'PY': '3.13'})
	>>> matrix_combinations(Task("t"))
	()
	>>> matrix_combinations(Parallel(
	...     Parallel(Task("t"), matrix={"PROFILE": ("release",), "PY": ("3.13",)}),
	...     Parallel(Task("t"), matrix={"PROFILE": ("debug",), "PY": ("3.12", "3.13")}),
	... ))
	({'PROFILE': 'release', 'PY': '3.13'}, {'PROFILE': 'debug', 'PY': '3.12'}, {'PROFILE': 'debug', 'PY': '3.13'})
	"""
	resolved: Final = tuple(matrix_axes(task)) if axes is None else axes
	return distinct_combinations(flatten_leaves(expand_matrix(task)), resolved)


def distinct_combinations(
	leaves: tuple[LeafInfo, ...], axes: tuple[str, ...]
) -> tuple[dict[str, str], ...]:
	"""The distinct axis bindings across ``leaves``, in first-seen order — the dedup core of
	:func:`matrix_combinations`, factored out so :func:`to_matrix_object` derives combos from the
	same expansion it checks for coverage instead of expanding the tree a second time. A leaf
	carrying none of ``axes`` yields an empty binding and contributes nothing (the coverage of
	such leaves is the caller's concern, checked in :func:`to_matrix_object`).
	"""
	combos: list[dict[str, str]] = []
	seen: set[tuple[tuple[str, str], ...]] = set()
	for info in leaves:
		combo = {a: info.task.env[a] for a in axes if a in info.task.env}
		key = tuple(sorted(combo.items()))
		if combo and key not in seen:
			seen.add(key)
			combos.append(combo)
	return tuple(combos)


def is_cross_product(combos: tuple[dict[str, str], ...], axes: tuple[str, ...]) -> bool:
	"""Whether ``combos`` is exactly the cartesian product of each axis's distinct values —
	i.e. representable as object-of-arrays without adding or dropping a single job.

	>>> is_cross_product(({'PY': '3.12'}, {'PY': '3.13'}), ('PY',))
	True
	>>> is_cross_product(
	...     ({'PROFILE': 'debug', 'PY': '3.12'}, {'PROFILE': 'debug', 'PY': '3.13'},
	...      {'PROFILE': 'release', 'PY': '3.12'}, {'PROFILE': 'release', 'PY': '3.13'}),
	...     ('PROFILE', 'PY'))
	True
	>>> is_cross_product(
	...     ({'PROFILE': 'debug', 'PY': '3.12'}, {'PROFILE': 'release', 'PY': '3.13'}),
	...     ('PROFILE', 'PY'))
	False
	>>> is_cross_product(({'PY': '3.12'}, {'PROFILE': 'debug'}), ('PY', 'PROFILE'))
	False
	"""
	if not combos or any(tuple(c) != axes for c in combos):
		return False
	sizes: Final = tuple(len(dict.fromkeys(c[a] for c in combos)) for a in axes)
	return math.prod(sizes) == len(combos)


def to_matrix_object(task: TaskNode) -> dict[str, list[str]]:
	"""Project ``task``'s matrix into the GHA object-of-arrays ``strategy.matrix`` consumes.

	Values come from the faithful run-set (:func:`matrix_combinations`), so the emitted object
	expands — under GHA's cross-product semantics — to exactly the jobs camas runs.

	Raises:
		ValueError: when ``task`` has no matrix axes, an axis has no values, a leaf runs under no
			matrix axis (a plain leaf beside matrixed siblings), or the run-set is not a clean
			cross-product (heterogeneous nested matrices, or independent fan-outs) — none of which
			has a faithful object-of-arrays.

	>>> from camas import Parallel, Task
	>>> to_matrix_object(Parallel(Task("test"), matrix={"PY": ("3.12", "3.13")}))
	{'PY': ['3.12', '3.13']}
	>>> to_matrix_object(Parallel(Task("t"), matrix={"PY": ("3.13",), "PROFILE": ("release",)}))
	{'PY': ['3.13'], 'PROFILE': ['release']}
	>>> to_matrix_object(Task("hi"))
	Traceback (most recent call last):
	    ...
	ValueError: task has no matrix axes to emit as a GitHub Actions job matrix
	>>> to_matrix_object(Parallel(Task("t"), matrix={"PY": ()}))
	Traceback (most recent call last):
	    ...
	ValueError: matrix axis 'PY' has no values
	>>> to_matrix_object(Parallel(
	...     Parallel(Task("t"), matrix={"PROFILE": ("release",), "PY": ("3.13",)}),
	...     Parallel(Task("t"), matrix={"PROFILE": ("debug",), "PY": ("3.12", "3.13")}),
	... ))
	Traceback (most recent call last):
	    ...
	ValueError: matrix is not a clean cross-product; object-of-arrays cannot represent a heterogeneous fan-out
	>>> to_matrix_object(Parallel(
	...     Parallel(Task("echo {X}"), matrix={"X": ("a", "b")}, name="matrixed"),
	...     Task("echo plain", name="plain"),
	... ))
	Traceback (most recent call last):
	    ...
	ValueError: matrix does not cover every leaf (plain): a leaf that runs under no matrix axis cannot be represented in a GitHub Actions object-of-arrays — mixing matrixed and plain leaves under one --github-matrix task is unsupported
	"""
	axes_map: Final = matrix_axes(task)
	if not axes_map:
		raise ValueError("task has no matrix axes to emit as a GitHub Actions job matrix")
	for name, values in axes_map.items():
		if not values:
			raise ValueError(f"matrix axis {name!r} has no values")
	axes: Final = tuple(axes_map)
	leaves: Final = flatten_leaves(expand_matrix(task))
	uncovered: Final = tuple(
		task_label(info.task) for info in leaves if not any(a in info.task.env for a in axes)
	)
	if uncovered:
		raise ValueError(
			f"matrix does not cover every leaf ({', '.join(uncovered)}): a leaf that runs under "
			"no matrix axis cannot be represented in a GitHub Actions object-of-arrays — mixing "
			"matrixed and plain leaves under one --github-matrix task is unsupported"
		)
	combos: Final = distinct_combinations(leaves, axes)
	if not is_cross_product(combos, axes):
		raise ValueError(
			"matrix is not a clean cross-product; object-of-arrays cannot represent a "
			"heterogeneous fan-out"
		)
	return {a: list(dict.fromkeys(c[a] for c in combos)) for a in axes}


def format_matrix_json(matrix: Mapping[str, list[str]], *, pretty: bool) -> str:
	"""Serialize the matrix object to JSON.

	Compact (no spaces, single line) when ``pretty`` is False — the canonical shape for
	``echo "matrix=$(...)" >> $GITHUB_OUTPUT``. Indented two spaces when ``pretty`` is True —
	readable preview for interactive use.

	>>> format_matrix_json({"PY": ["3.12", "3.13"]}, pretty=False)
	'{"PY":["3.12","3.13"]}'
	>>> print(format_matrix_json({"PY": ["3.12"]}, pretty=True))
	{
	  "PY": [
	    "3.12"
	  ]
	}
	"""
	if pretty:
		return json.dumps(matrix, indent=2)
	return json.dumps(matrix, separators=(",", ":"))


def emit(task: TaskNode, *, pretty: bool) -> str:
	"""Compose :func:`to_matrix_object` and :func:`format_matrix_json`.

	>>> from camas import Parallel, Task
	>>> emit(Parallel(Task("t"), matrix={"PY": ("3.12",)}), pretty=False)
	'{"PY":["3.12"]}'
	"""
	return format_matrix_json(to_matrix_object(task), pretty=pretty)
