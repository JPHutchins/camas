# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""``--github-matrix``: emit a GitHub Actions matrix as JSON.

Projects the task's matrix axes — collected from anywhere in the tree via
:func:`camas.core.matrix.matrix_axes`, outermost-wins on duplicate keys —
into the object-of-arrays form that GHA consumes natively::

    {"PY": ["3.10", "3.11"], "OS": ["linux", "macos"]}

Use the whole object as the matrix::

    matrix: ${{ fromJSON(needs.discover.outputs.matrix) }}

or one axis at a time, composed with other YAML-side axes::

    matrix:
      os: [ubuntu-latest, macos-latest]
      PY: ${{ fromJSON(needs.discover.outputs.matrix).PY }}

CLI overrides (``--PY 3.13``) flow through the same ``override_matrix``
pipeline used by the runner before emission, so the emitted JSON reflects
exactly what the run would have executed.

Output is TTY-aware: indented for interactive preview, compact one-line
for pipes — so ``camas matrix --github-matrix`` reads cleanly in a shell
*and* ``$(camas matrix --github-matrix)`` works directly with
``$GITHUB_OUTPUT``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from ..core.matrix import matrix_axes
from ..core.task import TaskNode


def to_matrix_object(task: TaskNode) -> dict[str, list[str]]:
	"""Project a task's matrix axes into a GHA-compatible object-of-arrays.

	Raises ``ValueError`` when the task has no matrix axes or any axis has no
	values — ``--github-matrix`` requires at least one cell to emit a workflow
	GHA will accept (the schema mandates ``minItems: 1`` on every axis array).

	>>> from camas import Parallel, Task
	>>> to_matrix_object(Parallel(Task("test"), matrix={"PY": ("3.12", "3.13")}))
	{'PY': ['3.12', '3.13']}
	>>> to_matrix_object(Parallel(Task("t"), matrix={"PY": ("3.13",), "OS": ("linux",)}))
	{'PY': ['3.13'], 'OS': ['linux']}
	>>> to_matrix_object(Task("hi"))
	Traceback (most recent call last):
	    ...
	ValueError: task has no matrix axes; --github-matrix requires at least one
	>>> to_matrix_object(Parallel(Task("t"), matrix={"PY": ()}))
	Traceback (most recent call last):
	    ...
	ValueError: matrix axis 'PY' has no values
	"""
	axes: Mapping[str, tuple[str, ...]] = matrix_axes(task)
	if not axes:
		raise ValueError("task has no matrix axes; --github-matrix requires at least one")
	for name, values in axes.items():
		if not values:
			raise ValueError(f"matrix axis {name!r} has no values")
	return {name: list(values) for name, values in axes.items()}


def format_matrix_json(matrix: Mapping[str, list[str]], *, pretty: bool) -> str:
	"""Serialize the matrix object to JSON.

	Compact (no spaces, single line) when ``pretty`` is False — the canonical
	shape for ``echo "matrix=$(...)" >> $GITHUB_OUTPUT``. Indented two spaces
	when ``pretty`` is True — readable preview for interactive use.

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
