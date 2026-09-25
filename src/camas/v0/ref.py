# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The expression mini-language's task reference, before resolution — parser-level, not a node."""

from __future__ import annotations

from typing import NamedTuple


class Ref(NamedTuple):
	"""Parser-only sentinel for a task referenced by name inside a config expression.

	Resolved by :func:`camas.main.expression.resolve_refs` before anything runs; a
	:class:`camas.v0.task.Pipe` stage admits one at construction for the same reason a group
	child does — the resolved value re-validates the stage check.

	>>> Ref("lint")
	Ref(name='lint')
	"""

	name: str
