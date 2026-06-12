# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins
"""The v0 generation of the public API — import from ``camas.v0`` and its submodules to pin it."""

import typing

from .effect import Effect as Effect
from .task import Parallel as Parallel
from .task import Sequential as Sequential
from .task import Task as Task

if typing.TYPE_CHECKING:
	from ..main.dispatch import run_cli as run_cli


def __getattr__(name: str) -> typing.Any:
	"""Lazily expose ``run_cli`` without importing ``camas.main`` / ``camas.core`` at
	import time, keeping the version namespace free of the engine it is consumed by.

	Returns ``Any`` (the PEP 562 convention): ``run_cli`` is typed precisely via the
	``TYPE_CHECKING`` import above, while ``object`` would mistype real module
	attributes (e.g. ``__file__``) for checkers that route them through ``__getattr__``.

	Raises:
		AttributeError: for any name other than ``run_cli``.
	"""
	if name == "run_cli":
		from ..main.dispatch import run_cli

		return run_cli
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
