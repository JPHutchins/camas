# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The error a failed ``Project`` load raises, attributed to the child that failed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
	from pathlib import Path


class ProjectLoadError(Exception):
	"""A failure loading a referenced ``Project``, carrying the child's ``source`` so the caller
	attributes it to the child file rather than the referencing parent.
	"""

	def __init__(self, source: Path, cause: Exception) -> None:
		super().__init__(f"{source}: {cause}")
		self.source: Final = source
		self.cause: Final = cause
