# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
	from pathlib import Path


class ProjectLoadError(Exception):
	def __init__(self, source: Path, cause: Exception) -> None:
		super().__init__(f"{source}: {cause}")
		self.source: Final = source
		self.cause: Final = cause
