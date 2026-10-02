# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Fixtures shared by the mcp tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from pathlib import Path


@pytest.fixture
def venv_camas() -> Callable[[Path], Path]:
	"""Build a ``camas`` inside a virtual environment rooted at the given path, and return it — the
	``pyvenv.cfg`` beside the script directory is what makes the environment one, so it is what the
	detection reads.
	"""

	def build(root: Path) -> Path:
		(root / "bin").mkdir(parents=True)
		(root / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
		(root / "bin" / "camas").write_text("", encoding="utf-8")
		return root / "bin" / "camas"

	return build
