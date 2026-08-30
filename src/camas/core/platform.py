# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Small platform facts importable by leaf modules without the engine graph."""

from __future__ import annotations

import sys


def env_case_insensitive() -> bool:
	"""Whether the platform's environment block is case-insensitive (Windows, MSYS/Cygwin)."""
	return sys.platform in ("win32", "msys", "cygwin")
