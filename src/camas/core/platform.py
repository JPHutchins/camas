# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import sys


def env_case_insensitive() -> bool:
	return sys.platform in ("win32", "msys", "cygwin")
