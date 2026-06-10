# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""ANSI codes and helpers for CLI output (listings and help, not Effects)."""

from typing import Final

from ..core.render import RESET

BLUE: Final = "\033[34m"
BOLD_BLUE: Final = "\033[1;34m"
BOLD_CYAN: Final = "\033[1;36m"
BOLD_YELLOW: Final = "\033[1;33m"


def wrap_ansi(text: str, code: str) -> str:
	"""Unconditionally wrap ``text`` in ``code``...``RESET``. Callers gate via
	``maybe_color`` (or by checking ``color_on()`` themselves).
	"""
	return f"{code}{text}{RESET}" if text else text


def maybe_color(text: str, code: str, on: bool) -> str:
	return wrap_ansi(text, code) if on else text
