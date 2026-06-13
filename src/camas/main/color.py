# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""ANSI codes and helpers for CLI output (listings and help, not Effects)."""

from ..core.color import RESET


def wrap_ansi(text: str, code: str) -> str:
	"""Unconditionally wrap ``text`` in ``code``...``RESET``. Callers gate via
	``maybe_color`` (or by checking ``color_on()`` themselves).
	"""
	return f"{code}{text}{RESET}" if text else text


def maybe_color(text: str, code: str, on: bool) -> str:
	return wrap_ansi(text, code) if on else text
