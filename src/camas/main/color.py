# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from ..core.color import RESET


def wrap_ansi(text: str, code: str) -> str:
	return f"{code}{text}{RESET}" if text else text


def maybe_color(text: str, code: str, on: bool) -> str:
	return wrap_ansi(text, code) if on else text
