# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations


def code(text: str) -> str:
	return f"`{cell(text)}`"


def cell(text: str) -> str:
	return " ".join(text.split()).replace("|", "\\|")
