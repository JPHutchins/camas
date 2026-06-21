# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import pytest

from camas.core.render import visual_width
from camas.effect.termtree import truncate_middle


@pytest.mark.parametrize(
	("text", "max_width", "expected"),
	[
		("hello", 10, "hello"),
		("hello", 5, "hello"),
		("ab", 2, "ab"),
	],
)
def test_fits_within_limit(text: str, max_width: int, expected: str) -> None:
	assert truncate_middle(text, max_width) == expected


@pytest.mark.parametrize(
	("text", "max_width"),
	[
		("abcdefghij", 7),
		("abcdefghij", 4),
		("a" * 100, 20),
	],
)
def test_truncates_with_ellipsis(text: str, max_width: int) -> None:
	result = truncate_middle(text, max_width)
	assert len(result) == max_width
	assert "..." in result


@pytest.mark.parametrize(
	("text", "max_width", "expected"),
	[
		("built", 3, "..."),
		("built", 2, ".."),
		("built", 1, "."),
		("built", 0, ""),
		("built", -5, ""),
	],
)
def test_small_max_width_never_exceeds(text: str, max_width: int, expected: str) -> None:
	result = truncate_middle(text, max_width)
	assert result == expected
	assert len(result) <= max(max_width, 0)


@pytest.mark.parametrize(
	("text", "max_width"),
	[
		("你好世界你好世界", 7),  # CJK, wide
		("\U0001f389" * 10, 7),  # 🎉, wide
		("a你b好c世d界e", 6),  # mixed narrow/wide
		("界", 1),  # a lone wide glyph cannot fit one column
	],
)
def test_wide_chars_budget_display_width_not_code_points(text: str, max_width: int) -> None:
	"""Regression for #64: the result's terminal-column width never exceeds ``max_width``,
	even when wide glyphs make ``len`` smaller than the rendered width."""
	assert visual_width(truncate_middle(text, max_width)) <= max_width
