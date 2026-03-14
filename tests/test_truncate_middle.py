from __future__ import annotations

import pytest

from camas import truncate_middle


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
