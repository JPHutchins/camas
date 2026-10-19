# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import itertools
import operator
import re
from typing import Final

FENCE: Final = re.compile(r"\s*(```|~~~)")
HEADING: Final = re.compile(r"#{1,6}\s+(?P<text>.+?)\s*")
LINK: Final = re.compile(r"\]\(#(?P<anchor>[^)\s]+)\)")


def broken_links(
	template: tuple[str, ...], rendered: tuple[str, ...]
) -> tuple[tuple[int, str], ...]:
	known = anchors(rendered)
	return tuple(
		(number, link["anchor"])
		for number, line in prose(template)
		for link in LINK.finditer(line)
		if link["anchor"] not in known
	)


def anchors(lines: tuple[str, ...]) -> frozenset[str]:
	slugs = tuple(
		slug(heading["text"]) for _, line in prose(lines) if (heading := HEADING.fullmatch(line))
	)
	return frozenset(
		name if repeats == 0 else f"{name}-{repeats}"
		for index, name in enumerate(slugs)
		for repeats in (slugs[:index].count(name),)
	)


def slug(heading: str) -> str:
	"""
	>>> slug("Time budget (`--under`)")
	'time-budget---under'
	>>> slug("Standalone `tasks.py` (PEP 723)")
	'standalone-taskspy-pep-723'
	"""
	return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")


def prose(lines: tuple[str, ...]) -> tuple[tuple[int, str], ...]:
	fences = tuple(FENCE.match(line) is not None for line in lines)
	inside = itertools.accumulate(fences, operator.xor)
	return tuple(
		(number, line)
		for number, (line, fence, fenced) in enumerate(zip(lines, fences, inside, strict=True), 1)
		if not fence and not fenced
	)
