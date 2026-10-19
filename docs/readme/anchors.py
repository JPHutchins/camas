# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import itertools
import re
from typing import Final, NamedTuple

FENCE: Final = re.compile(r"\s*(```|~~~)")
HEADING: Final = re.compile(r"(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*")
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
	return frozenset(heading.anchor for heading in headings(lines))


def contents(lines: tuple[str, ...]) -> tuple[str, ...]:
	listed = tuple(heading for heading in headings(lines) if heading.level in (2, 3))
	return tuple(
		f"{'  ' if heading.level == 3 and any(h.level == 2 for h in listed[:index]) else ''}"
		f"- [{heading.text}](#{heading.anchor})"
		for index, heading in enumerate(listed)
	)


class Heading(NamedTuple):
	level: int
	text: str
	anchor: str


def headings(lines: tuple[str, ...]) -> tuple[Heading, ...]:
	found = tuple(
		(len(heading["hashes"]), heading["text"])
		for _, line in prose(lines)
		if (heading := HEADING.fullmatch(line))
	)
	slugs = tuple(slug(text) for _, text in found)
	return tuple(
		Heading(level, text, name if repeats == 0 else f"{name}-{repeats}")
		for index, ((level, text), name) in enumerate(zip(found, slugs, strict=True))
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
	inside = itertools.accumulate(fences, lambda inside, fence: inside != fence)
	return tuple(
		(number, line)
		for number, (line, fence, fenced) in enumerate(zip(lines, fences, inside, strict=True), 1)
		if not fence and not fenced
	)
