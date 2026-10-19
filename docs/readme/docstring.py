# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import doctest
import inspect
import itertools
import re
from typing import Final, NamedTuple

SECTION: Final = re.compile(
	r"(?m)^(?P<header>Args|Arguments|Attributes|Raises|Returns|Warns|Yields):\n(?P<body>(?:[ \t]+\S.*(?:\n|$))+)"
)
ENTRY: Final = re.compile(r"(?P<name>[\w.]+): (?P<text>.*)")


class Text(NamedTuple):
	lines: tuple[str, ...]


class Examples(NamedTuple):
	lines: tuple[str, ...]


def split(docstring: str) -> tuple[Text | Examples, ...]:
	parsed = doctest.DocTestParser().parse(inspect.cleandoc(docstring))
	grouped = itertools.groupby(
		parsed, key=lambda piece: isinstance(piece, doctest.Example) or not str(piece).strip()
	)
	parts = tuple(
		Examples(
			tuple(
				line
				for piece in group
				if isinstance(piece, doctest.Example)
				for line in example(piece)
			)
		)
		if is_example
		else Text(
			tuple(
				SECTION.sub(bulleted, "".join(str(piece) for piece in group))
				.strip("\n")
				.split("\n")
			)
		)
		for is_example, group in grouped
	)
	return tuple(part for part in parts if part.lines)


def example(piece: doctest.Example) -> tuple[str, ...]:
	first, *rest = piece.source.rstrip("\n").split("\n")
	want = piece.want.rstrip("\n").split("\n") if piece.want else ()
	return (f">>> {first}", *(f"... {line}" for line in rest), *want)


def bulleted(section: re.Match[str]) -> str:
	body = section["body"].rstrip("\n").split("\n")
	base = len(body[0]) - len(body[0].lstrip())
	entries = re.split(r"\n(?=" + re.escape(body[0][:base]) + r"\S)", "\n".join(body))
	return (
		f"{section['header']}:\n\n"
		+ "\n".join(entry_bullet(" ".join(entry.split())) for entry in entries)
		+ "\n"
	)


def entry_bullet(entry: str) -> str:
	match = ENTRY.fullmatch(entry)
	return f"- {entry}" if match is None else f"- `{match['name']}`: {match['text']}"


def paragraphs(doc: str) -> tuple[str, ...]:
	return tuple(
		paragraph
		for paragraph in re.split(r"\n[ \t]*\n", inspect.cleandoc(doc))
		if paragraph.strip()
	)
