# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import re
import shlex
from typing import Final, NamedTuple

from .alias import alias_section
from .anchors import broken_links
from .callables import function_section, protocol_section
from .cli import cli_section
from .exports import exports_section
from .fields import fields_section
from .file import file_section
from .source import Unresolved

DIRECTIVE: Final = re.compile(r"<!-- GENERATED: (?P<directive>\S.*\S) -->")
DIRECTIVE_LIKE: Final = re.compile(r"<!--\s*GENERATED", re.IGNORECASE)
MALFORMED: Final = (
	"malformed directive; expected `<!-- GENERATED: <kind> <target> -->` alone on its line"
)


class TemplateError(NamedTuple):
	line: int
	message: str


def render(template: str) -> str | tuple[TemplateError, ...]:
	lines = tuple(template.split("\n"))
	expanded = tuple(expand(number, line) for number, line in enumerate(lines, 1))
	errors = tuple(piece for piece in expanded if isinstance(piece, TemplateError))
	if errors:
		return errors
	rendered = tuple(
		line for piece in expanded if not isinstance(piece, TemplateError) for line in piece
	)
	broken = tuple(
		TemplateError(number, f"no heading for the link `#{anchor}`")
		for number, anchor in broken_links(lines, rendered)
	)
	return broken or "\n".join(rendered)


def expand(number: int, line: str) -> tuple[str, ...] | TemplateError:
	if (directive := DIRECTIVE.fullmatch(line.strip())) is not None:
		generated = generate(directive["directive"])
		return (
			TemplateError(number, generated.message)
			if isinstance(generated, Unresolved)
			else generated
		)
	return TemplateError(number, MALFORMED) if DIRECTIVE_LIKE.search(line) else (line,)


def generate(directive: str) -> tuple[str, ...] | Unresolved:
	words = tokens(directive)
	if isinstance(words, Unresolved):
		return words
	match words:
		case ("fields", dotted):
			return fields_section(dotted)
		case ("function", dotted):
			return function_section(dotted)
		case ("protocol", dotted):
			return protocol_section(dotted)
		case ("alias", dotted):
			return alias_section(dotted)
		case ("file", name):
			return file_section(name)
		case ("exports", dotted):
			return exports_section(dotted)
		case ("cli", directory, *args):
			return cli_section(directory, tuple(args))
		case _:
			return Unresolved(f"unknown directive {directive!r}")


def tokens(directive: str) -> tuple[str, ...] | Unresolved:
	try:
		return tuple(shlex.split(directive))
	except ValueError as error:
		return Unresolved(f"unparsable directive {directive!r}: {error}")
