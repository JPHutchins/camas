# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import inspect


def code(text: str) -> str:
	return f"`{cell(text)}`"


def cell(text: str) -> str:
	return " ".join(text.split()).replace("|", "\\|")


def details(summary: str, entries: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
	if not entries:
		return ()
	return (
		"",
		f"<details><summary>{summary}</summary>",
		"",
		*(line for name, doc in entries for line in bullet(name, doc)),
		"",
		"</details>",
	)


def bullet(name: str, doc: str) -> tuple[str, ...]:
	first, *rest = inspect.cleandoc(doc).splitlines()
	return (f"- `{name}`: {first}", *(f"  {line}" if line else "" for line in rest))
