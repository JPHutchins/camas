# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The `toc` directive: the README's `##` and `###` headings as a nested list of links."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docs.readme import BANNER, main

if TYPE_CHECKING:
	from collections.abc import Callable
	from pathlib import Path


def test_the_contents_list_every_section_and_subsection_in_order(
	project: Path, template: Callable[[str], None]
) -> None:
	template(
		"# Title\n\n<!-- GENERATED: toc -->\n\n### Intro\n\n## One\n\n### Sub (`x`)\n\n#### Deep\n\n"
		"## One\n\n```\n## fenced\n```\n"
	)
	assert main([]) == 0
	assert (project / "README.md").read_text(encoding="utf-8").split("\n")[:8] == [
		BANNER,
		"# Title",
		"",
		"- [Intro](#intro)",
		"- [One](#one)",
		"  - [Sub (`x`)](#sub-x)",
		"- [One](#one-1)",
		"",
	]
