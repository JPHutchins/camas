# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""In-page links in the template must name a heading of the rendered README."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docs.readme import main

if TYPE_CHECKING:
	from collections.abc import Callable

	import pytest


def test_a_link_to_a_heading_passes(template: Callable[[str], None]) -> None:
	template("## Time budget (`--under`)\n\nSee [the budget](#time-budget---under).\n")
	assert main([]) == 0


def test_a_repeated_heading_gets_a_numbered_anchor(template: Callable[[str], None]) -> None:
	template("## Usage\n\n## Usage\n\n[second](#usage-1) and [first](#usage)\n")
	assert main([]) == 0


def test_a_broken_link_exits_2_naming_its_line(
	template: Callable[[str], None], capsys: pytest.CaptureFixture[str]
) -> None:
	template("## Config\n\nSee [config](#config) and [gone](#drift-gate).\n")
	assert main(["--check"]) == 2
	assert capsys.readouterr().err == "docs/README.in.md:3: no heading for the link `#drift-gate`\n"


def test_fenced_code_neither_defines_headings_nor_links(
	template: Callable[[str], None], capsys: pytest.CaptureFixture[str]
) -> None:
	template("```python\n# tasks.py\nx = '[a](#nowhere)'\n```\n\n[tasks](#taskspy)\n")
	assert main(["--check"]) == 2
	assert capsys.readouterr().err == "docs/README.in.md:6: no heading for the link `#taskspy`\n"
