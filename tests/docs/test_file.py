# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The `file <path>` directive: a file's contents in a fenced block, tabs expanded."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable
	from pathlib import Path


def test_a_python_file_renders_in_a_python_block_with_tabs_expanded(
	project: Path, generated: Callable[[str], tuple[str, ...]]
) -> None:
	(project / "tasks.py").write_text('def f():\n\treturn "x"\n\n', encoding="utf-8")
	assert generated("file tasks.py") == ("```python", "def f():", '    return "x"', "```")


def test_an_unknown_suffix_renders_in_a_plain_block(
	project: Path, generated: Callable[[str], tuple[str, ...]]
) -> None:
	(project / "notes.txt").write_text("plain\n", encoding="utf-8")
	assert generated("file notes.txt") == ("```", "plain", "```")
