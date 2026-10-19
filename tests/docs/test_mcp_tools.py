# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The `mcp-tools` directive: each camas MCP tool under its own heading, with its description."""

from __future__ import annotations

from typing import TYPE_CHECKING

from camas.mcp.serve import Compat, tools

if TYPE_CHECKING:
	from collections.abc import Callable


def test_every_tool_gets_a_heading_and_its_description(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	section = generated("mcp-tools")
	served = tools((), Compat())
	assert [line for line in section if line.startswith("#### ")] == [
		f"#### `{tool.name}`" for tool in served
	]
	assert section[:3] == (
		"#### `camas_list`",
		"",
		(served.discovery.description or "").split("\n")[0],
	)
	assert section[-1] == (served.github_matrix.description or "").split("\n")[-1]
