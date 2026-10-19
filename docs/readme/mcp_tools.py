# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from camas.mcp.serve import Compat, tools


def mcp_tools_section() -> tuple[str, ...]:
	sections = tuple(
		(f"#### `{tool.name}`", "", *(tool.description or "").split("\n"))
		for tool in tools((), Compat())
	)
	return tuple(
		line
		for index, section in enumerate(sections)
		for line in ((*(("",) if index else ()), *section))
	)
