# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Skip the MCP tests when the optional [mcp] extra is not installed."""

from __future__ import annotations

import importlib.util

collect_ignore_glob: list[str] = (
	[] if importlib.util.find_spec("mcp") else ["*"]
)  # pragma: no cover
