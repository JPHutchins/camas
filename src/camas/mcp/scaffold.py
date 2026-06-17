# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""``camas mcp init``: write this project's ``.mcp.json`` entry for the camas server.

Stdlib only — no ``mcp``/``pydantic`` — so scaffolding works without the ``[mcp]``
extra and stays off the import path of ``camas mcp`` (serving). The launch command
is ``{sys.executable} -m camas mcp``: the interpreter currently running camas, which
is correct regardless of install method (uv, pipx, pip) but absolute, hence
per-machine rather than committable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Final, cast

SERVER_NAME: Final = "camas"


def write_mcp_json(argv: list[str]) -> int:
	"""Merge a camas ``stdio`` server into ``./.mcp.json``, creating it if absent.

	``--rich`` opts the entry into the gated 2025-11-25 tool fields. Returns the
	process exit code: ``0`` on success, ``2`` if ``.mcp.json`` is unreadable, not
	a JSON object, or has a non-object ``mcpServers``.
	"""
	target: Final = Path.cwd() / ".mcp.json"
	existing = _load(target)
	if existing is None:
		print(f"error: {target} is not a readable JSON object", file=sys.stderr)
		return 2
	servers = existing.setdefault("mcpServers", {})
	if not isinstance(servers, dict):
		print(f"error: {target} has a non-object 'mcpServers'", file=sys.stderr)
		return 2
	servers[SERVER_NAME] = _entry(rich="--rich" in argv)
	target.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
	print(
		f"Wrote the {SERVER_NAME!r} MCP server to {target}\n"
		f"  command: {sys.executable} -m camas mcp{' --rich' if '--rich' in argv else ''}\n\n"
		"This command is specific to this environment, not portable. Reload Claude "
		"Code, approve the server, then ask it to call camas_list."
	)
	return 0


def _entry(*, rich: bool) -> dict[str, Any]:
	"""The ``.mcp.json`` stdio entry launching this interpreter's camas."""
	return {
		"type": "stdio",
		"command": sys.executable,
		"args": ["-m", "camas", "mcp", *(["--rich"] if rich else [])],
	}


def _load(path: Path) -> dict[str, Any] | None:
	"""Parse an existing ``.mcp.json`` (``{}`` if absent); ``None`` if not an object."""
	if not path.exists():
		return {}
	try:
		loaded: object = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return None
	return cast("dict[str, Any]", loaded) if isinstance(loaded, dict) else None
