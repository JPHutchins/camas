# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``camas mcp init`` command: write this project's ``.mcp.json`` entry for the camas server."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Final, cast

SERVER_NAME: Final = "camas"


def write_mcp_json(argv: list[str]) -> int:
	"""Merge a camas ``stdio`` server into ``./.mcp.json``; return 0, or 2 if it is malformed."""
	target: Final = Path.cwd() / ".mcp.json"
	existing = _load(target)
	if existing is None:
		print(f"error: {target} is not a readable JSON object", file=sys.stderr)
		return 2
	servers = existing.setdefault("mcpServers", {})
	if not isinstance(servers, dict):
		print(f"error: {target} has a non-object 'mcpServers'", file=sys.stderr)
		return 2
	command, args = _launch(rich="--rich" in argv)
	servers[SERVER_NAME] = {"type": "stdio", "command": command, "args": args}
	target.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
	print(
		f"Wrote the {SERVER_NAME!r} MCP server to {target}\n"
		f"  command: {command} {' '.join(args)}\n\n"
		f"{_portability_note(command)} Reload Claude Code, approve the server, "
		"then ask it to call camas_list."
	)
	return 0


def _launch(*, rich: bool) -> tuple[str, list[str]]:
	"""The most portable launch command for camas in this environment."""
	tail = ["mcp", "--rich"] if rich else ["mcp"]
	if shutil.which("uv") is not None and _uv_project_root() is not None:
		return "uv", ["run", "camas", *tail]
	if shutil.which("camas") is not None:
		return "camas", tail
	return sys.executable, ["-m", "camas", *tail]


def _uv_project_root() -> Path | None:
	"""The nearest directory (cwd or an ancestor) holding a ``uv.lock``."""
	cwd = Path.cwd()
	return next((d for d in (cwd, *cwd.parents) if (d / "uv.lock").is_file()), None)


def _portability_note(command: str) -> str:
	"""Whether the chosen command is committable, or an absolute per-machine path."""
	if command == sys.executable:
		return "This absolute command is specific to this machine, not portable."
	return "This entry is portable; commit it to share the server with your team."


def _load(path: Path) -> dict[str, Any] | None:
	"""Parse an existing ``.mcp.json`` (``{}`` if absent); ``None`` if not an object."""
	if not path.exists():
		return {}
	try:
		loaded: object = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return None
	return cast("dict[str, Any]", loaded) if isinstance(loaded, dict) else None
