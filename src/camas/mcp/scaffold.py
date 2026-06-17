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
	mcp_json_path: Final = Path.cwd() / ".mcp.json"
	existing = parse_json_object(mcp_json_path) if mcp_json_path.exists() else {}
	if existing is None:
		print(f"error: {mcp_json_path} is not a readable JSON object", file=sys.stderr)
		return 2
	servers = existing.setdefault("mcpServers", {})
	if not isinstance(servers, dict):
		print(f"error: {mcp_json_path} has a non-object 'mcpServers'", file=sys.stderr)
		return 2
	command, args = launch_command(rich="--rich" in argv)
	servers[SERVER_NAME] = {"type": "stdio", "command": command, "args": args}
	mcp_json_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
	print(
		f"Wrote the {SERVER_NAME!r} MCP server to {mcp_json_path}\n"
		f"  command: {command} {' '.join(args)}\n\n"
		f"{portability_note(command)} Reload Claude Code, approve the server, "
		"then ask it to call camas_list."
	)
	return 0


def launch_command(*, rich: bool) -> tuple[str, list[str]]:
	"""The most portable launch command for camas in this environment."""
	tail = ["mcp", "--rich"] if rich else ["mcp"]
	if shutil.which("uv") is not None and uv_project_root() is not None:
		return "uv", ["run", "camas", *tail]
	if shutil.which("camas") is not None:
		return "camas", tail
	return sys.executable, ["-m", "camas", *tail]


def uv_project_root() -> Path | None:
	"""The nearest directory (cwd or an ancestor) holding a ``uv.lock``."""
	cwd = Path.cwd()
	return next((d for d in (cwd, *cwd.parents) if (d / "uv.lock").is_file()), None)


def portability_note(command: str) -> str:
	"""How portable the chosen launch command is, for the committed ``.mcp.json``."""
	if command == "uv":
		return "This entry is portable; uv resolves camas from the lockfile."
	if command == sys.executable:
		return "This absolute command is specific to this machine, not portable."
	return "This entry is portable if your team installs camas on PATH; commit it to share."


def parse_json_object(path: Path) -> dict[str, Any] | None:
	"""Parse an existing ``.mcp.json`` as a JSON object; ``None`` if unreadable or not one."""
	try:
		loaded: object = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return None
	return cast("dict[str, Any]", loaded) if isinstance(loaded, dict) else None
