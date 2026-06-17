# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""``camas mcp init``: write this project's ``.mcp.json`` entry for the camas server.

Stdlib only — no ``mcp``/``pydantic`` — so scaffolding works without the ``[mcp]``
extra and stays off the import path of ``camas mcp`` (serving). The launch command
is chosen for portability: a uv project gets ``uv run camas mcp`` (committable, uses
the project's own camas), a ``camas`` on PATH is used directly, and only as a last
resort does it fall back to this interpreter's absolute path.
"""

from __future__ import annotations

import json
import shutil
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
	"""The most portable launch command for camas in this environment.

	A uv project (``uv.lock`` in cwd or an ancestor) runs ``uv run camas`` —
	committable and bound to the project's camas, not a stale global. Else a
	``camas`` on PATH is used directly. The final fallback runs this interpreter's
	``-m camas`` (an absolute, per-machine path).
	"""
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
