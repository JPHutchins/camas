# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``camas mcp init`` command: write this project's ``.mcp.json`` entry for the camas server."""

from __future__ import annotations

import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from ..core.timings import ensure_camas_dir
from ..v0.config import DEFAULT_CAMAS_DIR

SERVER_NAME: Final = "camas"
SETTINGS_PATH: Final = Path(".claude/settings.json")


class HookCommand(BaseModel):
	"""A single hook command entry in ``.claude/settings.json``."""

	model_config = ConfigDict(extra="forbid")

	type: Literal["command"]
	command: str


class HookGroup(BaseModel):
	"""A group of hooks that fire on the same event, with an optional matcher."""

	model_config = ConfigDict(extra="forbid")

	hooks: list[HookCommand]
	matcher: str | None = None


class SettingsFile(BaseModel):
	"""A ``.claude/settings.json`` file — validated, with extra fields preserved."""

	model_config = ConfigDict(extra="allow")

	hooks: dict[str, list[HookGroup]] = Field(default_factory=dict)


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
	launcher = launch_command(rich="--rich" in argv)
	if launcher is None:
		print(
			f"error: cannot write a portable {mcp_json_path} — no uv.lock found and camas is not "
			"on PATH.\n  Add camas to a uv project (uv add camas) or install it on PATH, then retry.",
			file=sys.stderr,
		)
		return 2
	command, args = launcher
	servers[SERVER_NAME] = {"type": "stdio", "command": command, "args": args}
	mcp_json_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
	camas_dir: Final = Path.cwd() / DEFAULT_CAMAS_DIR
	camas_note: Final = (
		f"  created {camas_dir} for run logs and timing estimates; delete it to opt out.\n"
		if not camas_dir.exists()
		else ""
	)
	ensure_camas_dir(camas_dir)
	print(
		f"Wrote the {SERVER_NAME!r} MCP server to {mcp_json_path}\n"
		f"  command: {command} {' '.join(args)}\n"
		f"{camas_note}"
		f"\n{portability_note(command)} Reload Claude Code, approve the server, "
		"then ask it to call camas_list."
	)
	return 0


def launch_command(*, rich: bool) -> tuple[str, list[str]] | None:
	"""The most portable launch command for camas, or None if none is portable enough to commit."""
	tail = ["mcp", "--rich"] if rich else ["mcp"]
	if shutil.which("uv") is not None and uv_project_root() is not None:
		return "uv", ["run", "camas", *tail]
	if shutil.which("camas") is not None:
		return "camas", tail
	return None


def uv_project_root() -> Path | None:
	"""The nearest directory (cwd or an ancestor) holding a ``uv.lock``."""
	cwd = Path.cwd()
	return next((d for d in (cwd, *cwd.parents) if (d / "uv.lock").is_file()), None)


def portability_note(command: str) -> str:
	"""How portable the chosen launch command is, for the committed ``.mcp.json``."""
	if command == "uv":
		return "This entry is portable; uv resolves camas from the lockfile."
	return "This entry is portable if your team installs camas on PATH; commit it to share."


def parse_json_object(path: Path) -> dict[str, Any] | None:
	"""Parse an existing JSON file as a JSON object; ``None`` if unreadable or not one."""
	try:
		loaded: object = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return None
	return cast("dict[str, Any]", loaded) if isinstance(loaded, dict) else None


def launch_command_str(*, rich: bool) -> str | None:
	"""The most portable launch command as a single shell string, or ``None``."""
	pair = launch_command(rich=rich)
	if pair is None:
		return None
	cmd, args = pair
	return shlex.join((cmd, *args))


def write_hooks(argv: list[str]) -> int:
	"""Write ``FileChanged`` / ``PostToolBatch`` hook entries into ``.claude/settings.json``
	using the same ``launch_command()`` resolution as the MCP server entry.
	"""
	settings_path = Path.cwd() / SETTINGS_PATH
	try:
		raw: object = json.loads(settings_path.read_text(encoding="utf-8"))
	except OSError:
		raw = {}
	except json.JSONDecodeError:
		print(f"error: {settings_path} is not valid JSON", file=sys.stderr)
		return 2
	if not isinstance(raw, dict):
		print(f"error: {settings_path} is not a JSON object", file=sys.stderr)
		return 2
	try:
		settings = SettingsFile.model_validate(raw)
	except Exception as e:
		print(f"error: {settings_path}: {e}", file=sys.stderr)
		return 2
	launcher = launch_command_str(rich="--rich" in argv)
	if launcher is None:
		print(
			"error: cannot write portable hooks — no uv.lock found and camas is not "
			"on PATH.\n  Add camas to a uv project (uv add camas) or install it on PATH, "
			"then retry.",
			file=sys.stderr,
		)
		return 2
	settings.hooks["FileChanged"] = [
		HookGroup(
			hooks=[
				HookCommand(
					type="command",
					command=f"{launcher} fix --paths ${{file_path}}",
				)
			]
		)
	]
	settings.hooks["PostToolBatch"] = [
		HookGroup(
			matcher="Edit|Write|MultiEdit|NotebookEdit",
			hooks=[
				HookCommand(
					type="command",
					command=f"{launcher} gate",
				)
			],
		)
	]
	settings_path.parent.mkdir(parents=True, exist_ok=True)
	settings_path.write_text(settings.model_dump_json(indent=2) + "\n", encoding="utf-8")
	print(
		f"Wrote camas hooks to {settings_path}\n"
		f"  FileChanged:    {launcher} fix --paths ${{file_path}}\n"
		f"  PostToolBatch:  {launcher} gate\n"
		f"\nReload Claude Code for hooks to take effect."
	)
	return 0
