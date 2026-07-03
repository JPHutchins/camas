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

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.timings import ensure_camas_dir
from ..v0.config import DEFAULT_CAMAS_DIR

SERVER_NAME: Final = "camas"
SETTINGS_PATH: Final = Path(".claude/settings.json")


class HookCommand(BaseModel):
	"""A single hook command entry in ``.claude/settings.json``, with extra fields preserved."""

	model_config = ConfigDict(extra="allow")

	type: Literal["command"]
	command: str


class HookGroup(BaseModel):
	"""A group of hooks that fire on the same event, with an optional matcher."""

	model_config = ConfigDict(extra="allow")

	hooks: list[HookCommand]
	matcher: str = ""


class SettingsFile(BaseModel):
	"""A ``.claude/settings.json`` file — validated, with extra fields preserved."""

	model_config = ConfigDict(extra="allow")

	hooks: dict[str, list[HookGroup]] = Field(default_factory=dict)


def tasks_py_path() -> Path | None:
	"""The nearest ``tasks.py`` up from cwd, or ``None`` if none found."""
	cwd = Path.cwd()
	return next((d / "tasks.py" for d in (cwd, *cwd.parents) if (d / "tasks.py").is_file()), None)


def resolve_pin() -> str | None:
	"""The camas MCP requirement with extras for pinning, resolved from the project's ``tasks.py``."""
	tasks_py = tasks_py_path()
	if tasks_py is None:
		return None
	from ..main.pep723 import camas_requirement_from, with_mcp_extra

	req = camas_requirement_from(tasks_py)
	if req is None:
		return None
	return with_mcp_extra(req)


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
	pin = resolve_pin()
	launcher = launch_command(pin=pin)
	if launcher is None:
		print(
			f"error: cannot write a portable {mcp_json_path} — no uv.lock found and camas is not "
			"on PATH.\n  Add camas to a uv project (uv add camas) or install it on PATH, then retry.",
			file=sys.stderr,
		)
		return 2
	command, args = launcher
	servers[SERVER_NAME] = {"type": "stdio", "command": command, "args": args}
	camas_dir: Final = Path.cwd() / DEFAULT_CAMAS_DIR
	camas_note: Final = (
		f"  created {camas_dir} for run logs and timing estimates; delete it to opt out.\n"
		if not camas_dir.exists()
		else ""
	)
	try:
		ensure_camas_dir(camas_dir)
	except OSError as exc:
		print(f"warning: cannot create {camas_dir}: {exc}", file=sys.stderr)
	mcp_json_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
	print(
		f"Wrote the {SERVER_NAME!r} MCP server to {mcp_json_path}\n"
		f"  command: {command} {' '.join(args)}\n"
		f"{camas_note}"
		f"\n{portability_note(command)} Reload Claude Code, approve the server, "
		"then ask it to call camas_list."
	)
	return 0


def launch_command(*, pin: str | None = None) -> tuple[str, list[str]] | None:
	"""The most portable launch command for camas, or None if none is portable enough to commit."""
	tail = ["mcp"]
	if shutil.which("uv") is not None and uv_project_root() is not None:
		return "uv", ["run", "camas", *tail]
	if shutil.which("uvx") is not None:
		spec = pin if pin is not None else "camas[mcp]"
		return "uvx", [spec, *tail]
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
	if command == "uvx":
		return "This entry is portable; uvx downloads and runs camas[mcp] from PyPI."
	return "This entry is portable if your team installs camas on PATH; commit it to share."


def parse_json_object(path: Path) -> dict[str, Any] | None:
	"""Parse an existing JSON file as a JSON object; ``None`` if unreadable or not one."""
	try:
		loaded: object = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return None
	return cast("dict[str, Any]", loaded) if isinstance(loaded, dict) else None


def launch_command_str(*, pin: str | None = None) -> str | None:
	"""The most portable launch command as a single shell string, or ``None``."""
	pair = launch_command(pin=pin)
	if pair is None:
		return None
	cmd, args = pair
	return shlex.join((cmd, *args))


def _object_list(value: object) -> list[dict[str, Any]]:
	"""The value as a list of JSON objects, or ``[]`` if it is not a list."""
	return cast("list[dict[str, Any]]", value) if isinstance(value, list) else []


def _is_camas_hook(command: str) -> bool:
	"""True when ``command`` is a camas autofix-hook invocation — the launcher (containing
	``camas``) plus the ``mcp fix`` subcommand, possibly with trailing ``--paths`` args.
	"""
	return "camas" in command and "mcp fix" in command


def _kept_hooks(group: dict[str, Any]) -> list[dict[str, Any]]:
	"""The group's hooks minus any camas autofix hook."""
	return [
		h for h in _object_list(group.get("hooks")) if not _is_camas_hook(cast("str", h["command"]))
	]


def _swept_event(groups: object) -> list[dict[str, Any]]:
	"""One event's hook groups with camas's autofix hooks removed and any group they emptied
	dropped.
	"""
	return [
		{**g, "hooks": remaining} for g in _object_list(groups) if (remaining := _kept_hooks(g))
	]


def _with_camas_hook(raw: dict[str, Any], camas_group: dict[str, Any]) -> dict[str, Any]:
	"""``raw`` with camas's own autofix hook removed from every event — dropping any group or event
	it empties, so a stale hook from an older camas (e.g. under ``FileChanged``) is swept out — and
	the current hook appended under ``PostToolBatch``. Every other key keeps its position, no
	defaults injected.
	"""
	hooks = raw.get("hooks")
	hooks_dict: dict[str, Any] = cast("dict[str, Any]", hooks) if isinstance(hooks, dict) else {}
	swept: dict[str, Any] = {
		event: groups
		for event in hooks_dict
		if (groups := _swept_event(hooks_dict[event])) or event == "PostToolBatch"
	}
	return {
		**raw,
		"hooks": {**swept, "PostToolBatch": [*swept.get("PostToolBatch", []), camas_group]},
	}


def write_hooks(argv: list[str], *, quiet: bool = False) -> int:
	"""Write the ``PostToolBatch`` autofix hook into ``.claude/settings.json``.

	``quiet`` suppresses the success message — set when called from ``write_claude``, which
	emits its own consolidated summary so the user sees one reload line, not two.
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
		SettingsFile.model_validate(raw)
	except ValidationError as e:
		print(f"error: {settings_path}: {e}", file=sys.stderr)
		return 2
	pin = resolve_pin()
	launcher = launch_command_str(pin=pin)
	if launcher is None:
		print(
			"error: cannot write portable hooks — no uv.lock found and camas is not "
			"on PATH.\n  Add camas to a uv project (uv add camas) or install it on PATH, "
			"then retry.",
			file=sys.stderr,
		)
		return 2
	camas_group: dict[str, Any] = {"hooks": [{"type": "command", "command": f"{launcher} fix"}]}
	settings_path.parent.mkdir(parents=True, exist_ok=True)
	settings_path.write_text(
		json.dumps(_with_camas_hook(cast("dict[str, Any]", raw), camas_group), indent=2) + "\n",
		encoding="utf-8",
	)
	if not quiet:
		print(
			f"Wrote the camas PostToolBatch autofix hook to {settings_path}\n"
			f"  PostToolBatch:  {launcher} fix\n"
			f"\nReload Claude Code for the hook to take effect."
		)
	return 0


def write_agent_skill_templates() -> None:
	"""Write camas's Claude Code agent and skill templates into ``.claude/``; idempotent."""
	cwd = Path.cwd()
	agent_dir = cwd / ".claude" / "agents"
	skill_dir = cwd / ".claude" / "skills" / "gate"
	agent_dir.mkdir(parents=True, exist_ok=True)
	skill_dir.mkdir(parents=True, exist_ok=True)
	templates_dir = Path(__file__).parent.parent / "main"
	(agent_dir / "camas-fixer.md").write_text(
		(templates_dir / "claude_agent.md").read_text(encoding="utf-8"), encoding="utf-8"
	)
	(skill_dir / "SKILL.md").write_text(
		(templates_dir / "claude_gate_skill.md").read_text(encoding="utf-8"), encoding="utf-8"
	)


def write_claude(argv: list[str]) -> int:
	"""Write ``.mcp.json``, the PostToolBatch autofix hook, and the camas-fixer/gate templates.

	Returns 0 on success, 2 on launcher-resolution or validation failure.
	"""
	rc = write_mcp_json(argv)
	if rc != 0:
		return rc
	rc = write_hooks(argv, quiet=True)
	if rc != 0:
		return rc
	write_agent_skill_templates()
	cwd = Path.cwd()
	print(
		f"Wrote camas-fixer agent to {cwd / '.claude' / 'agents' / 'camas-fixer.md'}\n"
		f"Wrote gate skill to {cwd / '.claude' / 'skills' / 'gate' / 'SKILL.md'}\n"
		f"\nClaude Code is configured. Reload for changes to take effect."
	)
	return 0
