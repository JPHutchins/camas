# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``camas mcp init`` command: write this project's ``.mcp.json`` entry for the camas server."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import sys
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.timings import ensure_camas_dir
from ..v0.config import DEFAULT_CAMAS_DIR

if TYPE_CHECKING:
	from collections.abc import Mapping

SERVER_NAME: Final = "camas"
SETTINGS_PATH: Final = Path(".claude/settings.json")

Launcher = Literal["uv", "uvx", "camas"]
LAUNCHERS: Final[tuple[Launcher, ...]] = ("uv", "uvx", "camas")

_RELEASE_VERSION_RE: Final = re.compile(r"\d+(?:\.\d+)*")


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


def pep723_tasks_py_in_cwd() -> Path | None:
	"""``Path.cwd() / "tasks.py"`` if it exists and has a PEP 723 header with a ``camas``
	dependency, otherwise ``None``.
	"""
	tasks_py = Path.cwd() / "tasks.py"
	if not tasks_py.is_file():
		return None
	from ..main.pep723 import camas_requirement_from

	if camas_requirement_from(tasks_py) is not None:
		return tasks_py
	return None


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


def write_mcp_json(argv: list[str], *, launcher: Launcher | None = None) -> int:
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
	resolved = launch_command(pin=pin, launcher=launcher)
	if resolved is None:
		print(no_launcher_error(f"a portable {mcp_json_path}", launcher), file=sys.stderr)
		return 2
	command, args = resolved
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


def installed_version_spec(installed: str) -> str:
	"""The ``uvx`` fallback spec pinned to the running camas ``installed`` version when it is a
	clean release, else the unpinned extra (a dev/local build isn't published on PyPI to pin).

	>>> installed_version_spec("0.1.18")
	'camas[mcp]==0.1.18'

	>>> installed_version_spec("0.1.22.dev3+g09f0fca")
	'camas[mcp]'
	"""
	return f"camas[mcp]=={installed}" if _RELEASE_VERSION_RE.fullmatch(installed) else "camas[mcp]"


def uvx_spec(pin: str | None) -> str:
	"""The ``uvx`` launch spec: ``pin`` (a PEP 723-derived requirement) verbatim when given, else
	the running camas version pinned when it's a clean release, else the unpinned extra.
	"""
	return pin if pin is not None else installed_version_spec(version("camas"))


def uv_command(tail: list[str]) -> tuple[str, list[str]] | None:
	"""The ``uv`` launch command: the lockfile project path, else PEP 723 ``tasks.py``, else
	``None`` when neither applies.
	"""
	if uv_project_root() is not None:
		return "uv", ["run", "camas", *tail]
	tasks_py = pep723_tasks_py_in_cwd()
	if tasks_py is not None:
		return "uv", ["run", tasks_py.name, *tail]
	return None


def launch_command(
	*, pin: str | None = None, launcher: Launcher | None = None
) -> tuple[str, list[str]] | None:
	"""The most portable launch command for camas, or None if none is portable enough to commit.

	``launcher`` forces a specific strategy instead of the auto-probe (``uv`` errors unless a
	uv.lock project or PEP 723 ``tasks.py`` applies; ``camas`` errors unless it's on PATH).
	"""
	tail = ["mcp"]
	if launcher == "uv":
		return uv_command(tail) if shutil.which("uv") is not None else None
	if launcher == "uvx":
		return ("uvx", [uvx_spec(pin), *tail]) if shutil.which("uvx") is not None else None
	if launcher == "camas":
		return ("camas", tail) if shutil.which("camas") is not None else None
	if shutil.which("uv") is not None:
		found = uv_command(tail)
		if found is not None:
			return found
	if shutil.which("uvx") is not None:
		return "uvx", [uvx_spec(pin), *tail]
	if shutil.which("camas") is not None:
		return "camas", tail
	return None


def no_launcher_error(target: str, launcher: Launcher | None) -> str:
	"""The actionable stderr line for when no launcher can be resolved to write ``target``."""
	if launcher == "uv":
		return (
			f"error: cannot write {target} — --launcher uv requires uv on PATH and either a "
			"uv.lock project or a PEP 723 tasks.py in this directory.\n  Add one, or drop "
			"--launcher to auto-detect."
		)
	if launcher == "uvx":
		return (
			f"error: cannot write {target} — --launcher uvx requires uvx on PATH.\n"
			"  Install uv, or drop --launcher to auto-detect."
		)
	if launcher == "camas":
		return (
			f"error: cannot write {target} — --launcher camas requires camas on PATH.\n"
			"  Install it on PATH, or drop --launcher to auto-detect."
		)
	return (
		f"error: cannot write {target} — no uv.lock found and camas is not on PATH.\n"
		"  Add camas to a uv project (uv add camas) or install it on PATH, then retry."
	)


def uv_project_root() -> Path | None:
	"""The nearest directory (cwd or an ancestor) holding a ``uv.lock``."""
	cwd = Path.cwd()
	return next((d for d in (cwd, *cwd.parents) if (d / "uv.lock").is_file()), None)


def portability_note(command: str) -> str:
	"""How portable the chosen launch command is, for the committed ``.mcp.json``."""
	if command == "uv":
		return "This entry is portable; uv resolves camas from the lockfile or the PEP 723 header in tasks.py."
	if command == "uvx":
		return "This entry is portable; uvx downloads and runs camas[mcp] from PyPI."
	return "This entry is portable if your team installs camas on PATH; commit it to share."


def parse_json_object(path: Path) -> dict[str, Any] | None:
	"""Parse an existing JSON file as a JSON object; ``None`` if unreadable or not one.

	``ValueError`` subsumes ``json.JSONDecodeError`` and the ``UnicodeDecodeError`` an invalid-
	UTF-8 file raises, so both corrupt-file cases fall back cleanly.
	"""
	try:
		loaded: object = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, ValueError):
		return None
	return cast("dict[str, Any]", loaded) if isinstance(loaded, dict) else None


def launch_command_str(*, pin: str | None = None, launcher: Launcher | None = None) -> str | None:
	"""The most portable launch command as a single shell string, or ``None``."""
	pair = launch_command(pin=pin, launcher=launcher)
	if pair is None:
		return None
	cmd, args = pair
	return shlex.join((cmd, *args))


def _object_list(value: object) -> list[dict[str, Any]]:
	"""The value as a list of JSON objects, or ``[]`` if it is not a list."""
	return cast("list[dict[str, Any]]", value) if isinstance(value, list) else []


_CAMAS_HOOK_RE: Final = re.compile(r"\bmcp (?:fix|gate)\b")


def _is_camas_hook(command: str) -> bool:
	"""True when ``command`` is a camas hook invocation — the camas-invented ``mcp fix`` autofix
	or ``mcp gate`` check/nudge subcommand, word-bounded so any launcher form matches (``camas``,
	``uv run tasks.py`` for PEP 723, ``uvx camas[mcp]``) while ``mcp gateway``/``mcp fixture`` do
	not.
	"""
	return _CAMAS_HOOK_RE.search(command) is not None


def _kept_hooks(group: dict[str, Any]) -> list[dict[str, Any]]:
	"""The group's hooks minus any camas hook."""
	return [
		h for h in _object_list(group.get("hooks")) if not _is_camas_hook(cast("str", h["command"]))
	]


def _swept_event(groups: object) -> list[dict[str, Any]]:
	"""One event's hook groups with camas's own hooks removed and any group they emptied
	dropped.
	"""
	return [
		{**g, "hooks": remaining} for g in _object_list(groups) if (remaining := _kept_hooks(g))
	]


def _with_camas_hooks(
	raw: dict[str, Any], groups_by_event: Mapping[str, list[dict[str, Any]]]
) -> dict[str, Any]:
	"""``raw`` with every camas hook removed from every event — dropping any group or event it
	empties, so a stale hook from an older camas (e.g. under ``FileChanged``) or a different event
	is swept out — and ``groups_by_event`` appended under their target events. Every other key
	keeps its position, no defaults injected.
	"""
	hooks = raw.get("hooks")
	hooks_dict: dict[str, Any] = cast("dict[str, Any]", hooks) if isinstance(hooks, dict) else {}
	swept: dict[str, Any] = {
		event: groups
		for event in hooks_dict
		if (groups := _swept_event(hooks_dict[event])) or event in groups_by_event
	}
	return {
		**raw,
		"hooks": {
			**swept,
			**{
				event: [*swept.get(event, ()), *new_groups]
				for event, new_groups in groups_by_event.items()
			},
		},
	}


STOP_NUDGE_UNDER: Final = "5s"
"""The wall-clock budget the async Stop-hook nudge time-boxes its check to — a headless,
turn-settle check must stay fast; a slower project simply reports more untimed/over-budget
leaves rather than delaying the nudge."""


def write_hooks(argv: list[str], *, quiet: bool = False, launcher: Launcher | None = None) -> int:
	"""Write the ``PostToolBatch`` autofix hook and the two ``Stop`` hooks (a settle-time fix,
	and an async check that nudges the main agent to launch the fixer ladder) into
	``.claude/settings.json``.

	``quiet`` suppresses the success message — set when called from ``write_claude``, which
	emits its own consolidated summary so the user sees one reload line, not two.
	"""
	settings_path = Path.cwd() / SETTINGS_PATH
	try:
		raw: object = json.loads(settings_path.read_text(encoding="utf-8"))
	except OSError:
		raw = {}
	except ValueError:
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
	launch_str = launch_command_str(pin=pin, launcher=launcher)
	if launch_str is None:
		print(no_launcher_error("portable hooks", launcher), file=sys.stderr)
		return 2
	fix_hook: dict[str, Any] = {"type": "command", "command": f"{launch_str} fix"}
	nudge_command = f"{launch_str} gate --under {STOP_NUDGE_UNDER} --nudge"
	nudge_hook: dict[str, Any] = {
		"type": "command",
		"command": nudge_command,
		"async": True,
		"asyncRewake": True,
	}
	settings_path.parent.mkdir(parents=True, exist_ok=True)
	settings_path.write_text(
		json.dumps(
			_with_camas_hooks(
				cast("dict[str, Any]", raw),
				{
					"PostToolBatch": [{"hooks": [fix_hook]}],
					"Stop": [{"hooks": [fix_hook, nudge_hook]}],
				},
			),
			indent=2,
		)
		+ "\n",
		encoding="utf-8",
	)
	if not quiet:
		print(
			f"Wrote the camas autofix and Stop hooks to {settings_path}\n"
			f"  PostToolBatch:      {launch_str} fix\n"
			f"  Stop (fix):         {launch_str} fix\n"
			f"  Stop (async nudge): {nudge_command}\n"
			f"\nReload Claude Code for the hooks to take effect."
		)
	return 0


AGENT_TEMPLATES: Final = (
	("claude_agent_lint_haiku.md", "camas-lint-fixer-haiku.md"),
	("claude_agent_lint_sonnet.md", "camas-lint-fixer-sonnet.md"),
	("claude_agent_test_fixer.md", "camas-test-fixer.md"),
)
"""The tiered camas-fixer agent templates, ``(source in src/camas/main/, dest in .claude/agents/)``."""


def write_agent_skill_templates() -> None:
	"""Write camas's Claude Code agent and skill templates into ``.claude/``; idempotent."""
	cwd = Path.cwd()
	agent_dir = cwd / ".claude" / "agents"
	skill_dir = cwd / ".claude" / "skills" / "gate"
	agent_dir.mkdir(parents=True, exist_ok=True)
	skill_dir.mkdir(parents=True, exist_ok=True)
	templates_dir = Path(__file__).parent.parent / "main"
	for source, dest in AGENT_TEMPLATES:
		(agent_dir / dest).write_text(
			(templates_dir / source).read_text(encoding="utf-8"), encoding="utf-8"
		)
	(skill_dir / "SKILL.md").write_text(
		(templates_dir / "claude_gate_skill.md").read_text(encoding="utf-8"), encoding="utf-8"
	)


def write_claude(argv: list[str], *, launcher: Launcher | None = None) -> int:
	"""Write ``.mcp.json``, the autofix/Stop hooks, and the tiered camas-fixer/gate templates.

	Returns 0 on success, 2 on launcher-resolution or validation failure.
	"""
	rc = write_mcp_json(argv, launcher=launcher)
	if rc != 0:
		return rc
	rc = write_hooks(argv, quiet=True, launcher=launcher)
	if rc != 0:
		return rc
	write_agent_skill_templates()
	cwd = Path.cwd()
	agent_dir = cwd / ".claude" / "agents"
	agent_lines = "\n".join(f"Wrote {dest} to {agent_dir / dest}" for _, dest in AGENT_TEMPLATES)
	print(
		f"{agent_lines}\n"
		f"Wrote gate skill to {cwd / '.claude' / 'skills' / 'gate' / 'SKILL.md'}\n"
		f"\nClaude Code is configured. Reload for changes to take effect."
	)
	return 0
