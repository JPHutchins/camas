# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The camas Claude Code plugin is shipped as committed files (distributed via the repo's
marketplace), not unit-tested machinery — so these guard that the static artifacts are
well-formed and self-consistent: the hook targets the real gate tool, and the marketplace
points at the plugin that exists."""

from __future__ import annotations

import json
from pathlib import Path

from camas.mcp.serve import ToolName

_REPO = Path(__file__).resolve().parents[2]
_PLUGIN = _REPO / "agent" / "claude" / "plugin"


def test_plugin_manifest_is_named_camas() -> None:
	assert json.loads((_PLUGIN / ".claude-plugin" / "plugin.json").read_text())["name"] == "camas"


def test_bundled_mcp_server_launches_camas_mcp() -> None:
	server = json.loads((_PLUGIN / ".mcp.json").read_text())["mcpServers"]["camas"]
	assert (server["command"], server["args"]) == ("camas", ["mcp"])


def test_gate_hook_targets_the_real_gate_tool() -> None:
	hooks = json.loads((_PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]
	hook = hooks["PostToolBatch"][0]["hooks"][0]
	assert hook["type"] == "mcp_tool"
	assert hook["server"] == "camas"
	assert hook["tool"] == ToolName.GATE.value


def test_filechanged_hook_runs_the_deterministic_autofix() -> None:
	hooks = json.loads((_PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]
	fc = hooks["FileChanged"][0]["hooks"][0]
	assert fc["type"] == "command"
	assert "camas fix" in fc["command"]


def test_marketplace_points_at_the_shipped_plugin() -> None:
	catalog = json.loads((_REPO / ".claude-plugin" / "marketplace.json").read_text())
	entry = next(p for p in catalog["plugins"] if p["name"] == "camas")
	assert (_REPO / entry["source"]).resolve() == _PLUGIN.resolve()


def test_plugin_ships_the_fixer_and_skill() -> None:
	assert "name: camas-fixer" in (_PLUGIN / "agents" / "camas-fixer.md").read_text()
	assert "name: gate" in (_PLUGIN / "skills" / "gate" / "SKILL.md").read_text()
