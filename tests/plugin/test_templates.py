# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Guards that the camas-fixer agent and gate skill templates shipped in the wheel are
well-formed: the agent gates via the MCP tool (not bare CLI), and both carry the correct
frontmatter."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TEMPLATES = _REPO / "src" / "camas" / "main"


def test_agent_template_gates_via_mcp_tool_not_bare_cli() -> None:
	agent = (_TEMPLATES / "claude_agent.md").read_text()
	assert "camas_gate" in agent
	assert "camas mcp gate" not in agent


def test_agent_template_has_correct_frontmatter_name() -> None:
	agent = (_TEMPLATES / "claude_agent.md").read_text()
	assert "name: camas-fixer" in agent


def test_skill_template_has_correct_frontmatter_name() -> None:
	skill = (_TEMPLATES / "claude_gate_skill.md").read_text()
	assert "name: gate" in skill
