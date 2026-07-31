# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Guards that the tiered camas-fixer agent and gate skill templates shipped in the wheel are
well-formed: each agent gates via the MCP tool (not bare CLI), carries the correct frontmatter,
budgets enough turns to finish the workflow it mandates, and the skill documents the escalation
ladder and the Stop-hook nudge."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TEMPLATES = _REPO / "src" / "camas" / "main"

_AGENT_TEMPLATES = (
	("claude_agent_lint_haiku.md", "camas-lint-fixer-haiku", "haiku"),
	("claude_agent_lint_sonnet.md", "camas-lint-fixer-sonnet", "sonnet"),
	("claude_agent_test_fixer.md", "camas-test-fixer", "sonnet"),
)


_MANDATED_TURNS = (
	("claude_agent_lint_haiku.md", 5),
	("claude_agent_lint_sonnet.md", 5),
	("claude_agent_test_fixer.md", 7),
)
"""``(template, turns its own instructions mandate)``: the conditional ``camas_gate``, the ``Read``
that Edit's contract requires first, the edit, ``camas_fix``, and the final report — plus, for the
test tier, the second read its diagnosis step calls for and its closing re-gate. Each step depends
on the previous one's result, so each costs an assistant turn; ``maxTurns`` caps those turns and
stops the agent *before* its final message, so a budget at or below this count silently hands the
delegating agent no report at all."""


def _max_turns(filename: str) -> int:
	return next(
		int(line.removeprefix("maxTurns:"))
		for line in (_TEMPLATES / filename).read_text().splitlines()
		if line.startswith("maxTurns:")
	)


@pytest.mark.parametrize(("filename", "mandated"), _MANDATED_TURNS)
def test_agent_template_budgets_more_turns_than_its_own_steps_mandate(
	filename: str, mandated: int
) -> None:
	assert _max_turns(filename) > mandated


def test_the_two_structurally_identical_lint_tiers_budget_the_same_turns() -> None:
	assert _max_turns("claude_agent_lint_haiku.md") == _max_turns("claude_agent_lint_sonnet.md")


@pytest.mark.parametrize(("filename", "name", "model"), _AGENT_TEMPLATES)
def test_agent_template_gates_via_mcp_tool_not_bare_cli(
	filename: str, name: str, model: str
) -> None:
	agent = (_TEMPLATES / filename).read_text()
	assert "camas_gate" in agent
	assert "camas mcp gate" not in agent


@pytest.mark.parametrize(("filename", "name", "model"), _AGENT_TEMPLATES)
def test_agent_template_has_correct_frontmatter(filename: str, name: str, model: str) -> None:
	agent = (_TEMPLATES / filename).read_text()
	assert f"name: {name}" in agent
	assert f"model: {model}" in agent
	assert "tools: Read, Edit, mcp__camas__camas_gate, mcp__camas__camas_fix" in agent


def test_skill_template_has_correct_frontmatter_name() -> None:
	skill = (_TEMPLATES / "claude_gate_skill.md").read_text()
	assert "name: gate" in skill


def test_skill_template_documents_the_escalation_ladder_and_stop_nudge() -> None:
	skill = (_TEMPLATES / "claude_gate_skill.md").read_text()
	assert "camas-lint-fixer-haiku" in skill
	assert "camas-lint-fixer-sonnet" in skill
	assert "camas-test-fixer" in skill
	assert "Stop" in skill
